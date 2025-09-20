"""
Main server application for AI game playing
"""
import os
import signal
import threading
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import base64
import logging
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Optional
from flask import Flask, request, jsonify, send_file, Response
import time
from flask_cors import CORS
import numpy as np
from PIL import Image

# Import configuration
try:
    from ...config import *
except ImportError:
    # Default configuration if config.py is not found
    HOST = "0.0.0.0"
    PORT = 5000
    DEBUG = True
    DEFAULT_EMULATOR = "gb"
    SCREEN_CAPTURE_FORMAT = "jpeg"
    SCREEN_CAPTURE_QUALITY = 85
    DEFAULT_AI_API = "gemini"
    AI_REQUEST_TIMEOUT = 30
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "{asctime} - {name} - {levelname} - {message}"
    LOG_FILE = "ai_game_server.log"
    MAX_ROM_SIZE = 100 * 1024 * 1024
    ALLOWED_ROM_EXTENSIONS = [".gb", ".gbc", ".gba"]
    ACTION_HISTORY_LIMIT = 1000

# Use absolute imports instead of relative imports
from emulators.pyboy_emulator import PyBoyEmulator
from emulators.pygba_emulator import PyGBAEmulator
from ai_apis.ai_provider_manager import ai_provider_manager

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format=LOG_FORMAT,
    style='{',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Configure CORS with specific settings for frontend
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176", "http://127.0.0.1:5173", "http://127.0.0.1:5174", "http://127.0.0.1:5175", "http://127.0.0.1:5176"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "max_age": 86400,
        "send_wildcard": False
    }
})

# Import and register RL API (conditional import to avoid breaking server if RL fails)
try:
    from api.rl_api import rl_bp
    app.register_blueprint(rl_bp)
    logger.info("RL API registered successfully")
except ImportError as e:
    logger.warning(f"Failed to import RL API: {e}. RL features will not be available.")

# Global emulator instances
emulators = {
    "gb": PyBoyEmulator(),
    "gba": PyGBAEmulator()
}

# AI provider manager is imported from ai_provider_manager

# Action history
action_history = []

# Game state
game_state = {
    "active_emulator": None,
    "rom_loaded": False,
    "ai_running": False,
    "current_goal": ""
}

def timeout_handler(timeout_seconds):
    """Decorator to add timeout handling to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def target():
                return func(*args, **kwargs)

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(target)
                try:
                    return future.result(timeout=timeout_seconds)
                except FutureTimeoutError:
                    logger.error(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
                    raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
        return wrapper
    return decorator

def numpy_to_base64_image(np_array: np.ndarray) -> str:
    """Convert numpy array to base64 encoded JPEG image using proper PyBoy connectors"""
    try:
        # Quick validation
        if np_array is None or np_array.size == 0:
            logger.error("Invalid numpy array: None or empty")
            return ""

        # Create a copy to avoid modifying the original array
        np_array = np_array.copy()

        # Handle different PyBoy formats more efficiently
        if len(np_array.shape) == 3:
            if np_array.shape[2] == 4:
                np_array = np_array[:, :, :3]  # RGBA to RGB
            elif np_array.shape[2] != 3:
                logger.error(f"Unsupported channels: {np_array.shape[2]}")
                return ""
        elif len(np_array.shape) == 2:
            np_array = np.stack([np_array] * 3, axis=2)  # Grayscale to RGB
        else:
            logger.error(f"Unsupported array shape: {np_array.shape}")
            return ""

        # Fast data type conversion and validation
        if np_array.dtype != np.uint8:
            if np_array.dtype in [np.float32, np.float64]:
                np_array = np.clip(np_array * 255, 0, 255)
            np_array = np_array.astype(np.uint8)

        # Ensure values are in valid range
        if np_array.min() < 0 or np_array.max() > 255:
            np_array = np.clip(np_array, 0, 255)

        # Create and optimize image efficiently
        try:
            image = Image.fromarray(np_array, mode='RGB')

            # Use smaller buffer and faster settings for streaming
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG', quality=75, optimize=False, progressive=False)
            img_buffer.seek(0)

            img_bytes = img_buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            return img_base64

        except Exception as e:
            logger.error(f"Image creation/conversion failed: {e}")
            return ""

    except Exception as e:
        logger.error(f"Error in numpy_to_base64_image: {e}")
        return ""

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint for monitoring"""
    return jsonify({"status": "healthy"}), 200

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get comprehensive status of the server"""
    status = game_state.copy()
    status['ai_providers'] = ai_provider_manager.get_provider_status()
    return jsonify(status), 200

@app.route('/api/providers/status', methods=['GET'])
def get_providers_status():
    """Get detailed status of all AI providers"""
    return jsonify(ai_provider_manager.get_provider_status()), 200

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get a list of available models for a given provider"""
    provider_name = request.args.get('provider')
    if not provider_name:
        return jsonify({"error": "Provider name is required"}), 400

    models = ai_provider_manager.get_models(provider_name)
    return jsonify({"models": models}), 200

@app.route('/api/upload-rom', methods=['POST'])
def upload_rom():
    """Upload a ROM file and load it into the specified emulator"""
    logger.info("=== ROM UPLOAD REQUEST RECEIVED ===")

    try:
        if 'rom_file' not in request.files:
            return jsonify({"error": "No ROM file provided"}), 400

        file = request.files['rom_file']
        emulator_type = request.form.get('emulator_type', 'gb')
        launch_ui = request.form.get('launch_ui', 'true').lower() == 'true'

        logger.info(f"File received: {file.filename}")
        logger.info(f"Emulator type: {emulator_type}")
        logger.info(f"Launch UI: {launch_ui}")
        logger.info(f"Available emulators: {list(emulators.keys())}")

        if file.filename == '':
            logger.error("No filename provided")
            return jsonify({"error": "No ROM file selected"}), 400

        _, ext = os.path.splitext(file.filename)
        logger.info(f"File extension: {ext}")
        if ext.lower() not in ALLOWED_ROM_EXTENSIONS:
            logger.error(f"Invalid extension: {ext}")
            return jsonify({"error": f"Invalid file extension. Allowed: {ALLOWED_ROM_EXTENSIONS}"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            file.save(temp_file.name)
            temp_rom_path = temp_file.name

        logger.info(f"ROM saved to temporary path: {temp_rom_path}")

        if emulator_type not in emulators:
            logger.error(f"Invalid emulator type: {emulator_type}")
            os.unlink(temp_rom_path)
            return jsonify({"error": f"Invalid emulator type. Available: {list(emulators.keys())}"}), 400

        logger.info(f"Loading ROM into {emulator_type} emulator...")
        success = emulators[emulator_type].load_rom(temp_rom_path)

        if success:
            game_state["active_emulator"] = emulator_type
            game_state["rom_loaded"] = True
            logger.info(f"=== ROM LOADED SUCCESSFULLY ===")
            logger.info(f"ROM: {file.filename}")
            logger.info(f"Emulator: {emulator_type}")
            logger.info(f"Temp path: {temp_rom_path}")

            emulator = emulators[emulator_type]
            if hasattr(emulator, 'pyboy') and emulator.pyboy:
                for _ in range(100):
                    emulator.pyboy.tick()

            # UI is now launched automatically by the emulator
            ui_status = emulator.get_ui_status() if hasattr(emulator, 'get_ui_status') else {"running": False}

            response_data = {
                "message": "ROM loaded successfully",
                "rom_name": file.filename,
                "ui_launched": ui_status.get("running", False),
                "ui_status": ui_status
            }

            # Check if UI auto-launch was attempted but failed
            auto_launch_enabled = ui_status.get("auto_launch_enabled", True)
            ui_process_running = ui_status.get("running", False)

            if not ui_process_running and auto_launch_enabled and launch_ui:
                logger.warning("UI process failed to launch automatically")
                # Add helpful information for manual UI launch
                response_data["ui_help"] = {
                    "message": "Automatic UI launch failed. You can:",
                    "actions": [
                        "Try launching UI manually using the UI control panel",
                        "Check if PyBoy is properly installed: pip install pyboy",
                        "Verify SDL2 libraries are available on your system"
                    ]
                }

            return jsonify(response_data), 200
        else:
            os.unlink(temp_rom_path)
            return jsonify({"error": "Failed to load ROM"}), 500

    except Exception as e:
        logger.error(f"Error uploading ROM: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/screen', methods=['GET'])
def get_screen():
    """Get the current screen from the active emulator using proper PyBoy connectors"""
    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[game_state["active_emulator"]]

        # Use proper PyBoy screen API - get screen directly without stepping
        screen_array = emulator.get_screen()

        # Validate screen data
        if screen_array is None or screen_array.size == 0:
            logger.error("Screen data is None or empty")
            return jsonify({"error": "Failed to capture screen"}), 500

        # Convert to base64 using optimized function
        img_base64 = numpy_to_base64_image(screen_array)

        if not img_base64:
            logger.error("Failed to convert screen to base64")
            return jsonify({"error": "Failed to process screen image"}), 500

        return jsonify({
            "image": img_base64,
            "shape": screen_array.shape,
            "timestamp": time.time(),
            "pyboy_frame": emulator.get_frame_count() if hasattr(emulator, 'get_frame_count') else None
        }), 200

    except Exception as e:
        logger.error(f"Error getting screen: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/screen/debug', methods=['GET'])
def get_screen_debug():
    """Debug endpoint to test PyBoy screen capture functionality"""
    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[game_state["active_emulator"]]

        # Get PyBoy info
        info = emulator.get_info() if hasattr(emulator, 'get_info') else {}

        # Get screen array
        screen_array = emulator.get_screen()

        debug_info = {
            "pyboy_info": info,
            "screen_shape": screen_array.shape if screen_array is not None else None,
            "screen_dtype": str(screen_array.dtype) if screen_array is not None else None,
            "screen_min": int(screen_array.min()) if screen_array is not None else None,
            "screen_max": int(screen_array.max()) if screen_array is not None else None,
            "screen_size": screen_array.size if screen_array is not None else None,
            "timestamp": time.time(),
            "emulator_type": game_state["active_emulator"]
        }

        # Try base64 conversion
        if screen_array is not None and screen_array.size > 0:
            img_base64 = numpy_to_base64_image(screen_array)
            debug_info["base64_success"] = img_base64 is not None and len(img_base64) > 0
            debug_info["base64_length"] = len(img_base64) if img_base64 else 0
            debug_info["image"] = img_base64 if debug_info["base64_success"] else None
        else:
            debug_info["base64_success"] = False
            debug_info["error"] = "Screen array is None or empty"

        return jsonify(debug_info), 200

    except Exception as e:
        logger.error(f"Error in screen debug: {e}", exc_info=True)
        return jsonify({"error": f"Debug error: {str(e)}"}), 500

@app.route('/api/stream', methods=['GET'])
def stream_screen():
    """SSE endpoint for live screen streaming using proper PyBoy connectors"""
    def generate():
        logger.info("SSE stream requested. Checking game state...")
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            logger.warning("SSE stream aborted: No ROM loaded.")
            yield f"data: {json.dumps({'error': 'No ROM loaded'})}\n\n"
            return

        emulator = emulators[game_state["active_emulator"]]
        logger.info(f"Starting SSE stream for {game_state['active_emulator']}.")

        frame_count = 0
        last_frame_time = time.time()
        target_fps = 20  # Reduced to 20 FPS for better stability
        frame_interval = 1.0 / target_fps

        # Dynamic frame rate adjustment
        frame_times = []
        max_frame_times = 30  # Keep last 30 frames for average
        min_fps = 10
        max_fps = 30

        # Performance monitoring
        pyboy_timeout_count = 0
        max_pyboy_timeouts = 3

        # Send initial frame immediately using proper PyBoy API
        try:
            # Use PyBoy's screen API directly - no stepping needed for initial frame
            screen_array = emulator.get_screen()

            # Validate screen data
            if screen_array is None or screen_array.size == 0:
                logger.warning("Initial screen data is invalid, using placeholder")
                import numpy as np
                screen_array = np.zeros((144, 160, 3), dtype=np.uint8)

            # Convert to base64 using proper PyBoy format
            img_base64 = numpy_to_base64_image(screen_array)

            if not img_base64:
                logger.error("Failed to convert initial screen to base64")
                yield f"data: {json.dumps({'error': 'Failed to process initial frame'})}\n\n"
                return

            initial_data = {
                'image': img_base64,
                'shape': screen_array.shape,
                'timestamp': time.time(),
                'frame': frame_count,
                'fps': target_fps,
                'status': 'stream_started'
            }
            yield f"data: {json.dumps(initial_data)}\n\n"
            frame_count += 1
            logger.info("Initial frame sent successfully")

        except Exception as e:
            logger.error(f"Error sending initial frame: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': f'Initial frame error: {str(e)}'})}\n\n"
            return

        # Main streaming loop with proper PyBoy integration
        consecutive_errors = 0
        max_consecutive_errors = 5
        last_pyboy_error = None

        # Create thread-safe executor for timeout handling
        with ThreadPoolExecutor(max_workers=2) as executor:
            last_client_activity = time.time()
            connection_timeout = 60  # 60 seconds without client response

            while True:
                try:
                    # Calculate timing for consistent frame rate
                    current_time = time.time()
                    elapsed = current_time - last_frame_time

                    # Only process frame if enough time has passed
                    if elapsed >= frame_interval:
                        try:
                            # Execute PyBoy operations with timeout protection
                            def step_emulator():
                                return emulator.step('NOOP', 1)

                            def get_screen():
                                return emulator.get_screen()

                            # Timeout-protected PyBoy calls with adaptive timeouts
                            step_timeout = 0.05 if pyboy_timeout_count == 0 else 0.1
                            try:
                                success = executor.submit(step_emulator).result(timeout=step_timeout)
                                pyboy_timeout_count = 0  # Reset timeout counter on success
                            except FutureTimeoutError:
                                pyboy_timeout_count += 1
                                logger.warning(f"PyBoy step timeout (count: {pyboy_timeout_count}), using last frame")
                                if pyboy_timeout_count >= max_pyboy_timeouts:
                                    raise TimeoutError(f"Too many PyBoy timeouts: {pyboy_timeout_count}")
                                raise TimeoutError("PyBoy step timeout")

                            screen_timeout = 0.05 if pyboy_timeout_count == 0 else 0.1
                            try:
                                screen_array = executor.submit(get_screen).result(timeout=screen_timeout)
                            except FutureTimeoutError:
                                logger.warning(f"PyBoy get_screen timeout (count: {pyboy_timeout_count}), using placeholder")
                                screen_array = np.zeros((144, 160, 3), dtype=np.uint8)

                        # Validate screen data - if invalid, use placeholder
                        if screen_array is None or screen_array.size == 0:
                            import numpy as np
                            screen_array = np.zeros((144, 160, 3), dtype=np.uint8)
                            logger.warning(f"Stream frame {frame_count}: Using placeholder screen (PyBoy API returned empty)")

                        # Additional validation for proper PyBoy format
                        if len(screen_array.shape) != 3 or screen_array.shape[2] not in [3, 4]:
                            logger.warning(f"Stream frame {frame_count}: Invalid shape {screen_array.shape}, correcting")
                            screen_array = np.zeros((144, 160, 3), dtype=np.uint8)

                        # Convert to base64 using timeout-protected function
                        def convert_image():
                            return numpy_to_base64_image(screen_array)

                        try:
                            img_base64 = executor.submit(convert_image).result(timeout=0.05)
                        except FutureTimeoutError:
                            logger.warning("Image conversion timeout")
                            img_base64 = None

                        if not img_base64:
                            # Send heartbeat without image if conversion failed
                            data = {
                                'heartbeat': True,
                                'frame': frame_count,
                                'timestamp': current_time,
                                'fps': target_fps,
                                'status': 'no_image',
                                'error': 'Screen conversion failed',
                                'elapsed_ms': round((time.time() - current_time) * 1000, 2)
                            }
                        else:
                            # Send normal frame with PyBoy metadata and timing info
                            frame_process_time = (time.time() - current_time) * 1000
                            data = {
                                'image': img_base64,
                                'shape': screen_array.shape,
                                'timestamp': current_time,
                                'frame': frame_count,
                                'fps': target_fps,
                                'actual_interval': elapsed,
                                'process_time_ms': round(frame_process_time, 2),
                                'pyboy_frame': emulator.get_frame_count() if hasattr(emulator, 'get_frame_count') else frame_count,
                                'status': 'streaming'
                            }

                        yield f"data: {json.dumps(data)}\n\n"

                        # Update counters and reset error counter on success
                        frame_count += 1
                        last_frame_time = current_time
                        consecutive_errors = 0

                        # Log progress every 60 frames (2 seconds at 30fps)
                        if frame_count % 60 == 0:
                            logger.info(f"Streamed frame {frame_count} successfully using PyBoy API")

                    except Exception as inner_e:
                        consecutive_errors += 1
                        last_pyboy_error = str(inner_e)
                        logger.error(f"Error processing stream frame {frame_count}: {inner_e} (consecutive errors: {consecutive_errors})", exc_info=True)

                        # Enhanced error recovery based on error type
                        if consecutive_errors >= max_consecutive_errors:
                            logger.warning(f"Too many consecutive errors ({consecutive_errors}), attempting recovery...")

                            # Send detailed recovery heartbeat
                            recovery_data = {
                                'heartbeat': True,
                                'frame': frame_count,
                                'timestamp': current_time,
                                'fps': target_fps,
                                'status': 'recovery_attempt',
                                'consecutive_errors': consecutive_errors,
                                'last_error': last_pyboy_error,
                                'message': 'Attempting to recover from stream errors',
                                'recovery_action': 'resetting_pyboy_state'
                            }
                            yield f"data: {json.dumps(recovery_data)}\n\n"

                            # Perform actual recovery
                            try:
                                # Attempt to reset PyBoy state
                                emulator.reset()
                                time.sleep(0.2)  # Allow time for reset
                                consecutive_errors = 0
                                logger.info("PyBoy state reset successful")
                            except Exception as reset_error:
                                logger.error(f"PyBoy reset failed: {reset_error}")
                                # If reset fails, send final error and break
                                error_data = {
                                    'heartbeat': True,
                                    'frame': frame_count,
                                    'timestamp': current_time,
                                    'fps': target_fps,
                                    'status': 'fatal_error',
                                    'consecutive_errors': consecutive_errors,
                                    'message': 'Recovery failed - stream terminated',
                                    'error': last_pyboy_error
                                }
                                yield f"data: {json.dumps(error_data)}\n\n"
                                break
                        else:
                            # Send detailed error heartbeat but continue streaming
                            error_data = {
                                'heartbeat': True,
                                'frame': frame_count,
                                'timestamp': current_time,
                                'fps': target_fps,
                                'error': str(inner_e),
                                'status': 'error_continuing',
                                'consecutive_errors': consecutive_errors,
                                'pyboy_status': 'PyBoy API error'
                            }
                            yield f"data: {json.dumps(error_data)}\n\n"

                        frame_count += 1
                        last_frame_time = current_time
                        continue

                else:
                    # Enhanced connection health monitoring
                    time_since_last_activity = current_time - last_client_activity

                    # Check for connection timeout
                    if time_since_last_activity > connection_timeout:
                        logger.warning(f"Connection timeout after {connection_timeout}s, closing stream")
                        timeout_data = {
                            'heartbeat': True,
                            'frame': frame_count,
                            'timestamp': current_time,
                            'fps': target_fps,
                            'status': 'connection_timeout',
                            'message': 'Client connection timed out'
                        }
                        yield f"data: {json.dumps(timeout_data)}\n\n"
                        break

                    # Send heartbeat every 3 seconds to keep connection alive during idle periods
                    if int(current_time) % 3 == 0 and int(current_time) != int(last_frame_time):
                        heartbeat_data = {
                            'heartbeat': True,
                            'frame': frame_count,
                            'timestamp': current_time,
                            'fps': target_fps,
                            'status': 'idle_heartbeat',
                            'message': 'Stream waiting for next frame interval',
                            'uptime': round(time_since_last_activity, 1),
                            'current_fps': target_fps,
                            'avg_frame_time': round(sum(frame_times) / len(frame_times) * 1000, 2) if frame_times else 0
                        }
                        yield f"data: {json.dumps(heartbeat_data)}\n\n"

                    # Dynamic frame rate adjustment based on performance
                    frame_process_time = (time.time() - current_time)
                    frame_times.append(frame_process_time)

                    if len(frame_times) > max_frame_times:
                        frame_times.pop(0)

                    # Adjust frame rate based on performance
                    if len(frame_times) >= 10:  # Only adjust after collecting enough data
                        avg_frame_time = sum(frame_times) / len(frame_times)
                        if avg_frame_time > frame_interval * 1.5:  # Taking too long
                            if target_fps > min_fps:
                                target_fps = max(min_fps, target_fps - 1)
                                frame_interval = 1.0 / target_fps
                                logger.info(f"Reducing FPS to {target_fps} due to slow performance")
                        elif avg_frame_time < frame_interval * 0.7:  # Processing quickly
                            if target_fps < max_fps:
                                target_fps = min(max_fps, target_fps + 1)
                                frame_interval = 1.0 / target_fps
                                logger.info(f"Increasing FPS to {target_fps} due to good performance")

                    # Sleep for the remaining time to maintain frame rate
                    remaining_time = frame_interval - elapsed
                    if remaining_time > 0:
                        time.sleep(remaining_time * 0.8)  # Sleep less to account for processing time

            except GeneratorExit:
                logger.info("SSE stream client disconnected gracefully.")
                break
            except KeyboardInterrupt:
                logger.info("SSE stream interrupted by user.")
                break
            except Exception as e:
                logger.error(f"Unexpected error in SSE stream: {e}", exc_info=True)
                # Send final error message before breaking
                try:
                    error_data = {
                        'heartbeat': True,
                        'frame': frame_count,
                        'timestamp': time.time(),
                        'fps': target_fps,
                        'status': 'stream_crashed',
                        'message': 'Stream encountered unrecoverable error',
                        'error': str(e)
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                except:
                    pass
                break
            finally:
                # Cleanup resources
                logger.info(f"SSE stream cleanup. Total frames: {frame_count}, Final FPS: {target_fps}")
                if 'executor' in locals():
                    executor.shutdown(wait=False)
            except Exception as e:
                logger.error(f"Critical error in SSE stream: {e}", exc_info=True)
                # Send final error and break
                error_data = {
                    'error': str(e),
                    'frame': frame_count,
                    'timestamp': time.time(),
                    'fps': target_fps,
                    'status': 'stream_ending',
                    'pyboy_status': 'PyBoy API failure'
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                break

    response = Response(generate(), mimetype='text/event-stream')
    # Add SSE-specific headers
    response.headers.add('Cache-Control', 'no-cache')
    response.headers.add('Connection', 'keep-alive')
    response.headers.add('X-Accel-Buffering', 'no')  # Disable buffering in nginx
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Cache-Control')
    return response

@app.route('/api/action', methods=['POST', 'OPTIONS'])
def execute_action():
    """Execute an action in the active emulator"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            logger.warning("Action requested but no ROM loaded")
            return jsonify({"error": "No ROM loaded"}), 400

        data = request.json
        if not data:
            logger.error("No JSON data received in action request")
            return jsonify({"error": "No request data provided"}), 400

        action = data.get('action', 'SELECT')
        frames = data.get('frames', 1)

        # Validate action
        valid_actions = {'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT', 'NOOP'}
        if action not in valid_actions:
            logger.warning(f"Invalid action requested: {action}")
            return jsonify({"error": f"Invalid action: {action}"}), 400

        logger.info(f"Executing action: {action} for {frames} frame(s)")
        emulator = emulators[game_state["active_emulator"]]
        success = emulator.step(action, frames)

        if success:
            action_history.append(action)
            logger.debug(f"Action {action} executed successfully")
            return jsonify({"message": "Action executed successfully"}), 200
        else:
            logger.error(f"Failed to execute action: {action}")
            return jsonify({"error": "Failed to execute action"}), 500

    except Exception as e:
        logger.error(f"Error executing action: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

def get_ai_connector(api_name, api_key, api_endpoint):
    """Get an AI connector using the provider manager."""
    # Note: We now use the global ai_provider_manager for provider management
    # This function is kept for backward compatibility
    try:
        # Set environment variables for this request if provided
        if api_key:
            if api_name == 'gemini':
                os.environ['GEMINI_API_KEY'] = api_key
            elif api_name == 'openrouter':
                os.environ['OPENROUTER_API_KEY'] = api_key
            elif api_name == 'openai-compatible':
                os.environ['OPENAI_API_KEY'] = api_key
                if api_endpoint:
                    os.environ['OPENAI_ENDPOINT'] = api_endpoint
            elif api_name == 'nvidia':
                os.environ['NVIDIA_API_KEY'] = api_key

        # Get the connector from the provider manager
        connector = ai_provider_manager.get_provider(api_name)
        if connector:
            logger.debug(f"Successfully obtained connector for {api_name}")
        else:
            logger.warning(f"Failed to obtain connector for {api_name}")
        return connector
    except Exception as e:
        logger.error(f"Error getting AI connector for {api_name}: {e}")
        return None

@app.route('/api/ai-action', methods=['POST', 'OPTIONS'])
def get_ai_action():
    """Get the next action from the specified AI API with automatic fallback"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            logger.warning("AI action requested but no ROM loaded")
            return jsonify({"error": "No ROM loaded"}), 400

        data = request.json
        if not data:
            logger.error("No JSON data received in AI action request")
            return jsonify({"error": "No request data provided"}), 400

        api_name = data.get('api_name')
        api_key = data.get('api_key')
        api_endpoint = data.get('api_endpoint')
        model = data.get('model')
        goal = data.get('goal', '')

        logger.info(f"AI action request: api={api_name or 'auto'}, model={model or 'default'}, goal='{goal}'")
        game_state["current_goal"] = goal

        emulator = emulators[game_state["active_emulator"]]

        # Get screen using proper PyBoy connectors
        screen_array = emulator.get_screen()

        if screen_array is None or screen_array.size == 0:
            logger.error("Failed to get screen from PyBoy emulator")
            return jsonify({"error": "Failed to capture screen"}), 500

        # Convert screen to bytes using proper PyBoy format handling
        img_bytes = emulator.get_screen_bytes()

        if len(img_bytes) == 0:
            logger.error("Failed to convert PyBoy screen to bytes")
            return jsonify({"error": "Failed to process screen image"}), 500

        logger.debug(f"PyBoy screen captured for AI: {len(img_bytes)} bytes")

        # Set environment variables for this request if provided
        if api_key:
            if api_name == 'gemini':
                os.environ['GEMINI_API_KEY'] = api_key
            elif api_name == 'openrouter':
                os.environ['OPENROUTER_API_KEY'] = api_key
            elif api_name == 'openai-compatible':
                os.environ['OPENAI_API_KEY'] = api_key
                if api_endpoint:
                    os.environ['OPENAI_ENDPOINT'] = api_endpoint
            elif api_name == 'nvidia':
                os.environ['NVIDIA_API_KEY'] = api_key
        if model:
            if api_name == 'openai-compatible':
                os.environ['OPENAI_MODEL'] = model
            elif api_name == 'nvidia':
                os.environ['NVIDIA_MODEL'] = model

        # Check if the requested provider is available
        if api_name and api_name not in ai_provider_manager.get_available_providers():
            available_providers = ai_provider_manager.get_available_providers()
            logger.warning(f"Requested provider '{api_name}' is not available. Available providers: {available_providers}")

            # If specific provider is requested but not available, return error with suggestions
            return jsonify({
                "error": f"Provider '{api_name}' is not available",
                "available_providers": available_providers,
                "suggestion": f"Please use one of the available providers: {', '.join(available_providers)}"
            }), 400

        # Use provider manager with automatic fallback
        logger.debug(f"Calling AI API: {api_name or 'auto'} with model: {model or 'default'}")
        action, actual_provider = ai_provider_manager.get_next_action(
            img_bytes, goal, action_history, api_name, model
        )

        action_history.append(action)

        logger.info(f"AI ({actual_provider or 'fallback'}) suggested action: {action}")
        return jsonify({
            "action": action,
            "provider_used": actual_provider,
            "history": action_history[-10:]
        }), 200

    except Exception as e:
        logger.error(f"Error getting AI action: {e}", exc_info=True)
        api_name = data.get('api_name') if 'data' in locals() else 'unknown'
        goal = data.get('goal', '') if 'data' in locals() else 'unknown'
        logger.error(f"AI action failed - API: {api_name}, Goal: '{goal}', Screen size: {len(img_bytes) if 'img_bytes' in locals() else 'unknown'}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def ai_chat():
    """Send a message to the AI and get a response with automatic fallback"""
    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        data = request.json
        user_message = data.get('message', '')
        api_name = data.get('api_name')
        api_key = data.get('api_key')
        api_endpoint = data.get('api_endpoint')
        model = data.get('model')

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        emulator = emulators[game_state["active_emulator"]]

        # Get screen bytes using proper PyBoy connectors
        img_bytes = emulator.get_screen_bytes()

        if len(img_bytes) == 0:
            logger.error("Failed to get screen bytes from PyBoy for chat")
            return jsonify({"error": "Failed to capture screen"}), 500

        logger.debug(f"PyBoy screen captured for chat: {len(img_bytes)} bytes")

        context = {
            "current_goal": game_state["current_goal"],
            "action_history": action_history[-20:],
            "game_type": game_state["active_emulator"].upper()
        }

        # Get model/provider with priority: game_state -> request params -> defaults
        current_provider = game_state.get('current_provider')
        current_model = game_state.get('current_model')

        # Use request parameters if provided, otherwise use stored values
        api_name = api_name or current_provider
        model = model or current_model

        logger.debug(f"Chat using provider: {api_name or 'auto'}, model: {model or 'default'}")
        logger.debug(f"Stored provider: {current_provider}, stored model: {current_model}")
        # Set environment variables for this request if provided
        if api_key:
            if api_name == 'gemini':
                os.environ['GEMINI_API_KEY'] = api_key
            elif api_name == 'openrouter':
                os.environ['OPENROUTER_API_KEY'] = api_key
            elif api_name == 'openai-compatible':
                os.environ['OPENAI_API_KEY'] = api_key
                if api_endpoint:
                    os.environ['OPENAI_ENDPOINT'] = api_endpoint
            elif api_name == 'nvidia':
                os.environ['NVIDIA_API_KEY'] = api_key
        if model:
            if api_name == 'openai-compatible':
                os.environ['OPENAI_MODEL'] = model
            elif api_name == 'nvidia':
                os.environ['NVIDIA_MODEL'] = model

        # Use provider manager with automatic fallback
        response_text, actual_provider = ai_provider_manager.chat_with_ai(
            user_message, img_bytes, context, api_name, model
        )

        logger.info(f"AI chat message from user: {user_message} (provider: {actual_provider or 'fallback'})")
        return jsonify({
            "response": response_text,
            "provider_used": actual_provider
        }), 200

    except Exception as e:
        logger.error(f"Error in AI chat: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/ui/launch', methods=['POST'])
def launch_ui():
    """Launch UI process for the current ROM"""
    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[game_state["active_emulator"]]

        if not hasattr(emulator, 'launch_ui'):
            return jsonify({"error": "UI control not supported by this emulator"}), 400

        success = emulator.launch_ui()

        if success:
            ui_status = emulator.get_ui_status()
            logger.info("UI process launched successfully")
            return jsonify({
                "message": "UI launched successfully",
                "ui_status": ui_status
            }), 200
        else:
            logger.error("Failed to launch UI process")
            return jsonify({"error": "Failed to launch UI"}), 500

    except Exception as e:
        logger.error(f"Error launching UI: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/ui/stop', methods=['POST'])
def stop_ui():
    """Stop the UI process"""
    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[game_state["active_emulator"]]

        if not hasattr(emulator, 'stop_ui'):
            return jsonify({"error": "UI control not supported by this emulator"}), 400

        success = emulator.stop_ui()

        if success:
            logger.info("UI process stopped successfully")
            return jsonify({"message": "UI stopped successfully"}), 200
        else:
            logger.error("Failed to stop UI process")
            return jsonify({"error": "Failed to stop UI"}), 500

    except Exception as e:
        logger.error(f"Error stopping UI: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/ui/restart', methods=['POST'])
def restart_ui():
    """Restart the UI process"""
    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[game_state["active_emulator"]]

        if not hasattr(emulator, 'restart_ui'):
            return jsonify({"error": "UI control not supported by this emulator"}), 400

        success = emulator.restart_ui()

        if success:
            ui_status = emulator.get_ui_status()
            logger.info("UI process restarted successfully")
            return jsonify({
                "message": "UI restarted successfully",
                "ui_status": ui_status
            }), 200
        else:
            logger.error("Failed to restart UI process")
            return jsonify({"error": "Failed to restart UI"}), 500

    except Exception as e:
        logger.error(f"Error restarting UI: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/ui/status', methods=['GET'])
def get_ui_status():
    """Get UI process status"""
    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[game_state["active_emulator"]]

        if not hasattr(emulator, 'get_ui_status'):
            return jsonify({"error": "UI control not supported by this emulator"}), 400

        ui_status = emulator.get_ui_status()

        return jsonify({
            "ui_status": ui_status,
            "rom_loaded": game_state["rom_loaded"],
            "active_emulator": game_state["active_emulator"]
        }), 200

    except Exception as e:
        logger.error(f"Error getting UI status: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/ui/sync', methods=['POST'])
def sync_ui():
    """Synchronize server state with UI process"""
    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[game_state["active_emulator"]]

        if not hasattr(emulator, 'sync_with_ui'):
            return jsonify({"error": "UI control not supported by this emulator"}), 400

        success = emulator.sync_with_ui()

        if success:
            logger.info("Server-UI synchronization completed")
            return jsonify({"message": "Synchronization completed"}), 200
        else:
            logger.warning("Server-UI synchronization failed or not supported")
            return jsonify({"error": "Synchronization failed"}), 500

    except Exception as e:
        logger.error(f"Error syncing with UI: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/audio/status', methods=['GET'])
def get_audio_status():
    """Get audio system status"""
    try:
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[game_state["active_emulator"]]

        # Check if emulator has audio support
        audio_status = {
            "enabled": hasattr(emulator, 'sound_enabled') and emulator.sound_enabled,
            "sample_rate": getattr(emulator, 'sample_rate', 48000),
            "buffer_size": 0,
            "volume": getattr(emulator, 'volume', 50)
        }

        return jsonify(audio_status), 200

    except Exception as e:
        logger.error(f"Error getting audio status: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/audio/stream', methods=['GET'])
def stream_audio():
    """SSE endpoint for live audio streaming"""
    def generate_audio():
        logger.info("Audio stream requested. Checking game state...")
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            logger.warning("Audio stream aborted: No ROM loaded.")
            yield f"data: {json.dumps({'error': 'No ROM loaded'})}\n\n"
            return

        emulator = emulators[game_state["active_emulator"]]

        # Check if audio is available
        if not hasattr(emulator, 'get_audio_buffer'):
            logger.warning("Audio stream aborted: Audio not supported.")
            yield f"data: {json.dumps({'error': 'Audio not supported'})}\n\n"
            return

        logger.info("Starting audio stream...")
        chunk_count = 0
        last_chunk_time = time.time()

        while True:
            try:
                current_time = time.time()

                # Send audio chunks at ~50Hz (20ms intervals)
                if current_time - last_chunk_time >= 0.02:
                    try:
                        # Get audio buffer from emulator
                        audio_data = emulator.get_audio_buffer()

                        if audio_data is not None and len(audio_data) > 0:
                            # Convert to base64 for streaming
                            audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')

                            chunk_data = {
                                'audio_data': audio_b64,
                                'timestamp': current_time,
                                'chunk': chunk_count,
                                'size': len(audio_data),
                                'sample_rate': getattr(emulator, 'sample_rate', 48000)
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                            chunk_count += 1

                        # Send heartbeat every 5 seconds
                        if chunk_count % 250 == 0:
                            heartbeat = {
                                'heartbeat': True,
                                'timestamp': current_time,
                                'chunks_sent': chunk_count,
                                'status': 'streaming_healthy'
                            }
                            yield f"data: {json.dumps(heartbeat)}\n\n"

                        last_chunk_time = current_time

                    except Exception as inner_e:
                        logger.error(f"Error processing audio chunk {chunk_count}: {inner_e}")
                        error_data = {
                            'error': str(inner_e),
                            'chunk': chunk_count,
                            'timestamp': current_time,
                            'status': 'continuing'
                        }
                        yield f"data: {json.dumps(error_data)}\n\n"
                        chunk_count += 1
                        last_chunk_time = current_time
                        continue
                else:
                    # Small sleep to prevent CPU overload
                    time.sleep(0.001)

            except GeneratorExit:
                logger.info("Audio stream client disconnected.")
                break
            except Exception as e:
                logger.error(f"Critical error in audio stream: {e}")
                try:
                    error_data = {
                        'error': str(e),
                        'timestamp': time.time(),
                        'status': 'stream_ending'
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                except:
                    pass
                break

    return Response(generate_audio(), mimetype='text/event-stream')


@app.route('/api/chat-stream', methods=['POST'])
def chat_stream():
    """SSE endpoint for streaming AI chat responses"""
    def generate_chat_stream():
        try:
            data = request.json
            if not data:
                yield f"data: {json.dumps({'error': 'No data provided'})}\n\n"
                return

            message = data.get('message', '')
            if not message:
                yield f"data: {json.dumps({'error': 'No message provided'})}\n\n"
                return

            # Get chat parameters with priority: game_state -> request params -> defaults
            current_provider = game_state.get('current_provider')
            current_model = game_state.get('current_model')

            # Use request parameters if provided, otherwise use stored values
            api_name = data.get('api_name') or current_provider
            model = data.get('model') or current_model
            context = data.get('context', {})

            logger.info(f"Chat stream request: message='{message[:50]}...', provider={api_name or 'auto'}, model={model or 'default'}")
            logger.debug(f"Stored provider: {current_provider}, stored model: {current_model}")

            # Send initial acknowledgment
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Processing your request...'})}\n\n"

            # Get AI response with fallback
            if not game_state["rom_loaded"] or not game_state["active_emulator"]:
                response_text = "Please load a ROM first to use AI chat features."
                actual_provider = None
            else:
                emulator = emulators[game_state["active_emulator"]]

                # Get screen bytes using proper PyBoy connectors for streaming
                img_bytes = emulator.get_screen_bytes()

                if len(img_bytes) == 0:
                    logger.error("Failed to get screen bytes from PyBoy for chat stream")
                    response_text = "Failed to capture screen for AI analysis."
                    actual_provider = None
                else:
                    logger.debug(f"PyBoy screen captured for chat stream: {len(img_bytes)} bytes")

                # Update context with game state information
                context.update({
                    "current_goal": game_state.get("current_goal", ""),
                    "action_history": action_history[-20:],
                    "game_type": game_state.get("active_emulator", "").upper()
                })

                response_text, actual_provider = ai_provider_manager.chat_with_ai(
                    message, img_bytes, context, api_name, model
                )

            # Stream the response in chunks
            words = response_text.split()
            chunk_size = 3  # Send 3 words at a time

            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                chunk_data = {
                    'chunk': chunk,
                    'is_final': i + chunk_size >= len(words),
                    'provider': actual_provider,
                    'timestamp': time.time()
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                time.sleep(0.05)  # Small delay for streaming effect

            # Send completion signal
            yield f"data: {json.dumps({'status': 'completed', 'provider': actual_provider})}\n\n"

        except Exception as e:
            logger.error(f"Error in chat stream: {e}")
            yield f"data: {json.dumps({'error': str(e), 'status': 'error'})}\n\n"

    return Response(generate_chat_stream(), mimetype='text/event-stream')


@app.route('/api/ai/provider/set', methods=['POST'])
def set_current_provider():
    """Set the current AI provider and model"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        provider = data.get('provider')
        model = data.get('model')

        if not provider:
            return jsonify({"error": "Provider name is required"}), 400

        # Validate provider exists
        if provider not in ai_provider_manager.get_available_providers():
            return jsonify({"error": f"Provider '{provider}' not available"}), 400

        # Store current provider/model in game state
        game_state['current_provider'] = provider
        game_state['current_model'] = model

        logger.info(f"Current provider set to: {provider}, model: {model}")

        return jsonify({
            "success": True,
            "provider": provider,
            "model": model,
            "message": f"Provider set to {provider}"
        }), 200

    except Exception as e:
        logger.error(f"Error setting provider: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/ai/provider/current', methods=['GET'])
def get_current_provider():
    """Get the currently selected AI provider and model"""
    try:
        current_provider = game_state.get('current_provider')
        current_model = game_state.get('current_model')

        return jsonify({
            "provider": current_provider,
            "model": current_model,
            "available_providers": ai_provider_manager.get_available_providers()
        }), 200

    except Exception as e:
        logger.error(f"Error getting current provider: {e}")
        return jsonify({"error": "Internal server error"}), 500


def main():
    """Main entry point for the server"""
    logger.info("=== STARTING AI GAME SERVER ===")

    # Check PyBoy availability
    try:
        from pyboy import PyBoy
        logger.info("[OK] PyBoy is available")
    except ImportError:
        logger.error("[ERROR] PyBoy is NOT available - install with 'pip install pyboy'")

    # Check SDL2 availability
    try:
        import sdl2
        logger.info("[OK] SDL2 is available")
    except ImportError:
        logger.warning("[WARN] SDL2 is not available - UI may not work")

    logger.info(f"Available AI providers: {ai_provider_manager.get_available_providers()}")

    if not ai_provider_manager.get_available_providers():
        logger.warning("[WARNING] No AI providers are available. AI features will be limited.")
        logger.info("To enable AI features, set the appropriate environment variables:")
        logger.info("  - GEMINI_API_KEY")
        logger.info("  - OPENROUTER_API_KEY")
        logger.info("  - NVIDIA_API_KEY (optional: NVIDIA_MODEL)")
        logger.info("  - OPENAI_API_KEY (optional: OPENAI_ENDPOINT for local providers)")
        logger.info("  - LM_STUDIO_URL (for local LM Studio instance)")
        logger.info("  - OLLAMA_URL (for local Ollama instance)")
        logger.info("Note: For local providers like LM Studio, you may not need an API key.")
    else:
        logger.info(f"[SUCCESS] {len(ai_provider_manager.get_available_providers())} AI provider(s) are ready for use:")
        for provider_name in ai_provider_manager.get_available_providers():
            logger.info(f"  - {provider_name}")

    app.run(host=HOST, port=PORT, debug=DEBUG)

if __name__ == '__main__':
    main()
