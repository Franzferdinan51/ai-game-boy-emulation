"""
Enhanced AI Game Boy Server with Stream Stability Fixes
"""
import os
import json
import time
import base64
import io
import logging
import signal
import threading
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
game_state = {
    "rom_loaded": False,
    "active_emulator": None,
    "rom_path": None,
    "fps": 60,
    "speed_multiplier": 1.0
}

# Import emulators
from emulators.pyboy_emulator import PyBoyEmulator
from emulators.pygba_emulator import PyGBAEmulator
from ai_apis.ai_provider_manager import AIProviderManager

# Initialize emulators
emulators = {
    "pyboy": PyBoyEmulator(),
    "pygba": PyGBAEmulator()
}

# Initialize AI provider manager
ai_provider_manager = AIProviderManager()

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

# Import Flask
from flask import Flask, request, jsonify, Response, stream_with_context

app = Flask(__name__)

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

@app.route('/api/tetris/train', methods=['POST'])
def train_tetris_ai():
    """Train the Tetris genetic AI"""
    try:
        data = request.get_json()
        population_size = data.get('population_size', 20)
        generations = data.get('generations', 5)

        # Get Tetris AI provider
        tetris_ai = ai_provider_manager.providers.get('tetris-genetic', {}).get('connector')

        if not tetris_ai:
            return jsonify({
                'success': False,
                'error': 'Tetris genetic AI not available'
            }), 400

        # Check if ROM is loaded and emulator is available
        if not game_state["rom_loaded"] or not game_state["active_emulator"]:
            return jsonify({
                'success': False,
                'error': 'No ROM loaded'
            }), 400

        emulator = emulators[game_state["active_emulator"]]

        # Start training
        logger.info(f"Starting Tetris AI training: population_size={population_size}, generations={generations}")

        # Train in a separate thread to avoid blocking
        def train_async():
            try:
                results = tetris_ai.train_generation(emulator, population_size, generations)
                logger.info(f"Tetris AI training completed: {results}")
            except Exception as e:
                logger.error(f"Tetris AI training failed: {e}")

        import threading
        training_thread = threading.Thread(target=train_async)
        training_thread.daemon = True
        training_thread.start()

        return jsonify({
            'success': True,
            'message': 'Tetris AI training started',
            'population_size': population_size,
            'generations': generations,
            'provider_status': tetris_ai.get_status()
        })

    except Exception as e:
        logger.error(f"Error starting Tetris AI training: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/tetris/status', methods=['GET'])
def get_tetris_status():
    """Get Tetris genetic AI status"""
    try:
        tetris_ai = ai_provider_manager.providers.get('tetris-genetic', {}).get('connector')

        if not tetris_ai:
            return jsonify({
                'available': False,
                'error': 'Tetris genetic AI not available'
            })

        return jsonify({
            'success': True,
            'status': tetris_ai.get_status()
        })

    except Exception as e:
        logger.error(f"Error getting Tetris AI status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/tetris/save', methods=['POST'])
def save_tetris_model():
    """Save Tetris AI model"""
    try:
        data = request.get_json()
        filepath = data.get('filepath')

        if not filepath:
            return jsonify({
                'success': False,
                'error': 'Filepath required'
            }), 400

        tetris_ai = ai_provider_manager.providers.get('tetris-genetic', {}).get('connector')

        if not tetris_ai:
            return jsonify({
                'success': False,
                'error': 'Tetris genetic AI not available'
            }), 400

        success = tetris_ai.save_training_state(filepath)

        return jsonify({
            'success': success,
            'message': 'Model saved successfully' if success else 'Failed to save model'
        })

    except Exception as e:
        logger.error(f"Error saving Tetris model: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/tetris/load', methods=['POST'])
def load_tetris_model():
    """Load Tetris AI model"""
    try:
        data = request.get_json()
        filepath = data.get('filepath')

        if not filepath:
            return jsonify({
                'success': False,
                'error': 'Filepath required'
            }), 400

        tetris_ai = ai_provider_manager.providers.get('tetris-genetic', {}).get('connector')

        if not tetris_ai:
            return jsonify({
                'success': False,
                'error': 'Tetris genetic AI not available'
            }), 400

        success = tetris_ai.load_training_state(filepath)

        return jsonify({
            'success': success,
            'message': 'Model loaded successfully' if success else 'Failed to load model'
        })

    except Exception as e:
        logger.error(f"Error loading Tetris model: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stream', methods=['GET'])
def stream_screen():
    """SSE endpoint for live screen streaming with stability fixes"""
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
                    # Update client activity timestamp
                    last_client_activity = time.time()

            # Cleanup resources
            logger.info(f"SSE stream cleanup. Total frames: {frame_count}, Final FPS: {target_fps}")
            if 'executor' in locals():
                executor.shutdown(wait=False)

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

def main():
    """Main server function"""
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

if __name__ == "__main__":
    main()