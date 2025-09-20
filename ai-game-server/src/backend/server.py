"""
Enhanced AI Game Boy Server with Stream Stability Fixes
"""
import os
import signal
import threading
import io
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import base64
import logging
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Optional
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
import time
from flask_cors import CORS
import numpy as np
from PIL import Image

# Performance optimization imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configuration (from server_original.py)
try:
    from ...config import *
except ImportError:
    # Default configuration if config.py is not found
    HOST = "0.0.0.0"
    PORT = 5000
    # Security: Debug mode should be disabled in production
    DEBUG = os.environ.get('FLASK_DEBUG', 'false').lower() in ('true', '1', 'yes', 'on')
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

    # Security settings
    # Rate limiting (requests per minute)
    RATE_LIMIT = int(os.environ.get('RATE_LIMIT', 60))
    # File upload size limit (bytes)
    MAX_UPLOAD_SIZE = int(os.environ.get('MAX_UPLOAD_SIZE', 50 * 1024 * 1024))  # 50MB
    # Allowed hostnames
    ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')
    # Secret key for sessions
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here-change-in-production')

# Set up logging
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

# Configure CORS with permissive settings for development
CORS(app, resources={
    r"/*": {
        "origins": ["*"],  # Allow all origins for development
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
    }
})

# Combined security middleware
@app.before_request
def security_middleware():
    """Combined security middleware for host validation and rate limiting"""
    # Skip validation for local development
    if DEBUG:
        return

    # Host validation
    host = request.host.split(':')[0]  # Remove port
    if host not in [h.strip() for h in ALLOWED_HOSTS]:
        logger.warning(f"Unauthorized host access attempt: {host}")
        return jsonify({"error": "Unauthorized"}), 403

    # Rate limiting
    client_ip = get_client_ip()
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return jsonify({
            "error": "Rate limit exceeded. Please try again later.",
            "retry_after": 60
        }), 429

# Security middleware for HTTP headers
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
    return response

# Import emulators
from emulators.pyboy_emulator import PyBoyEmulator, PyBoyEmulatorMP
from emulators.pygba_emulator import PyGBAEmulator
from ai_apis.ai_provider_manager import ai_provider_manager

# Configuration for emulator mode
USE_MULTI_PROCESS = os.environ.get('USE_MULTI_PROCESS', 'false').lower() == 'true'

# Initialize emulators based on configuration
if USE_MULTI_PROCESS:
    logger.info("Using multi-process emulator mode")
    emulators = {
        "pyboy": PyBoyEmulatorMP(),
        "pygba": PyGBAEmulator()  # GBA emulator doesn't have MP mode yet
    }
else:
    logger.info("Using single-process emulator mode")
    emulators = {
        "pyboy": PyBoyEmulator(),
        "pygba": PyGBAEmulator()
    }

# Thread-safe global state management
game_state_lock = threading.Lock()
action_history_lock = threading.Lock()

# Action history
action_history = []

# Game state
game_state = {
    "active_emulator": None,
    "rom_loaded": False,
    "ai_running": False,
    "current_goal": "",
    "rom_path": None,
    "fps": 60,
    "speed_multiplier": 1.0,
    "current_provider": None,
    "current_model": None
}

# AI provider manager is imported from ai_provider_manager

def get_game_state():
    """Thread-safe getter for game state"""
    with game_state_lock:
        return game_state.copy()

def update_game_state(updates):
    """Thread-safe updater for game state"""
    with game_state_lock:
        game_state.update(updates)

def get_action_history():
    """Thread-safe getter for action history"""
    with action_history_lock:
        return action_history.copy()

def add_to_action_history(action):
    """Thread-safe method to add to action history"""
    with action_history_lock:
        action_history.append(action)
        # Keep history within limits
        if len(action_history) > ACTION_HISTORY_LIMIT:
            action_history.pop(0)

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

def validate_string_input(value: str, field_name: str, min_length: int = 0, max_length: int = 1000,
                         allowed_chars: str = None, pattern: str = None) -> str:
    """Validate string input with comprehensive checks"""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    if len(value) < min_length:
        raise ValueError(f"{field_name} must be at least {min_length} characters long")

    if len(value) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters long")

    # Check for null bytes and other dangerous characters
    if '\x00' in value:
        raise ValueError(f"{field_name} contains null bytes")

    # Check for potential SQL injection patterns
    sql_patterns = ["'", '"', ';', '--', '/*', '*/', 'xp_', 'exec(', 'union ', 'select ', 'insert ', 'update ', 'delete ']
    value_lower = value.lower()
    for sql_pattern in sql_patterns:
        if sql_pattern in value_lower:
            raise ValueError(f"{field_name} contains potentially dangerous characters")

    # Check for path traversal attempts
    if '../' in value or '..\\' in value:
        raise ValueError(f"{field_name} contains path traversal attempts")

    # Check for command injection patterns
    cmd_patterns = ['|', '&', ';', '`', '$(', '&&', '||', '>', '<', '>>']
    for cmd_pattern in cmd_patterns:
        if cmd_pattern in value:
            raise ValueError(f"{field_name} contains command injection patterns")

    # Custom character validation
    if allowed_chars:
        if not all(c in allowed_chars for c in value):
            raise ValueError(f"{field_name} contains invalid characters")

    # Regex pattern validation
    if pattern:
        import re
        if not re.match(pattern, value):
            raise ValueError(f"{field_name} does not match required pattern")

    return value.strip()

def validate_integer_input(value, field_name: str, min_value: int = None, max_value: int = None) -> int:
    """Validate integer input with range checks"""
    try:
        if isinstance(value, str):
            # Remove whitespace
            value = value.strip()
            # Check for octal/hex that could be dangerous
            if value.startswith(('0o', '0x', '0b')):
                raise ValueError(f"{field_name} cannot use numeric prefixes")
            int_value = int(value)
        elif isinstance(value, int):
            int_value = value
        else:
            raise ValueError(f"{field_name} must be an integer")
    except (ValueError, TypeError):
        raise ValueError(f"{field_name} must be a valid integer")

    if min_value is not None and int_value < min_value:
        raise ValueError(f"{field_name} must be at least {min_value}")

    if max_value is not None and int_value > max_value:
        raise ValueError(f"{field_name} must be at most {max_value}")

    return int_value

def validate_file_upload(file_obj, field_name: str, allowed_extensions: list = None,
                        max_size: int = None, content_types: list = None) -> dict:
    """Validate file upload with comprehensive security checks"""
    if not hasattr(file_obj, 'filename') or not hasattr(file_obj, 'save'):
        raise ValueError(f"Invalid {field_name}: not a valid file object")

    filename = file_obj.filename
    if not filename or filename == '':
        raise ValueError(f"No {field_name} filename provided")

    # Check for dangerous filenames
    dangerous_patterns = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|', '\x00']
    for pattern in dangerous_patterns:
        if pattern in filename:
            raise ValueError(f"{field_name} filename contains dangerous characters")

    # Validate file extension
    if allowed_extensions:
        _, ext = os.path.splitext(filename)
        if ext.lower() not in allowed_extensions:
            raise ValueError(f"Invalid {field_name} extension. Allowed: {allowed_extensions}")

    # Check file size if content length is available
    if max_size and hasattr(file_obj, 'content_length') and file_obj.content_length:
        if file_obj.content_length > max_size:
            raise ValueError(f"{field_name} too large. Maximum size: {max_size} bytes")

    # Additional security checks for file content
    if hasattr(file_obj, 'stream') and hasattr(file_obj.stream, 'read'):
        try:
            # Reset file pointer and read first few bytes for magic number detection
            file_obj.stream.seek(0)
            file_header = file_obj.stream.read(1024)  # Read first 1KB
            file_obj.stream.seek(0)  # Reset pointer

            # Check for potentially dangerous file types by magic numbers
            dangerous_magic_numbers = {
                b'\x7fELF': 'ELF executable',
                b'MZ': 'Windows executable',
                b'#!': 'Script file',
                b'<html': 'HTML file',
                b'<?xml': 'XML file',
                b'%PDF': 'PDF file',
                b'\x1f\x8b': 'GZIP file',
                b'PK\x03\x04': 'ZIP file'
            }

            for magic, desc in dangerous_magic_numbers.items():
                if file_header.startswith(magic):
                    # Only allow if explicitly permitted
                    if content_types is None or desc not in content_types:
                        raise ValueError(f"{field_name} appears to be a {desc}, which is not allowed")

        except Exception as e:
            logger.warning(f"Could not validate {field_name} content: {e}")

    return {
        'filename': filename,
        'extension': os.path.splitext(filename)[1].lower(),
        'size': getattr(file_obj, 'content_length', 0)
    }

def validate_json_data(data: dict, required_fields: list = None, optional_fields: list = None,
                     field_validators: dict = None) -> dict:
    """Validate JSON data structure and fields"""
    if not isinstance(data, dict):
        raise ValueError("Request data must be a JSON object")

    validated_data = {}

    # Check required fields
    if required_fields:
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

    # Validate all fields
    for field_name, field_value in data.items():
        # Skip if field is not expected
        if required_fields and field_name not in required_fields:
            if optional_fields and field_name not in optional_fields:
                logger.warning(f"Unexpected field in request: {field_name}")
                continue

        # Apply field-specific validation
        if field_validators and field_name in field_validators:
            validator_func = field_validators[field_name]
            try:
                validated_data[field_name] = validator_func(field_value)
            except ValueError as e:
                raise ValueError(f"Invalid {field_name}: {str(e)}")
        else:
            validated_data[field_name] = field_value

    return validated_data

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and other attacks"""
    # Remove path components
    filename = os.path.basename(filename)

    # Remove dangerous characters
    dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|', '\x00']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')

    # Remove leading/trailing whitespace and dots
    filename = filename.strip('. ')

    # Ensure filename is not empty
    if not filename:
        filename = "uploaded_file"

    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext

    return filename

def secure_file_operation(operation_func, *args, **kwargs):
    """Execute file operations with proper error handling and security"""
    try:
        return operation_func(*args, **kwargs)
    except (OSError, IOError) as e:
        logger.error(f"File operation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in file operation: {e}")
        raise

def create_secure_temp_file(suffix: str = '', prefix: str = 'temp_', directory: str = None) -> str:
    """Create a secure temporary file with proper permissions"""
    import tempfile

    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=directory)

    try:
        # Set secure permissions (read/write for owner only)
        os.chmod(temp_path, 0o600)

        # Close the file descriptor
        os.close(fd)

        return temp_path
    except Exception as e:
        # Clean up on error
        try:
            os.close(fd)
            os.unlink(temp_path)
        except:
            pass
        raise

def secure_file_copy(src_path: str, dst_path: str, chunk_size: int = 8192) -> bool:
    """Securely copy a file with proper validation and error handling"""
    try:
        # Validate source file
        if not os.path.isfile(src_path):
            raise ValueError(f"Source file does not exist: {src_path}")

        # Check file size
        file_size = os.path.getsize(src_path)
        if file_size > MAX_UPLOAD_SIZE:
            raise ValueError(f"File too large: {file_size} bytes (max: {MAX_UPLOAD_SIZE})")

        # Create destination directory if it doesn't exist
        dst_dir = os.path.dirname(dst_path)
        if dst_dir and not os.path.exists(dst_dir):
            os.makedirs(dst_dir, mode=0o700)

        # Copy file in chunks to prevent memory issues
        with open(src_path, 'rb') as src_file:
            with open(dst_path, 'wb') as dst_file:
                while True:
                    chunk = src_file.read(chunk_size)
                    if not chunk:
                        break
                    dst_file.write(chunk)

        # Set secure permissions
        os.chmod(dst_path, 0o600)

        return True

    except Exception as e:
        logger.error(f"Secure file copy failed: {e}")
        # Clean up destination file if copy failed
        try:
            if os.path.exists(dst_path):
                os.unlink(dst_path)
        except:
            pass
        return False

def secure_file_delete(file_path: str) -> bool:
    """Securely delete a file with proper error handling"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
        return True
    except Exception as e:
        logger.error(f"Secure file delete failed: {e}")
        return False

# Rate limiting implementation
class RateLimiter:
    """Simple in-memory rate limiter"""
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.requests = {}  # {client_ip: [(timestamp, request_count)]}

    def is_allowed(self, client_ip: str) -> bool:
        """Check if client is allowed to make a request"""
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window

        # Clean up old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                (timestamp, count) for timestamp, count in self.requests[client_ip]
                if timestamp > window_start
            ]

        # Initialize if not exists
        if client_ip not in self.requests:
            self.requests[client_ip] = []

        # Count requests in the current window
        total_requests = sum(count for _, count in self.requests[client_ip])

        if total_requests >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return False

        # Add current request
        self.requests[client_ip].append((current_time, 1))

        return True

# Initialize rate limiter
rate_limiter = RateLimiter(RATE_LIMIT)

def get_client_ip() -> str:
    """Get client IP address, considering proxies"""
    # Check for forwarded IP (behind proxy)
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        # Get the first IP in the forwarded chain
        return forwarded_for.split(',')[0].strip()

    # Check for real IP (behind proxy)
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip

    # Fall back to direct IP
    return request.remote_addr or 'unknown'

def rate_limit_middleware():
    """Rate limiting middleware"""
    # Skip rate limiting for local development
    if DEBUG:
        return

    # Get client IP
    client_ip = get_client_ip()

    # Check if request is allowed
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return jsonify({
            "error": "Rate limit exceeded. Please try again later.",
            "retry_after": 60
        }), 429

def numpy_to_base64_image(np_array: np.ndarray) -> str:
    """Convert numpy array to base64 encoded JPEG image with performance optimizations"""
    try:
        # Quick validation
        if np_array is None or np_array.size == 0:
            logger.error("Invalid numpy array: None or empty")
            return ""

        start_time = time.time()

        # Create a view instead of copy if possible to save memory
        if np_array.base is not None:
            np_array = np_array.view()
        else:
            np_array = np_array.copy()

        # Optimized format handling
        if len(np_array.shape) == 3:
            if np_array.shape[2] == 4:
                # Fast RGBA to RGB using numpy slicing (no copy)
                np_array = np_array[:, :, :3]
            elif np_array.shape[2] != 3:
                logger.error(f"Unsupported channels: {np_array.shape[2]}")
                return ""
        elif len(np_array.shape) == 2:
            # Fast grayscale to RGB using numpy stacking
            np_array = np.expand_dims(np_array, axis=-1)
            np_array = np.repeat(np_array, 3, axis=-1)
        else:
            logger.error(f"Unsupported array shape: {np_array.shape}")
            return ""

        # Optimized data type conversion
        if np_array.dtype != np.uint8:
            if np_array.dtype in [np.float32, np.float64]:
                # Fast float to uint8 conversion
                np_array = np.multiply(np_array, 255, out=np_array, casting='unsafe')
            np_array = np.clip(np_array, 0, 255, out=np_array)
            np_array = np_array.astype(np.uint8, copy=False)

        # Use the fastest available encoding method
        img_bytes = None
        encoding_method = "unknown"

        try:
            # Priority 1: OpenCV (fastest for JPEG encoding)
            if CV2_AVAILABLE:
                success, encoded_img = cv2.imencode('.jpg', np_array, [
                    cv2.IMWRITE_JPEG_QUALITY, 75,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                    cv2.IMWRITE_JPEG_PROGRESSIVE, 0
                ])
                if success:
                    img_bytes = encoded_img.tobytes()
                    encoding_method = "opencv"

            # Priority 2: PyTorch GPU acceleration
            elif TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    # Convert to tensor and use GPU for processing
                    tensor = torch.from_numpy(np_array).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                    tensor = tensor.cuda()

                    # Use torchvision for JPEG encoding (if available)
                    try:
                        import torchvision.io
                        img_bytes = torchvision.io.encode_jpeg(tensor, quality=75)
                        encoding_method = "torch_gpu"
                    except ImportError:
                        # Fallback to CPU processing
                        tensor = tensor.cpu()
                        np_array = tensor.squeeze(0).permute(1, 2, 0).numpy() * 255
                        np_array = np_array.astype(np.uint8)
                except Exception as torch_error:
                    logger.debug(f"GPU encoding failed, falling back to CPU: {torch_error}")

            # Priority 3: Optimized PIL processing
            if img_bytes is None and PIL_AVAILABLE:
                # Use contiguous array for better PIL performance
                if not np_array.flags['C_CONTIGUOUS']:
                    np_array = np.ascontiguousarray(np_array)

                image = Image.fromarray(np_array, mode='RGB')

                # Use optimized JPEG settings
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='JPEG', quality=75, optimize=False, progressive=False)
                img_bytes = img_buffer.getvalue()
                encoding_method = "pil"

            # Fallback: raw bytes (no encoding)
            if img_bytes is None:
                img_bytes = np_array.tobytes()
                encoding_method = "raw"

            # Convert to base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            processing_time = time.time() - start_time
            logger.debug(f"Image conversion: {encoding_method}, {len(img_bytes)} bytes, {processing_time:.3f}s")

            return img_base64

        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            return ""

    except Exception as e:
        logger.error(f"Error in numpy_to_base64_image: {e}")
        return ""


# Global performance monitoring
performance_monitor = {
    'frame_times': [],
    'encoding_times': [],
    'fps_history': [],
    'last_fps_update': time.time(),
    'current_fps': 0,
    'adaptive_fps_target': 60,
    'min_fps': 15,
    'max_fps': 120
}

def update_performance_metrics(encoding_time: float, frame_time: float):
    """Update performance monitoring metrics"""
    current_time = time.time()

    # Track encoding performance
    performance_monitor['encoding_times'].append(encoding_time)
    if len(performance_monitor['encoding_times']) > 100:
        performance_monitor['encoding_times'].pop(0)

    # Track frame timing
    performance_monitor['frame_times'].append(frame_time)
    if len(performance_monitor['frame_times']) > 60:
        performance_monitor['frame_times'].pop(0)

    # Update FPS calculation every second
    if current_time - performance_monitor['last_fps_update'] >= 1.0:
        if len(performance_monitor['frame_times']) > 0:
            avg_frame_time = sum(performance_monitor['frame_times']) / len(performance_monitor['frame_times'])
            if avg_frame_time > 0:
                performance_monitor['current_fps'] = 1.0 / avg_frame_time

        # Adaptive FPS adjustment based on performance
        avg_encoding_time = sum(performance_monitor['encoding_times']) / len(performance_monitor['encoding_times']) if performance_monitor['encoding_times'] else 0.01

        if avg_encoding_time > 0.016:  # > 16ms encoding time (slower than 60 FPS)
            performance_monitor['adaptive_fps_target'] = max(
                performance_monitor['min_fps'],
                performance_monitor['adaptive_fps_target'] - 5
            )
        elif avg_encoding_time < 0.008 and performance_monitor['current_fps'] < 60:  # < 8ms encoding time
            performance_monitor['adaptive_fps_target'] = min(
                performance_monitor['max_fps'],
                performance_monitor['adaptive_fps_target'] + 5
            )

        performance_monitor['last_fps_update'] = current_time

def get_performance_stats() -> dict:
    """Get current performance statistics"""
    return {
        'current_fps': performance_monitor['current_fps'],
        'adaptive_fps_target': performance_monitor['adaptive_fps_target'],
        'avg_encoding_time': sum(performance_monitor['encoding_times']) / len(performance_monitor['encoding_times']) if performance_monitor['encoding_times'] else 0,
        'avg_frame_time': sum(performance_monitor['frame_times']) / len(performance_monitor['frame_times']) if performance_monitor['frame_times'] else 0,
        'cv2_available': CV2_AVAILABLE,
        'torch_available': TORCH_AVAILABLE,
        'torch_cuda_available': TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False
    }


# Global error handlers
@app.errorhandler(400)
def bad_request(error):
    """Handle 400 Bad Request errors"""
    logger.warning(f"Bad request: {error}")
    return jsonify({
        "error": "Bad request",
        "message": str(error),
        "status": 400
    }), 400

@app.errorhandler(404)
def not_found(error):
    """Handle 404 Not Found errors"""
    logger.warning(f"Endpoint not found: {request.path}")
    return jsonify({
        "error": "Endpoint not found",
        "message": f"The requested URL {request.path} was not found on this server",
        "status": 404
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 Method Not Allowed errors"""
    logger.warning(f"Method not allowed: {request.method} {request.path}")
    return jsonify({
        "error": "Method not allowed",
        "message": f"Method {request.method} not allowed for endpoint {request.path}",
        "status": 405
    }), 405

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle 413 Request Entity Too Large errors"""
    logger.warning(f"Request too large: {error}")
    return jsonify({
        "error": "Request too large",
        "message": f"File size exceeds maximum allowed size of {MAX_ROM_SIZE // (1024*1024)}MB",
        "status": 413
    }), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle 500 Internal Server Error"""
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred on the server",
        "status": 500
    }), 500

@app.before_request
def log_request_info():
    """Log information about each request"""
    logger.debug(f"Request: {request.method} {request.path} from {request.remote_addr}")

@app.after_request
def log_response_info(response):
    """Log information about each response"""
    logger.debug(f"Response: {response.status_code} for {request.method} {request.path}")
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint for monitoring"""
    return jsonify({"status": "healthy"}), 200

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get comprehensive status of the server"""
    status = get_game_state()
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
    """Upload a ROM file and load it into the specified emulator with enhanced security"""
    logger.info("=== ROM UPLOAD REQUEST RECEIVED ===")

    try:
        if 'rom_file' not in request.files:
            return jsonify({"error": "No ROM file provided"}), 400

        file = request.files['rom_file']
        emulator_type = request.form.get('emulator_type', 'gb')
        launch_ui = request.form.get('launch_ui', 'true')

        logger.info(f"File received: {file.filename}")
        logger.info(f"Emulator type: {emulator_type}")
        logger.info(f"Launch UI: {launch_ui}")
        logger.info(f"Available emulators: {list(emulators.keys())}")

        # Validate file upload with comprehensive security checks
        try:
            file_info = validate_file_upload(
                file,
                "ROM file",
                allowed_extensions=ALLOWED_ROM_EXTENSIONS,
                max_size=MAX_ROM_SIZE
            )
            logger.info(f"File validation passed: {file_info}")
        except ValueError as e:
            logger.error(f"File validation failed: {e}")
            return jsonify({"error": str(e)}), 400

        # Validate emulator type
        try:
            emulator_type = validate_string_input(
                emulator_type,
                "emulator_type",
                min_length=2,
                max_length=20,
                allowed_chars="abcdefghijklmnopqrstuvwxyz0123456789-"
            )
        except ValueError as e:
            return jsonify({"error": f"Invalid emulator type: {str(e)}"}), 400

        # Validate launch_ui parameter
        try:
            launch_ui = validate_string_input(
                launch_ui,
                "launch_ui",
                min_length=2,
                max_length=10,
                pattern=r'^(true|false)$'
            )
            launch_ui = launch_ui.lower() == 'true'
        except ValueError as e:
            return jsonify({"error": f"Invalid launch_ui parameter: {str(e)}"}), 400

        # Sanitize filename and create secure temporary file
        safe_filename = sanitize_filename(file_info['filename'])

        # Use secure temporary file creation
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_info['extension'],
            mode='wb'
        ) as temp_file:
            # Save file content securely
            file.save(temp_file.name)
            temp_rom_path = temp_file.name

            # Set secure file permissions (read/write for owner only)
            os.chmod(temp_rom_path, 0o600)

        logger.info(f"ROM saved to temporary path: {temp_rom_path}")

        # Map frontend emulator types to backend emulator keys
        emulator_type_mapping = {
            'gb': 'pyboy',
            'gba': 'pygba',
            'pyboy': 'pyboy',
            'pygba': 'pygba'
        }

        # Map the emulator type
        mapped_emulator_type = emulator_type_mapping.get(emulator_type)
        if not mapped_emulator_type or mapped_emulator_type not in emulators:
            logger.error(f"Invalid emulator type: {emulator_type}")
            os.unlink(temp_rom_path)
            return jsonify({"error": f"Invalid emulator type. Available: {list(emulator_type_mapping.keys())}"}), 400

        # Use the mapped emulator type
        emulator_type = mapped_emulator_type

        logger.info(f"Loading ROM into {emulator_type} emulator...")
        success = emulators[emulator_type].load_rom(temp_rom_path)

        if success:
            update_game_state({
                "active_emulator": emulator_type,
                "rom_loaded": True,
                "rom_path": temp_rom_path
            })
            logger.info(f"=== ROM LOADED SUCCESSFULLY ===")
            logger.info(f"ROM: {safe_filename}")
            logger.info(f"Emulator: {emulator_type}")
            logger.info(f"Temp path: {temp_rom_path}")

            emulator = emulators[emulator_type]
            if hasattr(emulator, 'pyboy') and emulator.pyboy:
                for _ in range(100):
                    emulator.pyboy.tick()

            # UI is now launched automatically by the emulator
            ui_status = emulator.get_ui_status() if hasattr(emulator, 'get_ui_status') else {"running": False}

        logger.info(f"Loading ROM into {emulator_type} emulator...")
        success = emulators[emulator_type].load_rom(temp_rom_path)

        if success:
            update_game_state({
                "active_emulator": emulator_type,
                "rom_loaded": True,
                "rom_path": temp_rom_path
            })
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

    # Track request for logging
    request_id = f"action_{time.time_ns()}"
    logger.info(f"[{request_id}] Action request received")

    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            logger.warning(f"[{request_id}] Action requested but no ROM loaded")
            return jsonify({"error": "No ROM loaded"}), 400

        # Validate and parse JSON data
        data = validate_json_data(request.get_data(as_text=True), "action request")

        action = data.get('action', 'SELECT')
        frames = data.get('frames', 1)

        # Validate action using comprehensive input validation
        try:
            action = validate_string_input(
                action,
                "action",
                min_length=1,
                max_length=10,
                allowed_chars="ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
            )
        except ValueError as e:
            logger.warning(f"[{request_id}] Invalid action format: {action} - {e}")
            return jsonify({"error": str(e)}), 400

        # Validate action is in allowed set
        valid_actions = {'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT', 'NOOP'}
        if action not in valid_actions:
            logger.warning(f"[{request_id}] Invalid action requested: {action}")
            return jsonify({
                "error": f"Invalid action: {action}",
                "valid_actions": list(valid_actions)
            }), 400

        # Validate frames using comprehensive input validation
        try:
            frames = validate_integer_input(
                frames,
                "frames",
                min_value=1,
                max_value=100
            )
        except ValueError as e:
            logger.warning(f"[{request_id}] Invalid frames value: {frames} - {e}")
            return jsonify({"error": str(e)}), 400

        logger.info(f"[{request_id}] Executing action: {action} for {frames} frame(s)")
        emulator = emulators[current_state["active_emulator"]]

        # Add timeout protection for emulator operations
        success = timeout_handler(5.0)(emulator.step)(action, frames)

        if success:
            add_to_action_history(action)
            logger.debug(f"[{request_id}] Action {action} executed successfully")
            return jsonify({
                "message": "Action executed successfully",
                "action": action,
                "frames": frames,
                "history_length": len(get_action_history())
            }), 200
        else:
            logger.error(f"[{request_id}] Failed to execute action: {action}")
            return jsonify({"error": "Failed to execute action"}), 500

    except Exception as e:
        logger.error(f"[{request_id}] Error executing action: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

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

    # Track request for logging
    request_id = f"ai_action_{time.time_ns()}"
    logger.info(f"[{request_id}] AI action request received")

    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            logger.warning(f"[{request_id}] AI action requested but no ROM loaded")
            return jsonify({"error": "No ROM loaded"}), 400

        data = request.json
        if not data:
            logger.error(f"[{request_id}] No JSON data received in AI action request")
            return jsonify({"error": "No request data provided"}), 400

        # Extract and validate parameters
        api_name = data.get('api_name')
        api_key = data.get('api_key')
        api_endpoint = data.get('api_endpoint')
        model = data.get('model')
        goal = data.get('goal', '')

        # Validate goal length
        if goal and len(goal) > 500:
            logger.warning(f"[{request_id}] Goal too long: {len(goal)} characters")
            return jsonify({"error": "Goal must be less than 500 characters"}), 400

        logger.info(f"[{request_id}] AI action request: api={api_name or 'auto'}, model={model or 'default'}, goal='{goal[:50]}...'")
        update_game_state({"current_goal": goal})

        current_state = get_game_state()
        emulator = emulators[current_state["active_emulator"]]

        # Get screen using timeout protection
        try:
            screen_array = timeout_handler(3.0)(emulator.get_screen)()
        except Exception as screen_error:
            logger.error(f"[{request_id}] Timeout getting screen: {screen_error}")
            return jsonify({"error": "Screen capture timeout"}), 500

        if screen_array is None or screen_array.size == 0:
            logger.error(f"[{request_id}] Failed to get screen from PyBoy emulator")
            return jsonify({"error": "Failed to capture screen"}), 500

        # Convert screen to bytes using timeout protection
        try:
            img_bytes = timeout_handler(3.0)(emulator.get_screen_bytes)()
        except Exception as convert_error:
            logger.error(f"[{request_id}] Timeout converting screen: {convert_error}")
            return jsonify({"error": "Screen processing timeout"}), 500

        if len(img_bytes) == 0:
            logger.error(f"[{request_id}] Failed to convert PyBoy screen to bytes")
            return jsonify({"error": "Failed to process screen image"}), 500

        logger.debug(f"[{request_id}] PyBoy screen captured for AI: {len(img_bytes)} bytes")

        # Set environment variables for this request if provided (with validation)
        if api_key:
            if len(api_key) < 10:
                logger.warning(f"[{request_id}] API key seems too short for provider: {api_name}")
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
            if len(model) < 1:
                logger.warning(f"[{request_id}] Model name is empty for provider: {api_name}")
            if api_name == 'openai-compatible':
                os.environ['OPENAI_MODEL'] = model
            elif api_name == 'nvidia':
                os.environ['NVIDIA_MODEL'] = model

        # Check if the requested provider is available
        if api_name and api_name not in ai_provider_manager.get_available_providers():
            available_providers = ai_provider_manager.get_available_providers()
            logger.warning(f"[{request_id}] Requested provider '{api_name}' is not available. Available providers: {available_providers}")

            # If specific provider is requested but not available, return error with suggestions
            return jsonify({
                "error": f"Provider '{api_name}' is not available",
                "available_providers": available_providers,
                "suggestion": f"Please use one of the available providers: {', '.join(available_providers)}"
            }), 400

        # Use provider manager with automatic fallback and timeout protection
        logger.debug(f"[{request_id}] Calling AI API: {api_name or 'auto'} with model: {model or 'default'}")
        current_history = get_action_history()

        try:
            action, actual_provider = timeout_handler(AI_REQUEST_TIMEOUT)(ai_provider_manager.get_next_action)(
                img_bytes, goal, current_history, api_name, model
            )
        except Exception as ai_timeout_error:
            logger.error(f"[{request_id}] AI request timeout: {ai_timeout_error}")
            return jsonify({"error": "AI request timeout"}), 500

        if not action or not isinstance(action, str):
            logger.error(f"[{request_id}] Invalid AI response: action={action}")
            return jsonify({"error": "AI returned invalid action"}), 500

        add_to_action_history(action)

        logger.info(f"[{request_id}] AI ({actual_provider or 'fallback'}) suggested action: {action}")
        return jsonify({
            "action": action,
            "provider_used": actual_provider,
            "history": get_action_history()[-10:]
        }), 200

    except Exception as e:
        logger.error(f"[{request_id}] Error getting AI action: {e}", exc_info=True)
        api_name = data.get('api_name') if 'data' in locals() else 'unknown'
        goal = data.get('goal', '') if 'data' in locals() else 'unknown'
        logger.error(f"[{request_id}] AI action failed - API: {api_name}, Goal: '{goal}', Screen size: {len(img_bytes) if 'img_bytes' in locals() else 'unknown'}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def ai_chat():
    """Send a message to the AI and get a response with automatic fallback"""
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        data = request.json
        user_message = data.get('message', '')
        api_name = data.get('api_name')
        api_key = data.get('api_key')
        api_endpoint = data.get('api_endpoint')
        model = data.get('model')

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        current_state = get_game_state()
        emulator = emulators[current_state["active_emulator"]]

        # Get screen bytes using proper PyBoy connectors
        img_bytes = emulator.get_screen_bytes()

        if len(img_bytes) == 0:
            logger.error("Failed to get screen bytes from PyBoy for chat")
            return jsonify({"error": "Failed to capture screen"}), 500

        logger.debug(f"PyBoy screen captured for chat: {len(img_bytes)} bytes")

        context = {
            "current_goal": current_state["current_goal"],
            "action_history": get_action_history()[-20:],
            "game_type": current_state["active_emulator"].upper()
        }

        # Get model/provider with priority: game_state -> request params -> defaults
        current_provider = current_state.get('current_provider')
        current_model = current_state.get('current_model')

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
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[current_state["active_emulator"]]

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
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[current_state["active_emulator"]]

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
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[current_state["active_emulator"]]

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
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[current_state["active_emulator"]]

        if not hasattr(emulator, 'get_ui_status'):
            return jsonify({"error": "UI control not supported by this emulator"}), 400

        ui_status = emulator.get_ui_status()

        return jsonify({
            "ui_status": ui_status,
            "rom_loaded": current_state["rom_loaded"],
            "active_emulator": current_state["active_emulator"]
        }), 200

    except Exception as e:
        logger.error(f"Error getting UI status: {e}")
        return jsonify({"error": "Internal server error"}), 500

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
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({
                'success': False,
                'error': 'No ROM loaded'
            }), 400

        emulator = emulators[current_state["active_emulator"]]

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
        # Validate and parse JSON data
        data = validate_json_data(request.get_data(as_text=True), "tetris save request")

        filepath = data.get('filepath')

        if not filepath:
            return jsonify({
                'success': False,
                'error': 'Filepath required'
            }), 400

        # Validate filepath for security
        try:
            filepath = validate_string_input(
                filepath,
                "filepath",
                min_length=1,
                max_length=500
            )

            # Additional path validation
            filepath = sanitize_filename(filepath)

            # Ensure filepath has proper extension
            if not filepath.endswith(('.pkl', '.model', '.dat')):
                raise ValueError("Filepath must have a valid model extension (.pkl, .model, .dat)")

        except ValueError as e:
            return jsonify({
                'success': False,
                'error': str(e)
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
        # Validate and parse JSON data
        data = validate_json_data(request.get_data(as_text=True), "tetris load request")

        filepath = data.get('filepath')

        if not filepath:
            return jsonify({
                'success': False,
                'error': 'Filepath required'
            }), 400

        # Validate filepath for security
        try:
            filepath = validate_string_input(
                filepath,
                "filepath",
                min_length=1,
                max_length=500
            )

            # Additional path validation
            filepath = sanitize_filename(filepath)

            # Ensure filepath has proper extension
            if not filepath.endswith(('.pkl', '.model', '.dat')):
                raise ValueError("Filepath must have a valid model extension (.pkl, .model, .dat)")

        except ValueError as e:
            return jsonify({
                'success': False,
                'error': str(e)
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

@app.route('/api/screen', methods=['GET'])
def get_screen():
    """Get the current screen from the active emulator with performance monitoring"""
    try:
        start_time = time.time()
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[current_state["active_emulator"]]

        # Get screen array
        screen_array = emulator.get_screen()

        # Validate screen data - don't use placeholders
        if screen_array is None or screen_array.size == 0:
            logger.error("Screen data is None or empty")
            return jsonify({"error": "Failed to capture screen"}), 500

        # Convert to base64 with timing
        conversion_start = time.time()
        img_base64 = numpy_to_base64_image(screen_array)
        conversion_time = time.time() - conversion_start

        if not img_base64:
            logger.error("Failed to convert screen to base64")
            return jsonify({"error": "Failed to process screen image"}), 500

        total_time = time.time() - start_time

        # Update performance metrics
        update_performance_metrics(conversion_time, total_time)

        return jsonify({
            "image": img_base64,
            "shape": screen_array.shape,
            "timestamp": time.time(),
            "pyboy_frame": emulator.get_frame_count() if hasattr(emulator, 'get_frame_count') else None,
            "performance": {
                "total_time_ms": round(total_time * 1000, 2),
                "conversion_time_ms": round(conversion_time * 1000, 2),
                "current_fps": round(performance_monitor['current_fps'], 1),
                "adaptive_fps_target": performance_monitor['adaptive_fps_target']
            }
        }), 200

    except Exception as e:
        logger.error(f"Error getting screen: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Get performance monitoring statistics"""
    try:
        stats = get_performance_stats()

        # Get emulator-specific stats if available
        emulator_stats = {}
        current_state = get_game_state()
        if current_state["rom_loaded"] and current_state["active_emulator"]:
            emulator = emulators[current_state["active_emulator"]]
            if hasattr(emulator, 'get_performance_stats'):
                emulator_stats = emulator.get_performance_stats()

        return jsonify({
            "server_performance": stats,
            "emulator_performance": emulator_stats,
            "system_info": {
                "cpu_count": multiprocessing.cpu_count(),
                "memory_usage_mb": _get_memory_usage(),
                "multi_process_mode": USE_MULTI_PROCESS,
                "timestamp": time.time()
            }
        }), 200

    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/emulator/mode', methods=['GET'])
def get_emulator_mode():
    """Get current emulator mode"""
    return jsonify({
        "multi_process_mode": USE_MULTI_PROCESS,
        "available_modes": ["single-process", "multi-process"],
        "current_mode": "multi-process" if USE_MULTI_PROCESS else "single-process"
    }), 200


@app.route('/api/emulator/clear-cache', methods=['POST'])
def clear_emulator_cache():
    """Clear emulator caches for performance optimization"""
    try:
        current_state = get_game_state()
        cleared = []

        if current_state["rom_loaded"] and current_state["active_emulator"]:
            emulator = emulators[current_state["active_emulator"]]

            # Clear screen cache if available
            if hasattr(emulator, 'clear_screen_cache'):
                emulator.clear_screen_cache()
                cleared.append("screen_cache")

            # Clear server performance cache
            performance_monitor['frame_times'].clear()
            performance_monitor['encoding_times'].clear()
            cleared.append("performance_cache")

            logger.info(f"Cleared emulator caches: {cleared}")
            return jsonify({
                "message": "Caches cleared successfully",
                "cleared_caches": cleared,
                "cache_size_after": {
                    "performance_monitor": {
                        "frame_times": len(performance_monitor['frame_times']),
                        "encoding_times": len(performance_monitor['encoding_times'])
                    }
                }
            }), 200
        else:
            return jsonify({"error": "No emulator running"}), 400

    except Exception as e:
        logger.error(f"Error clearing caches: {e}")
        return jsonify({"error": str(e)}), 500


def _get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        return 0.0

@app.route('/api/screen/debug', methods=['GET'])
def get_screen_debug():
    """Debug endpoint to test screen capture functionality"""
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[current_state["active_emulator"]]

        # Get emulator info
        info = emulator.get_info() if hasattr(emulator, 'get_info') else {}

        # Get screen array
        screen_array = emulator.get_screen()

        debug_info = {
            "emulator_info": info,
            "screen_shape": screen_array.shape if screen_array is not None else None,
            "screen_dtype": str(screen_array.dtype) if screen_array is not None else None,
            "screen_min": int(screen_array.min()) if screen_array is not None else None,
            "screen_max": int(screen_array.max()) if screen_array is not None else None,
            "screen_size": screen_array.size if screen_array is not None else None,
            "timestamp": time.time(),
            "emulator_type": current_state["active_emulator"],
            "rom_loaded": current_state["rom_loaded"],
            "rom_path": current_state["rom_path"]
        }

        # Try base64 conversion
        if screen_array is not None and screen_array.size > 0:
            img_base64 = numpy_to_base64_image(screen_array)
            debug_info["base64_success"] = img_base64 is not None and len(img_base64) > 0
            debug_info["base64_length"] = len(img_base64) if img_base64 else 0
            debug_info["base64_preview"] = img_base64[:100] + "..." if img_base64 and len(img_base64) > 100 else img_base64
        else:
            debug_info["base64_success"] = False
            debug_info["base64_length"] = 0
            debug_info["base64_preview"] = None

        return jsonify(debug_info), 200

    except Exception as e:
        logger.error(f"Error in debug screen endpoint: {e}", exc_info=True)
        return jsonify({"error": f"Debug endpoint error: {str(e)}"}), 500

@app.route('/api/stream', methods=['GET'])
def stream_screen():
    """SSE endpoint for live screen streaming with stability fixes"""
    def generate():
        logger.info("SSE stream requested. Checking game state...")
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            logger.warning("SSE stream aborted: No ROM loaded.")
            yield f"data: {json.dumps({'error': 'No ROM loaded'})}\n\n"
            return

        emulator = emulators[current_state["active_emulator"]]
        logger.info(f"Starting SSE stream for {current_state['active_emulator']}.")

        frame_count = 0
        last_frame_time = time.time()
        target_fps = performance_monitor['adaptive_fps_target']  # Start with adaptive target
        frame_interval = 1.0 / target_fps

        # Dynamic frame rate adjustment (use global performance monitor)
        frame_times = []
        max_frame_times = 60  # Keep last 60 frames for average
        min_fps = performance_monitor['min_fps']
        max_fps = performance_monitor['max_fps']

        # Performance monitoring
        pyboy_timeout_count = 0
        max_pyboy_timeouts = 3
        consecutive_errors = 0
        max_consecutive_errors = 5
        last_pyboy_error = None

        # Create thread-safe executor for timeout handling - managed manually for proper cleanup
        executor = ThreadPoolExecutor(max_workers=2)
        last_client_activity = time.time()
        connection_timeout = 60  # 60 seconds without client response

        try:
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

                            # Re-check ROM loaded state before proceeding
                            current_state = get_game_state()
                            if not current_state["rom_loaded"] or not current_state["active_emulator"]:
                                logger.warning("Stream aborted: ROM no longer loaded")
                                error_data = {
                                    'heartbeat': True,
                                    'frame': frame_count,
                                    'timestamp': current_time,
                                    'fps': target_fps,
                                    'status': 'rom_unloaded',
                                    'message': 'ROM is no longer loaded',
                                    'error': 'No ROM loaded'
                                }
                                yield f"data: {json.dumps(error_data)}\n\n"
                                break

                            screen_timeout = 0.05 if pyboy_timeout_count == 0 else 0.1
                            try:
                                screen_array = executor.submit(get_screen).result(timeout=screen_timeout)
                            except FutureTimeoutError:
                                pyboy_timeout_count += 1
                                logger.warning(f"PyBoy get_screen timeout (count: {pyboy_timeout_count})")

                                # Instead of using placeholder, send error heartbeat
                                if pyboy_timeout_count >= max_pyboy_timeouts:
                                    error_data = {
                                        'heartbeat': True,
                                        'frame': frame_count,
                                        'timestamp': current_time,
                                        'fps': target_fps,
                                        'status': 'pyboy_timeout',
                                        'message': 'PyBoy timeout threshold reached',
                                        'error': 'PyBoy API timeout',
                                        'timeout_count': pyboy_timeout_count
                                    }
                                    yield f"data: {json.dumps(error_data)}\n\n"
                                    break
                                else:
                                    # Send timeout error but continue streaming
                                    error_data = {
                                        'heartbeat': True,
                                        'frame': frame_count,
                                        'timestamp': current_time,
                                        'fps': target_fps,
                                        'status': 'timeout_error',
                                        'message': 'PyBoy get_screen timeout',
                                        'error': 'Screen capture timeout',
                                        'timeout_count': pyboy_timeout_count
                                    }
                                    yield f"data: {json.dumps(error_data)}\n\n"
                                    frame_count += 1
                                    last_frame_time = current_time
                                    continue

                            # Validate screen data - if invalid, send error instead of placeholder
                            if screen_array is None or screen_array.size == 0:
                                logger.warning(f"Stream frame {frame_count}: PyBoy API returned empty screen")
                                error_data = {
                                    'heartbeat': True,
                                    'frame': frame_count,
                                    'timestamp': current_time,
                                    'fps': target_fps,
                                    'status': 'empty_screen',
                                    'message': 'Screen data is empty',
                                    'error': 'Screen capture failed'
                                }
                                yield f"data: {json.dumps(error_data)}\n\n"
                                frame_count += 1
                                last_frame_time = current_time
                                continue

                            # Additional validation for proper PyBoy format
                            if len(screen_array.shape) != 3 or screen_array.shape[2] not in [3, 4]:
                                logger.warning(f"Stream frame {frame_count}: Invalid shape {screen_array.shape}")
                                error_data = {
                                    'heartbeat': True,
                                    'frame': frame_count,
                                    'timestamp': current_time,
                                    'fps': target_fps,
                                    'status': 'invalid_format',
                                    'message': 'Invalid screen format',
                                    'error': f'Invalid screen shape: {screen_array.shape}'
                                }
                                yield f"data: {json.dumps(error_data)}\n\n"
                                frame_count += 1
                                last_frame_time = current_time
                                continue

                            # Convert to base64 using timeout-protected function
                            def convert_image():
                                return numpy_to_base64_image(screen_array)

                            conversion_start = time.time()
                            try:
                                img_base64 = executor.submit(convert_image).result(timeout=0.05)
                                conversion_time = time.time() - conversion_start
                                # Update global performance metrics
                                update_performance_metrics(conversion_time, time.time() - current_time)
                            except FutureTimeoutError:
                                logger.warning("Image conversion timeout")
                                img_base64 = None
                                conversion_time = 0.05

                            if not img_base64:
                                # Send heartbeat without image if conversion failed
                                data = {
                                    'heartbeat': True,
                                    'frame': frame_count,
                                    'timestamp': current_time,
                                    'fps': target_fps,
                                    'adaptive_fps_target': performance_monitor['adaptive_fps_target'],
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
                                    'adaptive_fps_target': performance_monitor['adaptive_fps_target'],
                                    'actual_interval': elapsed,
                                    'process_time_ms': round(frame_process_time, 2),
                                    'conversion_time_ms': round(conversion_time * 1000, 2),
                                    'current_fps': round(performance_monitor['current_fps'], 1),
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

                    # Dynamic frame rate adjustment based on performance - use global adaptive target
                    frame_process_time = (time.time() - current_time)
                    frame_times.append(frame_process_time)

                    if len(frame_times) > max_frame_times:
                        frame_times.pop(0)

                    # Update target FPS from global performance monitor
                    new_target_fps = performance_monitor['adaptive_fps_target']
                    if new_target_fps != target_fps:
                        old_fps = target_fps
                        target_fps = new_target_fps
                        frame_interval = 1.0 / target_fps
                        logger.info(f"Adaptive FPS adjusted: {old_fps} -> {target_fps}")

                    # Local frame rate adjustment as backup
                    if len(frame_times) >= 30:  # Only adjust after collecting enough data
                        avg_frame_time = sum(frame_times) / len(frame_times)
                        if avg_frame_time > frame_interval * 1.5:  # Taking too long
                            if target_fps > min_fps:
                                target_fps = max(min_fps, target_fps - 2)
                                frame_interval = 1.0 / target_fps
                                logger.info(f"Local adjustment: Reducing FPS to {target_fps} due to slow performance")
                        elif avg_frame_time < frame_interval * 0.7:  # Processing quickly
                            if target_fps < max_fps:
                                target_fps = min(max_fps, target_fps + 1)
                                frame_interval = 1.0 / target_fps
                                logger.info(f"Local adjustment: Increasing FPS to {target_fps} due to good performance")

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
                    consecutive_errors += 1
                    last_pyboy_error = str(e)
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

        except Exception as outer_e:
            logger.error(f"Critical error in SSE stream generator: {outer_e}", exc_info=True)
            try:
                error_data = {
                    'error': str(outer_e),
                    'frame': frame_count,
                    'timestamp': time.time(),
                    'fps': target_fps,
                    'status': 'stream_ending',
                    'pyboy_status': 'Critical failure'
                }
                yield f"data: {json.dumps(error_data)}\n\n"
            except:
                pass
        finally:
            # Cleanup resources - this is guaranteed to run
            logger.info(f"SSE stream cleanup. Total frames: {frame_count}, Final FPS: {target_fps}")
            try:
                if 'executor' in locals() and executor:
                    executor.shutdown(wait=False)
                    logger.info("ThreadPoolExecutor shutdown successfully")
            except Exception as cleanup_error:
                logger.error(f"Error during executor cleanup: {cleanup_error}")

            # Clean up any remaining resources
            try:
                # Force garbage collection to clean up any remaining references
                import gc
                gc.collect()
                logger.info("Garbage collection completed")
            except Exception as gc_error:
                logger.warning(f"Garbage collection failed: {gc_error}")

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

def cleanup_server_resources():
    """Clean up server resources before shutdown"""
    logger.info("Cleaning up server resources...")

    # Clean up emulators
    try:
        for emulator_name, emulator in emulators.items():
            if hasattr(emulator, 'cleanup'):
                emulator.cleanup()
                logger.info(f"Cleaned up {emulator_name} emulator")
    except Exception as e:
        logger.error(f"Error cleaning up emulators: {e}")

    # Clean up AI provider manager
    try:
        if hasattr(ai_provider_manager, 'cleanup'):
            ai_provider_manager.cleanup()
            logger.info("Cleaned up AI provider manager")
    except Exception as e:
        logger.error(f"Error cleaning up AI provider manager: {e}")

    # Force garbage collection
    try:
        import gc
        gc.collect()
        logger.info("Server garbage collection completed")
    except Exception as e:
        logger.error(f"Error during server garbage collection: {e}")

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {sig}, shutting down gracefully...")
    cleanup_server_resources()
    logger.info("Server shutdown complete")
    exit(0)

def main():
    """Main server function"""
    logger.info("=== STARTING AI GAME SERVER ===")

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

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

    try:
        app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)
    except Exception as e:
        logger.error(f"Server failed to start: {e}", exc_info=True)
    finally:
        cleanup_server_resources()

if __name__ == "__main__":
    main()