#!/usr/bin/env python3
"""
PyBoy UI Launcher
Standalone script for launching PyBoy with UI configuration
"""
import sys
import os
import argparse
import logging
import time
from pathlib import Path

# Add the backend directory to Python path
current_dir = Path(__file__).parent
backend_dir = current_dir / "ai-game-server" / "src" / "backend"
if backend_dir.exists():
    sys.path.insert(0, str(backend_dir))

# Add the PyBoy directory to Python path
pyboy_dir = current_dir / "PyBoy"
if pyboy_dir.exists():
    sys.path.insert(0, str(pyboy_dir))

try:
    from pyboy import PyBoy
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("PyBoy not available. Please install it with 'pip install pyboy'")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PyBoy UI Launcher')

    parser.add_argument('rom_path', help='Path to the ROM file')
    parser.add_argument('--window', choices=['SDL2', 'OpenGL', 'null'],
                       default='SDL2', help='Window type (default: SDL2)')
    parser.add_argument('--scale', type=int, default=2,
                       help='Window scaling factor (default: 2)')
    parser.add_argument('--sound', action='store_true', default=True,
                       help='Enable sound (default: True)')
    parser.add_argument('--no-sound', action='store_true',
                       help='Disable sound')
    parser.add_argument('--sound-volume', type=int, default=50,
                       help='Sound volume 0-100 (default: 50)')
    parser.add_argument('--sound-sample-rate', type=int, default=44100,
                       help='Sound sample rate (default: 44100)')
    parser.add_argument('--color-palette', default='grayscale',
                       help='Color palette (default: grayscale)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Log level (default: INFO)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--show-fps', action='store_true',
                       help='Show FPS counter')
    parser.add_argument('--fullscreen', action='store_true',
                       help='Start in fullscreen mode')

    return parser.parse_args()

def validate_args(args):
    """Validate command line arguments"""
    if not os.path.exists(args.rom_path):
        print(f"ROM file not found: {args.rom_path}")
        return False

    if args.scale < 1 or args.scale > 8:
        print(f"Invalid scale value: {args.scale}. Must be between 1 and 8")
        return False

    if args.sound_volume < 0 or args.sound_volume > 100:
        print(f"Invalid sound volume: {args.sound_volume}. Must be between 0 and 100")
        return False

    if args.sound_sample_rate not in [22050, 44100, 48000]:
        print(f"Invalid sample rate: {args.sound_sample_rate}. Must be 22050, 44100, or 48000")
        return False

    return True

def create_pyboy_config(args):
    """Create PyBoy configuration from arguments"""
    config = {
        'window': args.window,
        'scale': args.scale,
        'log_level': args.log_level,
        'sound_emulated': True,
        'sound_volume': args.sound_volume,
        'sound_sample_rate': args.sound_sample_rate
    }

    # Only add color_palette if it's supported (not all PyBoy versions support it)
    # Note: This parameter may not be available in all PyBoy versions

    # Handle sound configuration
    if args.no_sound:
        config['sound'] = False
        config['sound_emulated'] = False
        config['sound_volume'] = 0
    else:
        config['sound'] = args.sound

    # Add debug options
    if args.debug:
        config['debug'] = True

    return config

def run_pyboy_ui(rom_path, config):
    """Run PyBoy with the specified configuration"""
    try:
        logger.info(f"Starting PyBoy with ROM: {rom_path}")
        logger.info(f"Configuration: {config}")

        # Initialize PyBoy
        pyboy = PyBoy(rom_path, **config)

        logger.info("PyBoy UI initialized successfully")
        print("UI_READY")  # Signal that UI is ready

        # Main emulation loop
        frame_count = 0
        start_time = time.time()

        while True:
            # Tick the emulator
            if not pyboy.tick():
                break

            frame_count += 1

            # Log FPS every 60 frames
            if frame_count % 60 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                logger.debug(f"Running at {fps:.1f} FPS")

        logger.info("PyBoy emulation ended")

    except KeyboardInterrupt:
        logger.info("PyBoy stopped by user")
    except Exception as e:
        logger.error(f"Error running PyBoy: {e}")
        return False
    finally:
        try:
            if 'pyboy' in locals():
                pyboy.stop()
        except:
            pass

    return True

def main():
    """Main entry point"""
    try:
        # Parse arguments
        args = parse_arguments()

        # Validate arguments
        if not validate_args(args):
            sys.exit(1)

        # Create configuration
        config = create_pyboy_config(args)

        # Run PyBoy
        success = run_pyboy_ui(args.rom_path, config)

        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()