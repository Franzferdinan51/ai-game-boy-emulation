"""
PyBoy UI Launcher
Separate process for running PyBoy with UI while server handles headless operations
"""
import sys
import os
import signal
import time
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    """Handle termination signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PyBoy UI Launcher')
    parser.add_argument('rom_path', help='Path to the ROM file to load')
    parser.add_argument('--window', default='sdl2', choices=['sdl2', 'opengl', 'null'],
                        help='Window type to use')
    parser.add_argument('--scale', type=int, default=2,
                        help='Scaling factor for the window')
    parser.add_argument('--color-palette', default='grayscale',
                        help='Color palette to use')
    parser.add_argument('--sound', action='store_true', default=True,
                        help='Enable sound')
    parser.add_argument('--sound-volume', type=int, default=50,
                        help='Sound volume (0-100)')
    parser.add_argument('--sound-sample-rate', type=int, default=44100,
                        help='Sound sample rate')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Log level')
    return parser.parse_args()

def main():
    """Main entry point for UI process"""
    # Parse command line arguments
    args = parse_arguments()

    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Set log level
        numeric_level = getattr(logging, args.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {args.log_level}')
        logging.getLogger().setLevel(numeric_level)

        # Validate ROM path
        if not os.path.exists(args.rom_path):
            logger.error(f"ROM file not found: {args.rom_path}")
            sys.exit(1)

        logger.info(f"Starting PyBoy UI with ROM: {args.rom_path}")
        logger.info(f"Configuration: window={args.window}, scale={args.scale}, sound={args.sound}")

        # Import PyBoy here to avoid issues if not available
        try:
            from pyboy import PyBoy
            PYBOY_AVAILABLE = True
            logger.info("PyBoy imported successfully")
        except ImportError as e:
            logger.error(f"PyBoy not available: {e}")
            logger.error("Please install PyBoy with: pip install pyboy")
            logger.error("Also ensure SDL2 libraries are installed on your system")
            sys.exit(1)

        # Initialize PyBoy with UI enabled
        try:
            pyboy_args = {
                'window': args.window,
                'sound': args.sound,
                'sound_volume': args.sound_volume,
                'sound_sample_rate': args.sound_sample_rate,
                'log_level': args.log_level,
                'color_palette': args.color_palette,
                'scale': args.scale
            }

            logger.info(f"PyBoy arguments: {pyboy_args}")
            pyboy = PyBoy(args.rom_path, **pyboy_args)

            logger.info("PyBoy UI initialized successfully")
            logger.info("UI_READY")  # Signal to the manager that UI is ready

            # Main loop
            frame_count = 0
            last_status_time = time.time()
            status_interval = 60  # Log status every 60 seconds

            while pyboy.tick():
                frame_count += 1

                # Log status periodically
                current_time = time.time()
                if current_time - last_status_time >= status_interval:
                    fps = frame_count / status_interval
                    logger.info(f"Running smoothly - Frame: {frame_count}, FPS: {fps:.1f}")
                    frame_count = 0
                    last_status_time = current_time

                # Handle any UI commands from stdin if needed
                # This could be extended for more sophisticated control

        except Exception as e:
            logger.error(f"Error running PyBoy UI: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            sys.exit(1)

        logger.info("PyBoy UI process completed normally")

    except Exception as e:
        logger.error(f"Unexpected error in UI process: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()