"""
PyBoy emulator implementation
"""
import numpy as np
from typing import List, Tuple, Optional, Any
import io
import os
import sys
import logging
import subprocess
import threading
import multiprocessing
import time

# Try to import PyBoy
try:
    from pyboy import PyBoy
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("PyBoy not available. Install with 'pip install pyboy'")

from .emulator_interface import EmulatorInterface

# UI process manager for separate PyBoy process
UI_MANAGER_AVAILABLE = True

logger = logging.getLogger(__name__)


class PyBoyEmulator(EmulatorInterface):
    """PyBoy emulator implementation using official API patterns"""

    def __init__(self):
        self.pyboy = None
        self.rom_path = None
        self.initialized = False
        self.game_title = ""
        self.auto_launch_ui = True
        self.ui_launched = False
        self.game_wrapper = None
        self.ui_process = None
        self.ui_thread = None

    def load_rom(self, rom_path: str) -> bool:
        """Load a ROM file into the PyBoy emulator using official API"""
        if not PYBOY_AVAILABLE:
            raise RuntimeError("PyBoy is not available. Please install it with 'pip install pyboy'")

        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"ROM file not found: {rom_path}")

        try:
            # Initialize PyBoy using the official API pattern
            logger.info(f"Initializing PyBoy with ROM: {os.path.basename(rom_path)}")

            # Use null window for server process - UI will be in separate process
            self.pyboy = PyBoy(
                rom_path,
                window="null",  # Always use null for server process
                scale=2,
                sound_emulated=False,  # Disable sound to prevent buffer overrun crashes
                sound_volume=0
            )

            # Set emulation speed to unlimited for AI training
            self.pyboy.set_emulation_speed(0)

            # Initialize game wrapper if available
            self.game_wrapper = self.pyboy.game_wrapper

            # Store basic info
            self.rom_path = rom_path
            self.initialized = True
            self.game_title = self.pyboy.cartridge_title

            # Log successful initialization
            logger.info(f"PyBoy initialized successfully")
            logger.info(f"Game title: {self.game_title}")
            logger.info(f"Window type: null")
            logger.info(f"Emulation speed: unlimited (0)")

            # Tick once to start the emulator
            self.pyboy.tick(1, False)

            # Launch UI in separate process if requested
            if self.auto_launch_ui:
                self._launch_ui_process()

            return True

        except Exception as e:
            logger.error(f"Error loading ROM: {e}")
            return False

    def _launch_ui_process(self):
        """Launch PyBoy UI in a separate process"""
        if not self.rom_path or not os.path.exists(self.rom_path):
            logger.error("Cannot launch UI - no ROM loaded")
            return

        try:
            logger.info("=== LAUNCHING PYBOY UI PROCESS ===")
            logger.info(f"ROM path: {self.rom_path}")
            logger.info(f"ROM exists: {os.path.exists(self.rom_path)}")
            logger.info(f"Working directory: {os.getcwd()}")

            # Create a simple script to run PyBoy with UI
            ui_script = f'''
import sys
import os
import time

print("=== PYBOY UI SCRIPT STARTING ===")
print(f"Python version: {{sys.version}}")
print(f"Working directory: {{os.getcwd()}}")
print(f"ROM path: {repr(self.rom_path)}")

# Add PyBoy path to Python path
sys.path.insert(0, "C:\\Users\\Ryan\\Desktop\\ROMS\\PyGB\\PyBoy")
sys.path.insert(0, "C:\\Users\\Ryan\\Desktop\\ROMS\\PyGB")

try:
    print("Importing PyBoy...")
    from pyboy import PyBoy
    print("PyBoy imported successfully")

    print(f"Loading ROM: {repr(self.rom_path)}")
    if not os.path.exists(self.rom_path):
        print(f"ERROR: ROM file not found: {repr(self.rom_path)}")
        sys.exit(1)

    pyboy = PyBoy(self.rom_path, window="SDL2", scale=2, sound_emulated=False, debug=False)
    print("PyBoy initialized successfully")
    pyboy.set_emulation_speed(1)
    print("Emulation speed set to 1")

    frame_count = 0
    print("Starting UI loop...")
    # Keep the UI running
    while True:
        try:
            pyboy.tick(1, True)
            frame_count += 1
            if frame_count % 60 == 0:  # Log every 2 seconds at 30fps
                print(f"UI frame: {{frame_count}}")
        except Exception as tick_error:
            print(f"Tick error: {{tick_error}}")
            time.sleep(0.1)
        except KeyboardInterrupt:
            print("UI interrupted by user")
            break

except KeyboardInterrupt:
    print("UI interrupted by user")
except Exception as e:
    print(f"UI process error: {{e}}")
    import traceback
    traceback.print_exc()
finally:
    print("UI process ending")
'''

            # Write the script to a temporary file
            script_path = os.path.join(os.path.dirname(self.rom_path), f"ui_script_{os.getpid()}.py")
            logger.info(f"Writing UI script to: {script_path}")

            with open(script_path, 'w') as f:
                f.write(ui_script)

            logger.info(f"UI script written successfully")

            # Launch the UI process with enhanced logging
            logger.info("Starting UI subprocess...")
            self.ui_process = subprocess.Popen([
                sys.executable, script_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, text=True)

            # Wait a moment and check if process started
            time.sleep(1)

            if self.ui_process.poll() is None:
                self.ui_launched = True
                logger.info(f"=== UI PROCESS LAUNCHED SUCCESSFULLY ===")
                logger.info(f"UI process PID: {self.ui_process.pid}")

                # Start a thread to monitor UI process output
                def monitor_ui_process():
                    try:
                        stdout, stderr = self.ui_process.communicate(timeout=5)
                        logger.info(f"UI process stdout: {stdout}")
                        if stderr:
                            logger.error(f"UI process stderr: {stderr}")
                    except subprocess.TimeoutExpired:
                        logger.info("UI process is running (timeout reading output)")
                    except Exception as monitor_e:
                        logger.error(f"Error monitoring UI process: {monitor_e}")

                monitor_thread = threading.Thread(target=monitor_ui_process)
                monitor_thread.daemon = True
                monitor_thread.start()

            else:
                # Process already terminated
                stdout, stderr = self.ui_process.communicate()
                logger.error(f"=== UI PROCESS FAILED TO START ===")
                logger.error(f"Exit code: {self.ui_process.returncode}")
                logger.error(f"stdout: {stdout}")
                logger.error(f"stderr: {stderr}")
                self.ui_launched = False

            # Clean up the script file after a delay
            def cleanup_script():
                time.sleep(5)
                try:
                    if os.path.exists(script_path):
                        os.remove(script_path)
                        logger.info(f"Cleaned up UI script: {script_path}")
                except Exception as cleanup_e:
                    logger.error(f"Failed to clean up script: {cleanup_e}")

            cleanup_thread = threading.Thread(target=cleanup_script)
            cleanup_thread.daemon = True
            cleanup_thread.start()

        except Exception as e:
            logger.error(f"=== FAILED TO LAUNCH UI PROCESS ===", exc_info=True)
            logger.error(f"Error: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.ui_launched = False

    def step(self, action: str, frames: int = 1) -> bool:
        """Execute an action for a number of frames using official PyBoy API"""
        if not self.initialized or self.pyboy is None:
            return False

        # Map actions to PyBoy buttons
        action_map = {
            'UP': 'up',
            'DOWN': 'down',
            'LEFT': 'left',
            'RIGHT': 'right',
            'A': 'a',
            'B': 'b',
            'START': 'start',
            'SELECT': 'select'
        }

        try:
            if action in action_map:
                # Use the official PyBoy button API - pyboy.button()
                button = action_map[action]

                # Process frames with button input
                for i in range(frames):
                    # Press button on even frames, release on odd frames for natural input
                    if i % 2 == 0:
                        self.pyboy.button(button)

                    # Render only on last frame for performance
                    render_this_frame = (i == frames - 1)
                    self.pyboy.tick(1, render_this_frame)

                # Ensure button is released
                self.pyboy.button(button)
            else:
                # For NOOP actions, just tick the emulator
                for i in range(frames):
                    render_this_frame = (i == frames - 1)
                    self.pyboy.tick(1, render_this_frame)

            return True
        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            return False

    def get_screen(self) -> np.ndarray:
        """Get the current screen as a numpy array using proper PyBoy API"""
        if not self.initialized or self.pyboy is None:
            logger.warning("PyBoy not initialized, returning black screen")
            return np.zeros((144, 160, 3), dtype=np.uint8)

        try:
            # Use PyBoy's official screen API - get the screen buffer directly
            screen_array = self.pyboy.screen.ndarray

            # Validate screen array dimensions
            if screen_array is None or screen_array.size == 0:
                logger.warning("Screen buffer is empty, returning black screen")
                return np.zeros((144, 160, 3), dtype=np.uint8)

            # PyBoy returns RGBA format (144, 160, 4) according to official documentation
            # Convert to RGB by removing alpha channel
            if len(screen_array.shape) == 3 and screen_array.shape[2] == 4:
                screen_array = screen_array[:, :, :3]  # Remove alpha channel
            elif len(screen_array.shape) != 3 or screen_array.shape[2] != 3:
                logger.error(f"Unexpected screen format from PyBoy: {screen_array.shape}")
                return np.zeros((144, 160, 3), dtype=np.uint8)

            # Ensure proper data type (uint8)
            if screen_array.dtype != np.uint8:
                screen_array = screen_array.astype(np.uint8)

            # Validate final dimensions
            if screen_array.shape != (144, 160, 3):
                logger.warning(f"Screen shape {screen_array.shape} doesn't match expected (144, 160, 3)")
                return np.zeros((144, 160, 3), dtype=np.uint8)

            return screen_array

        except Exception as e:
            logger.error(f"Error getting screen from PyBoy: {e}")
            return np.zeros((144, 160, 3), dtype=np.uint8)

    def get_screen_bytes(self) -> bytes:
        """Get the current screen as bytes for AI processing using proper PyBoy API"""
        try:
            # Get screen as numpy array first
            screen_array = self.get_screen()

            # Convert to JPEG bytes
            import io
            from PIL import Image

            img_buffer = io.BytesIO()
            Image.fromarray(screen_array).save(img_buffer, format='JPEG', quality=85, optimize=True)
            img_bytes = img_buffer.getvalue()

            logger.debug(f"Screen converted to bytes: {len(img_bytes)} bytes")
            return img_bytes

        except Exception as e:
            logger.error(f"Error converting screen to bytes: {e}")
            # Return minimal valid JPEG data as fallback
            return b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\xff\xd9'

    def get_memory(self, address: int, size: int = 1) -> bytes:
        """Read memory from the emulator"""
        if not self.initialized or self.pyboy is None:
            logger.warning("PyBoy not initialized, returning zeros")
            return b'\x00' * size

        # Validate memory address range (Game Boy has 64KB address space)
        if address < 0 or address > 0xFFFF:
            logger.error(f"Invalid memory address: {hex(address)}")
            return b'\x00' * size

        if size < 1 or (address + size) > 0x10000:
            logger.error(f"Invalid memory size or range: address={hex(address)}, size={size}")
            return b'\x00' * size

        try:
            if size == 1:
                return bytes([self.pyboy.memory[address]])
            else:
                return bytes(self.pyboy.memory[address:address + size])
        except Exception as e:
            logger.error(f"Error reading memory at {hex(address)}: {e}")
            return b'\x00' * size

    def set_memory(self, address: int, value: bytes) -> bool:
        """Write memory to the emulator"""
        if not self.initialized or self.pyboy is None:
            logger.warning("PyBoy not initialized, cannot write memory")
            return False

        # Validate memory address range
        if address < 0 or address > 0xFFFF:
            logger.error(f"Invalid memory address: {hex(address)}")
            return False

        if len(value) < 1 or (address + len(value)) > 0x10000:
            logger.error(f"Invalid memory write range: address={hex(address)}, size={len(value)}")
            return False

        try:
            if len(value) == 1:
                self.pyboy.memory[address] = value[0]
            else:
                self.pyboy.memory[address:address + len(value)] = list(value)
            logger.debug(f"Memory written to {hex(address)}: {value.hex()}")
            return True
        except Exception as e:
            logger.error(f"Error writing memory at {hex(address)}: {e}")
            return False

    def reset(self) -> bool:
        """Reset the emulator using simplified approach"""
        if not self.initialized or self.pyboy is None:
            return False

        try:
            # Stop the current emulator
            self.pyboy.stop()

            # Re-initialize PyBoy with simplified configuration
            self.pyboy = PyBoy(
                self.rom_path,
                window="SDL2" if self.auto_launch_ui else "null",
                scale=2,
                sound_emulated=True,
                sound_volume=50
            )

            # Set emulation speed to unlimited for AI training
            self.pyboy.set_emulation_speed(0)

            # Re-initialize game wrapper
            self.game_wrapper = self.pyboy.game_wrapper

            # Start with one tick
            self.pyboy.tick(1, self.auto_launch_ui)

            logger.info("Emulator reset successfully")
            return True

        except Exception as e:
            logger.error(f"Error resetting emulator: {e}")
            return False

    def save_state(self) -> bytes:
        """Save the current state of the emulator"""
        if not self.initialized or self.pyboy is None:
            logger.warning("PyBoy not initialized, cannot save state")
            return b''

        try:
            # Save state to bytes using PyBoy's save_state method
            state_buffer = io.BytesIO()
            self.pyboy.save_state(state_buffer)
            state_data = state_buffer.getvalue()
            logger.info(f"State saved successfully: {len(state_data)} bytes")
            return state_data
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return b''

    def load_state(self, state: bytes) -> bool:
        """Load a saved state into the emulator"""
        if not self.initialized or self.pyboy is None:
            logger.warning("PyBoy not initialized, cannot load state")
            return False

        if not state:
            logger.warning("Empty state data provided")
            return False

        try:
            # Load state from bytes using PyBoy's load_state method
            state_buffer = io.BytesIO(state)
            self.pyboy.load_state(state_buffer)
            logger.info(f"State loaded successfully: {len(state)} bytes")
            return True
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False

    def get_info(self) -> dict:
        """Get information about the current game state"""
        if not self.initialized or self.pyboy is None:
            return {"error": "PyBoy not initialized"}

        try:
            info = {
                "rom_title": self.pyboy.cartridge_title,
                "frame_count": self.pyboy.frame_count,
                "initialized": self.initialized,
                "game_title": self.game_title,
                "rom_path": self.rom_path,
                "emulation_speed": "unlimited"  # We set this to 0 for AI training
            }

            # Add screen information if available
            try:
                screen_shape = self.pyboy.screen.ndarray.shape
                info["screen_size"] = screen_shape
                info["screen_format"] = "RGBA" if len(screen_shape) == 3 and screen_shape[2] == 4 else "RGB"
            except Exception as screen_e:
                logger.warning(f"Could not get screen info: {screen_e}")
                info["screen_size"] = "unknown"
                info["screen_format"] = "unknown"

            return info
        except Exception as e:
            logger.error(f"Error getting info: {e}")
            return {"error": str(e), "initialized": self.initialized}

    def get_game_state_analysis(self) -> dict:
        """Get a detailed analysis of the current game state"""
        if not self.initialized or self.pyboy is None:
            return {}

        try:
            # Get basic info
            info = self.get_info()

            # Get screen analysis
            screen = self.get_screen()

            # Get memory regions of interest (this would be game-specific)
            # For now, we'll just get some general memory values
            memory_analysis = {}

            # Add game-specific analysis based on the game title
            game_specific = self._get_game_specific_analysis()

            return {
                "basic_info": info,
                "screen_analysis": {
                    "shape": screen.shape,
                    "mean_color": screen.mean(axis=(0,1)).tolist(),
                    "unique_colors": len(np.unique(screen.reshape(-1, screen.shape[2]), axis=0))
                },
                "memory_analysis": memory_analysis,
                "game_specific": game_specific
            }
        except Exception as e:
            print(f"Error getting game state analysis: {e}")
            return {}

    def _get_game_specific_analysis(self) -> dict:
        """Get game-specific analysis based on the game title"""
        if not self.initialized or self.pyboy is None:
            return {}

        game_title = self.game_title.lower()

        # Placeholder for game-specific analysis
        # In a real implementation, this would have specific logic for each game
        if "pokemon" in game_title:
            return self._get_pokemon_analysis()
        elif "tetris" in game_title:
            return self._get_tetris_analysis()
        elif "mario" in game_title:
            return self._get_mario_analysis()
        else:
            return {"game_type": "unknown", "analysis": "No specific analysis available for this game"}

    def _get_pokemon_analysis(self) -> dict:
        """Get Pokemon-specific game analysis"""
        # Placeholder for Pokemon-specific analysis
        # This would read memory addresses specific to Pokemon games
        return {
            "game_type": "pokemon",
            "player_position": "unknown",
            "current_party": [],
            "battle_status": "unknown"
        }

    def _get_tetris_analysis(self) -> dict:
        """Get Tetris-specific game analysis"""
        # Placeholder for Tetris-specific analysis
        return {
            "game_type": "tetris",
            "current_piece": "unknown",
            "next_piece": "unknown",
            "lines_cleared": 0
        }

    def _get_mario_analysis(self) -> dict:
        """Get Mario-specific game analysis"""
        # Placeholder for Mario-specific analysis
        return {
            "game_type": "mario",
            "player_position": "unknown",
            "lives": 0,
            "coins": 0
        }

    # UI Management Methods (Simplified)
    def set_auto_launch_ui(self, enabled: bool):
        """Enable or disable automatic UI launching"""
        self.auto_launch_ui = enabled
        logger.info(f"Auto-launch UI set to: {enabled}")

    def get_auto_launch_ui(self) -> bool:
        """Get current auto-launch UI setting"""
        return self.auto_launch_ui

    def get_ui_status(self) -> dict:
        """Get UI status (simplified)"""
        return {
            "running": self.auto_launch_ui and self.initialized,
            "ready": self.initialized,
            "rom_path": self.rom_path,
            "window_type": "SDL2" if self.auto_launch_ui else "null",
            "auto_launch_enabled": self.auto_launch_ui,
            "ui_launched": self.ui_launched
        }

    def cleanup(self) -> bool:
        """Clean up resources and stop the emulator"""
        try:
            logger.info("Cleaning up PyBoy emulator resources")

            # Stop PyBoy emulator
            if self.pyboy is not None:
                logger.info("Stopping PyBoy emulator")
                self.pyboy.stop()
                self.pyboy = None

            # Reset state
            self.initialized = False
            self.rom_path = None
            self.game_title = ""
            self.ui_launched = False
            self.game_wrapper = None

            logger.info("PyBoy emulator cleanup completed")
            return True

        except Exception as e:
            logger.error(f"Error during PyBoy cleanup: {e}")
            return False

    def is_running(self) -> bool:
        """Check if the emulator is currently running"""
        return self.initialized and self.pyboy is not None

    def get_frame_count(self) -> int:
        """Get the current frame count"""
        if not self.initialized or self.pyboy is None:
            return 0
        try:
            return self.pyboy.frame_count
        except Exception as e:
            logger.error(f"Error getting frame count: {e}")
            return 0

    def set_emulation_speed(self, speed: int) -> bool:
        """Set the emulation speed (0 = unlimited, 1 = normal, >1 = faster)"""
        if not self.initialized or self.pyboy is None:
            return False
        try:
            self.pyboy.set_emulation_speed(speed)
            logger.info(f"Emulation speed set to: {speed}")
            return True
        except Exception as e:
            logger.error(f"Error setting emulation speed: {e}")
            return False

    def run_game_loop(self, max_frames: int = 1000, render: bool = True) -> bool:
        """
        Run a standard game loop based on official PyBoy examples
        This method follows the pattern shown in PyBoy documentation

        Args:
            max_frames: Maximum number of frames to run
            render: Whether to render frames (for UI display)

        Returns:
            bool: True if loop completed successfully
        """
        if not self.initialized or self.pyboy is None:
            logger.error("PyBoy not initialized, cannot run game loop")
            return False

        try:
            logger.info(f"Starting game loop: max_frames={max_frames}, render={render}")

            frame_count = 0
            while frame_count < max_frames:
                # Progress the emulator by one frame
                # Based on official example: while pyboy.tick(): pass
                if not self.pyboy.tick(1, render):
                    logger.info("Game loop ended naturally (tick returned False)")
                    break

                frame_count += 1

                # Log progress every 60 frames (1 second at 60fps)
                if frame_count % 60 == 0:
                    logger.debug(f"Game loop progress: {frame_count}/{max_frames} frames")

            logger.info(f"Game loop completed: {frame_count} frames processed")
            return True

        except Exception as e:
            logger.error(f"Error in game loop: {e}")
            return False

    def run_ai_training_loop(self, episodes: int = 10, max_frames_per_episode: int = 1000) -> dict:
        """
        Run an AI training loop with proper episode management
        This is optimized for AI training with minimal rendering

        Args:
            episodes: Number of training episodes to run
            max_frames_per_episode: Maximum frames per episode

        Returns:
            dict: Training statistics
        """
        if not self.initialized or self.pyboy is None:
            logger.error("PyBoy not initialized, cannot run training loop")
            return {"error": "PyBoy not initialized"}

        try:
            logger.info(f"Starting AI training loop: episodes={episodes}, max_frames={max_frames_per_episode}")

            training_stats = {
                "episodes_completed": 0,
                "total_frames": 0,
                "episode_lengths": [],
                "success": True
            }

            for episode in range(episodes):
                logger.info(f"Starting episode {episode + 1}/{episodes}")

                # Reset emulator for new episode
                if episode > 0:
                    self.reset()

                episode_frames = 0
                episode_reward = 0  # Placeholder for reward tracking

                # Run episode with minimal rendering for performance
                while episode_frames < max_frames_per_episode:
                    # For AI training, we typically don't render every frame
                    # Only render on last frame for any needed screen capture
                    render_this_frame = (episode_frames == max_frames_per_episode - 1)

                    if not self.pyboy.tick(1, render_this_frame):
                        logger.info(f"Episode {episode + 1} ended naturally")
                        break

                    episode_frames += 1

                    # Here you would typically:
                    # 1. Get screen state
                    # 2. AI makes decision
                    # 3. Execute action
                    # 4. Calculate reward
                    # For now, just tick with NOOP

                    # Log progress
                    if episode_frames % 300 == 0:  # Every 5 seconds at 60fps
                        logger.debug(f"Episode {episode + 1}: {episode_frames}/{max_frames_per_episode} frames")

                # Record episode stats
                training_stats["episodes_completed"] += 1
                training_stats["total_frames"] += episode_frames
                training_stats["episode_lengths"].append(episode_frames)

                logger.info(f"Episode {episode + 1} completed: {episode_frames} frames")

            logger.info(f"Training loop completed: {training_stats}")
            return training_stats

        except Exception as e:
            logger.error(f"Error in training loop: {e}")
            return {"error": str(e), "success": False}

    def capture_screenshot(self, filepath: str = None) -> bool:
        """
        Capture a screenshot using PyBoy's official screen API
        Based on the official example: pyboy.screen.image.save()

        Args:
            filepath: Path to save screenshot. If None, uses temp file.

        Returns:
            bool: True if screenshot captured successfully
        """
        if not self.initialized or self.pyboy is None:
            logger.error("PyBoy not initialized, cannot capture screenshot")
            return False

        try:
            # Ensure we have a rendered frame
            self.pyboy.tick(1, True)

            # Use PyBoy's screen.image property (PIL Image)
            screen_image = self.pyboy.screen.image

            if screen_image is None:
                logger.error("Screen image is None")
                return False

            # Generate filename if not provided
            if filepath is None:
                import tempfile
                filepath = tempfile.mktemp(suffix=".png")

            # Save the screenshot
            screen_image.save(filepath)
            logger.info(f"Screenshot saved to: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}")
            return False