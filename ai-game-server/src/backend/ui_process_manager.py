"""
UI Process Manager for PyBoy integration
Handles launching and managing separate PyBoy UI processes
"""
import os
import sys
import subprocess
import time
import signal
import tempfile
import json
import logging
import threading
import multiprocessing
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from .ui_config import get_ui_config
except ImportError:
    # Fallback for different import contexts
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from ui_config import get_ui_config

logger = logging.getLogger(__name__)


class PyBoyUIProcessManager:
    """Manages separate PyBoy UI process for local display"""

    def __init__(self):
        self.ui_process = None
        self.rom_path = None
        self.process_lock = threading.Lock()
        self.ui_ready = False
        self.ui_port = None
        self.config = get_ui_config()
        self.config.update_from_env()

        # Load timeouts from config
        process_config = self.config.get_process_config()
        self.startup_timeout = process_config.get("startup_timeout", 10)
        self.shutdown_timeout = process_config.get("shutdown_timeout", 5)

    def is_ui_running(self) -> bool:
        """Check if UI process is currently running"""
        if self.ui_process is None:
            return False

        try:
            # Check if process is still alive
            return self.ui_process.poll() is None
        except Exception as e:
            logger.error(f"Error checking UI process status: {e}")
            return False

    def launch_ui(self, rom_path: str, additional_args: Optional[Dict[str, Any]] = None) -> bool:
        """Launch a separate PyBoy UI process"""
        with self.process_lock:
            if self.is_ui_running():
                logger.warning("UI process already running, stopping it first")
                self.stop_ui()

            self.rom_path = rom_path
            self.ui_ready = False

            try:
                # Create a temporary directory for UI process communication
                temp_dir = tempfile.mkdtemp(prefix="pyboy_ui_")

                # Prepare UI process arguments
                ui_args = self._prepare_ui_args(rom_path, temp_dir, additional_args)

                # Prepare environment variables
                env = os.environ.copy()
                env.update({
                    'PYBOY_UI_MODE': 'true',
                    'PYBOY_ROM_PATH': rom_path,
                    'PYBOY_TEMP_DIR': temp_dir,
                    'PYBOY_LOG_LEVEL': 'INFO'
                })

                # Launch UI process
                logger.info(f"Launching PyBoy UI process: {' '.join(ui_args)}")

                # Platform-specific subprocess configuration
                if sys.platform == 'win32':
                    # Windows-specific configuration
                    self.ui_process = subprocess.Popen(
                        ui_args,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        stdin=subprocess.PIPE,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                    )
                else:
                    # Unix-like systems
                    self.ui_process = subprocess.Popen(
                        ui_args,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        stdin=subprocess.PIPE,
                        start_new_session=True
                    )

                # Wait for UI process to start up
                if self._wait_for_ui_startup():
                    logger.info("PyBoy UI process started successfully")
                    return True
                else:
                    logger.error("UI process failed to start within timeout")
                    self.stop_ui()
                    return False

            except Exception as e:
                logger.error(f"Error launching UI process: {e}")
                self.stop_ui()
                return False

    def _prepare_ui_args(self, rom_path: str, temp_dir: str, additional_args: Optional[Dict[str, Any]] = None) -> list:
        """Prepare arguments for UI process"""
        # Get the path to the UI launcher script
        current_dir = Path(__file__).parent
        ui_launcher = current_dir / "ui_launcher.py"

        if not ui_launcher.exists():
            logger.error(f"UI launcher script not found at: {ui_launcher}")
            # Fallback to direct PyBoy execution
            ui_args = [sys.executable, "-c", f"""
import sys
sys.path.insert(0, r'{current_dir}')
from ui_launcher import main
sys.argv = ['ui_launcher.py', r'{rom_path}', '--window=sdl2', '--scale=2', '--sound']
main()
"""]
        else:
            ui_args = [sys.executable, str(ui_launcher)]

        # Add ROM path
        ui_args.extend([rom_path])

        # Add configuration from UI config
        window_args = self.config.get_window_args()
        sound_args = self.config.get_sound_args()
        debug_args = self.config.get_debug_args()

        # Convert configuration to command line arguments
        ui_args.extend([
            f"--window={window_args['window']}",
            f"--scale={window_args['scale']}",
            f"--color-palette={window_args['color_palette']}",
            f"--sound={sound_args['sound']}",
            f"--sound-volume={sound_args['sound_volume']}",
            f"--sound-sample-rate={sound_args['sound_sample_rate']}",
            f"--log-level={debug_args['log_level']}"
        ])

        # Add any additional arguments
        if additional_args:
            for key, value in additional_args.items():
                if key.startswith("--"):
                    ui_args.extend([key, str(value)])
                else:
                    ui_args.extend([f"--{key}", str(value)])

        logger.info(f"UI process arguments: {ui_args}")
        return ui_args

    def _wait_for_ui_startup(self) -> bool:
        """Wait for UI process to start up and become ready"""
        start_time = time.time()

        while time.time() - start_time < self.startup_timeout:
            if not self.is_ui_running():
                logger.error("UI process died during startup")
                return False

            # Check for UI ready signal
            if self._check_ui_ready():
                self.ui_ready = True
                return True

            time.sleep(0.5)

        return False

    def _check_ui_ready(self) -> bool:
        """Check if UI process is ready by reading its output"""
        if self.ui_process is None:
            return False

        try:
            # Platform-specific check for UI readiness
            if sys.platform == 'win32':
                # Windows approach - just check if process is running
                return self.is_ui_running()
            else:
                # Unix approach - check stdout for signals
                try:
                    import select
                    import fcntl

                    # Set stdout to non-blocking
                    stdout_fd = self.ui_process.stdout.fileno()
                    flags = fcntl.fcntl(stdout_fd, fcntl.F_GETFL)
                    fcntl.fcntl(stdout_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

                    # Try to read available data
                    ready, _, _ = select.select([stdout_fd], [], [], 0)
                    if ready:
                        output = self.ui_process.stdout.read(1024)
                        if output:
                            output_str = output.decode('utf-8', errors='ignore')
                            logger.debug(f"UI output: {output_str}")

                            # Look for ready signal
                            if "UI_READY" in output_str or "PyBoy window created" in output_str:
                                return True
                except Exception:
                    pass

                # Fallback - just check if process is running
                return self.is_ui_running()

        except Exception as e:
            logger.debug(f"Error checking UI ready status: {e}")

        return False

    def stop_ui(self) -> bool:
        """Stop the UI process gracefully"""
        with self.process_lock:
            if self.ui_process is None:
                return True

            try:
                logger.info("Stopping PyBoy UI process")

                # Try graceful shutdown first
                if self.ui_process.poll() is None:
                    # Platform-specific termination
                    if sys.platform == 'win32':
                        # Windows termination
                        try:
                            self.ui_process.terminate()
                        except Exception:
                            pass
                    else:
                        # Unix-like systems - send SIGTERM to process group
                        try:
                            os.killpg(os.getpgid(self.ui_process.pid), signal.SIGTERM)
                        except (ProcessLookupError, AttributeError):
                            pass

                    # Wait for graceful shutdown
                    try:
                        self.ui_process.wait(timeout=self.shutdown_timeout)
                        logger.info("UI process stopped gracefully")
                    except subprocess.TimeoutExpired:
                        logger.warning("UI process did not stop gracefully, forcing termination")
                        # Force terminate
                        if sys.platform == 'win32':
                            try:
                                self.ui_process.kill()
                            except Exception:
                                pass
                        else:
                            try:
                                os.killpg(os.getpgid(self.ui_process.pid), signal.SIGKILL)
                            except (ProcessLookupError, AttributeError):
                                pass
                        self.ui_process.wait()

                # Clean up
                self.ui_process = None
                self.ui_ready = False
                self.rom_path = None

                return True

            except Exception as e:
                logger.error(f"Error stopping UI process: {e}")
                return False

    def get_ui_status(self) -> Dict[str, Any]:
        """Get current UI process status"""
        return {
            "running": self.is_ui_running(),
            "ready": self.ui_ready,
            "rom_path": self.rom_path,
            "pid": self.ui_process.pid if self.ui_process else None
        }

    def restart_ui(self, new_rom_path: Optional[str] = None) -> bool:
        """Restart UI process with optional new ROM"""
        rom_to_use = new_rom_path or self.rom_path

        if not rom_to_use:
            logger.error("No ROM path available for UI restart")
            return False

        self.stop_ui()
        time.sleep(1)  # Brief pause to ensure clean shutdown

        return self.launch_ui(rom_to_use)

    def send_ui_command(self, command: str, args: Optional[Dict[str, Any]] = None) -> bool:
        """Send a command to the UI process"""
        if not self.is_ui_running() or not self.ui_ready:
            logger.warning("UI process not running or not ready")
            return False

        try:
            # Prepare command data
            command_data = {
                "command": command,
                "args": args or {},
                "timestamp": time.time()
            }

            # Send command to UI process stdin
            command_json = json.dumps(command_data) + "\n"
            self.ui_process.stdin.write(command_json.encode('utf-8'))
            self.ui_process.stdin.flush()

            logger.debug(f"Sent command to UI: {command}")
            return True

        except Exception as e:
            logger.error(f"Error sending command to UI: {e}")
            return False

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_ui()


# Global UI process manager instance
ui_manager = PyBoyUIProcessManager()


def get_ui_manager() -> PyBoyUIProcessManager:
    """Get the global UI process manager instance"""
    return ui_manager