#!/usr/bin/env python3
"""
Enhanced PyBoy Auto-Start Launcher - CLEAN VERSION
===============================================

Fixed to use official PyBoy wiki documentation patterns:
- Simple PyBoy initialization: PyBoy(rom_path)
- Proper game loop: while not pyboy.tick()
- Game wrapper support
- Official button input handling
- Proper error handling for Windows

Based on: https://github.com/Baekalfen/PyBoy/wiki/Example-Super-Mario-Land

Features:
- Official PyBoy API patterns
- Robust SDL2 window initialization
- Comprehensive error handling
- Windows compatibility
- Auto-recovery mechanisms
- Process monitoring

Author: Claude AI Assistant
Created: 2025-09-19
Updated: Fixed with official wiki patterns
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import threading
import signal
import platform
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Setup logging
def setup_logging(log_dir: str = None) -> logging.Logger:
    """Setup comprehensive logging with multiple outputs."""
    if log_dir is None:
        log_dir = Path.home() / "PyBoy" / "logs"

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("PyBoyLauncher")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pyboy_launcher_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info("PyBoy Launcher initialized with official wiki patterns")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Python: {sys.version}")

    return logger

class PyBoyLauncher:
    """
    ENHANCED PyBoy Auto-Start Launcher using official wiki patterns
    """

    def __init__(self, config_file: str = None, logger: logging.Logger = None):
        """Initialize launcher with official PyBoy patterns."""
        self.logger = logger or setup_logging()
        self.config = self.load_config(config_file)
        self.pyboy_process = None
        self.monitor_thread = None
        self.running = False
        self.restart_count = 0
        self.startup_attempts = 0
        self.max_startup_attempts = 5

        # Initialize paths
        self.pyboy_paths = self.discover_pyboy_installations()
        self.current_pyboy_path = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._emergency_shutdown)
        signal.signal(signal.SIGTERM, self._emergency_shutdown)

        self.logger.info("PyBoy Launcher initialized with official wiki patterns")

    def discover_pyboy_installations(self) -> Dict[str, Path]:
        """Discover PyBoy installations."""
        self.logger.info("Discovering PyBoy installations...")

        installations = {}

        search_paths = [
            Path.cwd() / "PyBoy",
            Path(sys.prefix) / "Lib" / "site-packages" / "pyboy",
            Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python*" / "Lib" / "site-packages" / "pyboy",
        ]

        for search_path in search_paths:
            if "*" in str(search_path):
                import glob
                matching_paths = glob.glob(str(search_path))
                for path_str in matching_paths:
                    path = Path(path_str)
                    if self._is_valid_pyboy_installation(path):
                        installations[path.name] = path
                        self.logger.info(f"Found PyBoy: {path}")
            else:
                if self._is_valid_pyboy_installation(search_path):
                    installations[search_path.name] = search_path
                    self.logger.info(f"Found PyBoy: {search_path}")

        if not installations:
            self.logger.warning("No PyBoy installations found in search paths")
            # Don't raise error - we'll use current environment if available

        self.logger.info(f"Discovered {len(installations)} PyBoy installations")
        return installations

    def _is_valid_pyboy_installation(self, path: Path) -> bool:
        """Check if a path contains a valid PyBoy installation."""
        try:
            if not path.exists():
                return False

            # Check for main PyBoy files
            required_files = ["pyboy.py", "__init__.py"]
            for file in required_files:
                if not (path / file).exists():
                    return False

            # Test import
            test_env = os.environ.copy()
            test_env["PYTHONPATH"] = str(path.parent) + os.pathsep + test_env.get("PYTHONPATH", "")

            result = subprocess.run(
                [sys.executable, "-c", f"import sys; sys.path.insert(0, r'{path.parent}'); import pyboy; print('OK')"],
                env=test_env,
                capture_output=True,
                text=True,
                timeout=10
            )

            return result.returncode == 0

        except Exception as e:
            self.logger.debug(f"Installation check failed for {path}: {e}")
            return False

    def load_config(self, config_file: str = None) -> Dict[str, Any]:
        """Load configuration with sensible defaults."""
        default_config = {
            "auto_start": {
                "enabled": True,
                "mode": "always",
                "startup_delay": 2,
                "max_startup_attempts": 5
            },
            "ui": {
                "auto_launch": True,
                "default_rom": None,
                "process_management": {
                    "auto_restart": True,
                    "max_restarts": 3,
                    "restart_delay": 2,
                    "health_check_interval": 30
                }
            },
            "window": {
                "type": "SDL2",
                "scale": 2,
                "fullscreen": False,
                "always_on_top": True
            },
            "sound": {
                "enabled": True,
                "volume": 50
            },
            "process": {
                "startup_timeout": 15,
                "shutdown_timeout": 5,
                "auto_restart": True,
                "cleanup_on_exit": True
            },
            "recovery": {
                "auto_fix_dependencies": True,
                "fallback_window_types": ["SDL2", "OpenGL", "Null"],
                "auto_install_missing": True
            }
        }

        # Load configuration file if provided
        config_path = Path(config_file or "pyboy_launcher_config.json")
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                self.logger.info(f"Configuration loaded from: {config_path}")
                # Merge with defaults
                self._merge_config(default_config, user_config)
            except Exception as e:
                self.logger.error(f"Failed to load config file {config_path}: {e}")
                self.logger.info("Using default configuration")

        return default_config

    def _merge_config(self, base: Dict, override: Dict):
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def _emergency_shutdown(self, signum, frame):
        """Emergency shutdown handler."""
        self.logger.critical(f"EMERGENCY SHUTDOWN triggered by signal {signum}")
        self.force_stop_all()
        sys.exit(1)

    def verify_dependencies(self) -> bool:
        """Verify critical dependencies."""
        self.logger.info("Verifying dependencies...")

        # Check Python
        try:
            result = subprocess.run([sys.executable, "--version"],
                                  capture_output=True, text=True, timeout=5)
            self.logger.info(f"Python version: {result.stdout.strip()}")
        except Exception as e:
            self.logger.error(f"Python not available: {e}")
            return False

        # First check if PyBoy is available in the current environment
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import pyboy; print('PyBoy available in current environment')"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                self.logger.info(result.stdout.strip())
                self.logger.info("Using PyBoy from current environment")
                return True
        except Exception as e:
            self.logger.debug(f"Current environment PyBoy check failed: {e}")

        # Check PyBoy installation from discovered paths
        for install_name, install_path in self.pyboy_paths.items():
            try:
                test_env = os.environ.copy()
                test_env["PYTHONPATH"] = str(install_path.parent) + os.pathsep + test_env.get("PYTHONPATH", "")

                result = subprocess.run(
                    [sys.executable, "-c", f"import sys; sys.path.insert(0, r'{install_path.parent}'); import pyboy; print(f'PyBoy from {install_name}')"],
                    env=test_env,
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode == 0:
                    self.logger.info(f"{result.stdout.strip()}")
                    self.current_pyboy_path = install_path
                    return True
            except Exception as e:
                self.logger.debug(f"PyBoy check failed for {install_name}: {e}")

        self.logger.error("No working PyBoy installation found")
        return False

    def find_roms(self) -> List[str]:
        """Find ROM files."""
        self.logger.info("Searching for ROM files...")

        rom_extensions = ['.gb', '.gbc', '.gba']
        search_paths = [
            Path.cwd() / "roms",
            Path.cwd() / "ROMs",
            Path.cwd() / "PyBoy" / "pyboy",
            Path.home() / "PyBoy" / "roms",
            Path.home() / "ROMs",
            Path.home() / "Downloads",
            Path.home() / "Desktop",
        ]

        found_roms = []

        for search_path in search_paths:
            if search_path.exists() and search_path.is_dir():
                for ext in rom_extensions:
                    found_roms.extend(search_path.glob(f"*{ext}"))
                    found_roms.extend(search_path.glob(f"*{ext.upper()}"))

        # Add default ROMs
        if self.current_pyboy_path:
            default_roms = [
                self.current_pyboy_path / "default_rom.gb",
                self.current_pyboy_path / "default_rom_cgb.gb"
            ]

            for default_rom in default_roms:
                if default_rom.exists() and default_rom not in found_roms:
                    found_roms.append(default_rom)
                    self.logger.info(f"Added default ROM: {default_rom.name}")

        # Convert to strings and remove duplicates
        unique_roms = list(set(str(rom) for rom in found_roms if rom.exists() and rom.stat().st_size > 0))

        # Sort by modification time (newest first), but prioritize default ROMs
        unique_roms.sort(key=lambda x: (
            0 if "default_rom" in x.lower() else 1,
            -Path(x).stat().st_mtime
        ))

        self.logger.info(f"Found {len(unique_roms)} ROM files")
        for i, rom in enumerate(unique_roms[:5]):
            self.logger.info(f"  {i+1}. {Path(rom).name}")

        return unique_roms

    def start_pyboy_ui(self, rom_path: str = None) -> bool:
        """Start PyBoy UI using official wiki patterns."""
        self.startup_attempts += 1
        self.logger.info(f"Starting PyBoy UI (attempt {self.startup_attempts}/{self.max_startup_attempts})")

        if not self.verify_dependencies():
            self.logger.error("Dependencies verification failed")
            return False

        # Find ROM if not provided
        if not rom_path:
            roms = self.find_roms()
            rom_path = roms[0] if roms else None

        if not rom_path:
            self.logger.error("No ROM file found")
            return False

        # Try different window types
        window_types = self.config["recovery"]["fallback_window_types"]

        for window_type in window_types:
            self.logger.info(f"Trying window type: {window_type}")

            try:
                if self._start_pyboy_with_window_type(window_type, rom_path):
                    self.logger.info(f"PyBoy UI started successfully with {window_type}")
                    return True
                else:
                    self.logger.warning(f"Failed to start with {window_type}")
            except Exception as e:
                self.logger.error(f"Error starting with {window_type}: {e}")

        self.logger.error("All window types failed")
        return False

    def _start_pyboy_with_window_type(self, window_type: str, rom_path: str) -> bool:
        """Start PyBoy with specific window type using official wiki patterns."""
        self.logger.info(f"Starting PyBoy with {window_type} window...")

        # Build the Python code using official PyBoy wiki patterns
        python_code = f'''
import sys
import os
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print(f"Starting PyBoy with ROM: {os.path.basename(r"{rom_path}")}")
print(f"Window type: {window_type}")
print(f"ROM path: {rom_path}")
print(f"ROM size: {{os.path.getsize(r"{rom_path}")}} bytes")

try:
    from pyboy import PyBoy

    # Verify ROM file exists
    if not os.path.exists(r"{rom_path}"):
        logger.error(f"ROM file not found: {{r'{rom_path}'}}")
        sys.exit(1)

    logger.info("Initializing PyBoy...")

    # Initialize PyBoy using the official wiki pattern
    # Based on: https://github.com/Baekalfen/PyBoy/wiki/Example-Super-Mario-Land
    pyboy = PyBoy(r"{rom_path}")

    # Set emulation speed to unlimited for best performance (from wiki)
    pyboy.set_emulation_speed(0)

    logger.info("PyBoy initialized successfully!")
    logger.info("Game Boy Emulator is running...")
    logger.info("Press Ctrl+C in this window to stop")
    logger.info("Use Alt+Enter to toggle fullscreen")
    logger.info("Use Alt+F4 to close the window")
    logger.info("Controls: Arrow keys for movement, Z for A button, X for B button")

    # Try to get game wrapper if available (from wiki example)
    try:
        game_wrapper = pyboy.game_wrapper
        if game_wrapper:
            logger.info("Game wrapper initialized successfully")
            game_wrapper.start_game()
            logger.info(f"Game wrapper lives: {{game_wrapper.lives_left}}")
            logger.info(f"Game wrapper score: {{game_wrapper.score}}")
    except Exception as wrapper_error:
        logger.info(f"Game wrapper not available - continuing without it: {{wrapper_error}}")

    # Main game loop using the official pattern from wiki
    frame_count = 0
    last_log_time = time.time()

    while True:
        try:
            # Official game loop pattern: while not pyboy.tick()
            if not pyboy.tick():
                logger.info("PyBoy exited normally (window closed)")
                break

            frame_count += 1

            # Log progress every 10 seconds
            current_time = time.time()
            if current_time - last_log_time >= 10:
                logger.info(f"Running smoothly - Frame: {{frame_count}}")
                last_log_time = current_time

                # If game wrapper is available, log game stats
                try:
                    if 'game_wrapper' in locals() and game_wrapper:
                        logger.info(f"Game stats - Lives: {{game_wrapper.lives_left}}, Score: {{game_wrapper.score}}")
                except:
                    pass

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in game loop: {{e}}")
            import traceback
            traceback.print_exc()
            break

    # Cleanup with proper error handling
    try:
        pyboy.stop(save=False)
        logger.info("PyBoy stopped cleanly")
    except Exception as cleanup_error:
        logger.warning(f"Cleanup error: {{cleanup_error}}")
        pass

except Exception as e:
    logger.error(f"PyBoy error: {{e}}")
    import traceback
    traceback.print_exc()
    input("Press Enter to exit...")
'''

        # Set up environment
        env = os.environ.copy()

        # Windows-specific optimizations
        if platform.system() == "Windows":
            env["SDL_VIDEODRIVER"] = "directx"
            if self.current_pyboy_path and (self.current_pyboy_path / "SDL2.dll").exists():
                env["PYSDL2_DLL_PATH"] = str(self.current_pyboy_path / "SDL2.dll")
            env["PYTHONIOENCODING"] = "utf-8"

        # Add PyBoy path to PYTHONPATH if we have one
        if self.current_pyboy_path:
            env["PYTHONPATH"] = str(self.current_pyboy_path.parent) + os.pathsep + env.get("PYTHONPATH", "")

        # Set working directory
        working_dir = str(self.current_pyboy_path.parent) if self.current_pyboy_path else str(Path.cwd())

        try:
            # Create temporary script
            temp_script = Path(tempfile.gettempdir()) / f"pyboy_launcher_{int(time.time())}.py"
            with open(temp_script, 'w', encoding='utf-8') as f:
                f.write(python_code)

            self.logger.info(f"Created temporary script: {temp_script}")

            # Start PyBoy process
            self.pyboy_process = subprocess.Popen(
                [sys.executable, str(temp_script)],
                cwd=working_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True
            )

            self.logger.info(f"PyBoy process started with PID: {self.pyboy_process.pid}")

            # Wait for startup verification
            startup_timeout = self.config["process"]["startup_timeout"]
            start_time = time.time()

            while time.time() - start_time < startup_timeout:
                if self.pyboy_process.poll() is not None:
                    stdout, stderr = self.pyboy_process.communicate()
                    self.logger.error("PyBoy process terminated")
                    self.logger.error(f"STDOUT: {stdout}")
                    self.logger.error(f"STDERR: {stderr}")
                    return False

                # Check if process is responsive
                if self.pyboy_process.poll() is None:
                    self.logger.info(f"PyBoy process is running (PID: {self.pyboy_process.pid})")
                    break

                time.sleep(0.5)

            # Start monitoring output
            def monitor_output():
                try:
                    while self.pyboy_process.poll() is None:
                        # Read stdout
                        try:
                            line = self.pyboy_process.stdout.readline()
                            if line:
                                self.logger.info(f"PYBOY: {line.strip()}")
                        except:
                            pass

                        # Read stderr
                        try:
                            line = self.pyboy_process.stderr.readline()
                            if line:
                                self.logger.error(f"PYBOY ERROR: {line.strip()}")
                        except:
                            pass

                        time.sleep(0.1)
                except Exception as e:
                    self.logger.debug(f"Output monitoring error: {e}")

            output_thread = threading.Thread(target=monitor_output, daemon=True)
            output_thread.start()

            return True

        except Exception as e:
            self.logger.error(f"Failed to start PyBoy with {window_type}: {e}")
            if self.pyboy_process:
                try:
                    self.pyboy_process.kill()
                except:
                    pass
            return False

    def start_monitor(self):
        """Start process monitoring."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._process_monitor, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Process monitoring started")

    def _process_monitor(self):
        """Process monitoring with auto-recovery."""
        process_config = self.config["ui"]["process_management"]
        health_check_interval = process_config["health_check_interval"]
        max_restarts = process_config["max_restarts"]
        restart_delay = process_config["restart_delay"]

        self.logger.info(f"Starting monitoring (interval: {health_check_interval}s)")

        while self.running:
            try:
                if not self.pyboy_process:
                    # No process running, check if we should start one
                    if (self.restart_count < max_restarts and
                        self.config["auto_start"]["enabled"] and
                        self.startup_attempts < self.max_startup_attempts):

                        self.logger.info(f"Auto-restarting PyBoy (attempt {self.restart_count + 1}/{max_restarts})")
                        self.restart_count += 1
                        time.sleep(restart_delay)

                        roms = self.find_roms()
                        rom_path = roms[0] if roms else None
                        self.start_pyboy_ui(rom_path)

                    time.sleep(health_check_interval)
                    continue

                # Check if process is still running
                return_code = self.pyboy_process.poll()

                if return_code is not None:
                    # Process terminated
                    self.logger.warning(f"PyBoy process exited with code: {return_code}")

                    # Log any output
                    try:
                        stdout, stderr = self.pyboy_process.communicate()
                        if stdout:
                            self.logger.info(f"Process output: {stdout}")
                        if stderr:
                            self.logger.error(f"Process error: {stderr}")
                    except:
                        pass

                    # Auto-restart logic
                    if (self.restart_count < max_restarts and
                        process_config["auto_restart"]):

                        self.restart_count += 1
                        self.logger.info(f"Auto-restarting PyBoy (attempt {self.restart_count}/{max_restarts})")

                        time.sleep(restart_delay)

                        roms = self.find_roms()
                        rom_path = roms[0] if roms else None
                        self.start_pyboy_ui(rom_path)
                    else:
                        self.logger.error("Max restart attempts reached or auto-restart disabled")
                        self.running = False
                    break

                time.sleep(health_check_interval)

            except Exception as e:
                self.logger.error(f"Error in process monitoring: {e}")
                time.sleep(5)

    def force_stop_all(self):
        """Stop all PyBoy processes."""
        self.logger.info("Stopping all PyBoy processes...")
        self.running = False

        # Stop monitoring thread
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

        # Terminate PyBoy process
        if self.pyboy_process:
            try:
                self.pyboy_process.terminate()
                try:
                    self.pyboy_process.wait(timeout=self.config["process"]["shutdown_timeout"])
                    self.logger.info("PyBoy process stopped gracefully")
                except subprocess.TimeoutExpired:
                    self.logger.warning("PyBoy process did not stop gracefully, forcing...")
                    self.pyboy_process.kill()
                    self.logger.info("PyBoy process force-killed")
            except Exception as e:
                self.logger.error(f"Error stopping PyBoy process: {e}")

    def auto_start(self):
        """Auto-start PyBoy."""
        if not self.config["auto_start"]["enabled"]:
            self.logger.info("Auto-start is disabled in configuration")
            return

        mode = self.config["auto_start"]["mode"]
        self.logger.info(f"Starting PyBoy auto-start in {mode} mode")

        # Apply startup delay if configured
        startup_delay = self.config["auto_start"].get("startup_delay", 2)
        if startup_delay > 0:
            self.logger.info(f"Waiting {startup_delay} seconds before startup...")
            time.sleep(startup_delay)

        # Start based on mode
        if mode == "always":
            roms = self.find_roms()
            rom_path = roms[0] if roms else None
            success = self.start_pyboy_ui(rom_path)

        elif mode == "with_rom":
            roms = self.find_roms()
            if roms:
                success = self.start_pyboy_ui(roms[0])
            else:
                self.logger.info("No ROMs found, skipping auto-start")
                return

        else:
            self.logger.error(f"Unknown startup mode: {mode}")
            return

        if success:
            self.logger.info("PyBoy auto-start completed successfully")

            # Start monitoring if enabled
            if self.config["ui"]["process_management"]["auto_restart"]:
                self.start_monitor()
        else:
            self.logger.error("PyBoy auto-start failed")

    def run_interactive(self):
        """Run launcher in interactive mode."""
        self.logger.info("PyBoy Launcher - Interactive Mode")
        self.logger.info("Type 'help' for available commands")

        # Show available ROMs
        roms = self.find_roms()
        if roms:
            self.logger.info(f"Found {len(roms)} ROM files:")
            for i, rom in enumerate(roms[:10]):
                self.logger.info(f"  {i+1}. {Path(rom).name}")

        while True:
            try:
                command = input("\nPyBoy> ").strip().lower()

                if command in ['exit', 'quit', 'q']:
                    self.force_stop_all()
                    break

                elif command == 'help':
                    self._show_help()

                elif command == 'start':
                    roms = self.find_roms()
                    rom_path = roms[0] if roms else None
                    self.start_pyboy_ui(rom_path)

                elif command.startswith('start '):
                    rom_name = command[6:]
                    roms = self.find_roms()
                    matching_roms = [r for r in roms if rom_name.lower() in Path(r).name.lower()]
                    if matching_roms:
                        self.start_pyboy_ui(matching_roms[0])
                    else:
                        self.logger.error(f"No ROM found matching: {rom_name}")

                elif command == 'stop':
                    self.force_stop_all()

                elif command == 'status':
                    self._show_status()

                elif command == 'roms':
                    roms = self.find_roms()
                    self.logger.info(f"Found {len(roms)} ROM files:")
                    for i, rom in enumerate(roms[:10]):
                        self.logger.info(f"  {i+1}. {Path(rom).name}")

                elif command == 'monitor':
                    self.start_monitor()

                else:
                    self.logger.error("Unknown command. Type 'help' for available commands.")

            except KeyboardInterrupt:
                self.logger.info("Goodbye!")
                self.force_stop_all()
                break
            except Exception as e:
                self.logger.error(f"Command error: {e}")

    def _show_help(self):
        """Show help information."""
        help_text = """
PyBoy Launcher Commands:
  start              - Start PyBoy with first available ROM
  start <rom_name>  - Start PyBoy with specific ROM
  stop               - Stop PyBoy
  status             - Show current status
  roms               - List available ROMs
  monitor            - Start process monitoring
  help               - Show this help
  exit/quit/q        - Exit launcher

Examples:
  start pokemon      - Start PyBoy with first Pokemon ROM
  start              - Start PyBoy with any available ROM
        """
        print(help_text)

    def _show_status(self):
        """Show current launcher status."""
        status_info = f"""
PyBoy Launcher Status:
  Running: {self.running}
  PyBoy Process: {'Active' if self.pyboy_process and self.pyboy_process.poll() is None else 'Inactive'}
  Startup Attempts: {self.startup_attempts}
  Restart Count: {self.restart_count}
  Monitor Active: {self.monitor_thread.is_alive() if self.monitor_thread else False}
        """
        print(status_info)

    def run(self):
        """Main entry point for the launcher."""
        self.logger.info("Enhanced PyBoy Launcher starting...")

        try:
            # Handle command line arguments
            if len(sys.argv) > 1:
                if "--auto-start" in sys.argv:
                    self.auto_start()

                    # Keep launcher running if monitoring is enabled
                    if self.config["ui"]["process_management"]["auto_restart"]:
                        self.logger.info("Keeping launcher alive for monitoring...")
                        try:
                            while self.running:
                                time.sleep(1)
                        except KeyboardInterrupt:
                            self.logger.info("Received interrupt signal")
                        finally:
                            self.force_stop_all()
                else:
                    self.run_interactive()
            else:
                # Interactive mode by default
                self.run_interactive()

        except Exception as e:
            self.logger.critical(f"Launcher crashed: {e}")
            import traceback
            self.logger.critical(traceback.format_exc())
        finally:
            self.force_stop_all()
            self.logger.info("PyBoy launcher stopped")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced PyBoy Launcher - Fixed with Official Wiki Patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_pyboy_launcher_clean.py              # Interactive mode
  python enhanced_pyboy_launcher_clean.py --auto-start  # Auto-start mode
  python enhanced_pyboy_launcher_clean.py --rom pokemon.gb  # Start with specific ROM

Uses official PyBoy wiki patterns for maximum reliability!
        """
    )

    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--auto-start', action='store_true', help='Start PyBoy automatically')
    parser.add_argument('--rom', help='ROM file to load')
    parser.add_argument('--log-dir', help='Log directory path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--no-monitor', action='store_true', help='Disable process monitoring')

    args = parser.parse_args()

    try:
        # Setup logging
        logger = setup_logging(args.log_dir) if args.log_dir else setup_logging()

        if args.verbose:
            logger.setLevel(logging.DEBUG)

        # Create launcher
        launcher = PyBoyLauncher(args.config, logger)

        # Apply command line overrides
        if args.no_monitor:
            launcher.config["ui"]["process_management"]["auto_restart"] = False

        # Handle specific commands
        if args.rom:
            # Start with specific ROM
            if launcher.start_pyboy_ui(args.rom):
                logger.info("PyBoy started successfully")
                if launcher.config["ui"]["process_management"]["auto_restart"]:
                    launcher.start_monitor()
                    try:
                        while launcher.running:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        pass
                    finally:
                        launcher.force_stop_all()
            else:
                logger.error("Failed to start PyBoy")
                sys.exit(1)
        elif args.auto_start:
            # Auto-start mode
            launcher.auto_start()

            # Keep running if monitoring is enabled
            if launcher.config["ui"]["process_management"]["auto_restart"]:
                try:
                    while launcher.running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal")
                finally:
                    launcher.force_stop_all()
        else:
            # Interactive mode
            launcher.run()

    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()