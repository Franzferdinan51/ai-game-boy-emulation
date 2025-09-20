"""
UI Configuration Module
Handles configuration settings for the PyBoy UI system
"""
import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Default UI configuration
DEFAULT_UI_CONFIG = {
    "window": {
        "type": "sdl2",
        "scale": 2,
        "color_palette": "grayscale",
        "fullscreen": False
    },
    "sound": {
        "enabled": True,
        "volume": 50,
        "sample_rate": 44100,
        "buffer_size": 1024,
        "emulated": True
    },
    "performance": {
        "frame_skip": 0,
        "target_fps": 60,
        "enable_vsync": True
    },
    "process": {
        "startup_timeout": 10,
        "shutdown_timeout": 5,
        "auto_restart": True,
        "cleanup_on_exit": True
    },
    "debug": {
        "log_level": "INFO",
        "show_fps": False,
        "enable_debug_overlay": False
    }
}

class UIConfig:
    """Manages UI configuration settings"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = DEFAULT_UI_CONFIG.copy()
        self.config_path = config_path or self._get_default_config_path()
        self.load_config()

    def _get_default_config_path(self) -> str:
        """Get the default configuration file path"""
        # Check for config file in various locations
        possible_paths = [
            os.path.join(os.getcwd(), "ui_config.json"),
            os.path.join(os.path.expanduser("~"), ".pyboy_ui_config.json"),
            os.path.join(Path(__file__).parent, "ui_config.json")
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Default to creating config in current directory
        return os.path.join(os.getcwd(), "ui_config.json")

    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)

                # Merge with default configuration
                self._merge_config(self.config, file_config)
                logger.info(f"UI configuration loaded from: {self.config_path}")
                return True
            else:
                logger.info("No UI config file found, using defaults")
                self.save_config()  # Save default config
                return False

        except Exception as e:
            logger.error(f"Error loading UI configuration: {e}")
            return False

    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"UI configuration saved to: {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving UI configuration: {e}")
            return False

    def _merge_config(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Recursively merge configuration dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'window.scale')"""
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any) -> bool:
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config

        try:
            # Navigate to the parent of the target key
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]

            # Set the value
            config[keys[-1]] = value
            return True
        except Exception as e:
            logger.error(f"Error setting config value '{key_path}': {e}")
            return False

    def get_window_args(self) -> Dict[str, Any]:
        """Get window configuration arguments for PyBoy"""
        return {
            "window": self.get("window.type", "sdl2"),
            "scale": self.get("window.scale", 2),
            "color_palette": self.get("window.color_palette", "grayscale")
        }

    def get_sound_args(self) -> Dict[str, Any]:
        """Get sound configuration arguments for PyBoy"""
        return {
            "sound": self.get("sound.enabled", True),
            "sound_volume": self.get("sound.volume", 50),
            "sound_sample_rate": self.get("sound.sample_rate", 44100),
            "sound_emulated": self.get("sound.emulated", True)
        }

    def get_process_config(self) -> Dict[str, Any]:
        """Get process management configuration"""
        return {
            "startup_timeout": self.get("process.startup_timeout", 10),
            "shutdown_timeout": self.get("process.shutdown_timeout", 5),
            "auto_restart": self.get("process.auto_restart", True),
            "cleanup_on_exit": self.get("process.cleanup_on_exit", True)
        }

    def get_performance_args(self) -> Dict[str, Any]:
        """Get performance configuration arguments"""
        return {
            "frame_skip": self.get("performance.frame_skip", 0),
            "target_fps": self.get("performance.target_fps", 60),
            "enable_vsync": self.get("performance.enable_vsync", True)
        }

    def get_debug_args(self) -> Dict[str, Any]:
        """Get debug configuration arguments"""
        return {
            "log_level": self.get("debug.log_level", "INFO"),
            "show_fps": self.get("debug.show_fps", False),
            "enable_debug_overlay": self.get("debug.enable_debug_overlay", False)
        }

    def update_from_env(self) -> None:
        """Update configuration from environment variables"""
        env_mappings = {
            "PYBOY_UI_WINDOW_TYPE": "window.type",
            "PYBOY_UI_SCALE": "window.scale",
            "PYBOY_UI_COLOR_PALETTE": "window.color_palette",
            "PYBOY_UI_SOUND_ENABLED": "sound.enabled",
            "PYBOY_UI_SOUND_VOLUME": "sound.volume",
            "PYBOY_UI_SAMPLE_RATE": "sound.sample_rate",
            "PYBOY_UI_STARTUP_TIMEOUT": "process.startup_timeout",
            "PYBOY_UI_SHUTDOWN_TIMEOUT": "process.shutdown_timeout",
            "PYBOY_UI_LOG_LEVEL": "debug.log_level",
            "PYBOY_UI_SHOW_FPS": "debug.show_fps"
        }

        for env_var, config_path in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if config_path.endswith(".enabled") or config_path.endswith(".show_fps"):
                    env_value = env_value.lower() in ("true", "1", "yes", "on")
                elif config_path.endswith((".scale", ".volume", ".timeout", ".sample_rate", ".target_fps", ".frame_skip")):
                    try:
                        env_value = int(env_value)
                    except ValueError:
                        logger.warning(f"Invalid integer value for {env_var}: {env_value}")
                        continue

                self.set(config_path, env_value)
                logger.debug(f"Updated config from environment: {config_path} = {env_value}")

    def validate_config(self) -> bool:
        """Validate the current configuration"""
        errors = []

        # Validate window settings
        window_type = self.get("window.type")
        if window_type not in ["sdl2", "opengl", "null"]:
            errors.append(f"Invalid window type: {window_type}")

        scale = self.get("window.scale")
        if not isinstance(scale, int) or scale < 1 or scale > 8:
            errors.append(f"Invalid scale value: {scale}")

        # Validate sound settings
        volume = self.get("sound.volume")
        if not isinstance(volume, int) or volume < 0 or volume > 100:
            errors.append(f"Invalid volume value: {volume}")

        sample_rate = self.get("sound.sample_rate")
        if sample_rate not in [22050, 44100, 48000]:
            errors.append(f"Invalid sample rate: {sample_rate}")

        # Validate timeouts
        startup_timeout = self.get("process.startup_timeout")
        if not isinstance(startup_timeout, int) or startup_timeout < 1 or startup_timeout > 60:
            errors.append(f"Invalid startup timeout: {startup_timeout}")

        if errors:
            logger.error("Configuration validation errors:")
            for error in errors:
                logger.error(f"  - {error}")
            return False

        return True

# Global configuration instance
ui_config = UIConfig()

def get_ui_config() -> UIConfig:
    """Get the global UI configuration instance"""
    return ui_config

def reload_ui_config() -> bool:
    """Reload the UI configuration"""
    return ui_config.load_config()