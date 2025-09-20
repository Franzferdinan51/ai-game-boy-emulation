"""
Configuration classes for PyBoy RL environment.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json


class ObservationType(Enum):
    """Types of observation spaces."""
    SCREEN = "screen"
    GAME_AREA = "game_area"
    MEMORY = "memory"
    TILES = "tiles"
    MULTI = "multi"


class ActionType(Enum):
    """Types of action spaces."""
    DISCRETE = "discrete"
    MULTI_DISCRETE = "multi_discrete"
    CONTINUOUS = "continuous"
    HYBRID = "hybrid"


@dataclass
class ObservationConfig:
    """Configuration for observation space."""
    type: ObservationType = ObservationType.SCREEN
    grayscale: bool = False
    resize_observation: bool = False
    target_size: Tuple[int, int] = (84, 84)
    memory_start: int = 0xC000
    memory_size: int = 1024
    tiles_width: int = 20
    tiles_height: int = 18
    include_screen: bool = True
    include_game_area: bool = True
    include_memory: bool = False
    stack_frames: int = 1  # Number of frames to stack


@dataclass
class ActionConfig:
    """Configuration for action space."""
    type: ActionType = ActionType.DISCRETE
    include_noop: bool = True
    allow_multiple_buttons: bool = False
    allow_button_holding: bool = True
    max_action_duration: int = 10
    button_press_duration: int = 1
    action_repeat: int = 1
    available_buttons: List[str] = field(default_factory=lambda: [
        "UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"
    ])
    custom_mappings: List[Dict[str, Any]] = field(default_factory=list)
    continuous_action_threshold: float = 0.5


@dataclass
class RewardConfig:
    """Configuration for reward system."""
    type: str = "memory_based"  # "memory_based", "score_based", "custom"
    weight: float = 1.0
    memory_addresses: List[Dict[str, Any]] = field(default_factory=list)
    reward_on_increase: bool = True
    reward_on_decrease: bool = False
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None
    cooldown: int = 0
    custom_function: Optional[str] = None  # Function name or path


@dataclass
class StateTrackingConfig:
    """Configuration for state tracking."""
    track_basic_state: bool = True
    track_memory_regions: bool = True
    track_screen_analysis: bool = True
    track_sprites: bool = True
    track_tiles: bool = True
    track_input_history: bool = True
    track_performance: bool = True
    memory_regions: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0xC000, 0xCFFF),  # WRAM
        (0xD000, 0xDFFF),  # WRAM bank switchable
        (0xFF00, 0xFFFF),  # Hardware registers
    ])
    memory_scan_interval: int = 60
    state_history_length: int = 1000
    input_history_length: int = 100
    custom_trackers: Dict[str, str] = field(default_factory=dict)  # name -> function path


@dataclass
class RLEnvironmentConfig:
    """
    Complete configuration for PyBoy RL environment.
    """

    # Basic configuration
    headless: bool = True
    window_scale: int = 1
    log_level: str = "WARNING"

    # PyBoy additional arguments
    pyboy_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Observation configuration
    observation_config: ObservationConfig = field(default_factory=ObservationConfig)

    # Action configuration
    action_config: ActionConfig = field(default_factory=ActionConfig)

    # Reward system configuration
    reward_configs: List[RewardConfig] = field(default_factory=list)
    base_reward: float = 0.0
    step_penalty: float = 0.0
    time_penalty: float = 0.0
    death_penalty: float = 0.0
    combo_multiplier: float = 1.0
    max_combo_length: int = 10
    normalize_rewards: bool = True
    reward_history_length: int = 1000

    # State tracking configuration
    state_config: StateTrackingConfig = field(default_factory=StateTrackingConfig)

    # Episode configuration
    frames_per_action: int = 1
    max_steps: Optional[int] = None
    max_episode_length: Optional[int] = None
    time_limit: Optional[float] = None
    max_history_length: int = 10000
    button_press_duration: int = 1

    # Game-specific configurations
    auto_detect_game: bool = True
    game_specific_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Training configuration
    save_states: bool = False
    state_save_interval: int = 1000
    log_interval: int = 100
    evaluation_interval: int = 10000

    # Rendering configuration
    render_mode: Optional[str] = None
    render_fps: int = 60

    def __post_init__(self):
        """Post-initialization setup."""
        # Setup game-specific configurations if auto-detect is enabled
        if self.auto_detect_game and not self.game_specific_configs:
            self.game_specific_configs = self._get_default_game_configs()

        # Setup default reward configurations if none provided
        if not self.reward_configs:
            self.reward_configs = self._get_default_reward_configs()

    def _get_default_game_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get default game-specific configurations."""
        return {
            "pokemon": {
                "observation_config": ObservationConfig(
                    type=ObservationType.MULTI,
                    include_screen=True,
                    include_game_area=True,
                    include_memory=True,
                    memory_start=0xD18C,  # Pokemon experience
                    memory_size=256
                ),
                "action_config": ActionConfig(
                    type=ActionType.DISCRETE,
                    include_noop=True,
                    available_buttons=["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]
                ),
                "reward_configs": [
                    RewardConfig(
                        type="experience",
                        memory_addresses=[{"address": 0xD18C, "name": "current_exp"}],
                        reward_on_increase=True,
                        weight=1.0
                    ),
                    RewardConfig(
                        type="level",
                        memory_addresses=[{"address": 0xD18D, "name": "pokemon_level"}],
                        reward_on_increase=True,
                        weight=5.0
                    ),
                    RewardConfig(
                        type="badges",
                        memory_addresses=[{"address": 0xD362, "name": "badges_obtained"}],
                        reward_on_increase=True,
                        weight=2.0
                    )
                ]
            },
            "mario": {
                "observation_config": ObservationConfig(
                    type=ObservationType.SCREEN,
                    grayscale=False,
                    stack_frames=4
                ),
                "action_config": ActionConfig(
                    type=ActionType.DISCRETE,
                    include_noop=True,
                    available_buttons=["UP", "DOWN", "LEFT", "RIGHT", "A", "B"]
                ),
                "reward_configs": [
                    RewardConfig(
                        type="score",
                        memory_addresses=[
                            {"address": 0xC0A0, "name": "score_high", "data_type": "uint16"},
                            {"address": 0xC0A2, "name": "score_low", "data_type": "uint16"}
                        ],
                        reward_on_increase=True,
                        weight=1.0
                    ),
                    RewardConfig(
                        type="coins",
                        memory_addresses=[{"address": 0xC0AD, "name": "coins"}],
                        reward_on_increase=True,
                        weight=2.0
                    ),
                    RewardConfig(
                        type="progress",
                        memory_addresses=[
                            {"address": 0xC0AB, "name": "current_level"},
                            {"address": 0xC0A4, "name": "world_position"}
                        ],
                        reward_on_increase=True,
                        weight=1.5
                    )
                ]
            },
            "tetris": {
                "observation_config": ObservationConfig(
                    type=ObservationType.GAME_AREA,
                    tiles_width=10,
                    tiles_height=20
                ),
                "action_config": ActionConfig(
                    type=ActionType.DISCRETE,
                    include_noop=True,
                    available_buttons=["LEFT", "RIGHT", "DOWN", "A", "B", "START"]
                ),
                "reward_configs": [
                    RewardConfig(
                        type="score",
                        memory_addresses=[
                            {"address": 0xC0A0, "name": "score_high", "data_type": "uint16"},
                            {"address": 0xC0A2, "name": "score_low", "data_type": "uint16"}
                        ],
                        reward_on_increase=True,
                        weight=1.0
                    ),
                    RewardConfig(
                        type="lines",
                        memory_addresses=[{"address": 0xC0B0, "name": "lines_cleared"}],
                        reward_on_increase=True,
                        weight=3.0
                    ),
                    RewardConfig(
                        type="level",
                        memory_addresses=[{"address": 0xC0AE, "name": "current_level"}],
                        reward_on_increase=True,
                        weight=5.0
                    )
                ]
            },
            "zelda": {
                "observation_config": ObservationConfig(
                    type=ObservationType.MULTI,
                    include_screen=True,
                    include_memory=True,
                    memory_start=0xC0F0,  # Health
                    memory_size=128
                ),
                "action_config": ActionConfig(
                    type=ActionType.HYBRID,
                    include_noop=True,
                    available_buttons=["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]
                ),
                "reward_configs": [
                    RewardConfig(
                        type="health",
                        memory_addresses=[
                            {"address": 0xC0F0, "name": "current_hearts"},
                            {"address": 0xC0F1, "name": "max_hearts"}
                        ],
                        reward_on_increase=True,
                        weight=2.0
                    ),
                    RewardConfig(
                        type="items",
                        memory_addresses=[{"address": 0xC0E0, "name": "items_collected"}],
                        reward_on_increase=True,
                        weight=1.0
                    ),
                    RewardConfig(
                        type="progress",
                        memory_addresses=[{"address": 0xC0C0, "name": "dungeons_completed"}],
                        reward_on_increase=True,
                        weight=3.0
                    )
                ]
            }
        }

    def _get_default_reward_configs(self) -> List[RewardConfig]:
        """Get default reward configurations."""
        return [
            RewardConfig(
                type="exploration",
                weight=0.1,
                memory_addresses=[
                    {"address": 0xC0A0, "name": "position_x"},
                    {"address": 0xC0A1, "name": "position_y"}
                ]
            ),
            RewardConfig(
                type="progress",
                weight=0.5,
                memory_addresses=[{"address": 0xC0B0, "name": "game_progress"}]
            )
        ]

    def get_game_config(self, game_title: str) -> Dict[str, Any]:
        """Get game-specific configuration."""
        game_title_lower = game_title.lower()

        # Check for exact matches
        for game_name, config in self.game_specific_configs.items():
            if game_name in game_title_lower:
                return config

        # Check for partial matches
        for game_name, config in self.game_specific_configs.items():
            if any(keyword in game_title_lower for keyword in game_name.split('_')):
                return config

        # Return default configuration
        return {}

    def update_for_game(self, game_title: str):
        """Update configuration for specific game."""
        game_config = self.get_game_config(game_title)

        if game_config:
            # Update observation config
            if 'observation_config' in game_config:
                self.observation_config = game_config['observation_config']

            # Update action config
            if 'action_config' in game_config:
                self.action_config = game_config['action_config']

            # Update reward configs
            if 'reward_configs' in game_config:
                self.reward_configs = game_config['reward_configs']

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                result = {}
                for field_name, field_value in obj.__dict__.items():
                    result[field_name] = dataclass_to_dict(field_value)
                return result
            elif isinstance(obj, (list, tuple)):
                return [dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: dataclass_to_dict(value) for key, value in obj.items()}
            elif isinstance(obj, Enum):
                return obj.value
            else:
                return obj

        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RLEnvironmentConfig':
        """Create configuration from dictionary."""
        # Create nested dataclass objects
        observation_config = ObservationConfig(**config_dict.get('observation_config', {}))
        action_config = ActionConfig(**config_dict.get('action_config', {}))
        state_config = StateTrackingConfig(**config_dict.get('state_config', {}))

        # Create reward configs
        reward_configs = []
        for reward_dict in config_dict.get('reward_configs', []):
            reward_configs.append(RewardConfig(**reward_dict))

        # Create main config
        config = cls(
            observation_config=observation_config,
            action_config=action_config,
            state_config=state_config,
            reward_configs=reward_configs
        )

        # Set other attributes
        for key, value in config_dict.items():
            if key not in ['observation_config', 'action_config', 'state_config', 'reward_configs']:
                setattr(config, key, value)

        return config

    def save(self, filepath: str):
        """Save configuration to file."""
        config_dict = self.to_dict()

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'RLEnvironmentConfig':
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    def copy(self) -> 'RLEnvironmentConfig':
        """Create a copy of the configuration."""
        return RLEnvironmentConfig.from_dict(self.to_dict())

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Validate observation config
        if self.observation_config.memory_size <= 0:
            errors.append("memory_size must be positive")

        if self.observation_config.memory_start < 0 or self.observation_config.memory_start > 0xFFFF:
            errors.append("memory_start must be between 0 and 0xFFFF")

        # Validate action config
        if not self.action_config.available_buttons:
            errors.append("available_buttons cannot be empty")

        # Validate reward configs
        for reward_config in self.reward_configs:
            if reward_config.weight < 0:
                errors.append(f"reward weight cannot be negative: {reward_config.type}")

        # Validate episode config
        if self.frames_per_action <= 0:
            errors.append("frames_per_action must be positive")

        if self.max_steps is not None and self.max_steps <= 0:
            errors.append("max_steps must be positive")

        if self.max_episode_length is not None and self.max_episode_length <= 0:
            errors.append("max_episode_length must be positive")

        return errors

    def __str__(self) -> str:
        """String representation of configuration."""
        return f"RLEnvironmentConfig(observation={self.observation_config.type.value}, action={self.action_config.type.value})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"RLEnvironmentConfig("
                f"observation={self.observation_config.type.value}, "
                f"action={self.action_config.type.value}, "
                f"headless={self.headless}, "
                f"frames_per_action={self.frames_per_action}, "
                f"max_steps={self.max_steps}, "
                f"reward_configs={len(self.reward_configs)})")