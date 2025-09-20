"""
Action space management for PyBoy RL environment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import gym
from gym.spaces import Discrete, MultiDiscrete, Box, Tuple, Dict


class ActionType(Enum):
    """Types of action spaces."""
    DISCRETE = "discrete"
    MULTI_DISCRETE = "multi_discrete"
    CONTINUOUS = "continuous"
    HYBRID = "hybrid"


class ButtonAction(Enum):
    """Game Boy button actions."""
    NOOP = "NOOP"
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    A = "A"
    B = "B"
    START = "START"
    SELECT = "SELECT"


@dataclass
class ActionMapping:
    """Mapping between environment action and PyBoy button."""
    action_id: int
    buttons: List[str]
    duration: int = 1
    description: str = ""


@dataclass
class ActionConfig:
    """Configuration for action space."""
    action_type: ActionType = ActionType.DISCRETE
    include_noop: bool = True
    allow_multiple_buttons: bool = False
    allow_button_holding: bool = True
    max_action_duration: int = 10
    button_press_duration: int = 1
    action_repeat: int = 1
    available_buttons: List[str] = field(default_factory=lambda: [
        "UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"
    ])
    custom_mappings: List[ActionMapping] = field(default_factory=list)
    continuous_action_threshold: float = 0.5


class ActionManager:
    """
    Action space management for Game Boy games.

    This system handles different action space types:
    - Discrete: Single action from a predefined set
    - Multi-discrete: Multiple discrete actions
    - Continuous: Continuous action values
    - Hybrid: Combination of discrete and continuous
    """

    def __init__(self, pyboy, config: ActionConfig):
        """
        Initialize the action manager.

        Args:
            pyboy: PyBoy instance
            config: Action configuration
        """
        self.pyboy = pyboy
        self.config = config

        # Initialize action mappings
        self.action_mappings = []
        self._setup_action_mappings()

        # Initialize action space
        self.action_space = self._create_action_space()

        # Track button states
        self.button_states = {button: False for button in ButtonAction}
        self.button_timers = {button: 0 for button in ButtonAction}

        # Action history
        self.action_history = []

    def _setup_action_mappings(self):
        """Setup action mappings based on configuration."""
        if self.config.custom_mappings:
            # Use custom mappings
            self.action_mappings = self.config.custom_mappings
        else:
            # Generate default mappings
            self.action_mappings = self._generate_default_mappings()

    def _generate_default_mappings(self) -> List[ActionMapping]:
        """Generate default action mappings."""
        mappings = []

        if self.config.action_type == ActionType.DISCRETE:
            # Simple discrete actions
            if self.config.include_noop:
                mappings.append(ActionMapping(
                    action_id=0,
                    buttons=["NOOP"],
                    description="No operation"
                ))

            # Single button actions
            action_id = 1
            for button in self.config.available_buttons:
                mappings.append(ActionMapping(
                    action_id=action_id,
                    buttons=[button],
                    description=f"Press {button}"
                ))
                action_id += 1

            # Direction combinations
            if "UP" in self.config.available_buttons and "A" in self.config.available_buttons:
                mappings.append(ActionMapping(
                    action_id=action_id,
                    buttons=["UP", "A"],
                    description="Press UP + A"
                ))
                action_id += 1

            if "DOWN" in self.config.available_buttons and "B" in self.config.available_buttons:
                mappings.append(ActionMapping(
                    action_id=action_id,
                    buttons=["DOWN", "B"],
                    description="Press DOWN + B"
                ))
                action_id += 1

        elif self.config.action_type == ActionType.MULTI_DISCRETE:
            # Multi-discrete actions (each button can be pressed independently)
            button_list = ["NOOP"] + self.config.available_buttons

            for i, button in enumerate(button_list):
                mappings.append(ActionMapping(
                    action_id=i,
                    buttons=[button],
                    description=f"Action {i}: {button}"
                ))

        elif self.config.action_type == ActionType.CONTINUOUS:
            # Continuous actions (mapping to button presses)
            # This is handled differently in action_to_pyboy
            pass

        elif self.config.action_type == ActionType.HYBRID:
            # Hybrid actions (discrete + continuous)
            # This is handled differently in action_to_pyboy
            pass

        return mappings

    def _create_action_space(self) -> gym.spaces.Space:
        """Create the action space based on configuration."""
        if self.config.action_type == ActionType.DISCRETE:
            return Discrete(len(self.action_mappings))

        elif self.config.action_type == ActionType.MULTI_DISCRETE:
            if self.config.allow_multiple_buttons:
                # Each button can be pressed or not
                num_buttons = len(self.config.available_buttons)
                return MultiDiscrete([2] * num_buttons)
            else:
                # Single button selection
                return Discrete(len(self.config.available_buttons) + (1 if self.config.include_noop else 0))

        elif self.config.action_type == ActionType.CONTINUOUS:
            # Continuous actions (e.g., joystick-like input)
            if len(self.config.available_buttons) >= 4:
                # Assume first 4 buttons are directions
                return Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # x, y
            else:
                return Box(low=0.0, high=1.0, shape=(len(self.config.available_buttons),), dtype=np.float32)

        elif self.config.action_type == ActionType.HYBRID:
            # Hybrid: discrete buttons + continuous direction
            discrete_buttons = [b for b in self.config.available_buttons if b not in ["UP", "DOWN", "LEFT", "RIGHT"]]
            discrete_size = len(discrete_buttons)
            continuous_size = 2  # x, y direction

            return Dict({
                'discrete': MultiDiscrete([2] * discrete_size),
                'continuous': Box(low=-1.0, high=1.0, shape=(continuous_size,), dtype=np.float32)
            })

        else:
            raise ValueError(f"Unknown action type: {self.config.action_type}")

    def action_to_pyboy(self, action: Union[int, np.ndarray, Dict[str, np.ndarray]]) -> List[str]:
        """
        Convert environment action to PyBoy button presses.

        Args:
            action: Environment action

        Returns:
            List of PyBoy button actions to execute
        """
        if self.config.action_type == ActionType.DISCRETE:
            return self._discrete_to_pyboy(action)
        elif self.config.action_type == ActionType.MULTI_DISCRETE:
            return self._multi_discrete_to_pyboy(action)
        elif self.config.action_type == ActionType.CONTINUOUS:
            return self._continuous_to_pyboy(action)
        elif self.config.action_type == ActionType.HYBRID:
            return self._hybrid_to_pyboy(action)
        else:
            raise ValueError(f"Unknown action type: {self.config.action_type}")

    def _discrete_to_pyboy(self, action: int) -> List[str]:
        """Convert discrete action to PyBoy buttons."""
        if action < 0 or action >= len(self.action_mappings):
            return ["NOOP"]

        mapping = self.action_mappings[action]
        return mapping.buttons

    def _multi_discrete_to_pyboy(self, action: np.ndarray) -> List[str]:
        """Convert multi-discrete action to PyBoy buttons."""
        buttons = []

        if self.config.allow_multiple_buttons:
            # Multiple buttons can be pressed
            for i, button_state in enumerate(action):
                if button_state == 1 and i < len(self.config.available_buttons):
                    buttons.append(self.config.available_buttons[i])
        else:
            # Single button selection
            button_idx = np.argmax(action)
            if button_idx == 0 and self.config.include_noop:
                buttons.append("NOOP")
            elif button_idx > 0:
                button_idx -= 1  # Adjust for NOOP
                if button_idx < len(self.config.available_buttons):
                    buttons.append(self.config.available_buttons[button_idx])

        return buttons if buttons else ["NOOP"]

    def _continuous_to_pyboy(self, action: np.ndarray) -> List[str]:
        """Convert continuous action to PyBoy buttons."""
        buttons = []

        if len(action) >= 2:
            # Assume first two values are x, y direction
            x, y = action[0], action[1]

            # Convert to directional buttons
            if abs(x) > self.config.continuous_action_threshold:
                if x > 0:
                    buttons.append("RIGHT")
                else:
                    buttons.append("LEFT")

            if abs(y) > self.config.continuous_action_threshold:
                if y > 0:
                    buttons.append("DOWN")
                else:
                    buttons.append("UP")

        # Add additional button presses if specified
        if len(action) > 2:
            for i in range(2, min(len(action), len(self.config.available_buttons) - 4 + 2)):
                if action[i] > self.config.continuous_action_threshold:
                    button_name = self.config.available_buttons[i + 2]  # Skip direction buttons
                    if button_name in ["A", "B", "START", "SELECT"]:
                        buttons.append(button_name)

        return buttons if buttons else ["NOOP"]

    def _hybrid_to_pyboy(self, action: Dict[str, np.ndarray]) -> List[str]:
        """Convert hybrid action to PyBoy buttons."""
        buttons = []

        # Handle discrete part
        if 'discrete' in action:
            discrete_buttons = [b for b in self.config.available_buttons if b not in ["UP", "DOWN", "LEFT", "RIGHT"]]
            for i, button_state in enumerate(action['discrete']):
                if button_state == 1 and i < len(discrete_buttons):
                    buttons.append(discrete_buttons[i])

        # Handle continuous part (direction)
        if 'continuous' in action:
            continuous = action['continuous']
            if len(continuous) >= 2:
                x, y = continuous[0], continuous[1]

                if abs(x) > self.config.continuous_action_threshold:
                    if x > 0:
                        buttons.append("RIGHT")
                    else:
                        buttons.append("LEFT")

                if abs(y) > self.config.continuous_action_threshold:
                    if y > 0:
                        buttons.append("DOWN")
                    else:
                        buttons.append("UP")

        return buttons if buttons else ["NOOP"]

    def pyboy_to_action(self, buttons: List[str]) -> Union[int, np.ndarray, Dict[str, np.ndarray]]:
        """
        Convert PyBoy buttons to environment action (inverse mapping).

        Args:
            buttons: List of PyBoy button actions

        Returns:
            Environment action
        """
        if self.config.action_type == ActionType.DISCRETE:
            return self._pyboy_to_discrete(buttons)
        elif self.config.action_type == ActionType.MULTI_DISCRETE:
            return self._pyboy_to_multi_discrete(buttons)
        elif self.config.action_type == ActionType.CONTINUOUS:
            return self._pyboy_to_continuous(buttons)
        elif self.config.action_type == ActionType.HYBRID:
            return self._pyboy_to_hybrid(buttons)
        else:
            raise ValueError(f"Unknown action type: {self.config.action_type}")

    def _pyboy_to_discrete(self, buttons: List[str]) -> int:
        """Convert PyBoy buttons to discrete action."""
        for mapping in self.action_mappings:
            if set(mapping.buttons) == set(buttons):
                return mapping.action_id

        # Find closest match
        for mapping in self.action_mappings:
            if set(mapping.buttons).issubset(set(buttons)):
                return mapping.action_id

        return 0  # Default to NOOP

    def _pyboy_to_multi_discrete(self, buttons: List[str]) -> np.ndarray:
        """Convert PyBoy buttons to multi-discrete action."""
        action = np.zeros(len(self.config.available_buttons), dtype=np.int32)

        for i, button in enumerate(self.config.available_buttons):
            if button in buttons:
                action[i] = 1

        return action

    def _pyboy_to_continuous(self, buttons: List[str]) -> np.ndarray:
        """Convert PyBoy buttons to continuous action."""
        if len(self.config.available_buttons) >= 4:
            # Direction mapping
            x = 0.0
            y = 0.0

            if "RIGHT" in buttons:
                x += 1.0
            if "LEFT" in buttons:
                x -= 1.0
            if "DOWN" in buttons:
                y += 1.0
            if "UP" in buttons:
                y -= 1.0

            # Normalize
            if x != 0 or y != 0:
                magnitude = np.sqrt(x*x + y*y)
                x /= magnitude
                y /= magnitude

            return np.array([x, y], dtype=np.float32)
        else:
            # Binary button mapping
            action = np.zeros(len(self.config.available_buttons), dtype=np.float32)
            for button in buttons:
                if button in self.config.available_buttons:
                    idx = self.config.available_buttons.index(button)
                    action[idx] = 1.0

            return action

    def _pyboy_to_hybrid(self, buttons: List[str]) -> Dict[str, np.ndarray]:
        """Convert PyBoy buttons to hybrid action."""
        discrete_buttons = [b for b in self.config.available_buttons if b not in ["UP", "DOWN", "LEFT", "RIGHT"]]
        direction_buttons = ["UP", "DOWN", "LEFT", "RIGHT"]

        # Discrete part
        discrete_action = np.zeros(len(discrete_buttons), dtype=np.int32)
        for button in buttons:
            if button in discrete_buttons:
                idx = discrete_buttons.index(button)
                discrete_action[idx] = 1

        # Continuous part
        x = 0.0
        y = 0.0

        if "RIGHT" in buttons:
            x += 1.0
        if "LEFT" in buttons:
            x -= 1.0
        if "DOWN" in buttons:
            y += 1.0
        if "UP" in buttons:
            y -= 1.0

        # Normalize
        if x != 0 or y != 0:
            magnitude = np.sqrt(x*x + y*y)
            x /= magnitude
            y /= magnitude

        return {
            'discrete': discrete_action,
            'continuous': np.array([x, y], dtype=np.float32)
        }

    def get_action_space(self) -> gym.spaces.Space:
        """Get the action space."""
        return self.action_space

    def get_action_info(self) -> Dict[str, Any]:
        """Get information about the action space."""
        info = {
            'action_type': self.config.action_type.value,
            'action_space': str(self.action_space),
            'action_space_size': self.action_space.n if hasattr(self.action_space, 'n') else 'continuous',
            'available_buttons': self.config.available_buttons,
        }

        if self.config.action_type == ActionType.DISCRETE:
            info['action_mappings'] = [
                {
                    'action_id': m.action_id,
                    'buttons': m.buttons,
                    'description': m.description
                }
                for m in self.action_mappings
            ]

        return info

    def get_valid_actions(self) -> List[int]:
        """Get list of valid action IDs."""
        if self.config.action_type == ActionType.DISCRETE:
            return list(range(len(self.action_mappings)))
        else:
            # For other types, return sample actions
            return [0]

    def is_action_valid(self, action: Union[int, np.ndarray, Dict[str, np.ndarray]]) -> bool:
        """Check if action is valid."""
        return self.action_space.contains(action)

    def get_action_description(self, action: Union[int, np.ndarray, Dict[str, np.ndarray]]) -> str:
        """Get human-readable description of action."""
        if self.config.action_type == ActionType.DISCRETE:
            if 0 <= action < len(self.action_mappings):
                return self.action_mappings[action].description
            else:
                return "Invalid action"
        else:
            buttons = self.action_to_pyboy(action)
            return f"Buttons: {buttons}"

    def update_button_states(self, action: Union[int, np.ndarray, Dict[str, np.ndarray]]):
        """Update button states based on action."""
        buttons = self.action_to_pyboy(action)

        # Reset button timers
        for button in self.button_timers:
            if self.button_timers[button] > 0:
                self.button_timers[button] -= 1

        # Update button states
        for button in self.button_states:
            self.button_states[button] = button in buttons
            if self.button_states[button]:
                self.button_timers[button] = self.config.button_press_duration

    def get_pressed_buttons(self) -> List[str]:
        """Get list of currently pressed buttons."""
        return [button for button, pressed in self.button_states.items() if pressed]

    def release_all_buttons(self):
        """Release all buttons."""
        for button in self.button_states:
            self.button_states[button] = False
            self.button_timers[button] = 0

    def add_custom_mapping(self, mapping: ActionMapping):
        """Add a custom action mapping."""
        if self.config.action_type == ActionType.DISCRETE:
            mapping.action_id = len(self.action_mappings)
            self.action_mappings.append(mapping)
            self.action_space = Discrete(len(self.action_mappings))

    def remove_custom_mapping(self, action_id: int):
        """Remove a custom action mapping."""
        if self.config.action_type == ActionType.DISCRETE:
            self.action_mappings = [m for m in self.action_mappings if m.action_id != action_id]
            # Reassign action IDs
            for i, mapping in enumerate(self.action_mappings):
                mapping.action_id = i
            self.action_space = Discrete(len(self.action_mappings))

    def save_mappings(self, filepath: str):
        """Save action mappings to file."""
        import json

        mappings_data = []
        for mapping in self.action_mappings:
            mappings_data.append({
                'action_id': mapping.action_id,
                'buttons': mapping.buttons,
                'duration': mapping.duration,
                'description': mapping.description
            })

        config_data = {
            'action_type': self.config.action_type.value,
            'include_noop': self.config.include_noop,
            'allow_multiple_buttons': self.config.allow_multiple_buttons,
            'available_buttons': self.config.available_buttons,
            'mappings': mappings_data
        }

        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)

    def load_mappings(self, filepath: str):
        """Load action mappings from file."""
        import json

        with open(filepath, 'r') as f:
            config_data = json.load(f)

        # Update config
        self.config.action_type = ActionType(config_data['action_type'])
        self.config.include_noop = config_data['include_noop']
        self.config.allow_multiple_buttons = config_data['allow_multiple_buttons']
        self.config.available_buttons = config_data['available_buttons']

        # Load mappings
        self.action_mappings = []
        for mapping_data in config_data['mappings']:
            self.action_mappings.append(ActionMapping(
                action_id=mapping_data['action_id'],
                buttons=mapping_data['buttons'],
                duration=mapping_data['duration'],
                description=mapping_data['description']
            ))

        # Recreate action space
        self.action_space = self._create_action_space()