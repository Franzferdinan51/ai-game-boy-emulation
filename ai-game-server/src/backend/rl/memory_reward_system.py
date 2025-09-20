"""
Memory-based reward system for PyBoy RL environment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from collections import defaultdict, deque


class RewardType(Enum):
    """Types of reward functions."""
    SCORE = "score"
    PROGRESS = "progress"
    EXPLORATION = "exploration"
    HEALTH = "health"
    COINS = "coins"
    EXPERIENCE = "experience"
    ENEMIES_DEFEATED = "enemies_defeated"
    ITEMS_COLLECTED = "items_collected"
    LEVEL_REACHED = "level_reached"
    CUSTOM = "custom"


@dataclass
class MemoryAddress:
    """Configuration for a memory address to monitor."""
    address: int
    bank: Optional[int] = None
    data_type: str = "uint8"  # uint8, uint16, int8, int16
    scale: float = 1.0
    offset: float = 0.0
    name: str = ""


@dataclass
class RewardConfig:
    """Configuration for a reward function."""
    type: RewardType
    weight: float = 1.0
    memory_addresses: List[MemoryAddress] = field(default_factory=list)
    custom_function: Optional[Callable] = None
    custom_config: Dict[str, Any] = field(default_factory=dict)
    reward_on_increase: bool = True
    reward_on_decrease: bool = False
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None
    cooldown: int = 0  # Frames between rewards


@dataclass
class RewardSystemConfig:
    """Configuration for the memory reward system."""
    reward_configs: List[RewardConfig] = field(default_factory=list)
    base_reward: float = 0.0
    step_penalty: float = 0.0
    time_penalty: float = 0.0
    death_penalty: float = 0.0
    combo_multiplier: float = 1.0
    max_combo_length: int = 10
    normalize_rewards: bool = True
    reward_history_length: int = 1000


class MemoryRewardSystem:
    """
    Memory-based reward system for Game Boy games.

    This system monitors memory addresses and calculates rewards based on:
    - Score changes
    - Progress indicators
    - Game state changes
    - Custom reward functions
    """

    def __init__(self, pyboy, config: RewardSystemConfig):
        """
        Initialize the memory reward system.

        Args:
            pyboy: PyBoy instance
            config: Reward system configuration
        """
        self.pyboy = pyboy
        self.config = config

        # Initialize reward tracking
        self.current_values = {}
        self.previous_values = {}
        self.reward_timers = {}
        self.reward_history = deque(maxlen=config.reward_history_length)
        self.combo_counter = 0
        self.combo_timer = 0

        # Game-specific reward configurations
        self._setup_game_specific_rewards()

        # Initialize memory tracking
        self._initialize_memory_tracking()

    def _setup_game_specific_rewards(self):
        """Setup game-specific reward configurations based on game title."""
        game_title = self.pyboy.cartridge_title.lower()

        if not self.config.reward_configs:
            # Setup default reward configurations based on game type
            if "pokemon" in game_title:
                self.config.reward_configs = self._get_pokemon_rewards()
            elif "mario" in game_title:
                self.config.reward_configs = self._get_mario_rewards()
            elif "tetris" in game_title:
                self.config.reward_configs = self._get_tetris_rewards()
            elif "zelda" in game_title:
                self.config.reward_configs = self._get_zelda_rewards()
            else:
                self.config.reward_configs = self._get_generic_rewards()

    def _get_pokemon_rewards(self) -> List[RewardConfig]:
        """Get Pokemon-specific reward configurations."""
        return [
            RewardConfig(
                type=RewardType.EXPERIENCE,
                weight=1.0,
                memory_addresses=[
                    MemoryAddress(address=0xD18C, name="current_exp"),
                ],
                reward_on_increase=True,
            ),
            RewardConfig(
                type=RewardType.LEVEL_REACHED,
                weight=5.0,
                memory_addresses=[
                    MemoryAddress(address=0xD18D, name="pokemon_level"),
                ],
                reward_on_increase=True,
            ),
            RewardConfig(
                type=RewardType.PROGRESS,
                weight=0.5,
                memory_addresses=[
                    MemoryAddress(address=0xD362, name="badges_obtained"),
                ],
                reward_on_increase=True,
            ),
            RewardConfig(
                type=RewardType.ENEMIES_DEFEATED,
                weight=0.8,
                memory_addresses=[
                    MemoryAddress(address=0xD163, name="wild_battles_won"),
                ],
                reward_on_increase=True,
            ),
            RewardConfig(
                type=RewardType.ITEMS_COLLECTED,
                weight=0.3,
                memory_addresses=[
                    MemoryAddress(address=0xD31E, name="items_in_bag"),
                ],
                reward_on_increase=True,
            ),
        ]

    def _get_mario_rewards(self) -> List[RewardConfig]:
        """Get Mario-specific reward configurations."""
        return [
            RewardConfig(
                type=RewardType.SCORE,
                weight=1.0,
                memory_addresses=[
                    MemoryAddress(address=0xC0A0, name="score_high", data_type="uint16"),
                    MemoryAddress(address=0xC0A2, name="score_low", data_type="uint16"),
                ],
                reward_on_increase=True,
            ),
            RewardConfig(
                type=RewardType.COINS,
                weight=2.0,
                memory_addresses=[
                    MemoryAddress(address=0xC0AD, name="coins"),
                ],
                reward_on_increase=True,
            ),
            RewardConfig(
                type=RewardType.PROGRESS,
                weight=1.5,
                memory_addresses=[
                    MemoryAddress(address=0xC0AB, name="current_level"),
                    MemoryAddress(address=0xC0A4, name="world_position"),
                ],
                reward_on_increase=True,
            ),
            RewardConfig(
                type=RewardType.HEALTH,
                weight=3.0,
                memory_addresses=[
                    MemoryAddress(address=0xC07A, name="lives_remaining"),
                ],
                reward_on_increase=True,
            ),
        ]

    def _get_tetris_rewards(self) -> List[RewardConfig]:
        """Get Tetris-specific reward configurations."""
        return [
            RewardConfig(
                type=RewardType.SCORE,
                weight=1.0,
                memory_addresses=[
                    MemoryAddress(address=0xC0A0, name="score_high", data_type="uint16"),
                    MemoryAddress(address=0xC0A2, name="score_low", data_type="uint16"),
                ],
                reward_on_increase=True,
            ),
            RewardConfig(
                type=RewardType.PROGRESS,
                weight=2.0,
                memory_addresses=[
                    MemoryAddress(address=0xC0B0, name="lines_cleared"),
                ],
                reward_on_increase=True,
            ),
            RewardConfig(
                type=RewardType.LEVEL_REACHED,
                weight=5.0,
                memory_addresses=[
                    MemoryAddress(address=0xC0AE, name="current_level"),
                ],
                reward_on_increase=True,
            ),
        ]

    def _get_zelda_rewards(self) -> List[RewardConfig]:
        """Get Zelda-specific reward configurations."""
        return [
            RewardConfig(
                type=RewardType.HEALTH,
                weight=2.0,
                memory_addresses=[
                    MemoryAddress(address=0xC0F0, name="current_hearts"),
                    MemoryAddress(address=0xC0F1, name="max_hearts"),
                ],
                reward_on_increase=True,
            ),
            RewardConfig(
                type=RewardType.EXPERIENCE,
                weight=1.0,
                memory_addresses=[
                    MemoryAddress(address=0xC0D0, name="sword_level"),
                    MemoryAddress(address=0xC0D1, name="shield_level"),
                ],
                reward_on_increase=True,
            ),
            RewardConfig(
                type=RewardType.ITEMS_COLLECTED,
                weight=0.8,
                memory_addresses=[
                    MemoryAddress(address=0xC0E0, name="items_collected"),
                ],
                reward_on_increase=True,
            ),
            RewardConfig(
                type=RewardType.PROGRESS,
                weight=1.5,
                memory_addresses=[
                    MemoryAddress(address=0xC0C0, name="dungeons_completed"),
                ],
                reward_on_increase=True,
            ),
        ]

    def _get_generic_rewards(self) -> List[RewardConfig]:
        """Get generic reward configurations."""
        return [
            RewardConfig(
                type=RewardType.EXPLORATION,
                weight=0.1,
                memory_addresses=[
                    MemoryAddress(address=0xC0A0, name="position_x"),
                    MemoryAddress(address=0xC0A1, name="position_y"),
                ],
                custom_function=self._exploration_reward,
            ),
            RewardConfig(
                type=RewardType.PROGRESS,
                weight=0.5,
                memory_addresses=[
                    MemoryAddress(address=0xC0B0, name="game_progress"),
                ],
                reward_on_increase=True,
            ),
        ]

    def _initialize_memory_tracking(self):
        """Initialize memory address tracking."""
        for reward_config in self.config.reward_configs:
            for memory_address in reward_config.memory_addresses:
                key = f"{memory_address.bank or 0}:{memory_address.address}"
                self.previous_values[key] = self._read_memory_value(memory_address)
                self.current_values[key] = self.previous_values[key]
                self.reward_timers[key] = 0

    def _read_memory_value(self, memory_address: MemoryAddress) -> float:
        """Read value from memory address."""
        try:
            if memory_address.bank is not None:
                value = self.pyboy.memory[memory_address.bank, memory_address.address]
            else:
                value = self.pyboy.memory[memory_address.address]

            # Convert based on data type
            if memory_address.data_type == "uint8":
                value = float(value)
            elif memory_address.data_type == "uint16":
                # Read 16-bit value (little endian)
                if memory_address.bank is not None:
                    low = self.pyboy.memory[memory_address.bank, memory_address.address]
                    high = self.pyboy.memory[memory_address.bank, memory_address.address + 1]
                else:
                    low = self.pyboy.memory[memory_address.address]
                    high = self.pyboy.memory[memory_address.address + 1]
                value = float((high << 8) | low)
            elif memory_address.data_type == "int8":
                value = float(value if value < 128 else value - 256)
            elif memory_address.data_type == "int16":
                # Read 16-bit value (little endian, signed)
                if memory_address.bank is not None:
                    low = self.pyboy.memory[memory_address.bank, memory_address.address]
                    high = self.pyboy.memory[memory_address.bank, memory_address.address + 1]
                else:
                    low = self.pyboy.memory[memory_address.address]
                    high = self.pyboy.memory[memory_address.address + 1]
                value = float((high << 8) | low)
                if value >= 32768:
                    value -= 65536

            # Apply scale and offset
            return (value * memory_address.scale) + memory_address.offset

        except Exception as e:
            print(f"Error reading memory {memory_address.address}: {e}")
            return 0.0

    def _exploration_reward(self, values: List[float], previous_values: List[float]) -> float:
        """Calculate exploration reward based on position changes."""
        if len(values) >= 2 and len(previous_values) >= 2:
            # Calculate distance moved
            dx = values[0] - previous_values[0]
            dy = values[1] - previous_values[1]
            distance = np.sqrt(dx*dx + dy*dy)
            return min(distance * 0.1, 1.0)  # Cap exploration reward
        return 0.0

    def get_reward(self) -> float:
        """Calculate total reward for current frame."""
        total_reward = self.config.base_reward

        # Update memory values
        self._update_memory_values()

        # Calculate rewards for each configuration
        rewards = []
        for reward_config in self.config.reward_configs:
            reward = self._calculate_reward_for_config(reward_config)
            rewards.append(reward)

        # Apply penalties
        if self.config.step_penalty > 0:
            total_reward -= self.config.step_penalty

        if self.config.time_penalty > 0:
            total_reward -= self.config.time_penalty

        # Check for death/game over
        if self._check_death_condition():
            total_reward -= self.config.death_penalty

        # Sum weighted rewards
        for i, reward in enumerate(rewards):
            total_reward += reward * self.config.reward_configs[i].weight

        # Apply combo multiplier
        if self.combo_counter > 0:
            combo_bonus = min(self.combo_counter * self.config.combo_multiplier,
                            self.config.max_combo_length * self.config.combo_multiplier)
            total_reward += combo_bonus
            self.combo_timer -= 1
            if self.combo_timer <= 0:
                self.combo_counter = 0

        # Normalize rewards if configured
        if self.config.normalize_rewards:
            total_reward = np.tanh(total_reward)

        # Store in history
        self.reward_history.append(total_reward)

        return total_reward

    def _update_memory_values(self):
        """Update current memory values."""
        for reward_config in self.config.reward_configs:
            for memory_address in reward_config.memory_addresses:
                key = f"{memory_address.bank or 0}:{memory_address.address}"
                self.previous_values[key] = self.current_values[key]
                self.current_values[key] = self._read_memory_value(memory_address)

                # Update reward timer
                if self.reward_timers[key] > 0:
                    self.reward_timers[key] -= 1

    def _calculate_reward_for_config(self, reward_config: RewardConfig) -> float:
        """Calculate reward for a specific configuration."""
        values = []
        previous_values = []

        # Read all memory addresses for this config
        for memory_address in reward_config.memory_addresses:
            key = f"{memory_address.bank or 0}:{memory_address.address}"
            values.append(self.current_values[key])
            previous_values.append(self.previous_values[key])

        # Check cooldown
        if any(self.reward_timers.get(
                f"{memory_address.bank or 0}:{memory_address.address}", 0) > 0
               for memory_address in reward_config.memory_addresses):
            return 0.0

        # Calculate reward based on type
        if reward_config.type == RewardType.CUSTOM and reward_config.custom_function:
            reward = reward_config.custom_function(values, previous_values)
        else:
            reward = self._calculate_standard_reward(reward_config, values, previous_values)

        # Check thresholds
        if reward_config.min_threshold is not None and reward < reward_config.min_threshold:
            return 0.0
        if reward_config.max_threshold is not None and reward > reward_config.max_threshold:
            return 0.0

        # Update combo counter
        if reward > 0:
            self.combo_counter += 1
            self.combo_timer = 30  # Reset combo timer

        # Set cooldown
        if reward != 0 and reward_config.cooldown > 0:
            for memory_address in reward_config.memory_addresses:
                key = f"{memory_address.bank or 0}:{memory_address.address}"
                self.reward_timers[key] = reward_config.cooldown

        return reward

    def _calculate_standard_reward(self, reward_config: RewardConfig,
                                 values: List[float], previous_values: List[float]) -> float:
        """Calculate standard reward based on value changes."""
        reward = 0.0

        if len(values) == 1 and len(previous_values) == 1:
            # Single value comparison
            change = values[0] - previous_values[0]

            if change > 0 and reward_config.reward_on_increase:
                reward = change
            elif change < 0 and reward_config.reward_on_decrease:
                reward = abs(change)
        else:
            # Multi-value comparison (e.g., 16-bit values)
            current_value = sum(values)
            previous_value = sum(previous_values)
            change = current_value - previous_value

            if change > 0 and reward_config.reward_on_increase:
                reward = change
            elif change < 0 and reward_config.reward_on_decrease:
                reward = abs(change)

        return reward

    def _check_death_condition(self) -> bool:
        """Check if the player has died."""
        # Common death indicators in Game Boy games
        death_indicators = [
            # Lives counter (many games)
            (0xC07A, 0),  # Lives = 0
            # Health (Zelda-style games)
            (0xC0F0, 0),  # Hearts = 0
            # Game over flag (some games)
            (0xC050, 0xFF),  # Game over flag set
        ]

        for address, value in death_indicators:
            try:
                if self.pyboy.memory[address] == value:
                    return True
            except:
                continue

        return False

    def get_reward_breakdown(self) -> Dict[str, float]:
        """Get breakdown of rewards by type."""
        breakdown = {}

        for reward_config in self.config.reward_configs:
            reward_type = reward_config.type.value
            reward = self._calculate_reward_for_config(
                reward_config,
                [self.current_values[f"{addr.bank or 0}:{addr.address}"]
                for addr in reward_config.memory_addresses
            ],
                [self.previous_values[f"{addr.bank or 0}:{addr.address}"]
                for addr in reward_config.memory_addresses
            ]
            )
            breakdown[reward_type] = reward * reward_config.weight

        # Add penalties
        if self.config.step_penalty > 0:
            breakdown['step_penalty'] = -self.config.step_penalty
        if self.config.time_penalty > 0:
            breakdown['time_penalty'] = -self.config.time_penalty

        return breakdown

    def is_done(self) -> bool:
        """Check if the episode should end."""
        # Check for game over conditions
        return self._check_death_condition()

    def reset(self):
        """Reset the reward system."""
        self._initialize_memory_tracking()
        self.reward_history.clear()
        self.combo_counter = 0
        self.combo_timer = 0

    def get_reward_stats(self) -> Dict[str, Any]:
        """Get reward statistics."""
        if not self.reward_history:
            return {}

        rewards = list(self.reward_history)
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'total_reward': np.sum(rewards),
            'reward_count': len(rewards),
            'current_combo': self.combo_counter,
        }

    def get_memory_values(self) -> Dict[str, float]:
        """Get current memory values being tracked."""
        return {
            f"{memory_address.name or memory_address.address}": self.current_values.get(
                f"{memory_address.bank or 0}:{memory_address.address}", 0.0
            )
            for config in self.config.reward_configs
            for memory_address in config.memory_addresses
        }

    def save_config(self, filepath: str):
        """Save reward configuration to file."""
        config_data = {
            'reward_configs': [
                {
                    'type': config.type.value,
                    'weight': config.weight,
                    'memory_addresses': [
                        {
                            'address': addr.address,
                            'bank': addr.bank,
                            'data_type': addr.data_type,
                            'scale': addr.scale,
                            'offset': addr.offset,
                            'name': addr.name,
                        }
                        for addr in config.memory_addresses
                    ],
                    'reward_on_increase': config.reward_on_increase,
                    'reward_on_decrease': config.reward_on_decrease,
                    'min_threshold': config.min_threshold,
                    'max_threshold': config.max_threshold,
                    'cooldown': config.cooldown,
                }
                for config in self.config.reward_configs
            ],
            'base_reward': self.config.base_reward,
            'step_penalty': self.config.step_penalty,
            'time_penalty': self.config.time_penalty,
            'death_penalty': self.config.death_penalty,
            'combo_multiplier': self.config.combo_multiplier,
            'max_combo_length': self.config.max_combo_length,
            'normalize_rewards': self.config.normalize_rewards,
            'reward_history_length': self.config.reward_history_length,
        }

        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)

    @classmethod
    def load_config(cls, filepath: str) -> RewardSystemConfig:
        """Load reward configuration from file."""
        with open(filepath, 'r') as f:
            config_data = json.load(f)

        reward_configs = []
        for config_dict in config_data['reward_configs']:
            memory_addresses = []
            for addr_dict in config_dict['memory_addresses']:
                memory_addresses.append(MemoryAddress(**addr_dict))

            reward_configs.append(RewardConfig(
                type=RewardType(config_dict['type']),
                weight=config_dict['weight'],
                memory_addresses=memory_addresses,
                reward_on_increase=config_dict['reward_on_increase'],
                reward_on_decrease=config_dict['reward_on_decrease'],
                min_threshold=config_dict['min_threshold'],
                max_threshold=config_dict['max_threshold'],
                cooldown=config_dict['cooldown'],
            ))

        return RewardSystemConfig(
            reward_configs=reward_configs,
            base_reward=config_data['base_reward'],
            step_penalty=config_data['step_penalty'],
            time_penalty=config_data['time_penalty'],
            death_penalty=config_data['death_penalty'],
            combo_multiplier=config_data['combo_multiplier'],
            max_combo_length=config_data['max_combo_length'],
            normalize_rewards=config_data['normalize_rewards'],
            reward_history_length=config_data['reward_history_length'],
        )