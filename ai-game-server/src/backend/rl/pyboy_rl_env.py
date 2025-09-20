"""
PyBoyEnv Reinforcement Learning Integration for AI Game Server
"""
import numpy as np
import os
from typing import List, Dict, Any, Optional, Tuple
import gym
from pyboyenv.env.env import PyBoyEnv

class PyBoyRLEnvironment:
    """Reinforcement Learning environment wrapper for the AI Game Server"""

    def __init__(self, rom_path: str, config: Dict[str, Any] = None):
        self.rom_path = rom_path
        self.config = config or {}
        self.env = None
        self.initialized = False
        self.reward_rules = []
        self.done_rules = []

    def initialize(self) -> bool:
        """Initialize the RL environment"""
        try:
            if not os.path.exists(self.rom_path):
                raise FileNotFoundError(f"ROM file not found: {self.rom_path}")

            # Initialize PyBoyEnv with configuration
            self.env = PyBoyEnv(
                game=self.rom_path,
                window='headless',  # No GUI for server use
                visible=False,
                colors=self.config.get('colors', (0xfff6d3, 0xf9a875, 0xeb6b6f, 0x7c3f58)),
                buttons_press_mode='toggle'
            )

            # Apply default reward rules for common games
            self._apply_default_reward_rules()

            self.initialized = True
            return True

        except Exception as e:
            print(f"Failed to initialize PyBoy RL environment: {e}")
            return False

    def _apply_default_reward_rules(self):
        """Apply default reward rules based on game type"""
        game_name = os.path.basename(self.rom_path).lower()

        # Pokemon-specific rules
        if any(pokemon in game_name for pokemon in ['pokemon', 'poke']):
            # Experience points reward
            self.add_reward_rule(0xD18D, 'increase', 1.0, 'EXP gained')
            # Level up reward
            self.add_reward_rule(0xD18C, 'increase', 10.0, 'Level up')
            # HP change rewards
            self.add_reward_rule(0xD16C, 'decrease', -5.0, 'HP lost')
            self.add_reward_rule(0xD16C, 'increase', 5.0, 'HP restored')

        # Mario-specific rules
        elif any(mario in game_name for mario in ['mario', 'super']):
            # Score reward
            self.add_reward_rule(0x07DE, 'increase', 0.1, 'Score increase')
            # Lives reward
            self.add_reward_rule(0x075A, 'decrease', -20.0, 'Life lost')
            # Coin reward
            self.add_reward_rule(0x07ED, 'increase', 1.0, 'Coin collected')

        # Tetris-specific rules
        elif any(tetris in game_name for tetris in ['tetris', 'blocks']):
            # Score reward
            self.add_reward_rule(0xC0A0, 'increase', 0.1, 'Lines cleared')
            # Level progression
            self.add_reward_rule(0xC0A1, 'increase', 5.0, 'Level up')
            # Game over detection
            self.add_done_rule(0xC0A3, 'equal 255', 'Game Over')

    def add_reward_rule(self, address: int, operator: str, reward: float, label: str):
        """Add a custom reward rule

        Operators:
        - increase: reward when value increases
        - decrease: reward when value decreases
        - equal X: reward when value equals X
        - bigger X: reward when value > X
        - smaller X: reward when value < X
        - in X1,X2,...,XN: reward when value is in the list
        """
        if self.env and self.initialized:
            self.env.set_reward_rule(address, operator, reward, label)
            self.reward_rules.append({
                'address': address,
                'operator': operator,
                'reward': reward,
                'label': label
            })

    def add_done_rule(self, address: int, operator: str, label: str):
        """Add a custom done rule"""
        if self.env and self.initialized:
            self.env.set_done_rule(address, operator, label)
            self.done_rules.append({
                'address': address,
                'operator': operator,
                'label': label
            })

    def step(self, action: str) -> Dict[str, Any]:
        """Execute an action and return RL environment results"""
        if not self.initialized or not self.env:
            return {
                'success': False,
                'error': 'Environment not initialized'
            }

        try:
            # Map action string to environment action ID
            action_map = {
                'UP': 0,
                'DOWN': 1,
                'LEFT': 2,
                'RIGHT': 3,
                'A': 4,
                'B': 5,
                'SELECT': 6,
                'START': 7,
                'RELEASE_UP': 8,
                'RELEASE_DOWN': 9,
                'RELEASE_LEFT': 10,
                'RELEASE_RIGHT': 11,
                'RELEASE_A': 12,
                'RELEASE_B': 13,
                'RELEASE_SELECT': 14,
                'RELEASE_START': 15,
                'PASS': 16
            }

            action_id = action_map.get(action, 16)  # Default to PASS

            # Execute the action
            observation, reward, done, info = self.env.step(action_id)

            return {
                'success': True,
                'observation': observation.tolist(),
                'reward': reward,
                'done': done,
                'info': info,
                'action': action,
                'action_id': action_id
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def reset(self) -> Dict[str, Any]:
        """Reset the environment"""
        if not self.initialized or not self.env:
            return {
                'success': False,
                'error': 'Environment not initialized'
            }

        try:
            observation = self.env.reset()
            return {
                'success': True,
                'observation': observation.tolist()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def get_screen(self) -> np.ndarray:
        """Get current screen state"""
        if not self.initialized or not self.env:
            return np.zeros((160, 144, 3), dtype=np.uint8)

        try:
            return self.env._get_observation()
        except Exception:
            return np.zeros((160, 144, 3), dtype=np.uint8)

    def get_memory_value(self, address: int) -> int:
        """Get memory value at specific address"""
        if not self.initialized or not self.env:
            return 0

        try:
            return self.env._pyboy.get_memory_value(address)
        except Exception:
            return 0

    def get_state_info(self) -> Dict[str, Any]:
        """Get comprehensive state information"""
        if not self.initialized or not self.env:
            return {}

        try:
            return {
                'action_space_size': self.env.action_space.n,
                'observation_space_shape': self.env.observation_space.shape,
                'reward_rules': self.reward_rules,
                'done_rules': self.done_rules,
                'current_rewards': [
                    {
                        'address': rule['address'],
                        'current_value': self.get_memory_value(rule['address']),
                        'operator': rule['operator'],
                        'reward': rule['reward'],
                        'label': rule['label']
                    }
                    for rule in self.reward_rules
                ]
            }
        except Exception as e:
            return {'error': str(e)}

    def close(self):
        """Close the environment"""
        if self.env:
            del self.env
            self.env = None
        self.initialized = False


# Global RL environment instance
_rl_environment = None

def get_rl_environment(rom_path: str = None, config: Dict[str, Any] = None) -> PyBoyRLEnvironment:
    """Get or create the RL environment instance"""
    global _rl_environment

    if _rl_environment is None and rom_path:
        _rl_environment = PyBoyRLEnvironment(rom_path, config)
        if not _rl_environment.initialize():
            _rl_environment = None

    return _rl_environment

def reset_rl_environment(rom_path: str = None, config: Dict[str, Any] = None) -> PyBoyRLEnvironment:
    """Reset the RL environment with new ROM"""
    global _rl_environment

    if _rl_environment:
        _rl_environment.close()

    _rl_environment = None
    return get_rl_environment(rom_path, config)