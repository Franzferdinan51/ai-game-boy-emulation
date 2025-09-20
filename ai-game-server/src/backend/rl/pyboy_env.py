"""
Reinforcement Learning Environment for PyBoy
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
import io
import time
import json
from collections import deque, defaultdict
import threading
from dataclasses import dataclass, field
import copy

try:
    from pyboy import PyBoy
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("PyBoy not available. Install with 'pip install pyboy'")

from .memory_reward_system import MemoryRewardSystem
from .game_state_tracker import GameStateTracker
from .action_manager import ActionManager
from .rl_environment_config import RLEnvironmentConfig


@dataclass
class RLStepResult:
    """Result of a single RL step"""
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]
    truncated: bool = False


class PyBoyEnv(gym.Env):
    """
    Gym-compatible environment for Game Boy games using PyBoy emulator.

    This environment provides reinforcement learning capabilities with:
    - Multiple observation spaces (screen, game_area, memory, tiles)
    - Configurable action spaces (discrete, multi-discrete, continuous)
    - Memory-based reward systems
    - State tracking and serialization
    - Training pipeline integration

    Example:
        ```python
        env = PyBoyEnv(
            rom_path="game.gb",
            config=RLEnvironmentConfig(
                observation_type="screen",
                action_type="discrete",
                reward_system="memory_based"
            )
        )

        obs = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                break
        ```
    """

    metadata = {
        'render_modes': ['human', 'rgb_array', 'ansi'],
        'render_fps': 60,
        'video.frames_per_second': 60
    }

    def __init__(
        self,
        rom_path: str,
        config: Optional[RLEnvironmentConfig] = None,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize PyBoy RL environment.

        Args:
            rom_path: Path to Game Boy ROM file
            config: RL environment configuration
            render_mode: Rendering mode ('human', 'rgb_array', 'ansi')
            **kwargs: Additional configuration options
        """
        super().__init__()

        if not PYBOY_AVAILABLE:
            raise ImportError("PyBoy is required. Install with 'pip install pyboy'")

        self.rom_path = rom_path
        self.config = config or RLEnvironmentConfig()
        self.render_mode = render_mode

        # Initialize PyBoy emulator
        self._init_pyboy()

        # Initialize RL components
        self.reward_system = MemoryRewardSystem(
            self.pyboy,
            self.config.reward_config
        )
        self.game_state_tracker = GameStateTracker(
            self.pyboy,
            self.config.state_config
        )
        self.action_manager = ActionManager(
            self.pyboy,
            self.config.action_config
        )

        # Setup action and observation spaces
        self._setup_spaces()

        # Environment state
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.done = False
        self.truncated = False

        # State management
        self.initial_state = None
        self.state_history = deque(maxlen=self.config.max_history_length)
        self.action_history = deque(maxlen=self.config.max_history_length)

        # Training metrics
        self.metrics = defaultdict(list)
        self.start_time = time.time()

        # Lock for thread safety
        self._lock = threading.Lock()

    def _init_pyboy(self):
        """Initialize PyBoy emulator with appropriate settings."""
        pyboy_kwargs = {
            'window': 'null' if self.config.headless else 'SDL2',
            'sound': False if self.config.headless else True,
            'sound_volume': 0 if self.config.headless else 50,
            'scale': self.config.window_scale,
            'log_level': self.config.log_level,
        }

        # Add any additional PyBoy configuration
        pyboy_kwargs.update(self.config.pyboy_kwargs)

        self.pyboy = PyBoy(self.rom_path, **pyboy_kwargs)

        # Start game if game wrapper is available
        if self.pyboy.game_wrapper:
            self.pyboy.game_wrapper.start_game()

        # Save initial state
        self.initial_state = self._save_state()

    def _setup_spaces(self):
        """Setup action and observation spaces based on configuration."""
        # Setup action space
        self.action_space = self.action_manager.get_action_space()

        # Setup observation space
        if self.config.observation_config.type == "screen":
            # Screen observation (144x160x3 for RGB)
            screen_shape = (144, 160, 3)
            self.observation_space = spaces.Box(
                low=0, high=255, shape=screen_shape, dtype=np.uint8
            )
        elif self.config.observation_config.type == "game_area":
            # Game area observation (simplified tiles)
            game_area_shape = self.pyboy.game_wrapper.shape if self.pyboy.game_wrapper else (32, 32)
            self.observation_space = spaces.Box(
                low=0, high=384, shape=game_area_shape, dtype=np.uint32
            )
        elif self.config.observation_config.type == "memory":
            # Memory observation
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.config.observation_config.memory_size,), dtype=np.uint8
            )
        elif self.config.observation_config.type == "tiles":
            # Tiles observation
            self.observation_space = spaces.Box(
                low=0, high=384, shape=(self.config.observation_config.tiles_height,
                                         self.config.observation_config.tiles_width), dtype=np.uint32
            )
        elif self.config.observation_config.type == "multi":
            # Multi-modal observation
            obs_dict = {}

            if self.config.observation_config.include_screen:
                obs_dict['screen'] = spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)

            if self.config.observation_config.include_game_area:
                game_area_shape = self.pyboy.game_wrapper.shape if self.pyboy.game_wrapper else (32, 32)
                obs_dict['game_area'] = spaces.Box(low=0, high=384, shape=game_area_shape, dtype=np.uint32)

            if self.config.observation_config.include_memory:
                obs_dict['memory'] = spaces.Box(
                    low=0, high=255, shape=(self.config.observation_config.memory_size,), dtype=np.uint8
                )

            self.observation_space = spaces.Dict(obs_dict)
        else:
            raise ValueError(f"Unknown observation type: {self.config.observation_config.type}")

    def reset(self, **kwargs) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Reset the environment to initial state.

        Args:
            **kwargs: Additional reset options

        Returns:
            Initial observation
        """
        with self._lock:
            # Reset emulator to initial state
            if self.initial_state:
                self._load_state(self.initial_state)
            else:
                # Fallback: restart PyBoy
                self.pyboy.stop(save=False)
                self._init_pyboy()

            # Reset environment state
            self.current_step = 0
            self.episode_reward = 0.0
            self.episode_length = 0
            self.done = False
            self.truncated = False

            # Clear history
            self.state_history.clear()
            self.action_history.clear()

            # Reset reward system
            self.reward_system.reset()

            # Reset game state tracker
            self.game_state_tracker.reset()

            # Get initial observation
            observation = self._get_observation()

            # Log episode start
            self._log_metrics("episode_start", {
                "timestamp": time.time(),
                "rom_path": self.rom_path
            })

            return observation

    def step(self, action: Union[int, np.ndarray, List[int]]) -> RLStepResult:
        """
        Execute one step in the environment.

        Args:
            action: Action to execute

        Returns:
            RLStepResult containing observation, reward, done, info
        """
        with self._lock:
            # Validate action
            if not self.action_space.contains(action):
                raise ValueError(f"Invalid action: {action}")

            # Convert action to PyBoy inputs
            pyboy_actions = self.action_manager.action_to_pyboy(action)

            # Execute action for specified number of frames
            total_reward = 0.0
            for i in range(self.config.frames_per_action):
                # Execute action
                if i < len(pyboy_actions):
                    self._execute_pyboy_action(pyboy_actions[i])
                else:
                    # No action or repeat last action
                    pass

                # Tick emulator
                self.pyboy.tick(1, render=self.render_mode == 'human')

                # Calculate reward for this frame
                frame_reward = self.reward_system.get_reward()
                total_reward += frame_reward

            # Update state
            self.current_step += self.config.frames_per_action
            self.episode_length += 1
            self.episode_reward += total_reward

            # Store action in history
            self.action_history.append(action)

            # Check if episode is done
            self.done = self._check_done()
            self.truncated = self._check_truncated()

            # Get observation
            observation = self._get_observation()

            # Update game state tracker
            self.game_state_tracker.update()

            # Prepare info dictionary
            info = self._get_info()

            # Log step metrics
            self._log_metrics("step", {
                "step": self.current_step,
                "reward": total_reward,
                "episode_reward": self.episode_reward,
                "episode_length": self.episode_length,
                "done": self.done,
                "truncated": self.truncated
            })

            return RLStepResult(
                observation=observation,
                reward=total_reward,
                done=self.done or self.truncated,
                info=info,
                truncated=self.truncated
            )

    def _execute_pyboy_action(self, action: str):
        """Execute a single PyBoy action."""
        if action == "NOOP":
            return

        # Press button
        self.pyboy.button_press(action)

        # Release button (immediately or after delay)
        if self.config.button_press_duration > 0:
            # Button remains pressed for duration
            pass
        else:
            # Release immediately
            self.pyboy.button_release(action)

    def _get_observation(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Get current observation based on configuration."""
        if self.config.observation_config.type == "screen":
            return self._get_screen_observation()
        elif self.config.observation_config.type == "game_area":
            return self._get_game_area_observation()
        elif self.config.observation_config.type == "memory":
            return self._get_memory_observation()
        elif self.config.observation_config.type == "tiles":
            return self._get_tiles_observation()
        elif self.config.observation_config.type == "multi":
            return self._get_multi_observation()
        else:
            raise ValueError(f"Unknown observation type: {self.config.observation_config.type}")

    def _get_screen_observation(self) -> np.ndarray:
        """Get screen observation."""
        screen = self.pyboy.screen.ndarray

        # Convert RGBA to RGB if needed
        if screen.shape[2] == 4:
            screen = screen[:, :, :3]

        # Apply preprocessing if configured
        if self.config.observation_config.grayscale:
            screen = np.dot(screen[..., :3], [0.2989, 0.5870, 0.1140])
            screen = np.expand_dims(screen, axis=-1)

        if self.config.observation_config.resize_observation:
            # Resize observation (implement resize logic)
            pass

        return screen

    def _get_game_area_observation(self) -> np.ndarray:
        """Get game area observation."""
        if self.pyboy.game_wrapper:
            return np.array(self.pyboy.game_wrapper.game_area(), dtype=np.uint32)
        else:
            return np.zeros((32, 32), dtype=np.uint32)

    def _get_memory_observation(self) -> np.ndarray:
        """Get memory observation."""
        memory_start = self.config.observation_config.memory_start
        memory_size = self.config.observation_config.memory_size

        memory_data = []
        for i in range(memory_size):
            addr = memory_start + i
            try:
                memory_data.append(self.pyboy.memory[addr])
            except:
                memory_data.append(0)

        return np.array(memory_data, dtype=np.uint8)

    def _get_tiles_observation(self) -> np.ndarray:
        """Get tiles observation."""
        width = self.config.observation_config.tiles_width
        height = self.config.observation_config.tiles_height

        if self.pyboy.game_wrapper:
            # Extract tiles from game area
            game_area = self.pyboy.game_wrapper.game_area()
            return np.array(game_area[:height, :width], dtype=np.uint32)
        else:
            return np.zeros((height, width), dtype=np.uint32)

    def _get_multi_observation(self) -> Dict[str, np.ndarray]:
        """Get multi-modal observation."""
        observation = {}

        if self.config.observation_config.include_screen:
            observation['screen'] = self._get_screen_observation()

        if self.config.observation_config.include_game_area:
            observation['game_area'] = self._get_game_area_observation()

        if self.config.observation_config.include_memory:
            observation['memory'] = self._get_memory_observation()

        return observation

    def _check_done(self) -> bool:
        """Check if episode is done."""
        # Check game over condition
        if self.pyboy.game_wrapper and hasattr(self.pyboy.game_wrapper, 'game_over'):
            if self.pyboy.game_wrapper.game_over():
                return True

        # Check reward system done condition
        if self.reward_system.is_done():
            return True

        # Check max steps
        if self.config.max_steps and self.current_step >= self.config.max_steps:
            return True

        return False

    def _check_truncated(self) -> bool:
        """Check if episode should be truncated."""
        # Check max episode length
        if self.config.max_episode_length and self.episode_length >= self.config.max_episode_length:
            return True

        # Check time limit
        if self.config.time_limit and (time.time() - self.start_time) >= self.config.time_limit:
            return True

        return False

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info dictionary."""
        info = {
            'current_step': self.current_step,
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
            'frame_count': self.pyboy.frame_count,
            'game_title': self.pyboy.cartridge_title,
            'game_state': self.game_state_tracker.get_state(),
            'reward_breakdown': self.reward_system.get_reward_breakdown(),
        }

        # Add action space info
        info.update(self.action_manager.get_action_info())

        # Add observation space info
        if hasattr(self, '_last_observation_stats'):
            info['observation_stats'] = self._last_observation_stats

        return info

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """Render the environment."""
        if mode is None:
            mode = self.render_mode

        if mode == 'human':
            # PyBoy handles human rendering
            pass
        elif mode == 'rgb_array':
            return self._get_screen_observation()
        elif mode == 'ansi':
            # Return ASCII representation of game area
            if self.pyboy.game_wrapper:
                return str(self.pyboy.game_wrapper)
            else:
                return "No game wrapper available"

        return None

    def close(self):
        """Close the environment."""
        with self._lock:
            if hasattr(self, 'pyboy') and self.pyboy:
                self.pyboy.stop(save=False)
                self.pyboy = None

            # Log final metrics
            self._log_metrics("episode_end", {
                "timestamp": time.time(),
                "total_reward": self.episode_reward,
                "total_length": self.episode_length,
                "total_steps": self.current_step,
            })

    def _save_state(self) -> bytes:
        """Save current emulator state."""
        state_buffer = io.BytesIO()
        self.pyboy.save_state(state_buffer)
        return state_buffer.getvalue()

    def _load_state(self, state: bytes):
        """Load emulator state."""
        state_buffer = io.BytesIO(state)
        self.pyboy.load_state(state_buffer)

    def save_state(self) -> bytes:
        """Save environment state including RL state."""
        state_data = {
            'emulator_state': self._save_state(),
            'current_step': self.current_step,
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
            'done': self.done,
            'truncated': self.truncated,
            'config': self.config.to_dict(),
        }

        return json.dumps(state_data).encode('utf-8')

    def load_state(self, state: bytes):
        """Load environment state including RL state."""
        state_data = json.loads(state.decode('utf-8'))

        self._load_state(state_data['emulator_state'])
        self.current_step = state_data['current_step']
        self.episode_reward = state_data['episode_reward']
        self.episode_length = state_data['episode_length']
        self.done = state_data['done']
        self.truncated = state_data['truncated']

    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        return dict(self.metrics)

    def _log_metrics(self, key: str, value: Any):
        """Log metrics."""
        self.metrics[key].append({
            'timestamp': time.time(),
            'value': value
        })

    def seed(self, seed: Optional[int] = None):
        """Set random seed."""
        np.random.seed(seed)
        return [seed]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()