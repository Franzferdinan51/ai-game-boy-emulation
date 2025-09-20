"""
Enhanced game state tracking for PyBoy RL environment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict, deque
import json
import copy


class GameStateType(Enum):
    """Types of game state information."""
    BASIC = "basic"
    MEMORY = "memory"
    SCREEN = "screen"
    SPRITES = "sprites"
    TILES = "tiles"
    INPUT = "input"
    PERFORMANCE = "performance"
    CUSTOM = "custom"


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
    memory_scan_interval: int = 60  # Scan memory every N frames
    state_history_length: int = 1000
    input_history_length: int = 100
    custom_trackers: Dict[str, Callable] = field(default_factory=dict)


@dataclass
class ScreenAnalysis:
    """Screen analysis results."""
    mean_color: Tuple[float, float, float]
    std_color: Tuple[float, float, float]
    unique_colors: int
    dominant_color: Tuple[int, int, int]
    brightness: float
    contrast: float
    entropy: float
    motion_score: float


@dataclass
class SpriteInfo:
    """Sprite information."""
    index: int
    x: int
    y: int
    tile_identifier: int
    on_screen: bool
    size: Tuple[int, int]
    palette: int
    flags: int


@dataclass
class GameStateSnapshot:
    """Complete game state snapshot."""
    timestamp: float
    frame_count: int
    basic_state: Dict[str, Any]
    memory_state: Dict[str, Any]
    screen_analysis: ScreenAnalysis
    sprites: List[SpriteInfo]
    tile_state: Dict[str, Any]
    input_state: Dict[str, Any]
    performance_metrics: Dict[str, float]
    custom_state: Dict[str, Any]


class GameStateTracker:
    """
    Enhanced game state tracking system.

    This system tracks various aspects of the game state:
    - Basic game information
    - Memory region analysis
    - Screen analysis and metrics
    - Sprite tracking
    - Tile state monitoring
    - Input history
    - Performance metrics
    - Custom tracking functions
    """

    def __init__(self, pyboy, config: StateTrackingConfig):
        """
        Initialize the game state tracker.

        Args:
            pyboy: PyBoy instance
            config: State tracking configuration
        """
        self.pyboy = pyboy
        self.config = config

        # Initialize tracking variables
        self.current_frame = 0
        self.last_memory_scan = 0
        self.memory_snapshots = deque(maxlen=config.state_history_length)
        self.state_history = deque(maxlen=config.state_history_length)
        self.input_history = deque(maxlen=config.input_history_length)
        self.screen_history = deque(maxlen=60)  # Keep last 60 frames for motion detection
        self.previous_screen = None

        # Initialize performance tracking
        self.performance_metrics = defaultdict(list)
        self.last_update_time = time.time()

        # Initialize custom trackers
        self.custom_state = {}
        for name, tracker in config.custom_trackers.items():
            self.custom_state[name] = tracker(self.pyboy)

    def update(self):
        """Update all tracking components."""
        current_time = time.time()

        # Update basic state
        basic_state = self._get_basic_state()

        # Update memory state (less frequently)
        memory_state = {}
        if self.config.track_memory_regions and self.current_frame - self.last_memory_scan >= self.config.memory_scan_interval:
            memory_state = self._get_memory_state()
            self.memory_snapshots.append(memory_state)
            self.last_memory_scan = self.current_frame

        # Update screen analysis
        screen_analysis = self._get_screen_analysis()

        # Update sprite tracking
        sprites = self._get_sprite_info() if self.config.track_sprites else []

        # Update tile state
        tile_state = self._get_tile_state() if self.config.track_tiles else {}

        # Update input state
        input_state = self._get_input_state() if self.config.track_input_history else {}

        # Update performance metrics
        if self.config.track_performance:
            self._update_performance_metrics(current_time)

        # Update custom trackers
        for name, tracker in self.config.custom_trackers.items():
            self.custom_state[name] = tracker(self.pyboy)

        # Create and store state snapshot
        snapshot = GameStateSnapshot(
            timestamp=current_time,
            frame_count=self.pyboy.frame_count,
            basic_state=basic_state,
            memory_state=memory_state,
            screen_analysis=screen_analysis,
            sprites=sprites,
            tile_state=tile_state,
            input_state=input_state,
            performance_metrics=self._get_performance_metrics(),
            custom_state=copy.deepcopy(self.custom_state)
        )

        self.state_history.append(snapshot)

        # Increment frame counter
        self.current_frame += 1

    def _get_basic_state(self) -> Dict[str, Any]:
        """Get basic game state information."""
        return {
            'frame_count': self.pyboy.frame_count,
            'cartridge_title': self.pyboy.cartridge_title,
            'is_cgb': self.pyboy.mb.cgb,
            'is_paused': self.pyboy.paused,
            'is_stopped': self.pyboy.stopped,
            'emulation_speed': getattr(self.pyboy, 'target_emulationspeed', 1.0),
            'current_tick_time': getattr(self.pyboy, 'avg_tick', 0.0),
            'current_emu_time': getattr(self.pyboy, 'avg_emu', 0.0),
        }

    def _get_memory_state(self) -> Dict[str, Any]:
        """Get memory state analysis."""
        memory_state = {}

        for start_addr, end_addr in self.config.memory_regions:
            region_name = f"region_{start_addr:04X}_{end_addr:04X}"
            memory_data = []

            # Read memory region
            for addr in range(start_addr, end_addr):
                try:
                    memory_data.append(self.pyboy.memory[addr])
                except:
                    memory_data.append(0)

            # Analyze memory region
            memory_array = np.array(memory_data, dtype=np.uint8)

            memory_state[region_name] = {
                'start_addr': start_addr,
                'end_addr': end_addr,
                'size': len(memory_data),
                'mean': float(memory_array.mean()),
                'std': float(memory_array.std()),
                'min': int(memory_array.min()),
                'max': int(memory_array.max()),
                'unique_values': int(len(np.unique(memory_array))),
                'entropy': self._calculate_entropy(memory_array),
                'zero_bytes': int(np.sum(memory_array == 0)),
                'non_zero_bytes': int(np.sum(memory_array != 0)),
                'checksum': int(np.sum(memory_array) & 0xFFFF),
            }

        return memory_state

    def _get_screen_analysis(self) -> ScreenAnalysis:
        """Get screen analysis."""
        screen = self.pyboy.screen.ndarray

        # Convert RGBA to RGB if needed
        if screen.shape[2] == 4:
            screen = screen[:, :, :3]

        # Calculate basic statistics
        mean_color = tuple(screen.mean(axis=(0, 1)))
        std_color = tuple(screen.std(axis=(0, 1)))

        # Calculate unique colors
        unique_pixels = screen.reshape(-1, screen.shape[2])
        unique_colors = len(np.unique(unique_pixels, axis=0))

        # Find dominant color
        unique, counts = np.unique(unique_pixels, axis=0, return_counts=True)
        dominant_color = tuple(unique[np.argmax(counts)])

        # Calculate brightness and contrast
        gray = np.dot(screen[..., :3], [0.2989, 0.5870, 0.1140])
        brightness = float(gray.mean())
        contrast = float(gray.std())

        # Calculate entropy
        entropy = self._calculate_entropy(gray.astype(np.uint8))

        # Calculate motion score
        motion_score = 0.0
        if self.previous_screen is not None and len(self.screen_history) > 0:
            motion_score = float(np.mean(np.abs(screen - self.previous_screen)))

        # Store for next frame
        self.previous_screen = screen.copy()
        self.screen_history.append(screen.copy())

        return ScreenAnalysis(
            mean_color=mean_color,
            std_color=std_color,
            unique_colors=unique_colors,
            dominant_color=dominant_color,
            brightness=brightness,
            contrast=contrast,
            entropy=entropy,
            motion_score=motion_score
        )

    def _get_sprite_info(self) -> List[SpriteInfo]:
        """Get sprite information."""
        sprites = []

        try:
            for i in range(40):  # Game Boy supports 40 sprites
                sprite = self.pyboy.get_sprite(i)

                sprites.append(SpriteInfo(
                    index=i,
                    x=sprite.x,
                    y=sprite.y,
                    tile_identifier=sprite.tile_identifier,
                    on_screen=sprite.on_screen,
                    size=(sprite.width, sprite.height),
                    palette=sprite.palette,
                    flags=sprite.flags
                ))
        except:
            # Fallback if sprite access fails
            pass

        return sprites

    def _get_tile_state(self) -> Dict[str, Any]:
        """Get tile state information."""
        tile_state = {}

        if self.pyboy.game_wrapper:
            try:
                # Get game area
                game_area = self.pyboy.game_wrapper.game_area()
                game_area_array = np.array(game_area)

                tile_state['game_area'] = {
                    'shape': game_area_array.shape,
                    'unique_tiles': int(len(np.unique(game_area_array))),
                    'tile_distribution': dict(zip(*np.unique(game_area_array, return_counts=True))),
                    'mean_tile': float(game_area_array.mean()),
                    'std_tile': float(game_area_array.std()),
                }

                # Get background and window tilemaps
                if hasattr(self.pyboy, 'tilemap_background'):
                    bg_tilemap = self.pyboy.tilemap_background
                    tile_state['background'] = {
                        'shape': bg_tilemap.shape,
                        'unique_tiles': int(len(np.unique(bg_tilemap))),
                    }

                if hasattr(self.pyboy, 'tilemap_window'):
                    window_tilemap = self.pyboy.tilemap_window
                    tile_state['window'] = {
                        'shape': window_tilemap.shape,
                        'unique_tiles': int(len(np.unique(window_tilemap))),
                    }

            except Exception as e:
                print(f"Error getting tile state: {e}")

        return tile_state

    def _get_input_state(self) -> Dict[str, Any]:
        """Get input state."""
        # This would track the current input state
        # Since PyBoy doesn't directly expose this, we'll track what we can
        return {
            'frame_count': self.current_frame,
            'input_history_length': len(self.input_history),
        }

    def _update_performance_metrics(self, current_time: float):
        """Update performance metrics."""
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        fps = 1.0 / dt if dt > 0 else 0.0

        self.performance_metrics['fps'].append(fps)
        self.performance_metrics['tick_time'].append(getattr(self.pyboy, 'avg_tick', 0.0))
        self.performance_metrics['emu_time'].append(getattr(self.pyboy, 'avg_emu', 0.0))

        # Keep only recent values
        for key in self.performance_metrics:
            if len(self.performance_metrics[key]) > 100:
                self.performance_metrics[key] = self.performance_metrics[key][-100:]

    def _get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        metrics = {}

        for key, values in self.performance_metrics.items():
            if values:
                metrics[key] = {
                    'current': values[-1],
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }

        return metrics

    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy of data."""
        if len(data) == 0:
            return 0.0

        # Calculate histogram
        hist, _ = np.histogram(data, bins=256, range=(0, 256))

        # Normalize to probabilities
        prob = hist / float(np.sum(hist))

        # Remove zero probabilities
        prob = prob[prob > 0]

        # Calculate entropy
        entropy = -np.sum(prob * np.log2(prob))

        return float(entropy)

    def get_state(self) -> Dict[str, Any]:
        """Get current game state summary."""
        if not self.state_history:
            return {}

        latest_state = self.state_history[-1]

        return {
            'frame_count': latest_state.frame_count,
            'timestamp': latest_state.timestamp,
            'basic_state': latest_state.basic_state,
            'screen_analysis': {
                'brightness': latest_state.screen_analysis.brightness,
                'contrast': latest_state.screen_analysis.contrast,
                'unique_colors': latest_state.screen_analysis.unique_colors,
                'motion_score': latest_state.screen_analysis.motion_score,
            },
            'sprite_count': len(latest_state.sprites),
            'visible_sprites': len([s for s in latest_state.sprites if s.on_screen]),
            'memory_regions': len(latest_state.memory_state),
            'performance_metrics': latest_state.performance_metrics,
            'custom_state': latest_state.custom_state,
        }

    def get_detailed_state(self) -> Dict[str, Any]:
        """Get detailed game state information."""
        if not self.state_history:
            return {}

        latest_state = self.state_history[-1]

        return {
            'snapshot': latest_state,
            'state_history_length': len(self.state_history),
            'memory_snapshots_count': len(self.memory_snapshots),
            'input_history_length': len(self.input_history),
            'screen_history_length': len(self.screen_history),
        }

    def get_memory_changes(self, window_size: int = 10) -> Dict[str, Any]:
        """Get memory change analysis over recent frames."""
        if len(self.memory_snapshots) < 2:
            return {}

        recent_snapshots = list(self.memory_snapshots)[-window_size:]
        changes = {}

        for region in recent_snapshots[0].keys():
            if region in recent_snapshots[-1]:
                old_data = recent_snapshots[0][region]
                new_data = recent_snapshots[-1][region]

                # Calculate changes
                checksum_change = new_data['checksum'] - old_data['checksum']
                mean_change = new_data['mean'] - old_data['mean']
                std_change = new_data['std'] - old_data['std']

                changes[region] = {
                    'checksum_change': checksum_change,
                    'mean_change': mean_change,
                    'std_change': std_change,
                    'unique_values_change': new_data['unique_values'] - old_data['unique_values'],
                }

        return changes

    def get_sprite_analysis(self) -> Dict[str, Any]:
        """Get sprite analysis."""
        if not self.state_history:
            return {}

        latest_state = self.state_history[-1]

        if not latest_state.sprites:
            return {'sprite_count': 0}

        # Analyze sprite positions
        x_positions = [s.x for s in latest_state.sprites]
        y_positions = [s.y for s in latest_state.sprites]

        return {
            'sprite_count': len(latest_state.sprites),
            'visible_sprites': len([s for s in latest_state.sprites if s.on_screen]),
            'x_range': {'min': min(x_positions), 'max': max(x_positions)} if x_positions else {},
            'y_range': {'min': min(y_positions), 'max': max(y_positions)} if y_positions else {},
            'center_x': float(np.mean(x_positions)) if x_positions else 0,
            'center_y': float(np.mean(y_positions)) if y_positions else 0,
            'unique_tiles': len(set(s.tile_identifier for s in latest_state.sprites)),
        }

    def get_screen_history_analysis(self) -> Dict[str, Any]:
        """Get screen history analysis."""
        if len(self.screen_history) < 2:
            return {}

        # Calculate motion over recent frames
        motion_scores = []
        for i in range(1, len(self.screen_history)):
            motion = np.mean(np.abs(self.screen_history[i] - self.screen_history[i-1]))
            motion_scores.append(float(motion))

        return {
            'frame_count': len(self.screen_history),
            'avg_motion': float(np.mean(motion_scores)) if motion_scores else 0.0,
            'max_motion': float(np.max(motion_scores)) if motion_scores else 0.0,
            'motion_trend': 'increasing' if len(motion_scores) > 1 and motion_scores[-1] > motion_scores[-2] else 'stable',
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.performance_metrics:
            return {}

        summary = {}
        for key, values in self.performance_metrics.items():
            if values:
                summary[key] = {
                    'current': values[-1],
                    'mean_1min': float(np.mean(values[-60:])) if len(values) >= 60 else float(np.mean(values)),
                    'mean_5min': float(np.mean(values[-300:])) if len(values) >= 300 else float(np.mean(values)),
                    'trend': 'improving' if len(values) > 10 and values[-1] < np.mean(values[-10:]) else 'stable',
                }

        return summary

    def export_state(self, filepath: str):
        """Export current state to file."""
        if not self.state_history:
            return

        state_data = {
            'current_state': self.get_state(),
            'detailed_state': self.get_detailed_state(),
            'memory_changes': self.get_memory_changes(),
            'sprite_analysis': self.get_sprite_analysis(),
            'screen_history_analysis': self.get_screen_history_analysis(),
            'performance_summary': self.get_performance_summary(),
            'export_timestamp': time.time(),
        }

        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)

    def reset(self):
        """Reset the state tracker."""
        self.current_frame = 0
        self.last_memory_scan = 0
        self.memory_snapshots.clear()
        self.state_history.clear()
        self.input_history.clear()
        self.screen_history.clear()
        self.previous_screen = None
        self.performance_metrics.clear()
        self.last_update_time = time.time()

        # Reset custom trackers
        for name, tracker in self.config.custom_trackers.items():
            self.custom_state[name] = tracker(self.pyboy)

    def add_custom_tracker(self, name: str, tracker_func: Callable):
        """Add a custom tracking function."""
        self.config.custom_trackers[name] = tracker_func
        self.custom_state[name] = tracker_func(self.pyboy)

    def remove_custom_tracker(self, name: str):
        """Remove a custom tracking function."""
        if name in self.config.custom_trackers:
            del self.config.custom_trackers[name]
        if name in self.custom_state:
            del self.custom_state[name]