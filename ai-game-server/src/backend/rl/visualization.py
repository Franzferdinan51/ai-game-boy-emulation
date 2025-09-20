"""
Visualization and monitoring features for PyBoy RL environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import json
from pathlib import Path
import threading
import queue
from datetime import datetime

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Install with 'pip install plotly'")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available. Install with 'pip install opencv-python'")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available. Install with 'pip install Pillow'")

from .pyboy_env import PyBoyEnv
from .game_state_tracker import GameStateSnapshot


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    window_size: Tuple[int, int] = (800, 600)
    update_interval: float = 0.1  # seconds
    max_history_length: int = 1000
    save_screenshots: bool = False
    screenshot_dir: str = "./screenshots"
    save_videos: bool = False
    video_dir: str = "./videos"
    video_fps: int = 30
    enable_live_plotting: bool = True
    enable_3d_visualization: bool = False
    color_scheme: str = "default"  # "default", "dark", "viridis"


class LiveVisualizer:
    """
    Live visualization system for PyBoy RL environment.

    Features:
    - Real-time screen display
    - Action visualization
    - Reward plotting
    - State metrics
    - Memory visualization
    - Performance monitoring
    """

    def __init__(
        self,
        env: PyBoyEnv,
        config: VisualizationConfig,
        queue_size: int = 100
    ):
        """
        Initialize the live visualizer.

        Args:
            env: PyBoy RL environment
            config: Visualization configuration
            queue_size: Size of the visualization queue
        """
        self.env = env
        self.config = config

        # Initialize data queues
        self.screen_queue = queue.Queue(maxsize=queue_size)
        self.metrics_queue = queue.Queue(maxsize=queue_size)
        self.action_queue = queue.Queue(maxsize=queue_size)

        # Initialize data storage
        self.screens = deque(maxlen=config.max_history_length)
        self.metrics_history = deque(maxlen=config.max_history_length)
        self.action_history = deque(maxlen=config.max_history_length)
        self.reward_history = deque(maxlen=config.max_history_length)
        self.state_history = deque(maxlen=config.max_history_length)

        # Initialize visualization state
        self.running = False
        self.paused = False
        self.current_frame = 0
        self.start_time = time.time()

        # Setup directories
        self._setup_directories()

        # Setup plotting
        self._setup_plotting()

        # Setup live plotting thread
        if config.enable_live_plotting:
            self.plotting_thread = threading.Thread(target=self._plotting_loop)
            self.plotting_thread.daemon = True

    def _setup_directories(self):
        """Setup output directories."""
        if self.config.save_screenshots:
            Path(self.config.screenshot_dir).mkdir(parents=True, exist_ok=True)

        if self.config.save_videos:
            Path(self.config.video_dir).mkdir(parents=True, exist_ok=True)

    def _setup_plotting(self):
        """Setup matplotlib plotting."""
        plt.style.use('dark_background' if self.config.color_scheme == "dark" else 'default')

        # Create figure with subplots
        self.fig, self.axes = plt.subplots(
            2, 3,
            figsize=(15, 10),
            gridspec_kw={'height_ratios': [2, 1]}
        )
        self.fig.suptitle('PyBoy RL Environment Visualization', fontsize=16)

        # Setup individual plots
        self.screen_ax = self.axes[0, 0]
        self.reward_ax = self.axes[0, 1]
        self.metrics_ax = self.axes[0, 2]
        self.action_ax = self.axes[1, 0]
        self.memory_ax = self.axes[1, 1]
        self.performance_ax = self.axes[1, 2]

        # Configure axes
        self.screen_ax.set_title('Game Screen')
        self.screen_ax.axis('off')

        self.reward_ax.set_title('Episode Rewards')
        self.reward_ax.set_xlabel('Episode')
        self.reward_ax.set_ylabel('Reward')

        self.metrics_ax.set_title('Game Metrics')
        self.metrics_ax.set_xlabel('Frame')
        self.metrics_ax.set_ylabel('Value')

        self.action_ax.set_title('Action Distribution')
        self.action_ax.set_xlabel('Action')
        self.action_ax.set_ylabel('Count')

        self.memory_ax.set_title('Memory Regions')
        self.memory_ax.set_xlabel('Region')
        self.memory_ax.set_ylabel('Activity')

        self.performance_ax.set_title('Performance Metrics')
        self.performance_ax.set_xlabel('Time')
        self.performance_ax.set_ylabel('Value')

        plt.tight_layout()

    def start(self):
        """Start the visualization."""
        self.running = True

        if self.config.enable_live_plotting:
            self.plotting_thread.start()

        self.update()

    def stop(self):
        """Stop the visualization."""
        self.running = False
        if hasattr(self, 'plotting_thread'):
            self.plotting_thread.join(timeout=1.0)

        plt.close(self.fig)

    def update(self):
        """Update visualization with current environment state."""
        if not self.running:
            return

        # Get current state
        screen = self.env.render(mode='rgb_array')
        info = self.env.get_info()

        # Update data
        self._update_data(screen, info)

        # Update queues
        self._update_queues(screen, info)

        # Save screenshot if configured
        if self.config.save_screenshots and self.current_frame % 60 == 0:
            self._save_screenshot(screen)

        self.current_frame += 1

    def _update_data(self, screen: np.ndarray, info: Dict[str, Any]):
        """Update internal data storage."""
        # Store screen
        self.screens.append(screen)

        # Store metrics
        self.metrics_history.append({
            'frame': self.current_frame,
            'timestamp': time.time(),
            'episode_reward': info.get('episode_reward', 0),
            'episode_length': info.get('episode_length', 0),
            'frame_count': info.get('frame_count', 0),
            'game_state': info.get('game_state', {}),
        })

        # Store reward
        if 'reward_breakdown' in info:
            self.reward_history.append(info['reward_breakdown'])

        # Store state
        if hasattr(self.env, 'game_state_tracker'):
            state = self.env.game_state_tracker.get_state()
            self.state_history.append(state)

    def _update_queues(self, screen: np.ndarray, info: Dict[str, Any]):
        """Update visualization queues."""
        try:
            # Add to queues (non-blocking)
            self.screen_queue.put_nowait(screen)
            self.metrics_queue.put_nowait(info)
        except queue.Full:
            # Skip if queue is full
            pass

    def _plotting_loop(self):
        """Main plotting loop running in separate thread."""
        while self.running:
            try:
                # Get data from queues
                if not self.screen_queue.empty():
                    screen = self.screen_queue.get(timeout=0.1)
                    self._update_screen_plot(screen)

                if not self.metrics_queue.empty():
                    info = self.metrics_queue.get(timeout=0.1)
                    self._update_metrics_plots(info)

                # Update all plots
                self._update_all_plots()

                # Sleep to control update rate
                time.sleep(self.config.update_interval)

            except (queue.Empty, KeyboardInterrupt):
                continue

    def _update_screen_plot(self, screen: np.ndarray):
        """Update screen plot."""
        self.screen_ax.clear()
        self.screen_ax.imshow(screen)
        self.screen_ax.set_title(f'Game Screen (Frame {self.current_frame})')
        self.screen_ax.axis('off')

    def _update_metrics_plots(self, info: Dict[str, Any]):
        """Update metrics plots."""
        # Update reward plot
        if self.metrics_history:
            rewards = [m['episode_reward'] for m in self.metrics_history]
            episodes = range(len(rewards))

            self.reward_ax.clear()
            self.reward_ax.plot(episodes, rewards, 'b-', linewidth=2)
            self.reward_ax.set_title('Episode Rewards')
            self.reward_ax.set_xlabel('Episode')
            self.reward_ax.set_ylabel('Reward')
            self.reward_ax.grid(True, alpha=0.3)

        # Update metrics plot
        if self.state_history:
            latest_state = self.state_history[-1]

            # Extract various metrics
            metrics = {}
            if 'screen_analysis' in latest_state:
                metrics['brightness'] = latest_state['screen_analysis'].get('brightness', 0)
                metrics['contrast'] = latest_state['screen_analysis'].get('contrast', 0)
                metrics['unique_colors'] = latest_state['screen_analysis'].get('unique_colors', 0)

            if metrics:
                self.metrics_ax.clear()
                names = list(metrics.keys())
                values = list(metrics.values())

                bars = self.metrics_ax.bar(names, values)
                self.metrics_ax.set_title('Game Metrics')
                self.metrics_ax.set_ylabel('Value')

                # Color bars
                for i, (name, value) in enumerate(zip(names, values)):
                    if 'brightness' in name:
                        bars[i].set_color('yellow')
                    elif 'contrast' in name:
                        bars[i].set_color('orange')
                    else:
                        bars[i].set_color('cyan')

        # Update action distribution plot
        if hasattr(self.env, 'action_manager'):
            action_info = self.env.action_manager.get_action_info()
            if 'action_mappings' in action_info:
                actions = [m['action_id'] for m in action_info['action_mappings']]
                descriptions = [m['description'] for m in action_info['action_mappings']]

                self.action_ax.clear()
                self.action_ax.bar(actions, [1] * len(actions))  # Placeholder for actual counts
                self.action_ax.set_title('Available Actions')
                self.action_ax.set_xlabel('Action ID')
                self.action_ax.set_ylabel('Available')
                self.action_ax.set_xticks(actions)
                self.action_ax.set_xticklabels([d[:10] + '...' if len(d) > 10 else d for d in descriptions], rotation=45)

        # Update memory activity plot
        if self.state_history and 'memory_regions' in self.state_history[-1]:
            memory_regions = self.state_history[-1]['memory_regions']
            if memory_regions:
                regions = list(memory_regions.keys())
                activities = [memory_regions[r].get('mean', 0) for r in regions]

                self.memory_ax.clear()
                self.memory_ax.bar(range(len(regions)), activities)
                self.memory_ax.set_title('Memory Region Activity')
                self.memory_ax.set_xlabel('Region')
                self.memory_ax.set_ylabel('Mean Value')
                self.memory_ax.set_xticks(range(len(regions)))
                self.memory_ax.set_xticklabels([r[:8] + '...' if len(r) > 8 else r for r in regions], rotation=45)

        # Update performance plot
        if hasattr(self.env, 'game_state_tracker'):
            perf_summary = self.env.game_state_tracker.get_performance_summary()
            if perf_summary:
                self.performance_ax.clear()

                metrics = list(perf_summary.keys())
                values = [perf_summary[m].get('current', 0) for m in metrics]

                self.performance_ax.bar(metrics, values)
                self.performance_ax.set_title('Performance Metrics')
                self.performance_ax.set_ylabel('Value')
                self.performance_ax.set_xticklabels(metrics, rotation=45)

    def _update_all_plots(self):
        """Update all plots and refresh display."""
        plt.tight_layout()
        plt.pause(0.001)  # Small pause to allow GUI update

    def _save_screenshot(self, screen: np.ndarray):
        """Save screenshot to file."""
        if not PIL_AVAILABLE:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}_{self.current_frame}.png"
        filepath = Path(self.config.screenshot_dir) / filename

        # Convert numpy array to PIL Image
        if screen.dtype == np.uint8:
            image = Image.fromarray(screen)
        else:
            image = Image.fromarray((screen * 255).astype(np.uint8))

        # Add timestamp overlay
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        text = f"Frame: {self.current_frame} | Time: {timestamp}"
        draw.text((10, 10), text, fill=(255, 255, 255), font=font)

        # Save image
        image.save(filepath)

    def create_reward_animation(self, save_path: str, fps: int = 10):
        """Create animation of reward progression."""
        if not self.metrics_history:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        def animate(frame):
            ax.clear()
            if frame < len(self.metrics_history):
                rewards = [m['episode_reward'] for m in self.metrics_history[:frame+1]]
                episodes = range(len(rewards))
                ax.plot(episodes, rewards, 'b-', linewidth=2)
                ax.set_title(f'Episode Rewards (Frame {frame})')
                ax.set_xlabel('Episode')
                ax.set_ylabel('Reward')
                ax.grid(True, alpha=0.3)

        ani = animation.FuncAnimation(
            fig, animate, frames=len(self.metrics_history),
            interval=1000//fps, blit=False
        )

        # Save animation
        writer = animation.PillowWriter(fps=fps)
        ani.save(save_path, writer=writer)
        plt.close(fig)

    def create_memory_heatmap(self, save_path: str):
        """Create memory activity heatmap."""
        if not self.state_history:
            return

        # Collect memory data over time
        memory_data = []
        for state in self.state_history:
            if 'memory_regions' in state:
                row = []
                for region in state['memory_regions'].values():
                    row.append(region.get('mean', 0))
                memory_data.append(row)

        if not memory_data:
            return

        memory_array = np.array(memory_data)

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            memory_array.T,
            cmap='viridis',
            cbar=True,
            xticklabels=False,
            yticklabels=True
        )
        plt.title('Memory Region Activity Over Time')
        plt.xlabel('Time (Frames)')
        plt.ylabel('Memory Region')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def create_interactive_dashboard(self, save_path: str):
        """Create interactive dashboard using Plotly."""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available for interactive dashboard")
            return

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Episode Rewards', 'Performance Metrics', 'Action Distribution', 'Memory Activity'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Episode rewards
        if self.metrics_history:
            episodes = list(range(len(self.metrics_history)))
            rewards = [m['episode_reward'] for m in self.metrics_history]

            fig.add_trace(
                go.Scatter(x=episodes, y=rewards, mode='lines', name='Episode Reward'),
                row=1, col=1
            )

        # Performance metrics
        if hasattr(self.env, 'game_state_tracker'):
            perf_summary = self.env.game_state_tracker.get_performance_summary()
            if perf_summary:
                metrics = list(perf_summary.keys())
                values = [perf_summary[m].get('current', 0) for m in metrics]

                fig.add_trace(
                    go.Bar(x=metrics, y=values, name='Performance'),
                    row=1, col=2
                )

        # Action distribution (placeholder)
        if hasattr(self.env, 'action_manager'):
            action_info = self.env.action_manager.get_action_info()
            if 'action_mappings' in action_info:
                actions = [m['action_id'] for m in action_info['action_mappings']]
                descriptions = [m['description'] for m in action_info['action_mappings']]

                fig.add_trace(
                    go.Bar(x=actions, y=[1] * len(actions), name='Actions'),
                    row=2, col=1
                )

        # Memory activity
        if self.state_history and 'memory_regions' in self.state_history[-1]:
            memory_regions = self.state_history[-1]['memory_regions']
            if memory_regions:
                regions = list(memory_regions.keys())
                activities = [memory_regions[r].get('mean', 0) for r in regions]

                fig.add_trace(
                    go.Bar(x=regions, y=activities, name='Memory Activity'),
                    row=2, col=2
                )

        # Update layout
        fig.update_layout(
            title_text="PyBoy RL Dashboard",
            showlegend=False,
            height=800
        )

        # Save interactive dashboard
        pyo.plot(fig, filename=save_path, auto_open=False)

    def export_visualization_data(self, save_path: str):
        """Export visualization data to JSON."""
        data = {
            'config': {
                'window_size': self.config.window_size,
                'update_interval': self.config.update_interval,
                'max_history_length': self.config.max_history_length,
            },
            'metrics': list(self.metrics_history),
            'rewards': list(self.reward_history),
            'states': list(self.state_history),
            'total_frames': self.current_frame,
            'total_time': time.time() - self.start_time,
        }

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def get_visualization_summary(self) -> Dict[str, Any]:
        """Get summary of visualization data."""
        summary = {
            'total_frames': self.current_frame,
            'total_time': time.time() - self.start_time,
            'fps': self.current_frame / max(1, time.time() - self.start_time),
            'screens_saved': len(self.screens),
            'metrics_points': len(self.metrics_history),
            'reward_points': len(self.reward_history),
            'state_points': len(self.state_history),
        }

        if self.metrics_history:
            rewards = [m['episode_reward'] for m in self.metrics_history]
            summary['rewards'] = {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards),
            }

        return summary


class VideoRecorder:
    """
    Video recording system for PyBoy RL environment.
    """

    def __init__(
        self,
        env: PyBoyEnv,
        output_path: str,
        fps: int = 30,
        codec: str = 'mp4v',
        quality: int = 90
    ):
        """
        Initialize video recorder.

        Args:
            env: PyBoy RL environment
            output_path: Output video file path
            fps: Frames per second
            codec: Video codec
            quality: Video quality (0-100)
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for video recording")

        self.env = env
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.quality = quality

        self.recording = False
        self.video_writer = None
        self.frame_count = 0

    def start_recording(self):
        """Start video recording."""
        if self.recording:
            return

        # Get screen dimensions
        screen = self.env.render(mode='rgb_array')
        height, width = screen.shape[:2]

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.video_writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (width, height)
        )

        self.recording = True
        self.frame_count = 0

    def record_frame(self):
        """Record a single frame."""
        if not self.recording or not self.video_writer:
            return

        # Get screen and convert to BGR (OpenCV format)
        screen = self.env.render(mode='rgb_array')
        frame_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

        # Write frame
        self.video_writer.write(frame_bgr)
        self.frame_count += 1

    def stop_recording(self):
        """Stop video recording."""
        if not self.recording:
            return

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        self.recording = False
        print(f"Video saved: {self.output_path} ({self.frame_count} frames)")

    def __enter__(self):
        self.start_recording()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_recording()