"""
Visualization example for PyBoy RL environment.
"""

import sys
import os
import time
import threading

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.backend.rl import (
    PyBoyEnv,
    RLEnvironmentConfig,
    LiveVisualizer,
    VideoRecorder,
    VisualizationConfig
)


def check_visualization_dependencies():
    """Check if visualization dependencies are available."""
    missing = []

    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")

    try:
        import seaborn
    except ImportError:
        missing.append("seaborn")

    try:
        import plotly
    except ImportError:
        missing.append("plotly")

    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")

    if missing:
        print("Missing visualization dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("Install with: pip install matplotlib seaborn plotly opencv-python Pillow")
        return False

    print("All visualization dependencies are available!")
    return True


def live_visualization_example():
    """Example of live visualization during training."""
    print("=== Live Visualization Example ===")

    if not check_visualization_dependencies():
        return

    # Create environment configuration
    env_config = RLEnvironmentConfig(
        headless=False,  # Need non-headless for visualization
        frames_per_action=4,
        max_steps=1000
    )

    # Configure for visualization
    env_config.observation_config.type = "screen"
    env_config.action_config.type = "discrete"

    # Create visualization configuration
    viz_config = VisualizationConfig(
        window_size=(1200, 800),
        update_interval=0.05,  # Update every 50ms
        save_screenshots=True,
        screenshot_dir="./visualization_screenshots",
        enable_live_plotting=True,
        color_scheme="dark"
    )

    # Note: Replace with actual ROM path
    rom_path = "path/to/your/game.gb"

    try:
        print(f"Creating environment with live visualization: {rom_path}")

        # Create environment
        env = PyBoyEnv(rom_path, config=env_config)

        # Create visualizer
        visualizer = LiveVisualizer(env, viz_config)

        print("Starting visualization...")
        visualizer.start()

        # Run environment with visualization
        obs = env.reset()
        total_reward = 0

        print("Running environment with visualization...")
        print("Close the plot window to stop")

        # Run for a limited time with visualization
        for step in range(200):
            # Sample random action
            action = env.action_space.sample()

            # Execute action
            obs, reward, done, info = env.step(action)

            # Update visualization
            visualizer.update()

            total_reward += reward

            # Print progress
            if step % 50 == 0:
                print(f"Step {step}: Reward={reward:.2f}, Total={total_reward:.2f}")

            if done:
                print(f"Episode finished at step {step}")
                break

            # Small delay to allow visualization updates
            time.sleep(0.01)

        print(f"Visualization completed!")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Frames processed: {step}")

        # Get visualization summary
        summary = visualizer.get_visualization_summary()
        print(f"Visualization summary: {summary}")

        # Stop visualization
        visualizer.stop()

        # Close environment
        env.close()

    except Exception as e:
        print(f"Error during live visualization: {e}")


def video_recording_example():
    """Example of recording video during gameplay."""
    print("\n=== Video Recording Example ===")

    if not check_visualization_dependencies():
        return

    # Create environment configuration
    env_config = RLEnvironmentConfig(
        headless=True,  # Headless for video recording
        frames_per_action=4,
        max_steps=1000
    )

    # Note: Replace with actual ROM path
    rom_path = "path/to/your/game.gb"

    try:
        print(f"Creating environment for video recording: {rom_path}")

        # Create environment
        env = PyBoyEnv(rom_path, config=env_config)

        # Create video recorder
        video_path = "./recorded_gameplay.mp4"
        print(f"Recording video to: {video_path}")

        with VideoRecorder(env, video_path, fps=30) as recorder:
            obs = env.reset()
            total_reward = 0

            # Run episode and record video
            for step in range(300):  # Record 300 steps
                # Sample action
                action = env.action_space.sample()

                # Execute action
                obs, reward, done, info = env.step(action)

                # Record frame
                recorder.record_frame()

                total_reward += reward

                if step % 100 == 0:
                    print(f"Step {step}: Recording...")

                if done:
                    print(f"Episode finished at step {step}")
                    break

        print(f"Video recording completed!")
        print(f"Video saved to: {video_path}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Frames recorded: {recorder.frame_count}")

        # Close environment
        env.close()

    except Exception as e:
        print(f"Error during video recording: {e}")


def metrics_visualization_example():
    """Example of visualizing training metrics."""
    print("\n=== Metrics Visualization Example ===")

    if not check_visualization_dependencies():
        return

    # Create environment and collect metrics
    env_config = RLEnvironmentConfig(
        headless=True,
        frames_per_action=4,
        max_steps=500
    )

    # Note: Replace with actual ROM path
    rom_path = "path/to/your/game.gb"

    try:
        print(f"Collecting metrics for visualization: {rom_path}")

        # Create environment
        env = PyBoyEnv(rom_path, config=env_config)

        # Create visualizer
        viz_config = VisualizationConfig(
            enable_live_plotting=False,  # Disable live plotting
            save_screenshots=False
        )
        visualizer = LiveVisualizer(env, viz_config)

        # Collect data
        obs = env.reset()
        for step in range(200):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            # Update visualizer data
            visualizer.update()

            if done:
                obs = env.reset()

        print("Creating visualizations...")

        # Create reward animation
        visualizer.create_reward_animation("./reward_animation.gif", fps=10)
        print("Reward animation saved: ./reward_animation.gif")

        # Create memory heatmap
        visualizer.create_memory_heatmap("./memory_heatmap.png")
        print("Memory heatmap saved: ./memory_heatmap.png")

        # Create interactive dashboard
        visualizer.create_interactive_dashboard("./interactive_dashboard.html")
        print("Interactive dashboard saved: ./interactive_dashboard.html")

        # Export visualization data
        visualizer.export_visualization_data("./visualization_data.json")
        print("Visualization data exported: ./visualization_data.json")

        # Get summary
        summary = visualizer.get_visualization_summary()
        print(f"Visualization summary: {summary}")

        # Close environment
        env.close()
        visualizer.stop()

        print("Metrics visualization completed!")

    except Exception as e:
        print(f"Error during metrics visualization: {e}")


def real_time_monitoring_example():
    """Example of real-time monitoring during training."""
    print("\n=== Real-time Monitoring Example ===")

    if not check_visualization_dependencies():
        return

    # Create environment configuration
    env_config = RLEnvironmentConfig(
        headless=True,
        frames_per_action=4,
        max_steps=1000
    )

    # Configure detailed state tracking
    env_config.state_config = {
        "track_basic_state": True,
        "track_memory_regions": True,
        "track_screen_analysis": True,
        "track_sprites": True,
        "track_performance": True
    }

    # Note: Replace with actual ROM path
    rom_path = "path/to/your/game.gb"

    try:
        print(f"Setting up real-time monitoring: {rom_path}")

        # Create environment
        env = PyBoyEnv(rom_path, config=env_config)

        # Create monitoring thread
        stop_event = threading.Event()

        def monitoring_thread():
            """Background monitoring thread."""
            while not stop_event.is_set():
                # Get current metrics
                if hasattr(env, 'game_state_tracker'):
                    state = env.game_state_tracker.get_state()
                    perf_summary = env.game_state_tracker.get_performance_summary()

                    print(f"\\rMonitor: FPS={perf_summary.get('fps', {}).get('current', 0):.1f} "
                          f"Brightness={state.get('screen_analysis', {}).get('brightness', 0):.2f} "
                          f"Sprites={state.get('sprite_count', 0)}", end="")

                time.sleep(0.1)

        # Start monitoring thread
        monitor = threading.Thread(target=monitoring_thread)
        monitor.daemon = True
        monitor.start()

        print("Starting monitoring (press Ctrl+C to stop)...")

        try:
            # Run environment
            obs = env.reset()
            for step in range(300):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)

                if done:
                    obs = env.reset()

                if step % 50 == 0:
                    print(f"\\nStep {step}: Reward={reward:.2f}")

        except KeyboardInterrupt:
            print("\\nMonitoring stopped by user")

        # Stop monitoring
        stop_event.set()
        monitor.join(timeout=1.0)

        print("\\nFinal metrics:")
        if hasattr(env, 'game_state_tracker'):
            final_state = env.game_state_tracker.get_state()
            final_perf = env.game_state_tracker.get_performance_summary()

            print(f"  Final FPS: {final_perf.get('fps', {}).get('current', 0):.1f}")
            print(f"  Final brightness: {final_state.get('screen_analysis', {}).get('brightness', 0):.2f}")
            print(f"  Final sprite count: {final_state.get('sprite_count', 0)}")

        # Close environment
        env.close()

        print("Real-time monitoring completed!")

    except Exception as e:
        print(f"Error during real-time monitoring: {e}")


def comparative_visualization_example():
    """Example comparing different visualization approaches."""
    print("\n=== Comparative Visualization Example ===")

    if not check_visualization_dependencies():
        return

    # Test different observation types
    observation_types = ["screen", "game_area", "memory"]

    for obs_type in observation_types:
        print(f"\\nTesting {obs_type} visualization...")

        # Create environment configuration
        env_config = RLEnvironmentConfig(
            headless=True,
            frames_per_action=4,
            max_steps=100
        )

        env_config.observation_config.type = obs_type

        # Note: Replace with actual ROM path
        rom_path = "path/to/your/game.gb"

        try:
            # Create environment
            env = PyBoyEnv(rom_path, config=env_config)

            # Create visualizer
            viz_config = VisualizationConfig(
                enable_live_plotting=False,
                save_screenshots=False
            )
            visualizer = LiveVisualizer(env, viz_config)

            # Collect data
            obs = env.reset()
            for step in range(100):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                visualizer.update()

                if done:
                    obs = env.reset()

            # Create specific visualization for this observation type
            output_file = f"./{obs_type}_visualization.png"

            # Create a simple plot showing observation characteristics
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            if obs_type == "screen":
                axes[0].imshow(obs)
                axes[0].set_title("Screen Observation")
                axes[0].axis('off')

                # Show color histogram
                if len(obs.shape) == 3:
                    colors = ['Red', 'Green', 'Blue']
                    for i, color in enumerate(colors):
                        hist, bins = np.histogram(obs[:, :, i], bins=50)
                        axes[1].plot(bins[:-1], hist, label=color, alpha=0.7)
                    axes[1].set_title("Color Distribution")
                    axes[1].legend()

            elif obs_type == "game_area":
                axes[0].imshow(obs, cmap='tab10')
                axes[0].set_title("Game Area Observation")
                axes[0].axis('off')

                # Show tile distribution
                unique, counts = np.unique(obs, return_counts=True)
                axes[1].bar(unique, counts)
                axes[1].set_title("Tile Distribution")
                axes[1].set_xlabel("Tile ID")
                axes[1].set_ylabel("Count")

            elif obs_type == "memory":
                axes[0].plot(obs)
                axes[0].set_title("Memory Observation")
                axes[0].set_xlabel("Address")
                axes[0].set_ylabel("Value")

                # Show memory histogram
                axes[1].hist(obs, bins=50)
                axes[1].set_title("Memory Value Distribution")
                axes[1].set_xlabel("Value")
                axes[1].set_ylabel("Frequency")

            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  {obs_type} visualization saved: {output_file}")

            # Close environment
            env.close()
            visualizer.stop()

        except Exception as e:
            print(f"  Error with {obs_type}: {e}")

    print("Comparative visualization completed!")


if __name__ == "__main__":
    print("PyBoy RL Visualization Examples")
    print("=" * 50)

    # Run examples
    try:
        live_visualization_example()
        video_recording_example()
        metrics_visualization_example()
        real_time_monitoring_example()
        comparative_visualization_example()

        print("\\n" + "=" * 50)
        print("All visualization examples completed!")
        print("Check the current directory for output files")

    except KeyboardInterrupt:
        print("\\nVisualization examples interrupted by user")
    except Exception as e:
        print(f"\\nError running visualization examples: {e}")
        print("Make sure all visualization dependencies are installed")