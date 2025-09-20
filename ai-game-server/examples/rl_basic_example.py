"""
Basic example of using PyBoy RL environment.
"""

import sys
import os
import time

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.backend.rl import PyBoyEnv, RLEnvironmentConfig, ObservationConfig, ActionConfig


def basic_environment_example():
    """Basic environment usage example."""
    print("=== Basic PyBoy RL Environment Example ===")

    # Create environment configuration
    config = RLEnvironmentConfig(
        headless=True,  # Run without GUI
        frames_per_action=4,  # Execute 4 frames per action
        max_steps=1000,  # Max steps per episode
        button_press_duration=1  # Button press duration
    )

    # Configure observation space
    config.observation_config = ObservationConfig(
        type="screen",  # Use screen as observation
        grayscale=False,  # Keep RGB colors
        resize_observation=False  # Keep original size
    )

    # Configure action space
    config.action_config = ActionConfig(
        type="discrete",  # Discrete action space
        include_noop=True,  # Include no-operation action
        available_buttons=["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]
    )

    # Note: Replace with actual ROM path
    rom_path = "path/to/your/game.gb"

    try:
        # Create environment
        print(f"Creating environment for ROM: {rom_path}")
        env = PyBoyEnv(rom_path, config=config)

        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        # Run a few episodes
        for episode in range(3):
            print(f"\n--- Episode {episode + 1} ---")

            # Reset environment
            obs = env.reset()
            total_reward = 0
            step_count = 0

            print("Starting episode...")

            # Run episode
            for step in range(100):  # Limit steps for example
                # Sample random action
                action = env.action_space.sample()

                # Execute action
                obs, reward, done, info = env.step(action)

                total_reward += reward
                step_count += 1

                # Print progress every 20 steps
                if step % 20 == 0:
                    print(f"Step {step}: Reward={reward:.2f}, Total={total_reward:.2f}")

                if done:
                    print(f"Episode finished at step {step}")
                    break

            print(f"Episode {episode + 1} finished:")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Steps taken: {step_count}")
            print(f"  Final info: {info}")

        # Close environment
        env.close()
        print("\nExample completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. A valid Game Boy ROM file")
        print("2. PyBoy installed: pip install pyboy")
        print("3. Required dependencies: pip install gym numpy")


def observation_types_example():
    """Example showing different observation types."""
    print("\n=== Observation Types Example ===")

    observation_types = [
        ("screen", "RGB screen capture"),
        ("game_area", "Simplified game area"),
        ("memory", "Memory region values"),
        ("multi", "Multi-modal observations")
    ]

    for obs_type, description in observation_types:
        print(f"\nTesting {obs_type} observation ({description})")

        config = RLEnvironmentConfig(
            headless=True,
            frames_per_action=1
        )

        config.observation_config = ObservationConfig(
            type=obs_type,
            memory_start=0xC000,  # WRAM start
            memory_size=256,     # 256 bytes
            include_screen=obs_type in ["multi"],
            include_game_area=obs_type in ["multi"],
            include_memory=obs_type in ["multi"]
        )

        # Note: Replace with actual ROM path
        rom_path = "path/to/your/game.gb"

        try:
            env = PyBoyEnv(rom_path, config=config)
            obs = env.reset()

            print(f"  Observation space: {env.observation_space}")
            print(f"  Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")

            env.close()
        except Exception as e:
            print(f"  Error: {e}")


def action_types_example():
    """Example showing different action types."""
    print("\n=== Action Types Example ===")

    action_types = [
        ("discrete", "Single discrete actions"),
        ("multi_discrete", "Multiple button combinations"),
        ("continuous", "Continuous joystick-like input"),
        ("hybrid", "Hybrid discrete + continuous")
    ]

    for action_type, description in action_types:
        print(f"\nTesting {action_type} action space ({description})")

        config = RLEnvironmentConfig(
            headless=True,
            frames_per_action=1
        )

        config.action_config = ActionConfig(
            type=action_type,
            include_noop=True,
            available_buttons=["UP", "DOWN", "LEFT", "RIGHT", "A", "B"]
        )

        # Note: Replace with actual ROM path
        rom_path = "path/to/your/game.gb"

        try:
            env = PyBoyEnv(rom_path, config=config)
            obs = env.reset()

            print(f"  Action space: {env.action_space}")
            print(f"  Sample action: {env.action_space.sample()}")

            env.close()
        except Exception as e:
            print(f"  Error: {e}")


def game_specific_example():
    """Example showing game-specific configurations."""
    print("\n=== Game-Specific Configurations Example ===")

    games = ["pokemon", "mario", "tetris", "zelda"]

    for game in games:
        print(f"\nTesting {game} configuration")

        # Get game-specific configuration
        from src.backend.rl import get_game_config
        config = get_game_config(game)

        print(f"  Observation type: {config.observation_config.type}")
        print(f"  Action type: {config.action_config.type}")
        print(f"  Reward configs: {len(config.reward_configs)}")

        # Note: Replace with actual ROM path
        rom_path = "path/to/your/game.gb"

        try:
            env = PyBoyEnv(rom_path, config=config)
            obs = env.reset()

            print(f"  Environment created successfully")
            print(f"  Observation space: {env.observation_space}")
            print(f"  Action space: {env.action_space}")

            env.close()
        except Exception as e:
            print(f"  Error: {e}")


def environment_features_example():
    """Example showing advanced environment features."""
    print("\n=== Advanced Environment Features Example ===")

    config = RLEnvironmentConfig(
        headless=True,
        frames_per_action=4,
        max_steps=500,
        time_limit=60.0,  # 60 seconds per episode
        save_states=True,
        state_save_interval=100
    )

    # Configure advanced features
    config.observation_config = ObservationConfig(
        type="multi",
        include_screen=True,
        include_game_area=True,
        include_memory=True,
        memory_start=0xC000,
        memory_size=512
    )

    config.action_config = ActionConfig(
        type="discrete",
        include_noop=True,
        allow_button_holding=True,
        button_press_duration=4
    )

    # Configure reward system
    config.reward_configs = [
        {
            "type": "exploration",
            "weight": 0.1,
            "memory_addresses": [
                {"address": 0xC0A0, "name": "position_x"},
                {"address": 0xC0A1, "name": "position_y"}
            ]
        },
        {
            "type": "progress",
            "weight": 0.5,
            "memory_addresses": [{"address": 0xC0B0, "name": "game_progress"}]
        }
    ]

    # Note: Replace with actual ROM path
    rom_path = "path/to/your/game.gb"

    try:
        env = PyBoyEnv(rom_path, config=config)
        obs = env.reset()

        print("Advanced environment features:")
        print(f"  Multi-modal observation: {type(obs)}")
        print(f"  State saving enabled: {config.save_states}")
        print(f"  Time limit: {config.time_limit}s")
        print(f"  Max steps: {config.max_steps}")

        # Test state saving/loading
        state = env.save_state()
        print(f"  State saved: {len(state)} bytes")

        # Run a few steps
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"  Step {step}: Reward={reward:.2f}, Done={done}")

        # Load state
        env.load_state(state)
        print("  State loaded successfully")

        env.close()
        print("\nAdvanced features example completed!")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("PyBoy RL Environment Examples")
    print("=" * 50)

    # Check if PyBoy is available
    try:
        import pyboy
        print("PyBoy is available")
    except ImportError:
        print("PyBoy is not available. Install with 'pip install pyboy'")
        print("Examples will show configuration only, not actual execution")
        print()

    # Run examples
    try:
        basic_environment_example()
        observation_types_example()
        action_types_example()
        game_specific_example()
        environment_features_example()

        print("\n" + "=" * 50)
        print("All examples completed!")
        print("Note: Replace 'path/to/your/game.gb' with actual ROM path")
        print("Available ROMs can be found online or created with development tools")

    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure all dependencies are installed")