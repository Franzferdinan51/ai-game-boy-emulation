"""
Training example using PyBoy RL environment with Stable Baselines3.
"""

import sys
import os
import time

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.backend.rl import (
    PyBoyEnv,
    RLEnvironmentConfig,
    TrainingPipeline,
    TrainingConfig,
    EvaluationConfig,
    create_training_pipeline,
    quick_train
)


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []

    try:
        import pyboy
    except ImportError:
        missing.append("pyboy")

    try:
        import gym
    except ImportError:
        missing.append("gym")

    try:
        import stable_baselines3
    except ImportError:
        missing.append("stable-baselines3")

    try:
        import torch
    except ImportError:
        missing.append("torch")

    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep} (install with: pip install {dep})")
        return False

    print("All dependencies are available!")
    return True


def basic_training_example():
    """Basic training example with PPO."""
    print("=== Basic Training Example ===")

    if not check_dependencies():
        return

    # Create environment configuration
    env_config = RLEnvironmentConfig(
        headless=True,
        frames_per_action=4,
        max_steps=1000,
        log_level="WARNING"
    )

    # Configure observation space (simplified for faster training)
    env_config.observation_config.type = "game_area"  # Simplified observation
    env_config.observation_config.tiles_width = 10
    env_config.observation_config.tiles_height = 9

    # Configure action space
    env_config.action_config.type = "discrete"
    env_config.action_config.available_buttons = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B"]

    # Create training configuration
    training_config = TrainingConfig(
        algorithm="PPO",
        total_timesteps=50000,  # Reduced for example
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        tensorboard_log="./training_logs"
    )

    # Create evaluation configuration
    eval_config = EvaluationConfig(
        n_eval_episodes=5,
        eval_freq=5000,
        deterministic=True,
        eval_log_path="./eval_logs"
    )

    # Note: Replace with actual ROM path
    rom_path = "path/to/your/game.gb"

    try:
        print(f"Creating training pipeline for: {rom_path}")
        print(f"Environment config: {env_config}")
        print(f"Training config: {training_config}")

        # Create environment
        env = PyBoyEnv(rom_path, config=env_config)

        # Create training pipeline
        pipeline = TrainingPipeline(env, training_config, eval_config)

        # Create and train model
        print("Creating model...")
        pipeline.create_model()

        print("Starting training...")
        pipeline.train()

        # Evaluate model
        print("Evaluating model...")
        results = pipeline.evaluate()

        print("Training results:")
        print(f"  Mean reward: {results['mean_reward']:.2f}")
        print(f"  Std reward: {results['std_reward']:.2f}")
        print(f"  Mean length: {results['mean_length']:.2f}")

        # Save model and metrics
        pipeline.save_model("./models/ppo_model")
        pipeline.export_metrics("./metrics/training_metrics.json")

        print("Training completed successfully!")

    except Exception as e:
        print(f"Error during training: {e}")
        print("Make sure you have a valid ROM and all dependencies installed")


def quick_training_example():
    """Quick training example using helper function."""
    print("\n=== Quick Training Example ===")

    if not check_dependencies():
        return

    # Note: Replace with actual ROM path
    rom_path = "path/to/your/game.gb"

    try:
        print(f"Starting quick training for: {rom_path}")

        # Quick training with default parameters
        results = quick_train(
            rom_path=rom_path,
            algorithm="PPO",
            total_timesteps=25000,  # Reduced for example
            output_dir="./quick_training_output"
        )

        print("Quick training results:")
        print(f"  Mean reward: {results['mean_reward']:.2f}")
        print(f"  Std reward: {results['std_reward']:.2f}")

        print("Quick training completed!")

    except Exception as e:
        print(f"Error during quick training: {e}")


def multi_algorithm_training():
    """Example comparing different algorithms."""
    print("\n=== Multi-Algorithm Training Example ===")

    if not check_dependencies():
        return

    algorithms = ["PPO", "A2C", "DQN"]
    results = {}

    # Note: Replace with actual ROM path
    rom_path = "path/to/your/game.gb"

    # Create environment configuration
    env_config = RLEnvironmentConfig(
        headless=True,
        frames_per_action=4,
        max_steps=500
    )

    env_config.observation_config.type = "game_area"
    env_config.observation_config.tiles_width = 10
    env_config.observation_config.tiles_height = 9

    env_config.action_config.type = "discrete"
    env_config.action_config.available_buttons = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B"]

    for algorithm in algorithms:
        print(f"\nTraining with {algorithm}...")

        try:
            # Create training configuration
            training_config = TrainingConfig(
                algorithm=algorithm,
                total_timesteps=20000,  # Reduced for example
                learning_rate=3e-4,
                verbose=0  # Reduce output
            )

            # Create environment
            env = PyBoyEnv(rom_path, config=env_config)

            # Create training pipeline
            pipeline = TrainingPipeline(env, training_config)

            # Train model
            pipeline.create_model()
            pipeline.train()

            # Evaluate model
            eval_results = pipeline.evaluate()

            results[algorithm] = eval_results

            print(f"  {algorithm} - Mean reward: {eval_results['mean_reward']:.2f}")

            # Clean up
            env.close()

        except Exception as e:
            print(f"  Error with {algorithm}: {e}")

    # Compare results
    print("\nAlgorithm Comparison:")
    print("-" * 40)
    for algorithm, result in results.items():
        print(f"{algorithm:10} | {result['mean_reward']:8.2f} ± {result['std_reward']:6.2f}")

    best_algorithm = max(results.keys(), key=lambda k: results[k]['mean_reward'])
    print(f"\nBest algorithm: {best_algorithm}")


def custom_reward_training():
    """Example with custom reward configuration."""
    print("\n=== Custom Reward Training Example ===")

    if not check_dependencies():
        return

    # Create environment with custom rewards
    env_config = RLEnvironmentConfig(
        headless=True,
        frames_per_action=4,
        max_steps=1000
    )

    # Configure custom reward system
    env_config.reward_configs = [
        {
            "type": "exploration",
            "weight": 0.2,
            "memory_addresses": [
                {"address": 0xC0A0, "name": "position_x"},
                {"address": 0xC0A1, "name": "position_y"}
            ],
            "reward_on_increase": True
        },
        {
            "type": "progress",
            "weight": 0.5,
            "memory_addresses": [
                {"address": 0xC0B0, "name": "game_progress"}
            ],
            "reward_on_increase": True
        },
        {
            "type": "time_efficiency",
            "weight": -0.1,  # Penalty for taking too long
            "custom_function": "time_penalty"
        }
    ]

    env_config.base_reward = 0.01
    env_config.step_penalty = -0.001
    env_config.normalize_rewards = True

    # Note: Replace with actual ROM path
    rom_path = "path/to/your/game.gb"

    try:
        print("Training with custom reward system...")

        # Create environment
        env = PyBoyEnv(rom_path, config=env_config)

        # Create training configuration
        training_config = TrainingConfig(
            algorithm="PPO",
            total_timesteps=30000,
            learning_rate=3e-4,
            verbose=1
        )

        # Create training pipeline
        pipeline = TrainingPipeline(env, training_config)

        # Train model
        pipeline.create_model()
        pipeline.train()

        # Evaluate model
        results = pipeline.evaluate()

        print("Custom reward training results:")
        print(f"  Mean reward: {results['mean_reward']:.2f}")
        print(f"  Std reward: {results['std_reward']:.2f}")

        # Get reward breakdown
        if hasattr(env, 'reward_system'):
            reward_breakdown = env.reward_system.get_reward_breakdown()
            print("  Reward breakdown:")
            for reward_type, value in reward_breakdown.items():
                print(f"    {reward_type}: {value:.3f}")

        print("Custom reward training completed!")

    except Exception as e:
        print(f"Error during custom reward training: {e}")


def continue_training_example():
    """Example of continuing training from saved checkpoint."""
    print("\n=== Continue Training Example ===")

    if not check_dependencies():
        return

    # Note: Replace with actual ROM path
    rom_path = "path/to/your/game.gb"

    try:
        print("Starting initial training...")

        # Initial training
        results1 = quick_train(
            rom_path=rom_path,
            algorithm="PPO",
            total_timesteps=20000,
            output_dir="./continue_training_1"
        )

        print(f"Initial training - Mean reward: {results1['mean_reward']:.2f}")

        print("Continuing training...")

        # Create environment
        env_config = RLEnvironmentConfig(
            headless=True,
            frames_per_action=4,
            max_steps=1000
        )

        env = PyBoyEnv(rom_path, config=env_config)

        # Create training pipeline
        training_config = TrainingConfig(
            algorithm="PPO",
            total_timesteps=20000,
            verbose=1
        )

        pipeline = TrainingPipeline(env, training_config)

        # Load previous model
        pipeline.load_model("./continue_training_1/final_model")

        # Continue training
        pipeline.train()

        # Evaluate
        results2 = pipeline.evaluate()

        print(f"Continued training - Mean reward: {results2['mean_reward']:.2f}")
        print(f"Improvement: {results2['mean_reward'] - results1['mean_reward']:.2f}")

        print("Continue training example completed!")

    except Exception as e:
        print(f"Error during continue training: {e}")


def hyperparameter_search():
    """Simple hyperparameter search example."""
    print("\n=== Hyperparameter Search Example ===")

    if not check_dependencies():
        return

    # Hyperparameter combinations
    learning_rates = [1e-4, 3e-4, 1e-3]
    batch_sizes = [32, 64, 128]

    best_score = -float('inf')
    best_params = None

    # Note: Replace with actual ROM path
    rom_path = "path/to/your/game.gb"

    # Create environment configuration
    env_config = RLEnvironmentConfig(
        headless=True,
        frames_per_action=4,
        max_steps=500
    )

    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\nTesting LR={lr}, Batch={bs}")

            try:
                # Create training configuration
                training_config = TrainingConfig(
                    algorithm="PPO",
                    total_timesteps=10000,  # Short for search
                    learning_rate=lr,
                    batch_size=bs,
                    verbose=0
                )

                # Create environment
                env = PyBoyEnv(rom_path, config=env_config)

                # Create training pipeline
                pipeline = TrainingPipeline(env, training_config)

                # Train model
                pipeline.create_model()
                pipeline.train()

                # Evaluate model
                results = pipeline.evaluate()
                mean_reward = results['mean_reward']

                print(f"  Mean reward: {mean_reward:.2f}")

                # Update best parameters
                if mean_reward > best_score:
                    best_score = mean_reward
                    best_params = {'learning_rate': lr, 'batch_size': bs}

                # Clean up
                env.close()

            except Exception as e:
                print(f"  Error: {e}")

    print(f"\nBest hyperparameters:")
    print(f"  Learning rate: {best_params['learning_rate']}")
    print(f"  Batch size: {best_params['batch_size']}")
    print(f"  Best score: {best_score:.2f}")


if __name__ == "__main__":
    print("PyBoy RL Training Examples")
    print("=" * 50)

    # Run examples
    try:
        basic_training_example()
        quick_training_example()
        multi_algorithm_training()
        custom_reward_training()
        continue_training_example()
        hyperparameter_search()

        print("\n" + "=" * 50)
        print("All training examples completed!")
        print("Check the ./training_logs and ./models directories for outputs")

    except KeyboardInterrupt:
        print("\nTraining examples interrupted by user")
    except Exception as e:
        print(f"\nError running training examples: {e}")
        print("Make sure all dependencies are installed and you have a valid ROM")