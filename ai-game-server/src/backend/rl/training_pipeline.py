"""
Training pipeline integration for PyBoy RL environment.
"""

import numpy as np
import time
import json
import os
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import pickle
import logging
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with 'pip install torch'")

try:
    import stable_baselines3 as sb3
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Stable Baselines3 not available. Install with 'pip install stable-baselines3'")

from .pyboy_env import PyBoyEnv
from .rl_environment_config import RLEnvironmentConfig


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    algorithm: str = "PPO"  # PPO, A2C, DQN, SAC, TD3
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99
    n_steps: int = 2048
    n_epochs: int = 10
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95
    use_sde: bool = False
    sde_sample_freq: int = -1
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)
    tensorboard_log: Optional[str] = None
    create_new_env: bool = True
    verbose: int = 1
    seed: Optional[int] = None
    device: str = "auto"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    n_eval_episodes: int = 10
    deterministic: bool = True
    render: bool = False
    callback: Optional[Callable] = None
    eval_freq: int = 10000
    n_eval_envs: int = 1
    eval_log_path: Optional[str] = None


@dataclass
class TrainingMetrics:
    """Training metrics."""
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    episode_times: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    exploration_rates: List[float] = field(default_factory=list)
    evaluations: List[Dict[str, Any]] = field(default_factory=list)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)


class CustomCallback(BaseCallback):
    """Custom callback for training monitoring."""

    def __init__(self, training_pipeline, verbose=0):
        super().__init__(verbose)
        self.training_pipeline = training_pipeline

    def _on_step(self) -> bool:
        # Collect metrics during training
        if hasattr(self.training_env, 'envs'):
            for env in self.training_env.envs:
                if hasattr(env, 'get_metrics'):
                    metrics = env.get_metrics()
                    self.training_pipeline.update_metrics(metrics)

        return True

    def _on_rollout_end(self) -> None:
        # Log rollout end metrics
        if hasattr(self.training_pipeline, 'log_rollout_metrics'):
            self.training_pipeline.log_rollout_metrics()

    def _on_training_end(self) -> None:
        # Log training end
        if hasattr(self.training_pipeline, 'log_training_end'):
            self.training_pipeline.log_training_end()


class TrainingPipeline:
    """
    Training pipeline for PyBoy RL environments.

    This pipeline provides:
    - Integration with Stable Baselines3
    - Custom training loop support
    - Metrics collection and logging
    - Model checkpointing
    - Evaluation and monitoring
    """

    def __init__(
        self,
        env: PyBoyEnv,
        config: TrainingConfig,
        evaluation_config: Optional[EvaluationConfig] = None
    ):
        """
        Initialize the training pipeline.

        Args:
            env: PyBoy RL environment
            config: Training configuration
            evaluation_config: Evaluation configuration
        """
        if not SB3_AVAILABLE:
            raise ImportError("Stable Baselines3 is required for training pipeline")

        self.env = env
        self.config = config
        self.evaluation_config = evaluation_config or EvaluationConfig()

        # Initialize metrics
        self.metrics = TrainingMetrics()

        # Initialize model
        self.model = None
        self.training_start_time = None
        self.current_timestep = 0

        # Setup logging
        self.setup_logging()

        # Check environment
        self.check_environment()

    def setup_logging(self):
        """Setup logging for training."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Create file handler if tensorboard_log is specified
        if self.config.tensorboard_log:
            log_dir = Path(self.config.tensorboard_log)
            log_dir.mkdir(parents=True, exist_ok=True)

            fh = logging.FileHandler(log_dir / 'training.log')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def check_environment(self):
        """Check if environment is compatible with SB3."""
        try:
            check_env(self.env)
            self.logger.info("Environment check passed")
        except Exception as e:
            self.logger.error(f"Environment check failed: {e}")
            raise

    def create_model(self):
        """Create RL model based on configuration."""
        self.logger.info(f"Creating {self.config.algorithm} model")

        # Determine policy architecture
        if self.env.observation_space.shape is not None:
            # Image observation
            if len(self.env.observation_space.shape) == 3:
                # CNN policy for image observations
                policy = "CnnPolicy"
            else:
                # MLP policy for vector observations
                policy = "MlpPolicy"
        else:
            # Multi-discrete or dict observations
            policy = "MultiInputPolicy"

        # Create model based on algorithm
        if self.config.algorithm == "PPO":
            self.model = sb3.PPO(
                policy,
                self.env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                n_steps=self.config.n_steps,
                n_epochs=self.config.n_epochs,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                gae_lambda=self.config.gae_lambda,
                use_sde=self.config.use_sde,
                sde_sample_freq=self.config.sde_sample_freq,
                policy_kwargs=self.config.policy_kwargs,
                tensorboard_log=self.config.tensorboard_log,
                verbose=self.config.verbose,
                seed=self.config.seed,
                device=self.config.device
            )
        elif self.config.algorithm == "A2C":
            self.model = sb3.A2C(
                policy,
                self.env,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                n_steps=self.config.n_steps,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                use_sde=self.config.use_sde,
                sde_sample_freq=self.config.sde_sample_freq,
                policy_kwargs=self.config.policy_kwargs,
                tensorboard_log=self.config.tensorboard_log,
                verbose=self.config.verbose,
                seed=self.config.seed,
                device=self.config.device
            )
        elif self.config.algorithm == "DQN":
            self.model = sb3.DQN(
                policy,
                self.env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                learning_starts=1000,
                train_freq=4,
                target_update_interval=1000,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                max_grad_norm=self.config.max_grad_norm,
                policy_kwargs=self.config.policy_kwargs,
                tensorboard_log=self.config.tensorboard_log,
                verbose=self.config.verbose,
                seed=self.config.seed,
                device=self.config.device
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")

        self.logger.info(f"Model created successfully")

    def train(self):
        """Start training."""
        self.logger.info("Starting training...")
        self.training_start_time = time.time()

        # Create callback
        callback = CustomCallback(self, verbose=self.config.verbose)

        # Add evaluation callback if configured
        eval_callback = None
        if self.evaluation_config.eval_freq > 0:
            eval_callback = EvalCallback(
                self.env,
                callback_on_new_best=None,
                callback_after_eval=None,
                n_eval_episodes=self.evaluation_config.n_eval_episodes,
                eval_freq=self.evaluation_config.eval_freq,
                log_path=self.evaluation_config.eval_log_path,
                best_model_save_path=self.evaluation_config.eval_log_path,
                deterministic=self.evaluation_config.deterministic,
                render=self.evaluation_config.render,
                verbose=self.config.verbose
            )

        # Combine callbacks
        callbacks = [callback]
        if eval_callback is not None:
            callbacks.append(eval_callback)

        # Start training
        try:
            self.model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
                log_interval=10
            )

            self.logger.info("Training completed successfully")
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def evaluate(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate the trained model."""
        self.logger.info("Starting evaluation...")

        if model_path:
            # Load model from file
            self.model = self.model.load(model_path)

        # Run evaluation
        episode_rewards = []
        episode_lengths = []

        for i in range(self.evaluation_config.n_eval_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action, _ = self.model.predict(
                    obs,
                    deterministic=self.evaluation_config.deterministic
                )
                obs, reward, done, info = self.env.step(action)

                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        # Calculate statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)

        results = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_length': mean_length,
            'std_length': std_length,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'n_episodes': len(episode_rewards)
        }

        self.logger.info(f"Evaluation results: {results}")
        return results

    def save_model(self, path: str):
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save")

        self.model.save(path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model."""
        if self.config.algorithm == "PPO":
            self.model = sb3.PPO.load(path)
        elif self.config.algorithm == "A2C":
            self.model = sb3.A2C.load(path)
        elif self.config.algorithm == "DQN":
            self.model = sb3.DQN.load(path)
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")

        self.logger.info(f"Model loaded from {path}")

    def save_training_state(self, path: str):
        """Save training state including model and metrics."""
        state = {
            'config': self.config.__dict__,
            'metrics': self.metrics.__dict__,
            'current_timestep': self.current_timestep,
            'training_time': time.time() - self.training_start_time if self.training_start_time else 0
        }

        if self.model is not None:
            model_path = f"{path}_model"
            self.save_model(model_path)
            state['model_path'] = model_path

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        self.logger.info(f"Training state saved to {path}")

    def load_training_state(self, path: str):
        """Load training state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        # Load metrics
        for key, value in state['metrics'].items():
            setattr(self.metrics, key, value)

        self.current_timestep = state['current_timestep']

        # Load model if available
        if 'model_path' in state:
            self.load_model(state['model_path'])

        self.logger.info(f"Training state loaded from {path}")

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update training metrics."""
        for key, value in metrics.items():
            if key == 'episode_rewards':
                self.metrics.episode_rewards.extend(value)
            elif key == 'episode_lengths':
                self.metrics.episode_lengths.extend(value)
            elif key == 'losses':
                self.metrics.losses.extend(value)
            elif key == 'learning_rates':
                self.metrics.learning_rates.extend(value)
            elif key == 'exploration_rates':
                self.metrics.exploration_rates.extend(value)

    def log_rollout_metrics(self):
        """Log rollout metrics."""
        if self.model and hasattr(self.model, 'logger'):
            if self.metrics.episode_rewards:
                mean_reward = np.mean(self.metrics.episode_rewards[-100:])
                self.model.logger.record('rollout/ep_rew_mean', mean_reward)

            if self.metrics.episode_lengths:
                mean_length = np.mean(self.metrics.episode_lengths[-100:])
                self.model.logger.record('rollout/ep_len_mean', mean_length)

    def log_training_end(self):
        """Log training end metrics."""
        if self.model and hasattr(self.model, 'logger'):
            total_time = time.time() - self.training_start_time
            self.model.logger.record('time/total_time', total_time)
            self.model.logger.record('time/total_timesteps', self.current_timestep)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics."""
        summary = {}

        if self.metrics.episode_rewards:
            summary['episode_rewards'] = {
                'mean': np.mean(self.metrics.episode_rewards),
                'std': np.std(self.metrics.episode_rewards),
                'min': np.min(self.metrics.episode_rewards),
                'max': np.max(self.metrics.episode_rewards),
                'count': len(self.metrics.episode_rewards)
            }

        if self.metrics.episode_lengths:
            summary['episode_lengths'] = {
                'mean': np.mean(self.metrics.episode_lengths),
                'std': np.std(self.metrics.episode_lengths),
                'min': np.min(self.metrics.episode_lengths),
                'max': np.max(self.metrics.episode_lengths),
                'count': len(self.metrics.episode_lengths)
            }

        if self.metrics.losses:
            summary['losses'] = {
                'mean': np.mean(self.metrics.losses),
                'std': np.std(self.metrics.losses),
                'min': np.min(self.metrics.losses),
                'max': np.max(self.metrics.losses),
                'count': len(self.metrics.losses)
            }

        if self.training_start_time:
            summary['training_time'] = time.time() - self.training_start_time

        summary['current_timestep'] = self.current_timestep

        return summary

    def export_metrics(self, filepath: str):
        """Export metrics to file."""
        summary = self.get_metrics_summary()

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Metrics exported to {filepath}")

    def create_video(self, model_path: str, video_path: str, n_episodes: int = 1):
        """Create video of model performance."""
        self.logger.info(f"Creating video: {video_path}")

        # Load model
        self.load_model(model_path)

        # Create video recording environment
        eval_env = Monitor(self.env, filename=video_path)

        # Run episodes
        for i in range(n_episodes):
            obs = eval_env.reset()
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)

        eval_env.close()
        self.logger.info("Video creation completed")


def create_training_pipeline(
    rom_path: str,
    config: Optional[TrainingConfig] = None,
    env_config: Optional[RLEnvironmentConfig] = None
) -> TrainingPipeline:
    """
    Create a complete training pipeline.

    Args:
        rom_path: Path to Game Boy ROM
        config: Training configuration
        env_config: Environment configuration

    Returns:
        TrainingPipeline instance
    """
    if env_config is None:
        env_config = RLEnvironmentConfig()

    # Create environment
    env = PyBoyEnv(rom_path, config=env_config)

    # Create training pipeline
    if config is None:
        config = TrainingConfig()

    pipeline = TrainingPipeline(env, config)

    return pipeline


def quick_train(
    rom_path: str,
    algorithm: str = "PPO",
    total_timesteps: int = 100000,
    output_dir: str = "./training_output"
) -> Dict[str, Any]:
    """
    Quick training setup with default parameters.

    Args:
        rom_path: Path to Game Boy ROM
        algorithm: RL algorithm to use
        total_timesteps: Total training timesteps
        output_dir: Output directory for results

    Returns:
        Training results
    """
    # Create configurations
    env_config = RLEnvironmentConfig(
        headless=True,
        frames_per_action=4,
        max_steps=1000
    )

    training_config = TrainingConfig(
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        tensorboard_log=output_dir
    )

    # Create and run training
    pipeline = create_training_pipeline(rom_path, training_config, env_config)
    pipeline.create_model()
    pipeline.train()

    # Evaluate model
    results = pipeline.evaluate()

    # Save results
    pipeline.save_model(f"{output_dir}/final_model")
    pipeline.export_metrics(f"{output_dir}/metrics.json")

    return results