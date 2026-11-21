"""PPO training settings for the Trotter velocity task.

`register_mjlab_task` (see __init__.py) hands this config to mjlab's runner.
The values follow mjlab's Unitree Go1 velocity config, a well-tested starting
point for a small legged robot. Tune from here once training runs.
"""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def trotter_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO runner configuration for the Trotter velocity task."""
    return RslRlOnPolicyRunnerCfg(
        # Actor: observations -> action mean. Std is a single learned scalar per
        # action dimension, starting at 1.0 for broad early exploration.
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        # Critic: observations -> value estimate.
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            # "adaptive" adjusts the learning rate to keep KL near desired_kl.
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="trotter_velocity",
        wandb_project="mjlab_trotter",
        # Don't upload checkpoints (.pt, ~4.4 MB each, one every save_interval)
        # or exported .onnx files to wandb; metrics are still logged. The files
        # stay on the training machine in logs/rsl_rl/trotter_velocity/.
        upload_model=False,
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=10_000,
    )
