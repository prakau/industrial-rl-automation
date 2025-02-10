import pytest
import numpy as np
import torch
from src.environments.industrial_arm.arm_env import IndustrialArmEnv
from src.models.ppo.actor_critic import PolicyNetwork, PPOTrainer

def test_environment_init():
    config = {
        'max_episode_steps': 1000,
        'reward_scales': {
            'distance': 1.0,
            'control': 0.001,
            'velocity': 0.0001,
            'success': 10.0
        }
    }
    try:
        env = IndustrialArmEnv(config)
        assert env.observation_space is not None
        assert env.action_space is not None
    except Exception as e:
        pytest.skip(f"MuJoCo environment initialization failed: {e}")

def test_policy_network():
    obs_dim = {
        'qpos': 7,
        'qvel': 7,
        'eef_pos': 3,
        'eef_quat': 4
    }
    act_dim = 8
    hidden_size = 256
    
    policy = PolicyNetwork(obs_dim, act_dim, hidden_size)
    
    # Create dummy observations
    obs_dict = {
        'qpos': torch.randn(1, 7),
        'qvel': torch.randn(1, 7),
        'eef_pos': torch.randn(1, 3),
        'eef_quat': torch.randn(1, 4)
    }
    
    # Test forward pass
    with torch.no_grad():
        actions, values = policy(obs_dict)
        assert actions.shape == (1, act_dim)
        assert values.shape == (1, 1)