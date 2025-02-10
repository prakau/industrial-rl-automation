import gymnasium as gym
import mujoco
import numpy as np
from typing import Dict, Tuple, Any

class IndustrialArmEnv(gym.Env):
    def __init__(self, config_path: str):
        super().__init__()
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(config_path)
        self.data = mujoco.MjData(self.model)
        
        # Define spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Dict({
            "joint_pos": gym.spaces.Box(-np.pi, np.pi, (7,)),
            "joint_vel": gym.spaces.Box(-10.0, 10.0, (7,)),
            "ee_pos": gym.spaces.Box(-2.0, 2.0, (3,)),
            "target_pos": gym.spaces.Box(-2.0, 2.0, (3,))
        })
        
        self._setup_randomization()

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = False
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._randomize_target()
        return self._get_obs(), self._get_info()

    def _setup_randomization(self):
        self.rand_params = {
            'damping': (0.8, 1.2),
            'friction': (0.9, 1.1),
            'mass': (0.8, 1.2)
        }
        self.target_range = (-1.5, 1.5)

    def _apply_action(self, action: np.ndarray) -> None:
        # Scale actions from [-1, 1] to joint limits
        scaled_action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = scaled_action * self.model.actuator_ctrlrange[:,1]

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "joint_pos": self.data.qpos[:7].copy(),
            "joint_vel": self.data.qvel[:7].copy(),
            "ee_pos": self.data.site_xpos[0].copy(),
            "target_pos": self.target_pos.copy()
        }

    def _compute_reward(self) -> float:
        ee_pos = self.data.site_xpos[0]
        dist = np.linalg.norm(ee_pos - self.target_pos)
        return -dist - 0.1 * np.sum(np.square(self.data.ctrl))

    def _check_termination(self) -> bool:
        ee_pos = self.data.site_xpos[0]
        dist = np.linalg.norm(ee_pos - self.target_pos)
        return dist < 0.05 or self.steps >= self.max_steps

    def _get_info(self) -> Dict[str, Any]:
        ee_pos = self.data.site_xpos[0]
        return {
            "distance": np.linalg.norm(ee_pos - self.target_pos),
            "success": self._check_termination()
        }

    def _randomize_target(self) -> None:
        self.target_pos = np.random.uniform(
            low=self.target_range[0],
            high=self.target_range[1],
            size=3
        )