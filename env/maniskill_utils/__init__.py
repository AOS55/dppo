"""
ManiSkill3 environment wrapper for DPPO.

Bridges the gap between ManiSkill3's gymnasium-based GPU-parallelised
interface and DPPO's old gym-based AsyncVectorEnv interface.

Key differences handled:
- gymnasium API (reset returns obs, info) -> old gym API (reset returns obs)
- torch tensor outputs -> numpy arrays
- Batched envs natively (ManiSkill handles parallelism internally)
- obs is flat tensor, not dict — wrapped into {"state": obs} for DPPO
"""

import numpy as np
import torch


class ManiSkillVecEnv:
    def __init__(self, env, normalization_path=None):
        self.env = env
        self.n_envs = env.unwrapped.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Load normalization stats if provided
        self.norm = None
        if normalization_path is not None:
            norm = np.load(normalization_path)
            self.states_mean = torch.tensor(norm['states_mean'], dtype=torch.float32)
            self.states_std = torch.tensor(norm['states_std'], dtype=torch.float32).clamp(min=1e-8)

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return np.array(x)

    def _normalize_obs(self, obs):
        if self.norm is not None:
            obs = (obs - self.states_mean.to(obs.device)) / self.states_std.to(obs.device)
        return obs

    def _wrap_obs(self, obs):
        obs = self._normalize_obs(obs)
        obs_np = self._to_numpy(obs)  # [n_envs, obs_dim]
        obs_np = obs_np[:, np.newaxis, :]  # [n_envs, 1, obs_dim]
        return {"state": obs_np}

    def reset(self):
        obs, info = self.env.reset()
        return self._wrap_obs(obs)

    def reset_arg(self, options_list=None):
        obs, info = self.env.reset()
        return self._wrap_obs(obs)

    def reset_one_arg(self, env_ind, options=None):
        obs, info = self.env.reset()
        return self._wrap_obs(obs)

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float().cuda()
        # actions shape: [n_envs, act_steps, action_dim]
        if actions.dim() == 3:
            actions = actions[:, 0, :]
        obs, reward, terminated, truncated, info = self.env.step(actions)
        obs_dict = self._wrap_obs(obs)
        reward_np = self._to_numpy(reward.float())
        terminated_np = self._to_numpy(terminated)
        truncated_np = self._to_numpy(truncated)
        info_list = [{}] * self.n_envs
        return obs_dict, reward_np, terminated_np, truncated_np, info_list

    def seed(self, seeds):
        pass

    def close(self):
        self.env.close()


def make_maniskill(
    id,
    num_envs=1,
    obs_dim=42,
    action_dim=8,
    max_episode_steps=200,
    normalization_path=None,
    **kwargs,
):
    import mani_skill.envs  # noqa: F401
    import gymnasium as gym

    env = gym.make(
        id,
        obs_mode='state',
        num_envs=num_envs,
        max_episode_steps=max_episode_steps,
        reward_mode='sparse',
    )
    return ManiSkillVecEnv(env, normalization_path=normalization_path)
