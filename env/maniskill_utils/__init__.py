"""
ManiSkill3 GPU-native environment wrapper for DPPO.

"""

import numpy as np
import torch


class ManiSkillVecEnv:
    """
    Wraps a ManiSkill3 GPU-parallelised gymnasium env to match DPPO's
    VecEnv interface, but keeps data on GPU as tensors rather than
    converting to numpy.

    Expected usage in train_ppo_diffusion_agent.py:
        obs_dict = env.reset()          # {"state": Tensor[n_envs, 1, obs_dim]}
        obs_dict, rew, term, done, info = env.step(actions)
        # rew, term, done are Tensor[n_envs] on GPU
    """

    def __init__(
        self,
        env,
        device: str = "cuda",
        normalization_path: str = None,
        render_video: bool = False,
    ):
        self.env = env
        self.device = device
        self.n_envs = env.unwrapped.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.render_video = render_video

        # Load normalisation stats onto GPU if provided
        self.states_mean = None
        self.states_std = None
        if normalization_path is not None:
            norm = np.load(normalization_path)
            self.states_mean = torch.tensor(
                norm["states_mean"], dtype=torch.float32, device=device
            )
            self.states_std = torch.tensor(
                norm["states_std"], dtype=torch.float32, device=device
            ).clamp(min=1e-8)

    def _to_device(self, x: torch.Tensor) -> torch.Tensor:
        """Move tensor to env device if not already there."""
        if x.device != torch.device(self.device):
            return x.to(self.device)
        return x

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Z-score normalisation on GPU. No-op if no stats loaded."""
        if self.states_mean is not None:
            obs = (obs - self.states_mean) / self.states_std
        return obs

    def _wrap_obs(self, obs: torch.Tensor) -> dict:
        """
        Normalise and add history dimension.
        Input:  Tensor[n_envs, obs_dim]
        Output: {"state": Tensor[n_envs, 1, obs_dim]}  — all on GPU
        """
        obs = self._normalize_obs(obs.float())
        return {"state": obs.unsqueeze(1)}  # [n_envs, 1, obs_dim]

    def _actions_to_tensor(self, actions) -> torch.Tensor:
        """Accept numpy or tensor actions, return float32 GPU tensor."""
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float().to(self.device)
        elif isinstance(actions, torch.Tensor):
            actions = self._to_device(actions.float())
        else:
            raise TypeError(f"Unsupported action type: {type(actions)}")
        return actions

    def reset(self) -> dict:
        obs, _ = self.env.reset()
        return self._wrap_obs(obs)

    def reset_arg(self, options_list=None) -> dict:
        """DPPO calls this instead of reset(). options_list is ignored for now."""
        obs, _ = self.env.reset()
        return self._wrap_obs(obs)

    def reset_one_arg(self, env_ind: int = None, options=None) -> dict:
        """Single-env reset — resets the whole batch (ManiSkill limitation)."""
        obs, _ = self.env.reset()
        return self._wrap_obs(obs)

    def step(self, actions):
        """
        Step all envs.

        Args:
            actions: Tensor or ndarray of shape [n_envs, act_dim] or
                     [n_envs, act_steps, act_dim] (first step used if 3D).

        Returns:
            obs_dict:   {"state": Tensor[n_envs, 1, obs_dim]}
            reward:     Tensor[n_envs]   — float32, on GPU
            terminated: Tensor[n_envs]   — bool/float32, on GPU
            done:       Tensor[n_envs]   — terminated | truncated, on GPU
            info:       {} (ManiSkill info dicts are complex; pass empty for now)
        """
        actions = self._actions_to_tensor(actions)

        # DPPO passes [n_envs, act_steps, act_dim] — take first step only
        if actions.dim() == 3:
            actions = actions[:, 0, :]

        obs, reward, terminated, truncated, info = self.env.step(actions)

        obs_dict = self._wrap_obs(obs)
        reward = reward.float()          # [n_envs]
        terminated = terminated.float()  # [n_envs]
        done = (terminated.bool() | truncated.bool()).float()  # [n_envs]

        return obs_dict, reward, terminated, done, info

    def seed(self, seeds):
        """ManiSkill handles seeding internally"""
        pass

    def close(self):
        if self.render_video:
            try:
                self.env.flush_video()
            except AttributeError:
                pass
        self.env.close()


def make_maniskill(
    id: str,
    num_envs: int = 1,
    obs_dim: int = 42,
    action_dim: int = 8,
    max_episode_steps: int = 200,
    normalization_path: str = None,
    render_video: bool = False,
    video_dir: str = None,
    device: str = "cuda",
    sim_backend: str = "gpu",
    **kwargs,
) -> ManiSkillVecEnv:
    """
    Create a GPU-parallelised ManiSkill3 env wrapped for DPPO.

    Args:
        id:                 ManiSkill env id, e.g. "PickCube-v1"
        num_envs:           number of parallel envs
        obs_dim:            expected obs dimension (for validation only)
        action_dim:         expected action dimension (for validation only)
        max_episode_steps:  episode length
        normalization_path: path to .npz with states_mean / states_std
        render_video:       whether to record video
        video_dir:          where to save videos
        device:             torch device string
        sim_backend:        "gpu" (default) or "cpu"
    """
    import mani_skill.envs  # registers envs
    import gymnasium as gym

    render_mode = "rgb_array" if render_video else None

    env = gym.make(
        id,
        obs_mode="state",
        num_envs=num_envs,
        sim_backend=sim_backend,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
    )

    if render_video and video_dir is not None:
        from mani_skill.utils.wrappers.record import RecordEpisode
        env = RecordEpisode(
            env,
            output_dir=video_dir,
            save_video=True,
            save_trajectory=False,
            video_fps=20,
            max_steps_per_video=max_episode_steps,
        )

    return ManiSkillVecEnv(
        env,
        device=device,
        normalization_path=normalization_path,
        render_video=render_video,
    )