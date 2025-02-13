import gymnasium as gym
import numpy as np
import torch
import einops
from gymnasium import spaces
from gymnasium.core import ObservationWrapper
from torch import Tensor
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # map to expected inputs for the policy
    return_observations = {}
    if "pixels" in observations:
        if isinstance(observations["pixels"], dict):
            imgs = {f"observation.images.{key}": img for key, img in observations["pixels"].items()}
        else:
            imgs = {"observation.image": observations["pixels"]}

        for imgkey, img in imgs.items():
            img = torch.from_numpy(img)

            # sanity check that images are channel last
            h, w, c = img.shape
            assert c < h and c < w, f"expect channel first images, but instead {img.shape}"

            # sanity check that images are uint8
            assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

            # convert to channel first of type float32 in range [0,1]
            img = einops.rearrange(img, "h w c -> c h w").contiguous()
            img = img.type(torch.float32)
            img /= 255

            return_observations[imgkey] = img

    if "environment_state" in observations:
        return_observations["observation.environment_state"] = torch.from_numpy(
            observations["environment_state"]
        ).float()

    # TODO(rcadene): enable pixels only baseline with `obs_type="pixels"` in environment by removing
    # requirement for "agent_pos"
    return_observations["observation.state"] = torch.from_numpy(observations["agent_pos"]).float()

    if "task_index" in observations:
        return_observations["task_index"] = torch.from_numpy(observations["task_index"])

    return return_observations


class D3ILObservationWrapper(ObservationWrapper):
    """Wrapper that applies LeRobot's observation preprocessing.
    
    This wrapper automatically converts environment observations to the LeRobot 
    format, including proper tensor conversion and image preprocessing.
    """

    def __init__(self, env: gym.Env):
        """Initialize the wrapper.
        
        Args:
            env: The environment to wrap.
        """
        super().__init__(env)
        self.remapped_keys = {
            "agent_pos": "observation.state",
            "environment_state": "observation.environment_state",
            "task_index": "task_index",
        }
        pixels_dict = (
            'pixels' in self.env.observation_space.spaces and 
            isinstance(self.env.observation_space.spaces['pixels'], spaces.Dict)
        )

        if isinstance(self.env.observation_space, spaces.Dict):
            obs_spaces = self.env.observation_space.spaces 
            if pixels_dict:
                # Convert each camera's observation space
                pixel_spaces = {
                    f"observation.images.{camera}": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(obs_spaces["pixels"][camera].shape[-1],  # C
                              obs_spaces["pixels"][camera].shape[0],    # H
                              obs_spaces["pixels"][camera].shape[1]),   # W
                        dtype=np.float32
                    )
                    for camera in obs_spaces["pixels"]
                }
                self.remapped_keys.update({"pixels": pixel_spaces})
            else:
                # Convert single image observation space
                self.remapped_keys.update({
                    "pixels": "observation.image"
                })
                if "pixels" in obs_spaces:
                    new_observation_space["observation.image"] = spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(obs_spaces["pixels"].shape[-1],  # C
                              obs_spaces["pixels"].shape[0],    # H
                              obs_spaces["pixels"].shape[1]),   # W
                        dtype=np.float32
                    )

            new_observation_space = spaces.Dict({
                self.remapped_keys.get(key, f"observation.{key}"): obs_spaces[key]
                for key in self.remapped_keys 
                if (key in obs_spaces and not isinstance(obs_spaces[key], spaces.Dict))
            })
            if pixels_dict:
                new_observation_space.spaces.update(self.remapped_keys["pixels"])
            elif "pixels" in obs_spaces:
                new_observation_space[self.remapped_keys["pixels"]] = obs_spaces["pixels"]
                
        self.observation_space = new_observation_space

    def observation(self, obs):
        """Transform the observation using LeRobot's preprocessing.
        
        Args:
            obs: Raw observation from the environment.
            
        Returns:
            Preprocessed observation in LeRobot format.
        """
        if isinstance(obs.get('pixels', None), dict):
            for key in obs['pixels']:
                obs['pixels'][key] = np.ascontiguousarray(obs['pixels'][key])
        elif isinstance(obs.get('pixels', None), np.ndarray):
            obs['pixels'] = np.ascontiguousarray(obs['pixels'])
        return preprocess_observation(obs)

    def reset(self, **kwargs):
        seed = kwargs.get('seed', None)
        random = kwargs.get('random', True)
        context = kwargs.get('context', None)
        obs, info = self.env.reset(seed=seed, random=random, context=context)
        return self.observation(obs), info


class LerobotManiskillWrapper(ManiSkillVectorEnv):
    def __init__(self, env, num_envs, **kwargs):
        super().__init__(env, num_envs, **kwargs)
        
        # Assuming the original observation space is a Dict
        assert isinstance(self.observation_space, spaces.Dict)
        
        # Define the new observation space
        self.single_observation_space = spaces.Dict({
            "observation.state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
            ),
            "observation.environment_state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
            ),
            "observation.image": spaces.Box(
                low=0, high=255, shape=(128, 128, 3), dtype=np.uint8
            )
        })
        
        # Update the observation space for the vectorized environment
        self.observation_space = spaces.Dict({
            k: spaces.Box(
                low=np.stack([space.low] * self.num_envs),
                high=np.stack([space.high] * self.num_envs),
                shape=(self.num_envs,) + space.shape,
                dtype=space.dtype
            )
            for k, space in self.single_observation_space.spaces.items()
        })


    def observation(self, obs):
        # Extract relevant information from the original observation
        qpos = obs['agent']['qpos']
        qvel = obs['agent']['qvel']
        obs['extra'] = self._get_obs_extra(obs)
        obj_pose = obs['extra']['obj_pose']
        goal_pos = obs['extra']['goal_pos']
        
        # Combine qpos and qvel for observation.state
        agent_pos = torch.cat([qpos, qvel], dim=-1)  # (num_envs, 18)
        
        # Construct environment_state similar to the h5 format
        # Assuming obj_pose contains [pos, quat, vel, ang_vel] for the cube
        # and goal_pos contains [x, y, z] for the goal
        env_state = np.concatenate([
            obj_pose,  # cube state: pos, quat, vel, ang_vel
            goal_pos,  # goal position: x, y, z
        ], axis=-1)
        
        # Get the image from the environment
        image = obs['sensor_data']['base_camera']['rgb']  # (num_envs, 128, 128, 3)
        
        # Create the new observation dictionary
        new_obs = {
            "agent_pos": agent_pos,  # This will be used for "observation.state"
            "environment_state": env_state,
            "pixels": image,  # This should be uint8 and in HWC format
        }
        
        # Add task_index if it's available in the original observation
        if 'task_index' in obs:
            new_obs['task_index'] = obs['task_index']
        
        return new_obs

    def reset(self, **kwargs):
        seed = kwargs.get('seed', None)
        if isinstance(seed, list):
            kwargs['seed'] = seed[0]
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Compute success condition based on the PushCube task definition
        obj_pos = obs['extra']['obj_pose'][:, :2]  # Get XY position of the cube
        goal_pos = obs['extra']['goal_pos'][:, :2]  # Get XY position of the goal
        
        # Calculate distance between cube and goal
        distance = torch.linalg.norm(obj_pos - goal_pos, axis=1)
        
        # Check if the distance is less than the goal radius (0.1 by default)
        is_success = distance < 0.1
        
        # Update info with the computed success condition
        info["is_success"] = is_success
        
        return obs, reward, terminated, truncated, info

    def _get_obs_extra(self, info):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        obs.update(
            goal_pos=self.goal_region.pose.p,
            obj_pose=self.obj.pose.raw_pose,
        )
        return obs

    @property
    def num_envs(self):
        return self.env.num_envs