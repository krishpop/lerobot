import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

class LerobotManiskillWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
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