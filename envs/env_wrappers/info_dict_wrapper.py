# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An env wrapper that flattens the observation dictionary to an array."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym

from envs.utilities import env_utils


class InfoDictStateWrapper(gym.Env):
  """An env wrapper that flattens the observation dictionary to an array."""

  def __init__(self, gym_env):
    """Initializes the wrapper."""
    self._gym_env = gym_env
    self.observation_space = self._gym_env.observation_space
    self.action_space = self._gym_env.action_space

  def __getattr__(self, attr):
    return getattr(self._gym_env, attr)


  def _get_obs(self, obs):
      import numpy as np
#       print ("sim obs: ", len(obs))
#       print ("sim self.getVisualState()[0]: ", self.getVisualState()[0].shape)
      obs = [obs, self.getVisualState()[0]]
#       print ("sim obs: ", np.array(obs).shape)
      return obs
      
      
  def reset(self, initial_motor_angles=None, reset_duration=0.0):
    observation = self._gym_env.reset(
        initial_motor_angles=initial_motor_angles,
        reset_duration=reset_duration)
    return self._get_obs(observation)

  def step(self, action):
    """Steps the wrapped environment.

    Args:
      action: Numpy array. The input action from an NN agent.

    Returns:
      The tuple containing the flattened observation, the reward, the epsiode
        end indicator.
        
    """
    info_ = {}
    info_["agent_obs_before"] = self.getVisualState()[0]
    info_["expert_obs_before"] = self.getImitationVisualState()[0]
    observation, reward, done, info = self._gym_env.step(action)
    ### This is really the next state observation which is exactly what we want to computing rewards
    info["agent_pos"] = observation
    info["agent_obs"] = self.getVisualState()[0]
    info["expert_pos"] = self._task.build_target_obs()
    info["expert_obs"] = self.getImitationVisualState()[0]
    info.update(info_)
    return self._get_obs(observation), reward, done, info

  def render(self, mode='human'):
    return self._gym_env.render(mode)
