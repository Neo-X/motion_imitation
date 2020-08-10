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

import argparse
from mpi4py import MPI
import numpy as np
import os
import random
import tensorflow as tf
import time

import envs.env_builder as env_builder
import learning.imitation_policies as imitation_policies
import learning.ppo_imitation as ppo_imitation

from stable_baselines.common.callbacks import CheckpointCallback

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256

ENABLE_ENV_RANDOMIZER = True

def set_rand_seed(seed=None):
  if seed is None:
    seed = int(time.time())

  seed += 97 * MPI.COMM_WORLD.Get_rank()

  tf.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  return

def build_model(env, num_procs, timesteps_per_actorbatch, optim_batchsize, output_dir):
  policy_kwargs = {
      "net_arch": [{"pi": [512, 256],
                    "vf": [512, 256]}],
      "act_fun": tf.nn.relu
  }

  timesteps_per_actorbatch = int(np.ceil(float(timesteps_per_actorbatch) / num_procs))
  optim_batchsize = int(np.ceil(float(optim_batchsize) / num_procs))

  model = ppo_imitation.PPOImitation(
               policy=imitation_policies.ImitationPolicy,
               env=env,
               gamma=0.95,
               timesteps_per_actorbatch=timesteps_per_actorbatch,
               clip_param=0.2,
               optim_epochs=1,
               optim_stepsize=1e-5,
               optim_batchsize=optim_batchsize,
               lam=0.95,
               adam_epsilon=1e-5,
               schedule='constant',
               policy_kwargs=policy_kwargs,
               tensorboard_log=output_dir,
               verbose=1)
  return model


def train(model, env, total_timesteps, output_dir="", int_save_freq=0):
  if (output_dir == ""):
    save_path = None
  else:
    save_path = os.path.join(output_dir, "model.zip")
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  

  callbacks = []
  # Save a checkpoint every n steps
  if (output_dir != ""):
    if (int_save_freq > 0):
      int_dir = os.path.join(output_dir, "intermedate")
      callbacks.append(CheckpointCallback(save_freq=int_save_freq, save_path=int_dir,
                                          name_prefix='model'))

  model.learn(total_timesteps=total_timesteps, save_path=save_path, callback=callbacks)

  return

def test(model, env, num_procs, num_episodes=None):
  curr_return = 0
  sum_return = 0
  episode_count = 0

  if num_episodes is not None:
    num_local_episodes = int(np.ceil(float(num_episodes) / num_procs))
  else:
    num_local_episodes = np.inf

  o = env.reset()
  while episode_count < num_local_episodes:
    a, _ = model.predict(o, deterministic=True)
    o, r, done, info = env.step(a)
    curr_return += r

    if done:
        o = env.reset()
        sum_return += curr_return
        episode_count += 1

  sum_return = MPI.COMM_WORLD.allreduce(sum_return, MPI.SUM)
  episode_count = MPI.COMM_WORLD.allreduce(episode_count, MPI.SUM)

  mean_return = sum_return / episode_count

  if MPI.COMM_WORLD.Get_rank() == 0:
      print("Mean Return: " + str(mean_return))
      print("Episode Count: " + str(episode_count))

  return

def main():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
  arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
  arg_parser.add_argument("--motion_file", dest="motion_file", type=str, default="motion_imitation/data/motions/laikago_dog_pace.txt")
  arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
  arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output")
  arg_parser.add_argument("--num_test_episodes", dest="num_test_episodes", type=int, default=None)
  arg_parser.add_argument("--model_file", dest="model_file", type=str, default="")
  arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=2e8)
  arg_parser.add_argument("--int_save_freq", dest="int_save_freq", type=int, default=0) # save intermediate model every n policy steps

  args = arg_parser.parse_args()
  
  num_procs = MPI.COMM_WORLD.Get_size()
  os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
  
  enable_env_rand = ENABLE_ENV_RANDOMIZER and (args.mode != "test")
  env = env_builder.build_imitation_env(motion_files=[args.motion_file],
                                        num_parallel_envs=num_procs,
                                        mode=args.mode,
                                        enable_randomizer=enable_env_rand,
                                        enable_rendering=args.visualize)
  
  
  rewards = []
  states = []
  time_limit=2
  print ("env.action_space.high: ", env.action_space.high)
  for i_episode in range(50):
      observation = env.reset()
      for t in range(time_limit):
#         env.render()
        print(observation.shape)
        # action = ((actionSpace.getMaximum() - actionSpace.getMinimum()) * np.random.uniform(size=actionSpace.getMinimum().shape[0])  ) + actionSpace.getMinimum()
        vizData = env.getVisualState()
        vizImitateData = env.getImitationVisualState()
        for vd in range(len(vizData)):
            # print("viewData: ", viewData)
            viewData = vizData[vd]
            viewImitateData = vizImitateData[vd]
            ## Get and vis terrain data
            if (True):
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                # img_ = viewData
#                 viewData = viewData - viewImitateData
                img_ = np.reshape(viewData[:2304], (48,48))
#                     noise = np.random.normal(loc=0, scale=0.02, size=img_.shape)
#                     img_ = img_ + noise
                print("img_ shape", img_.shape, " sum: ", np.sum(viewData))
                fig1 = plt.figure(1)
                plt.imshow(img_, origin='lower')
                plt.title("visual Data: " +  str(vd))
                fig1.savefig("char_viz_state_"+str(i_episode)+"_"+str(t)+".svg")
 
                if (True):                    
                    img__ = viewImitateData
                    img__ = np.reshape(viewImitateData[:2304], (48, 48))
                    fig2 = plt.figure(2)
                    img__ = np.concatenate((img_, img__), axis=1)
                    plt.imshow(img__, origin='lower')
                    plt.title("visual Data: " +  str(vd))
                    fig2.savefig("char_viz_imitation_state_"+str(i_episode)+"_"+str(t)+".svg")
                plt.show()
        action = env.action_space.sample()
        # print("Actions: ", actions)
        observation, reward, done, info = env.step(action)
        print("Reward: ", reward)
        img = env.render(mode='rgb_array')
        rewards.append(reward)
        states.append(observation)
        if (t >= (time_limit-1)):
        # if (t >= (time_limit-1)):
            print("Episode finished after {} timesteps".format(t+1))
            print("mean reward: ", np.mean(rewards))
            print("std reward: ", np.std(rewards))
            break
        
  return

if __name__ == '__main__':
  main()
