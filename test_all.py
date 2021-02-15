import numpy as np
from progress.bar import Bar
import gym
import aoi_envs
import glob
import aoi_learner
import os
import sys
from aoi_learner.ppo2 import PPO2
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import DummyVecEnv

N_ENVS = 25


def eval_model(env, model, n_episodes):
    """
    Evaluate a model against an environment over N games.
    """
    results = {'reward': np.zeros(n_episodes)}
    # with Bar('Eval', max=n_episodes) as bar:
    for k in range(n_episodes // N_ENVS):
        done = [False]
        obs = env.reset()
        state = None
        timestep = 1
        # Run one game.
        while not np.any(done):
            action, state = model.predict(obs, state=state, deterministic=False)
            state = None
            obs, rewards, done, info = env.step(action)

            # Record results.
            results['reward'][k * N_ENVS:(k * N_ENVS + N_ENVS)] += np.array(rewards)
            timestep += 1
            # bar.next()

    return results


def test_one(ckpt, test_env, n_episodes=100):
    # load the dictionary of parameters from file
    model_params, params = BaseRLModel._load_from_file(ckpt)
    policy_kwargs = model_params['policy_kwargs']

    model = PPO2(
        policy=aoi_learner.gnn_policy.GNNPolicy,
        n_steps=10,
        policy_kwargs=policy_kwargs,
        env=test_env)

    # update new model's parameters
    model.load_parameters(params)
    print('Testing ' + ckpt + ' over ' + str(n_episodes) + ' episodes...')
    results = eval_model(test_env, model, n_episodes)

    mean_reward = np.mean(results['reward'])
    std_reward = np.std(results['reward'])
    # print('reward,          mean = {:.1f}, std = {:.1f}'.format(mean_reward, std_reward))
    # rewards.append(mean_reward)
    return mean_reward, std_reward


def find_best_model(all_ckpt_dir, test_env):
    # Get the path of the last checkpoint.
    try:
        ckpt_list = sorted(glob.glob(str(all_ckpt_dir) + '/ckpt_*.pkl'))
    except IndexError:
        print('Invalid experiment folder name!')
        raise
    if len(ckpt_list) is 0:
        print('Invalid experiment folder name!')
        raise IndexError('Invalid experiment folder name!')
    rewards = []

    # Test every 10th checkpoint.
    ckpt_list = ckpt_list[-10:]

    for ckpt in ckpt_list:
        mean_reward, std_reward = test_one(ckpt, test_env, 50)
        print('reward,          mean = {:.1f}, std = {:.1f}'.format(mean_reward, std_reward))
        rewards.append(mean_reward)

    best_ckpt = ckpt_list[rewards.index(max(rewards))]
    print("Best Model: " + best_ckpt)

    mean_reward, std_reward = test_one(best_ckpt, test_env, 100)
    print('reward,          mean = {:.1f}, std = {:.1f}'.format(mean_reward, std_reward))


if __name__ == '__main__':
    def make_env():
        env = gym.make('Stationary40Env-v0')
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=env.env.keys)
        return env


    env = DummyVecEnv([make_env] * N_ENVS)

    # Specify pre-trained model checkpoint folder (containing all checkpoints).
    all_ckpt_dir = 'models/' + sys.argv[1] + '/ckpt'
    find_best_model(all_ckpt_dir, env)
