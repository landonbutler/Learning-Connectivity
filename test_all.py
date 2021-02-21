import numpy as np
import csv
import gym
import configparser
from os import path
import sys
import glob
from functools import partial

import aoi_learner
import aoi_envs
from aoi_learner.ppo2 import PPO2
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import DummyVecEnv

N_ENVS = 25

# Usage:
# python3 test_all.py flocking_3_40_025 Flocking025Env-v0
# or
# python3 test_all.py cfg/flocking.cfg flocking


def eval_model(env, model, n_episodes):
    """
    Evaluate a model against an environment over N games.
    """
    results = {'reward': np.zeros(n_episodes)}
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
    return mean_reward, std_reward


def find_best_model(all_ckpt_dir, test_env, find_best=True):
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

    if find_best:
        # Test last 10 checkpoints
        #ckpt_list = ckpt_list[0:40][0::2]
        for ckpt in ckpt_list:
            mean_reward, std_reward = test_one(ckpt, test_env, 50)
            print('reward,          mean = {:.1f}, std = {:.1f}'.format(mean_reward, std_reward))
            rewards.append(mean_reward)

        best_ckpt = ckpt_list[rewards.index(max(rewards))]
        print("Best Model: " + best_ckpt)
    else:
        best_ckpt = ckpt_list[-1]

    mean_reward, std_reward = test_one(best_ckpt, test_env, 100)
    print('reward,          mean = {:.1f}, std = {:.1f}'.format(mean_reward, std_reward))

    return mean_reward, std_reward, best_ckpt


def make_env(env_name):
    env = gym.make(env_name)
    env = gym.wrappers.FlattenDictWrapper(env, dict_keys=env.env.keys)
    return env


if __name__ == '__main__':

    fname = sys.argv[1]

    if fname[-4:] == '.cfg':
        data_to_csv = []
        results_csv_fname = sys.argv[2] + '_results.csv'
        config_file = path.join(path.dirname(__file__), fname)
        config = configparser.ConfigParser()
        config.read(config_file)

        section_names = config.sections() if config.sections() else [config.default_section]
        for section_name in section_names:
            print(section_name)
            results = [config[section_name].get('name') + section_name]
            env_name = config[section_name].get('env', 'StationaryEnv-v0')
            test_env = DummyVecEnv([partial(make_env, env_name)] * N_ENVS)
            model_name = config[section_name].get('model_path', config[section_name].get('name') + section_name)
            find_best = config[section_name].getboolean('find_best', True)
            all_ckpt_dir = 'models/' + model_name + '/ckpt'
            print(all_ckpt_dir)
            try:
                mean, std, path = find_best_model(all_ckpt_dir, test_env, find_best)
                results.extend([mean, std, path])
                data_to_csv.append(results)
            except IndexError:
                print('Invalid experiment folder name!')
                break

        # writing to csv file
        with open(results_csv_fname, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the fields
            csvwriter.writerow(['Section', 'Mean', 'Std', 'Path'])

            # writing the data rows
            csvwriter.writerows(data_to_csv)

    else:
        env = DummyVecEnv([partial(make_env, sys.argv[2])] * N_ENVS)
        # Specify pre-trained model checkpoint folder (containing all checkpoints).
        all_ckpt_dir = 'models/' + sys.argv[1] + '/ckpt'
        find_best_model(all_ckpt_dir, env)
