import numpy as np
from progress.bar import Bar
import gym
import aoi_envs
import glob
import aoi_learner
import os
import sys
from aoi_learner.ppo2 import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.base_class import BaseRLModel


def eval_model(env, model, n_episodes):
    """
    Evaluate a model against an environment over N games.
    """
    results = {'reward': np.zeros(n_episodes)}
    with Bar('Eval', max=n_episodes) as bar:
        for k in range(n_episodes):
            done = False
            obs = env.reset()
            state = None
            timestep = 1
            # Run one game.
            while not done:
                action, state = model.predict(obs, state=state, deterministic=False)
                state = None
                obs, rewards, done, info = env.step(action)

                # Record results.
                results['reward'][k] += rewards
                timestep += 1
            bar.next()
    return results


def find_best_model(all_ckpt_dir, test_env, n_episodes=50):

    # Get the path of the last checkpoint.
    try:
        ckpt_list = sorted(glob.glob(str(all_ckpt_dir) + '/*.pkl'))
    except IndexError:
        print('Invalid experiment folder name!')
        raise
    if len(ckpt_list) is 0:
        print('Invalid experiment folder name!')
        raise IndexError('Invalid experiment folder name!')
    rewards = []

    # Test every 10th checkpoint.
    ckpt_list = ckpt_list[0::10]
  
    for ckpt in ckpt_list:
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
        print('reward,          mean = {:.1f}, std = {:.1f}'.format(mean_reward, std_reward))
        rewards.append(mean_reward)
    
    best_ckpt = ckpt_list[rewards.index(max(rewards))]
    print("Best Model: " + best_ckpt)

    best_model_params, best_params = BaseRLModel._load_from_file(best_ckpt)
    best_policy_kwargs = best_model_params['policy_kwargs']

    best_model = PPO2(
        policy=aoi_learner.gnn_policy.GNNPolicy,
        n_steps=10,
        policy_kwargs=best_policy_kwargs,
        env=test_env)

    best_n_episodes = 100
    # update new model's parameters
    best_model.load_parameters(best_params)
    print('Testing ' + best_ckpt + ' over ' + str(best_n_episodes) + ' episodes...')
    results = eval_model(test_env, model, best_n_episodes)

    mean_reward = np.mean(results['reward'])
    std_reward = np.std(results['reward'])
    print('reward,          mean = {:.1f}, std = {:.1f}'.format(mean_reward, std_reward))

    # Delete all but best ckpt
    # for i in ckpt_list:
        # if i is not best_ckpt:
            # os.remove(i)



if __name__ == '__main__':

    env = gym.make('StationaryEnv-v0')
    env = gym.wrappers.FlattenDictWrapper(env, dict_keys=env.env.keys)

    # Specify pre-trained model checkpoint folder (containing all checkpoints).
    all_ckpt_dir = 'models/' + sys.argv[1] + '/ckpt'
    find_best_model(all_ckpt_dir, env, 10)
