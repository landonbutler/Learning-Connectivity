import numpy as np
from progress.bar import Bar
import gym
import time
import aoi_envs
import imageio
import argparse
import os
import glob


def make_env():
    env_name = "StationaryEnv-v0"
    my_env = gym.make(env_name)
    my_env = gym.wrappers.FlattenDictWrapper(my_env, dict_keys=my_env.env.keys)
    return my_env


def eval_model(env, model, N, render=False):
    """
    Evaluate a model against an environment over N games.
    """
    results = {'reward': np.zeros(N)}
    with Bar('Eval', max=N) as bar:
        for k in range(N):
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


def save_gif(model_number, timestep, fp, controller):
    filename = fp + controller + str(model_number) + '.gif'
    with imageio.get_writer(filename, mode='I', duration=.3) as writer:
        for i in range(1, timestep+1):
            fileloc = fp + 'ts' +str(i) +'.png'
            image = imageio.imread(fileloc)
            writer.append_data(image)
            os.remove(fileloc)


def test_model(model_name, vec_env, env, n_episodes=10):
    # load the dictionary of parameters from file
    model_params, params = BaseRLModel._load_from_file(model_name)
    policy_kwargs = model_params['policy_kwargs']

    model = PPO2(
        policy=aoi_learner.gnn_policy.GNNPolicy,
        n_steps=10,
        policy_kwargs=policy_kwargs,
        env=vec_env)

    # update new model's parameters
    model.load_parameters(params)
    print('\nTest over ' + str(n_episodes) + ' episodes...')
    results = eval_model(env, model, n_episodes)

    mean_reward = np.mean(results['reward'])
    std_reward = np.std(results['reward'])
    return mean_reward, std_reward


if __name__ == '__main__':

    import aoi_learner
    from aoi_learner.ppo2 import PPO2
    from stable_baselines.common.vec_env import SubprocVecEnv
    from stable_baselines.common.base_class import BaseRLModel

    vec_env = SubprocVecEnv([make_env])

    # Specify pre-trained model checkpoint file.
    best_reward = -np.Inf
    best_index = 0
    env = make_env()

    ckpt_dir = 'models/rl_nonlinear_7_2/ckpt'
    try:
        ckpt_list = sorted(glob.glob(str(ckpt_dir) + '/*.pkl'))
    except IndexError:
        print('Invalid experiment folder name!')
        raise

    model_name = ckpt_list[-2]
    mean_reward, std_reward = test_model(model_name, vec_env, env, 100)
    print('reward,          mean = {:.1f}, std = {:.1f}'.format(mean_reward, std_reward))
    print('')




