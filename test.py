import numpy as np
from progress.bar import Bar
import gym
import time
import aoi_envs

import tensorflow as tf
import imageio
import argparse
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser(description="My parser")
parser.add_argument('-g', '--greedy', dest='greedy', action='store_true')
parser.add_argument('-m', '--mst', dest='mst', action='store_true')
parser.add_argument('-r', '--random', dest='random', action='store_true')
parser.add_argument('-n', '--neopolitan', dest='neopolitan', action='store_true')
parser.add_argument('-v', '--visualize', dest='visualize', action='store_true')
parser.add_argument('-l', '--learner', dest='learner', action='store_true')
parser.add_argument('-b', '--buffer', dest='buffer', action='store_true')

parser.set_defaults(random=False, mst=False, greedy=False, visualize=False, learner=False, buffer=False)
args = parser.parse_args()


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
            controller = ""
            gif_fp = "visuals/models/"
            while not done:
                if args.learner and model:
                    action, state = model.predict(obs, state=state, deterministic=False)
                    controller = "GNN"
                elif args.mst:
                    action = env.env.env.mst_controller()
                    controller = "MST"
                elif args.greedy:
                    action = env.env.env.greedy_controller()
                    controller = "Greedy"
                elif args.random:
                    action = env.env.env.random_controller()
                    controller = "Random"
                else:
                    action = env.env.env.neopolitan_controller()
                    controller = "Neopolitan"

                state = None
                obs, rewards, done, info = env.step(action)
                # env.render(mode=render_mode)
                if render:
                    if args.buffer:
                        env.env.env.render()
                    else:
                        env.env.env.render_interference(controller=controller)
                    time.sleep(0.1)

                # Record results.
                results['reward'][k] += rewards
                timestep += 1
            # save_gif(k, timestep, gif_fp, controller)
            # print(results['reward'][k])
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


if __name__ == '__main__':
    env = make_env()

    if args.learner:
        import aoi_learner
        from aoi_learner.ppo2 import PPO2
        from stable_baselines.common.vec_env import SubprocVecEnv
        from stable_baselines.common.base_class import BaseRLModel

        vec_env = SubprocVecEnv([make_env])

        # Specify pre-trained model checkpoint file.
        # model_name = 'models/rl_4/ckpt/ckpt_000.pkl'  # ent_coef  = 1e-6

        model_name = 'models/rl_Landon/RL_GNN_5_ENT5_1/RL_GNN_5_ENT5_1.pkl'
        # model_name = 'models/rl_15/ckpt/ckpt_481.pkl'

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
    else:
        model = None

    print('Model loaded')
    print('\nTest over 100 episodes...')
    results = eval_model(env, model, 100, render=args.visualize)
    print('reward,          mean = {:.1f}, std = {:.1f}'.format(np.mean(results['reward']), np.std(results['reward'])))
    print('')

    print('\nTest over 10  episodes live visualization...')
    eval_model(env, model, 10, render=True)
