import numpy as np
from progress.bar import Bar
import gym
import time
import aoi_envs
import imageio
import argparse
import os

# Example Usage:
# python3 test.py -v -e FlockingAOIEnv-v0 -p models/mobile12_05_2/ckpt/ckpt_040.pkl

parser = argparse.ArgumentParser(description="Testing AoI Environments and Models")
parser.add_argument('-v', '--visualize', dest='visualize', action='store_true')

parser.add_argument('-g', '--greedy', dest='greedy', action='store_true')
parser.add_argument('-m', '--mst', dest='mst', action='store_true')
parser.add_argument('-r', '--random', dest='random', action='store_true')
parser.add_argument('-l', '--learner', dest='learner', action='store_true')

parser.add_argument('-gif', '--gif', dest='gif', action='store_true')
parser.add_argument('-e', '--env', type=str)
parser.add_argument('-p', '--path', dest='path', type=str)
parser.add_argument('-n', '--n_episodes', dest='n_episodes', type=int)

parser.set_defaults(random=False, mst=False, greedy=False, visualize=False, learner=False,
                    gif=False, path='', env='StationaryEnv-v0', n_episodes=10)
args = parser.parse_args()


def make_env():
    my_env = gym.make(args.env)
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

            gif_fp = 'visuals/'
            while not done:
                if args.learner or model:
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
                    action = env.env.env.roundrobin_controller()
                    controller = "RoundRobin"

                state = None
                obs, rewards, done, info = env.step(action)

                if render:
                    env.env.env.render(controller=controller, save_plots=args.gif)
                    time.sleep(0.1)

                # Record results.
                results['reward'][k] += rewards
                timestep += 1
            if args.gif:
                save_gif(k, timestep, gif_fp, controller)
            print(results['reward'][k])
            bar.next()
    return results


def save_gif(model_number, timestep, fp, controller):
    filename = fp + controller + str(model_number) + '.gif'
    with imageio.get_writer(filename, mode='I', duration=.15) as writer:
        for i in range(1, timestep):
            fileloc = fp + 'ts' + str(int(i)) + '.png'
            image = imageio.imread(fileloc)
            writer.append_data(image)
            os.remove(fileloc)


if __name__ == '__main__':

    model_name = args.path

    if args.learner or len(model_name) > 0:
        import aoi_learner
        from aoi_learner.ppo2 import PPO2
        from stable_baselines.common.vec_env import DummyVecEnv
        from stable_baselines.common.base_class import BaseRLModel

        vec_env = DummyVecEnv([make_env])

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
        print('Model loaded')
    else:
        model = None

    env = make_env()

    if args.visualize:
        if args.gif:
            print('\nTest over 1  episode live visualization...')
            eval_model(env, model, 1, render=True)
        else:
            print('\nTest over 10  episodes live visualization...')
            eval_model(env, model, 10, render=True)
    else:
        print('\nTest over ' + str(args.n_episodes) + ' episodes...')
        results = eval_model(env, model, args.n_episodes)
        print('reward,          mean = {:.1f}, std = {:.1f}'.format(np.mean(results['reward']), np.std(results['reward'])))
        print('')