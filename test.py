import numpy as np
from progress.bar import Bar
import gym
import time
import aoi_envs
import aoi_learner
from aoi_learner.ppo2 import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.base_class import BaseRLModel
import tensorflow as tf
import imageio
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def make_env():
    env_name = "StationaryEnv-v0"
    my_env = gym.make(env_name)
    my_env = gym.wrappers.FlattenDictWrapper(my_env, dict_keys=my_env.env.keys)
    return my_env


def eval_model(env, model, N, render_mode='none'):
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
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, state = model.predict(obs, state=state, deterministic=True)
                state = None
                obs, rewards, done, info = env.step(action)
                env.render(mode=render_mode)

                if render_mode == 'human':
                    time.sleep(0.1)

                # Record results.
                results['reward'][k] += rewards
                timestep += 1
            # save_gif(k, timestep)
            # print(results['reward'][k])
            bar.next()
    return results

def save_gif(model_number, timestep):
    filename = 'visuals/bufferTrees/controller' + str(model_number) + '.gif'
    with imageio.get_writer(filename, mode='I', duration=.3) as writer:
        for i in range(1, timestep+1):
            fileloc = 'visuals/bufferTrees/ts'+str(i)+'.png'
            image = imageio.imread(fileloc)
            writer.append_data(image)
            os.remove(fileloc)

if __name__ == '__main__':
    env = make_env()
    vec_env = SubprocVecEnv([make_env])

    # # Specify pre-trained model checkpoint file.
    # model_name = 'models/rl_4/ckpt/ckpt_000.pkl'  # ent_coef  = 1e-6
    #
    # # load the dictionary of parameters from file
    # model_params, params = BaseRLModel._load_from_file(model_name)
    # policy_kwargs = model_params['policy_kwargs']
    #
    # model = PPO2(
    #     policy=aoi_learner.gnn_policy.GNNPolicy,
    #     n_steps=10,
    #     policy_kwargs=policy_kwargs,
    #     env=vec_env)

    # update new model's parameters
    # model.load_parameters(params)
    model = None

    # print('Model loaded')
    # print('\nTest over 100 episodes...')
    # results = eval_model(env, model, 100, render_mode='none')
    # print('reward,          mean = {:.1f}, std = {:.1f}'.format(np.mean(results['reward']), np.std(results['reward'])))
    # print('')

    print('\nTest over 10  episodes live visualization...')
    eval_model(env, model, 10, render_mode='human')
