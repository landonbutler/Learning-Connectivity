import numpy as np
from progress.bar import Bar
import gym
import aoi_envs
import glob
import aoi_learner
from aoi_learner.ppo2 import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.base_class import BaseRLModel


def make_env():
    env_name = "StationaryEnv-v0"
    my_env = gym.make(env_name)
    my_env = gym.wrappers.FlattenDictWrapper(my_env, dict_keys=my_env.env.keys)
    return my_env


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


if __name__ == '__main__':

    vec_env = SubprocVecEnv([make_env])
    env = make_env()

    # Specify pre-trained model checkpoint folder.
    ckpt_dir = 'models/rl_nonlinear_8_3/ckpt'
    n_episodes = 10

    # Get the path of the last checkpoint.
    try:
        ckpt_list = sorted(glob.glob(str(ckpt_dir) + '/*.pkl'))
    except IndexError:
        print('Invalid experiment folder name!')
        raise
    model_name = ckpt_list[-2]

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
    print('Testing ' + model_name + ' over ' + str(n_episodes) + ' episodes...')
    results = eval_model(env, model, n_episodes)

    mean_reward = np.mean(results['reward'])
    std_reward = np.std(results['reward'])

    print('reward,          mean = {:.1f}, std = {:.1f}'.format(mean_reward, std_reward))




