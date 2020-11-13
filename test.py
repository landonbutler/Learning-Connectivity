import numpy as np
from progress.bar import Bar
import gym
import aoi_envs
import aoi_learner.gnn_policy as GNNPolicy
from aoi_learner.ppo2 import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.base_class import BaseRLModel
import tensorflow as tf
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
            # Run one game.
            while not done:
                action, state = model.predict(obs, state=state, deterministic=True)
                state = None
                obs, rewards, done, info = env.step(action)
                env.render(mode=render_mode)

                # if render_mode == 'human':
                #     time.sleep(0.1)

                # Record results.
                results['reward'][k] += rewards

            bar.next()
    return results


if __name__ == '__main__':
    env = make_env()
    vec_env = SubprocVecEnv([make_env])

    # Specify pre-trained model checkpoint file.
    model_name = 'models/rl_1/ckpt/ckpt_001.pkl'  # ent_coef  = 1e-6

    # load the dictionary of parameters from file
    model_params, params = BaseRLModel._load_from_file(model_name)
    policy_kwargs = model_params['policy_kwargs']

    new_model = PPO2(
        policy=GNNPolicy,
        n_steps=10,
        policy_kwargs=policy_kwargs,
        env=vec_env)

    # update new model's parameters
    new_model.load_parameters(params)

    print('Model loaded')
    print('\nPlay 10 games and return scores...')
    results = eval_model(env, new_model, 100, render_mode='none')
    print('reward,          mean = {:.1f}, std = {:.1f}'.format(np.mean(results['reward']), np.std(results['reward'])))
    print('')

    print('\nPlay games with live visualization...')
    eval_model(env, new_model, 10, render_mode='human')
