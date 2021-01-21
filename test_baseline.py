import gym
import configparser
from os import path
import sys
import numpy as np
import aoi_envs


def eval_baseline(env, baseline, n_episodes=100):
    """
    Evaluate a model against an environment over N games.
    """
    results = {'reward': np.zeros(n_episodes)}

    for k in range(n_episodes):
        done = False
        obs = env.reset()
        timestep = 1
        while not done:
            if baseline == 'mst':
                action = env.env.mst_controller()
            elif baseline == 'greedy':
                action = env.env.greedy_controller()
            elif baseline == 'random':
                action = env.env.random_controller()
            elif baseline == 'roundrobin':
                action = env.env.roundrobin_controller()
            else:
                action = env.env.neopolitan_controller()

            obs, rewards, done, info = env.step(action)
            # env.render(mode=render_mode)

            # Record results.
            results['reward'][k] += rewards
            timestep += 1

    mean_reward = np.mean(results['reward'])
    std_reward = np.std(results['reward'])
    print(baseline + ', mean = {:.1f}, std = {:.1f}'.format(mean_reward, std_reward))
    return results


def main():
    fname = sys.argv[1]
    print(fname)
    config_file = path.join(path.dirname(__file__), fname)
    config = configparser.ConfigParser()
    config.read(config_file)
    baselines = ['mst', 'greedy', 'roundrobin', 'random']

    if config.sections():
        for section_name in config.sections():
            print(config[section_name].get('name') + section_name)
            env_name = config[section_name].get('env', 'StationaryEnv-v0')
            print(env_name)
            env = gym.make(config[section_name].get('env', 'StationaryEnv-v0'))
            for baseline in baselines:
                eval_baseline(env, baseline)
    else:
        env = gym.make(config.default_section.get('env', 'StationaryEnv-v0'))
        print(config[config.default_section].get('name'))
        for baseline in baselines:
            eval_baseline(env, baseline)
if __name__ == '__main__':
    main()
