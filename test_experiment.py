import gym
import configparser
from os import path
import sys
from pathlib import Path
import glob
import aoi_envs
from test_all import find_best_model, test_one


def main():
    fname = sys.argv[1]
    config_file = path.join(path.dirname(__file__), fname)
    config = configparser.ConfigParser()
    config.read(config_file)
    if config.sections():
        for section_name in config.sections():
            print(section_name)
            test_env = gym.make(config[section_name].get('env', 'StationaryEnv-v0'))
            test_env = gym.wrappers.FlattenDictWrapper(test_env, dict_keys=test_env.env.keys)
            directory = Path('models/' + config[section_name].get('name') + section_name)
            save_dir = Path(directory)
            ckpt_dir = save_dir / 'ckpt'

            ckpt_list = sorted(glob.glob(str(ckpt_dir) + '/ckpt_*.pkl'))

            mean_reward, std_reward = test_one(ckpt_list[-1], test_env)
            print('reward,          mean = {:.1f}, std = {:.1f}'.format(mean_reward, std_reward))
    else:
        test_env = gym.make(config.default_section.get('env', 'StationaryEnv-v0'))
        directory = Path('models/' + config[config.default_section].get('name'))
        save_dir = Path(directory)
        ckpt_dir = save_dir / 'ckpt'
        ckpt_list = sorted(glob.glob(str(ckpt_dir) + '/ckpt_*.pkl'))

        mean_reward, std_reward = test_one(ckpt_list[-1], test_env)
        print('reward,          mean = {:.1f}, std = {:.1f}'.format(mean_reward, std_reward))


if __name__ == '__main__':
    main()
