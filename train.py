import gym
import configparser
import json
from os import path
import functools
import glob
import sys
import argparse
from pathlib import Path
from stable_baselines.common import BaseRLModel
from stable_baselines.common.vec_env import SubprocVecEnv

from aoi_learner.gnn_policy import GNNPolicy
from aoi_learner.ppo2 import PPO2
from aoi_learner.utils import ckpt_file, callback
from test_all import find_best_model


def train_helper(env_param, test_env_param, train_param, policy_fn, policy_param, directory, env=None, test_env=None):
    save_dir = Path(directory)
    tb_dir = save_dir / 'tb'
    ckpt_dir = save_dir / 'ckpt'
    for d in [save_dir, tb_dir, ckpt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if env is None:
        env = SubprocVecEnv([env_param['make_env']] * train_param['n_env'])
    if test_env is None:
        test_env = SubprocVecEnv([test_env_param['make_env']])

    if train_param['use_checkpoint']:
        # Find latest checkpoint index.
        ckpt_list = sorted(glob.glob(str(ckpt_dir) + '/*.pkl'))
        if len(ckpt_list) == 0:
            ckpt_idx = None
        else:
            ckpt_idx = int(ckpt_list[-2][-7:-4])
    else:
        ckpt_idx = None

    # Load or create model.
    if ckpt_idx is not None:
        print('\nLoading model {}.\n'.format(ckpt_file(ckpt_dir, ckpt_idx).name))
        model = PPO2.load(str(ckpt_file(ckpt_dir, ckpt_idx)), env, tensorboard_log=str(tb_dir))
        ckpt_idx += 1
    else:
        print('\nCreating new model.\n')

        model = PPO2(
            policy=policy_fn,
            policy_kwargs=policy_param,
            env=env,
            learning_rate=train_param['train_lr'],
            cliprange=train_param['cliprange'],
            adam_epsilon=train_param['adam_epsilon'],
            n_steps=train_param['n_steps'],
            ent_coef=train_param['ent_coef'],
            vf_coef=train_param['vf_coef'],
            verbose=1,
            tensorboard_log=str(tb_dir),
            full_tensorboard_log=False,
            lr_decay_factor=train_param['lr_decay_factor'],
            lr_decay_steps=train_param['lr_decay_steps'],
        )

        ckpt_idx = 0

        if 'load_trained_policy' in train_param and len(train_param['load_trained_policy']) > 0:
            model_name = train_param['load_trained_policy']

            # load the dictionary of parameters from file
            _, params = BaseRLModel._load_from_file(model_name)

            # update new model's parameters
            model.load_parameters(params)

    # Training loop.
    print('\nBegin training.\n')
    while train_param['total_timesteps'] > 0 and model.num_timesteps <= train_param['total_timesteps']:
        print('\nLearning...\n')
        model.learn(
            total_timesteps=train_param['checkpoint_timesteps'],
            log_interval=500,
            reset_num_timesteps=False,
            callback=functools.partial(callback, test_env=test_env, interval=train_param['checkpoint_timesteps'], n_episodes=20))

        print('\nSaving model {}.\n'.format(ckpt_file(ckpt_dir, ckpt_idx).name))
        model.save(str(ckpt_file(ckpt_dir, ckpt_idx)))
        ckpt_idx += 1

    print('Finished.')
    # env.close()
    # test_env.close()
    del model

    return env, test_env


def run_experiment(args, section_name='', env=None, test_env=None):

    policy_param = {
        'num_processing_steps': args.getint('num_processing_steps', 5),
        'latent_size': args.getint('latent_size', 16),
        'n_layers': args.getint('n_layers', 3),
        'reducer': args.get('reducer', 'mean'),
        'model_type': args.get('model_type', 'identity'),
    }

    policy_fn = GNNPolicy
    policy_param['n_gnn_layers'] = args.getint('n_gnn_layers', 1)
    env_name = args.get('env', 'StationaryEnv-v0')


    def make_env():
        env = gym.make(env_name)
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=env.env.keys)
        return env

    env_param = {'make_env': make_env}
    test_env_param = {'make_env': make_env}

    train_param = {
        'use_checkpoint': args.getboolean('use_checkpoint', False),
        'load_trained_policy': args.get('load_trained_policy', ''),
        'normalize_reward': args.get('normalize_reward', False),
        'n_env': args.getint('n_env', 4),
        'n_steps': args.getint('n_steps', 10),
        'checkpoint_timesteps': args.getint('checkpoint_timesteps', 10000),
        'total_timesteps': args.getint('total_timesteps', 50000000),
        'train_lr': args.getfloat('train_lr', 1e-4),
        'cliprange': args.getfloat('cliprange', 0.2),
        'adam_epsilon': args.getfloat('adam_epsilon', 1e-6),
        'vf_coef': args.getfloat('vf_coef', 0.5),
        'ent_coef': args.getfloat('ent_coef', 1e-6),
        'lr_decay_factor': args.getfloat('lr_decay_factor', 1.0),
        # 'lr_decay_factor': args.getfloat('lr_decay_factor', 0.97),
        'lr_decay_steps': args.getfloat('lr_decay_steps', 10000),
    }

    directory = Path('models/' + args.get('name') + section_name)

    env, test_env = train_helper(
        env_param=env_param,
        test_env_param=test_env_param,
        train_param=train_param,
        policy_fn=policy_fn,
        policy_param=policy_param,
        directory=directory,
        env=env, test_env=test_env)
    return env, test_env


def main():
    fname = sys.argv[1]
    config_file = path.join(path.dirname(__file__), fname)
    config = configparser.ConfigParser()
    config.read(config_file)
    if config.sections():
        env = None
        test_env = None
        for section_name in config.sections():
            env, test_env = run_experiment(config[section_name], section_name, env, test_env)
            directory = Path('models/' + config[section_name].get('name') + section_name)
            save_dir = Path(directory)
            ckpt_dir = save_dir / 'ckpt'
            find_best_model(ckpt_dir, test_env)
    else:
        env, test_env = run_experiment(config[config.default_section])
        directory = Path('models/' + config[config.default_section].get('name'))
        save_dir = Path(directory)
        ckpt_dir = save_dir / 'ckpt'
        find_best_model(ckpt_dir, test_env)

if __name__ == '__main__':
    main()
