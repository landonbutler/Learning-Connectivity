import gym
import tensorflow as tf
from datetime import datetime

from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from aoi_learner.CommunicationPolicy import CommunicationPolicy


def run_experiment(args):
    env_name = 'StationaryEnv-v0'
    config_file = 'cfg/FlockingCommFailure.cfg'

    env = gym.make(env_name)
    config = configparser.ConfigParser()
    config.read(config_file)
    env.env.params_from_cfg(config[config.sections()[0]])

    # use seed
    seed = args.getint('seed')
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # torch.manual_seed(seed)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = PPO2(CommunicationPolicy, env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=100000)

    # obs should initially be a blank network buffer (NxNx5-7)
    obs = env.reset()

    for i in range(1000):
        # based on an observation, figure out what the model will do. This will 
        # be a loop over all agents for us
        action, _states = model.predict(obs)
        # based on the action(s), see how the environment is updated
        obs, rewards, dones, info = env.step(action)
        # show the outcome of the action
        env.render()

    model.save("/models/" + "ppo_commpolicy" + datetime.now().time())
    log_dir = "/tmp/"
    stats_path = os.path.join(log_dir, "training_states.pkl")
    env.save(stats_path)

    # figure out what to return
    return stats