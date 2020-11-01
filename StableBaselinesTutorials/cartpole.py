import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# create environment in open ai gym
env = gym.make('CartPole-v1')

# model takes in environment, verbose is logger information, 
# we will need to design our own policy, with the GNN as the actor
model = PPO2(MlpPolicy, env, verbose=1)
# train the model
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    # based on an observation, figure out what the model will do. This will 
    # be a loop over all agents for us
    action, _states = model.predict(obs)
    # based on the action(s), see how the environment is updated
    obs, rewards, dones, info = env.step(action)
    # show the outcome of the action
    env.render()