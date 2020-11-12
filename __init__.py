from gym.envs.registration import register
import gym

register(
    id='StationaryEnv-v0',
    entry_point='envs:StationaryEnv',
    max_episode_steps=100,
)
