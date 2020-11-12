from aoi_envs.Stationary import StationaryEnv
from gym.envs.registration import register

register(
    id='StationaryEnv-v0',
    entry_point='aoi_envs:StationaryEnv',
    max_episode_steps=100,
)
