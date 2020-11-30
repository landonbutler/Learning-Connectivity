from aoi_envs.Stationary import StationaryEnv
from aoi_envs.Mobile import MobileEnv
from gym.envs.registration import register

register(
    id='StationaryEnv-v0',
    entry_point='aoi_envs:StationaryEnv',
    max_episode_steps=100,
)

register(
    id='MobileEnv-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=100,
)
