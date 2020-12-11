from aoi_envs.MultiAgent import MultiAgentEnv
from aoi_envs.Mobile import MobileEnv
from aoi_envs.StationaryKnown import StationaryKnownEnv
from gym.envs.registration import register

register(
    id='StationaryEnv-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=200,
)

register(
    id='MobileEnv-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=200,
)

register(
    id='StationaryKnownEnv-v0',
    entry_point='aoi_envs:StationaryKnownEnv',
    max_episode_steps=200,
)