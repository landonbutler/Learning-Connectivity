from aoi_envs.MultiAgent import MultiAgentEnv
from aoi_envs.Mobile import MobileEnv
from aoi_envs.StationaryKnown import StationaryKnownEnv
from aoi_envs.Flocking import FlockingEnv
from gym.envs.registration import register

MAX_EPISODE_STEPS = 200

register(
    id='StationaryEnv-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
)

register(
    id='MobileEnv-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
)

register(
    id='StationaryKnownEnv-v0',
    entry_point='aoi_envs:StationaryKnownEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
)

register(
    id='FlockingEnv-v0',
    entry_point='aoi_envs:FlockingEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
)