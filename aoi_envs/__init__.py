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
    id='MobileEnv00-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 0.0},
)

register(
    id='MobileEnv05-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 0.5},
)

register(
    id='MobileEnv10-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 1.0},
)

register(
    id='MobileEnv15-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 1.5},
)

register(
    id='MobileEnv20-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 2.0},
)

register(
    id='MobileEnv50-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 5.0},
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
