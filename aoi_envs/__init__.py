from aoi_envs.MultiAgent import MultiAgentEnv
from aoi_envs.Mobile import MobileEnv
from gym.envs.registration import register

MAX_EPISODE_STEPS = 500

register(
    id='StationaryEnv-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
)

register(
    id='PowerLevelsEnv-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'fractional_power_levels': [0.25, 0.125]},
)

register(
    id='PowerLevels1Env-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'fractional_power_levels': [0.25, 0.125]},
)

register(
    id='PowerLevels2Env-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'fractional_power_levels': [0.5, 0.25]},
)

register(
    id='PowerLevels3Env-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'fractional_power_levels': [0.5, 0.125]},
)

register(
    id='PowerLevel10Env-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'fractional_power_levels': [1.0]},
)

register(
    id='PowerLevel075Env-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'fractional_power_levels': [0.75]},
)

register(
    id='PowerLevel05Env-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'fractional_power_levels': [0.5]},
)

register(
    id='PowerLevel025Env-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'fractional_power_levels': [0.25]},
)

register(
    id='PowerLevel02Env-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'fractional_power_levels': [0.2]},
)

register(
    id='PowerLevel015Env-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'fractional_power_levels': [0.15]},
)

register(
    id='StationaryGridEnv-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'initialization': "Grid"},
)

register(
    id='StationaryGrid40Env-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'initialization': "Grid", 'num_agents': 40},
)

register(
    id='StationaryGrid60Env-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'initialization': "Grid", 'num_agents': 60},
)

register(
    id='StationaryGrid80Env-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'initialization': "Grid", 'num_agents': 80},
)

register(
    id='StationaryGrid100Env-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'initialization': "Grid", 'num_agents': 100},
)

register(
    id='StationaryGrid150Env-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'initialization': "Grid", 'num_agents': 150},
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
    id='MobileEnv01-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 0.1},
)

register(
    id='MobileEnv025-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 0.25},
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
    id='MobileRandomEnv-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 1.0, 'random_acceleration': True},
)



register(
    id='MobileEnv20-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 2.0},
)

register(
    id='FlockingEnv-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 1.0, 'initialization': 'Grid', 'flocking': True, 'biased_velocities': False},
)

