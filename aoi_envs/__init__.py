from aoi_envs.MultiAgent import MultiAgentEnv
from aoi_envs.Mobile import MobileEnv
from gym.envs.registration import register

MAX_EPISODE_STEPS = 10000

register(
    id='StationaryEnv-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
)

###################################################################

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

###################################################################

register(
    id='Stationary30Env-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'num_agents': 30},
)

register(
    id='Stationary40Env-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'num_agents': 40},
)

register(
    id='Stationary60Env-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'num_agents': 60},
)

register(
    id='Stationary80Env-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'num_agents': 80},
)

register(
    id='Stationary100Env-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'num_agents': 100},
)

register(
    id='Stationary150Env-v0',
    entry_point='aoi_envs:MultiAgentEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'num_agents': 150},
)

###################################################################

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
    id='MobileEnv075-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 0.75},
)

register(
    id='MobileEnv10-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 1.0},
)

register(
    id='MobileEnv125-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 1.25},
)

register(
    id='MobileEnv15-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 1.5},
)

###################################################################

register(
    id='MobileEnv10N10-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 1.0, 'num_agents': 10},
)


register(
    id='MobileEnv10N40-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 1.0, 'num_agents': 40},
)

register(
    id='MobileEnv10N60-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 1.0, 'num_agents': 60},
)

register(
    id='MobileEnv10N80-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 1.0, 'num_agents': 80},
)

register(
    id='MobileEnv10N100-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 1.0, 'num_agents': 100},
)

###################################################################

register(
    id='FlockingEnv-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 1.0, 'flocking': True, 'aoi_reward': False},
)

register(
    id='FlockingAOIEnv-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 1.0, 'flocking': True},
)

###################################################################

register(
    id='Flocking025Env-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 0.25, 'flocking': True, 'aoi_reward': False},
)

register(
    id='Flocking0325Env-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 0.325, 'flocking': True, 'aoi_reward': False},
)

register(
    id='Flocking05Env-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 0.5, 'flocking': True, 'aoi_reward': False},
)

register(
    id='Flocking0625Env-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 0.625, 'flocking': True, 'aoi_reward': False},
)

register(
    id='Flocking075Env-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 0.75, 'flocking': True, 'aoi_reward': False},
)

# register(
#     id='Flocking10Env-v0',
#     entry_point='aoi_envs:MobileEnv',
#     max_episode_steps=MAX_EPISODE_STEPS,
#     kwargs={'agent_velocity': 1.0, 'flocking': True, 'aoi_reward': False},
# )
#
# register(
#     id='Flocking125Env-v0',
#     entry_point='aoi_envs:MobileEnv',
#     max_episode_steps=MAX_EPISODE_STEPS,
#     kwargs={'agent_velocity': 1.25, 'flocking': True, 'aoi_reward': False},
# )
#
# register(
#     id='Flocking15Env-v0',
#     entry_point='aoi_envs:MobileEnv',
#     max_episode_steps=MAX_EPISODE_STEPS,
#     kwargs={'agent_velocity': 1.5, 'flocking': True, 'aoi_reward': False},
# )

###################################################################

register(
    id='FlockingAOI025Env-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 0.25, 'flocking': True},
)

register(
    id='FlockingAOI0325Env-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 0.325, 'flocking': True},
)

register(
    id='FlockingAOI05Env-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 0.5, 'flocking': True},
)

register(
    id='FlockingAOI0625Env-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 0.625, 'flocking': True},
)


register(
    id='FlockingAOI075Env-v0',
    entry_point='aoi_envs:MobileEnv',
    max_episode_steps=MAX_EPISODE_STEPS,
    kwargs={'agent_velocity': 0.75, 'flocking': True},
)

# register(
#     id='FlockingAOI10Env-v0',
#     entry_point='aoi_envs:MobileEnv',
#     max_episode_steps=MAX_EPISODE_STEPS,
#     kwargs={'agent_velocity': 1.0, 'flocking': True},
# )
#
# register(
#     id='FlockingAOI125Env-v0',
#     entry_point='aoi_envs:MobileEnv',
#     max_episode_steps=MAX_EPISODE_STEPS,
#     kwargs={'agent_velocity': 1.25, 'flocking': True},
# )
#
# register(
#     id='FlockingAOI15Env-v0',
#     entry_point='aoi_envs:MobileEnv',
#     max_episode_steps=MAX_EPISODE_STEPS,
#     kwargs={'agent_velocity': 1.5, 'flocking': True},
# )
