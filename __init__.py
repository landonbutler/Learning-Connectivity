from gym.envs.registration import register

register(
    id='StationaryEnv-v0',
    entry_point='aoi_multi_agent_swarm.envs:StationaryEnv',
    max_episode_steps=100,
)