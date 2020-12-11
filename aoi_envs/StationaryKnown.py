from aoi_envs.MultiAgent import MultiAgentEnv
import numpy as np

class StationaryKnownEnv(MultiAgentEnv):

    def __init__(self):
        super().__init__()
        self.known_initial_positions = True