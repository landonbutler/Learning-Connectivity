from aoi_envs.Mobile import MobileEnv
import numpy as np


class FlockingEnv(MobileEnv):

    def __init__(self):
        super().__init__(agent_velocity=3.0)
        self.flocking = True
        self.biased_velocities = False