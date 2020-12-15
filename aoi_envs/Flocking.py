from aoi_envs.Mobile import MobileEnv
import numpy as np


class FlockingEnv(MobileEnv):

    def __init__(self):
        super().__init__()
        self.flocking = True
        self.biased_velocities = False