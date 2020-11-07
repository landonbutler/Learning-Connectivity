import unittest
import numpy as np
from Stationary import StationaryEnv

class StationaryTest(unittest.TestCase):
    def setUp(self):
        self.stat = StationaryEnv()
        self.stat.reset()

    def test_reset_and_render(self):
        print("attempting reset")
        self.stat.reset()
        self.stat.render()

    def test_interference_1(self):
        attempt_comm = self.generate_attempted_communications(1, 15)
        succ_comm = self.stat.interference(attempt_comm)
        print(succ_comm)

    def test_interference_k(self):
        attempt_comm = self.generate_attempted_communications(int(self.stat.n_agents / 2), 15)
        succ_comm = self.stat.interference(attempt_comm)
        print(succ_comm)

    def test_update_buffers(self):
        attempt_comm = self.generate_attempted_communications(int(self.stat.n_agents / 2), 15)
        print(attempt_comm)
        self.stat.update_buffers(attempt_comm)
        print(self.stat.network_buffer)

    def test_multiple_steps_without_interference(self):
        for i in range(5):
            attempt_comm = self.generate_attempted_communications(int(self.stat.n_agents / 2), 15)
            self.stat.is_interference = False
            observation, reward, done, info = self.stat.step(attempt_comm)
        print(observation)

    def generate_attempted_communications(self, trans_per_agent, max_trans_pow):
        comm_mat = np.zeros((self.stat.n_agents, self.stat.n_agents))
        
        rand_trans_pow = np.random.rand(self.stat.n_agents,self.stat.n_agents) * max_trans_pow
        for i in range(self.stat.n_agents):
            choice_comm = np.random.choice(self.stat.n_agents, trans_per_agent, replace=False)
            comm_mat[i,choice_comm] = 1
        return np.multiply(comm_mat, rand_trans_pow)

if __name__ == '__main__':
    unittest.main()