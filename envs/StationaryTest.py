import unittest
from Stationary import StationaryEnv

class StationaryTest(unittest.TestCase):
    def setUp(self):
        self.stat = StationaryEnv()

    def test_num_agents(self):
        self.assertEqual(100, self.stat.n_agents,'incorrect number of agents')

    def test_reset(self):
        self.assertEqual(self.stat.seed(), self.stat.seed(),'incorrect number of agents')

if __name__ == '__main__':
    unittest.main()