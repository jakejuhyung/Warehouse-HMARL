import numpy as np
import gymnasium as gym
from gymnasium import spaces

class WorkerNavEnv(gym.Env):
    """
    Single worker on an NxN grid. Goal-conditioned.
    Observation: [worker_x, worker_y, goal_x, goal_y] (normalized to [0,1])
    Actions: 0=stay, 1=up, 2=down, 3=left, 4=right
    Episode ends when worker reaches goal or max_steps.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=10, max_steps=100, seed=None):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # Observation space: normalized positions (worker + goal)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        # Action space: discrete movement commands
        self.action_space = spaces.Discrete(5)

        self.pos = None
        self.goal = None
        self.steps = 0

    def _sample_pos(self):
        """Random position within grid"""
        return np.array(
            [self.rng.integers(0, self.grid_size),
             self.rng.integers(0, self.grid_size)],
            dtype=np.int64
        )

    def _obs(self):
        """Return normalized observation"""
        return np.array([
            self.pos[0] / (self.grid_size - 1),
            self.pos[1] / (self.grid_size - 1),
            self.goal[0] / (self.grid_size - 1),
            self.goal[1] / (self.grid_size - 1)
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Start a new episode"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos = self._sample_pos()
        self.goal = self._sample_pos()
        while np.all(self.pos == self.goal):
            self.goal = self._sample_pos()
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        """Move the worker according to action"""
        self.steps += 1

        # Movement
        if action == 1:   # up
            self.pos[1] = min(self.pos[1] + 1, self.grid_size - 1)
        elif action == 2: # down
            self.pos[1] = max(self.pos[1] - 1, 0)
        elif action == 3: # left
            self.pos[0] = max(self.pos[0] - 1, 0)
        elif action == 4: # right
            self.pos[0] = min(self.pos[0] + 1, self.grid_size - 1)

        # Check termination
        done = np.all(self.pos == self.goal)
        truncated = self.steps >= self.max_steps

        # Reward shaping
        reward = -0.01  # small time penalty
        if done:
            reward += 1.0
        elif truncated:
            reward -= 0.5

        return self._obs(), reward, done, truncated, {}

    def render(self):
        """Simple print render"""
        print(f"Worker position={self.pos}, Goal={self.goal}")
