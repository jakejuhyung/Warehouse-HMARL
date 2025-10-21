import gymnasium as gym
import numpy as np
from gymnasium import spaces
from environments.worker_nav_env import WorkerNavEnv

class WarehouseManagerEnv(gym.Env):
    """
    High-level Manager environment coordinating multiple pre-trained WorkerNavEnvs.

    - Manager issues sub-goals (grid positions) to each worker.
    - Workers automatically navigate to those sub-goals using their frozen PPO policy.
    - Manager receives reward for completing all deliveries efficiently.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, num_workers=2, grid_size=10, episode_len=50, worker_policy=None, seed=None):
        super().__init__()
        self.num_workers = num_workers
        self.grid_size = grid_size
        self.episode_len = episode_len
        self.rng = np.random.default_rng(seed)
        self.worker_policy = worker_policy  # frozen PPO worker policy (optional)

        # Create N worker environments (each is goal-conditioned)
        self.workers = [WorkerNavEnv(grid_size=grid_size, max_steps=20) for _ in range(num_workers)]

        # Observation (for manager):
        # includes all workers' positions + all package goal positions
        obs_size = num_workers * 4  # each worker sees [worker_x, worker_y, goal_x, goal_y]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # Manager's action = assign a new sub-goal (x, y) for each worker
        # Each worker gets a 2D target on grid
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(num_workers * 2,), dtype=np.float32
        )

        # Track time
        self.timestep = 0

    def reset(self, seed=None, options=None):
        self.timestep = 0
        obs_list = []
        self.worker_goals = []

        for w in self.workers:
            obs, _ = w.reset()
            obs_list.append(obs)
            self.worker_goals.append(w.goal)

        return np.concatenate(obs_list).astype(np.float32), {}

    def step(self, action):
        """
        Manager proposes new sub-goals for workers (scaled 0–1 → grid coords).
        Each worker executes one rollout step towards its goal.
        """
        self.timestep += 1

        # Convert manager's normalized action into worker goal positions
        actions_reshaped = action.reshape(self.num_workers, 2)
        rewards, obs_list = [], []

        for i, worker in enumerate(self.workers):
            new_goal = (actions_reshaped[i] * (self.grid_size - 1)).astype(int)
            worker.goal = np.clip(new_goal, 0, self.grid_size - 1)

            # Let worker take 1 navigation step
            if self.worker_policy:
                # use frozen policy for worker
                obs = worker._obs()
                act, _ = self.worker_policy.predict(obs, deterministic=True)
            else:
                act = worker.action_space.sample()

            obs, r, done, trunc, _ = worker.step(act)
            obs_list.append(obs)
            rewards.append(r)

        # Manager’s reward: mean of worker progress - penalty for time
        manager_reward = np.mean(rewards) - 0.01
        done = self.timestep >= self.episode_len
        trunc = False

        return np.concatenate(obs_list).astype(np.float32), manager_reward, done, trunc, {}

    def render(self):
        """Simple text render"""
        print(f"\nTime step {self.timestep}")
        for i, w in enumerate(self.workers):
            print(f"Worker {i}: pos={w.pos}, goal={w.goal}")
