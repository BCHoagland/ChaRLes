import torch
import numpy as np
from charles.env import *
from charles.storage import *
from charles.visualize import *

class Agent:
    def __init__(self, algo, config):
        self.env = Env(config.env, config.actors)
        self.visualizer = Visualizer(config.env)
        self.config = config
        self.storage = Storage(config)

        self.algo = algo()
        self.algo.agent = self
        try:
            self.algo.env_wrappers
        except:
            pass
        else:
            for env_wrapper in self.algo.env_wrappers:
                self.env = env_wrapper(self.env)

        self.visualizer.reset_data_for_algo(self.algo.name)

        try:
            self.vis_title = config.vis_title
        except:
            self.vis_title = None

    def random_action(self):
        return np.stack([self.env.action_space.sample() for _ in range(self.config.actors)])

    def noisy_action(self, a, std, clip=None):
        noise = np.random.normal(0, std)
        if clip is not None:
            noise = np.clip(noise, -clip, clip)
        return np.clip(a.numpy() + noise, self.env.action_space.low, self.env.action_space.high)

    def explore(self):
        s = self.env.reset()
        T = int(self.config.explore_steps)
        for step in range(T):
            if step % self.config.vis_iter == self.config.vis_iter - 1:
                progress(step, T, 'Exploring')
            a = self.random_action()
            s2, r, done, _ = self.env.explore_step(a)
            self.storage.store((s, a, r, s2, done))
            s = s2

    def train(self):
        self.algo.setup()

        mean_r = np.zeros(self.config.actors)

        total_timesteps = 0
        progress(total_timesteps - 1, self.config.max_timesteps, 'Training')

        ep_reward = np.zeros(self.config.actors)
        final_ep_reward = np.zeros(self.config.actors)

        s = self.env.reset()
        while total_timesteps < self.config.max_timesteps:

            # collect trajectory
            for t in range(int(self.config.trajectory_length)):

                # collect transition
                with torch.no_grad():
                    s2, r, done, data = self.algo.interact(s)
                self.storage.store(data)
                s = s2

                # update stored rewards
                mean_r += (1 / (total_timesteps + 1)) * (r - mean_r)

                ep_reward += r
                mask = 1 - done
                final_ep_reward *= mask
                final_ep_reward += (1 - mask) * ep_reward
                ep_reward *= mask

                # visualize progress occasionally
                total_timesteps += 1
                if total_timesteps % self.config.vis_iter == 0:
                    progress(total_timesteps - 1, self.config.max_timesteps, 'Training')
                    self.visualizer.plot(self.algo.name, 'Episodic Reward', 'Timesteps', total_timesteps, final_ep_reward, self.algo.color)
                    # self.visualizer.plot(self.algo.name, 'Mean Reward', 'Timesteps', total_timesteps, mean_r, self.algo.color, title=self.vis_title)
                    # self.visualizer.plot(self.algo.name, 'Instances', 'Timesteps', total_timesteps, data[1][0][-1], self.algo.color, title='Num Instances')
                    # self.visualizer.plot(self.algo.name, 'Requests', 'Timesteps', total_timesteps, data[0][0][-1], self.algo.color, title='Num Active Requests')

            # run updates after trajectory has been collected
            for _ in range(self.config.epochs):
                self.algo.update(self.storage)
            if self.algo.type == 'on-policy':
                self.storage.clear()

        if total_timesteps % self.config.vis_iter != 0:
            progress(total_timesteps - 1, int(self.config.max_timesteps), 'Training')

    def demo(self):
        ep_reward = np.zeros(self.config.actors)
        final_ep_reward = np.zeros(self.config.actors)

        T = self.config.testing_steps

        s = self.env.reset()
        for t in range(T):
            if t % (T / 20) == (T / 20) - 1:
                progress(t, T, 'Testing')

            with torch.no_grad():
                s, r, done, _ = self.algo.interact(s)

            ep_reward += r
            mask = 1 - done
            final_ep_reward *= mask
            final_ep_reward += (1 - mask) * ep_reward
            ep_reward *= mask

            self.visualizer.plot(self.algo.name, 'Episodic Reward', 'Timesteps', t, final_ep_reward, self.algo.color, title='Testing')