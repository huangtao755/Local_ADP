import random
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, batch_size, state_dim, action_dim):
        self.mem_size = max_size
        self.mem_counter = 0
        self.batch_size = batch_size
        self.s_mem = np.zeros((self.mem_size, state_dim))
        self._s_mem = np.zeros((self.mem_size, state_dim))
        self.a_mem = np.zeros((self.mem_size, action_dim))
        self.r_mem = np.zeros(self.mem_size)
        self.sorted_index = []
        self.resort_count = 0

    def store_transition(self, state, action, reward, state_):
        index = self.mem_counter % self.mem_size
        self.s_mem[index] = state
        self.a_mem[index] = action
        self.r_mem[index] = reward
        self._s_mem[index] = state_
        self.mem_counter += 1

    def get_reward_sort(self):
        """
        :return:        根据奖励大小得到所有数据的索引值，升序，即从小到大
        """
        print('...sorting...')
        self.sorted_index = sorted(range(min(self.mem_counter, self.mem_size)), key=lambda k: self.r_mem[k], reverse=False)

    def store_transition_per_episode(self, states, actions, rewards, states_):
        self.resort_count += 1
        num = len(states)
        for i in range(num):
            self.store_transition(states[i], actions[i], rewards[i], states_[i])

    def sample_buffer(self, is_reward_ascent=True):
        max_mem = min(self.mem_counter, self.mem_size)
        if is_reward_ascent:
            batchNum = min(int(0.25 * max_mem), self.batch_size)
            batch = random.sample(self.sorted_index[-int(0.25 * max_mem):], batchNum)
        else:
            batch = np.random.choice(max_mem, self.batch_size)
        states = self.s_mem[batch]
        actions = self.a_mem[batch]
        rewards = self.r_mem[batch]
        actions_ = self._s_mem[batch]

        return states, actions, rewards, actions_
