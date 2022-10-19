import numpy as np
import torch as t
from torch.autograd import Variable


class Model(t.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.lay1 = t.nn.Linear(input_dim, 10, bias=True)
        self.lay1.weight.data.normal_(-0.1, 0.1)
        self.lay2 = t.nn.Linear(10, 10, bias=True)
        self.lay2.weight.data.normal_(-0.1, 0.1)
        self.lay3 = t.nn.Linear(10, output_dim, bias=True)
        self.lay3.weight.data.normal_(-0.1, 0.1)

    def forward(self, x):
        layer1 = self.lay1(x)
        layer1 = t.nn.functional.elu(layer1, alpha=1)
        layer2 = self.lay2(layer1)
        layer2 = t.nn.functional.elu(layer2, alpha=1)
        output = self.lay3(layer2)
        return output


class Evn(object):
    def __init__(self):
        self.state_dim = 2
        self.action_dim = 1
        self.state = t.tensor([0, 0], dtype=t.float32, requires_grad=True)
        self.action = t.tensor([0], dtype=t.float32, requires_grad=True)
        self.R_inver = 5 * t.eye(1)
        self.g_x_t = [0, 0.1]

    def step(self, u):
        u = t.tensor(u, dtype=t.float32, requires_grad=True)
        x1 = self.state[0]
        x2 = self.state[1]
        state = self.state
        state_ = t.tensor(([0.1 * x2 + x1, -0.49 * t.sin(x1) - 0.1 * 0.2 * x2 + x2])) + t.tensor([0, 0.1])*u
        reward = self.get_reward(self.state, u)
        self.state = state_
        return state, u, state_, reward

    @staticmethod
    def get_reward(x, u):
        # print(x, u)

        Q = t.tensor([[1, 0], [0, 1]], dtype=t.float32)
        R = 0.2 * t.eye(1)
        reward = t.matmul(t.matmul(x, Q), x) + t.matmul(t.matmul(u, R), u)
        return reward

    def reset(self):
        self.state = t.tensor([1, 1], dtype=t.float32, requires_grad=True)
        self.action = t.tensor([0], dtype=t.float32, requires_grad=True)


class LocalADP(object):
    def __init__(self):
        self.evn = Evn()

        learning_rate = 0.1

        self.critic_eval = Model(input_dim=self.evn.state_dim, output_dim=1)
        self.critic_target = Model(input_dim=self.evn.state_dim, output_dim=1)
        self.criterion_v = t.nn.MSELoss(reduction='mean')
        self.optimizer_v = t.optim.Adam(self.critic_eval.parameters(), lr=learning_rate)

        self.action_net = Model(input_dim=self.evn.state_dim, output_dim=self.evn.action_dim)
        self.criterion_a = t.nn.MSELoss(reduction='mean')
        self.optimizer_a = t.optim.Adam(self.action_net.parameters(), lr=learning_rate)

        self.replay_buffer = []
        self.loss = []

        self.batch_size = 3
        self.update_freq = 60

        self.buffer_init()
        print()

    def batch_sample(self):
        count_num = len(self.replay_buffer)

        state = t.zeros(self.batch_size, self.evn.state_dim)
        action = t.zeros(self.batch_size, self.evn.action_dim)
        state_ = t.zeros(self.batch_size, self.evn.state_dim)
        reward = t.zeros(self.batch_size, 1)

        if count_num >= self.batch_size:

            index = np.random.randint(0, count_num, self.batch_size)
            j = 0
            for i in iter(index):
                state[j, :] = (self.replay_buffer[i][0])
                action[j, :] = (self.replay_buffer[i][1])
                state_[j, :] = (self.replay_buffer[i][2])
                reward[j, :] = (self.replay_buffer[i][-1])
                j = j + 1

        if count_num < self.batch_size:
            state = t.zeros(count_num, self.evn.state_dim)
            action = t.zeros(count_num, self.evn.action_dim)
            state_ = t.zeros(count_num, self.evn.state_dim)
            reward = t.zeros(count_num, 1)
            t.zeros(count_num, self.evn.state_dim)
            for i in range(count_num):
                state[i, :] = (self.replay_buffer[i][0])
                action[i, :] = (self.replay_buffer[i][1])
                state_[i, :] = (self.replay_buffer[i][2])
                reward[i, :] = (self.replay_buffer[i][-1])

        state = Variable(t.tensor(state, dtype=t.float32, requires_grad=True))
        action = Variable(t.tensor(action, dtype=t.float32, requires_grad=True))
        state_ = Variable(t.tensor(state_, dtype=t.float32, requires_grad=True))
        reward = Variable(t.tensor(reward, dtype=t.float32, requires_grad=True))
        print(state)
        print(action)
        print(state_)
        print(reward)
        return state, action, state_, reward

    @staticmethod
    def update_rate(i):
        rate = 0.39 * np.sin(i + 1) + 0.6
        return rate

    # def choose_action(self, state):
    #     state = Variable(t.tensor(state, dtype=t.float32, requires_grad=True))
    #     action = self.action_net(state)
    #     return action

    def train(self, i):
        data = self.batch_sample()
        state = data[0]

        action = self.action_net(state)
        print(action, 'action')

        state_ = t.zeros(len(state), self.evn.state_dim)
        reward = t.zeros(len(state), 1)

        for i in range(len(state)):
            self.evn.state = state[i]
            u = action[i]
            data = self.evn.step(u)
            # _, _, state_[i, :], reward[i, :] = self.evn.step(u)
            state_[i, :] = data[2]
            reward[i, :] = data[3]

        rate = self.update_rate(i)

        # calculate v target
        v_target_new = self.critic_target(state)
        v_target_old = self.critic_target(state)
        v_target = (1 - rate) * v_target_old + rate * (reward + v_target_new)
        # calculate v target

        # update v net
        # state = t.tensor(state, dtype=t.float32, requires_grad=True)
        print(state.shape[0], 'state size')
        v_eval = self.critic_eval(state)
        v_loss = self.criterion_v(v_eval, v_target)
        self.optimizer_v.zero_grad()
        v_loss.backward(t.rand(state.shape[0], 2))
        self.optimizer_v.step()
        # update v net

        # update u net
        v_eval.backward(t.rand(state.shape[0], 2))
        v_grad = state.grad
        u_target = -0.5 * self.evn.R_inver * self.evn.g_x_t * v_grad
        print('v_grad', v_grad)
        print('u_target', u_target)
        # update u net

    def buffer_init(self):
        num = 21
        state = []
        action = []
        state_ = []
        reward = []
        for i in range(num):
            for j in range(num):
                state = t.tensor([1-0.1*i, 1-0.1*j])
                action = t.tensor([0], dtype=t.float32, requires_grad=True)
                self.evn.state = state
                _, _, state_, reward = self.evn.step(u=action)
                self.replay_buffer.s
        print(self.replay_buffer)

adp = LocalADP()
state = t.tensor([[1, 2],[3, 4]], dtype=t.float32)
adp.train(1)
