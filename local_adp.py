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

        self.batch_size = 32
        self.update_freq = 60

    def batch_sample(self):
        count_num = len(self.replay_buffer)
        state = []
        action = []
        state_ = []
        reward = []

        if count_num >= self.batch_size:
            index = np.random.randint(0, count_num, self.batch_size)
            for i in iter(index):
                state.append(self.replay_buffer[i, :self.evn.state_dim])
                action.append(self.replay_buffer[i, self.evn.state_dim: self.evn.state_dim + self.evn.action_dim-1])
                state_.append(self.replay_buffer[i, -self.evn.state_dim:-1])
                reward.append(self.replay_buffer[i, -1:])

        if count_num < self.batch_size:
            for i in range(count_num):
                state.append(self.replay_buffer[i, :self.evn.state_dim])
                action.append(self.replay_buffer[i, self.evn.state_dim: self.evn.state_dim + self.evn.action_dim-1])
                state_.append(self.replay_buffer[i, -self.evn.state_dim:-1])
                reward.append(self.replay_buffer[i, -1:])

        state = Variable(t.tensor(state, dtype=t.float32, requires_grad=True).data)
        action = Variable(t.tensor(action, dtype=t.float32, requires_grad=True).data)
        state_ = Variable(t.tensor(state_, dtype=t.float32, requires_grad=True).data)
        reward = Variable(t.tensor(reward, dtype=t.float32, requires_grad=True).data)

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
        state, _, _, _ = self.batch_sample()
        state = Variable(t.tensor(state, dtype=t.float32, requires_grad=True))
        action = self.action_net(state)

        state_ = t.zeros(len(state), self.evn.state_dim)
        reward = t.zeros(len(state))

        for i in range(len(state)):
            self.evn.state = state[i]
            u = action[i]
            _, _, state_[i, :], reward[i, :] = self.evn.step(u)

        rate = self.update_rate(i)

        # calculate v target
        v_target_new = self.critic_target(Variable(t.tensor(state_, dtype=t.float32, requires_grad=True)))
        v_target_old = self.critic_target(Variable(t.tensor(state, dtype=t.float32, requires_grad=True)))
        v_target = (1 - rate) * v_target_old + rate * (reward + v_target_new)
        # calculate v target

        v_eval = self.critic_eval(Variable(t.tensor(state, dtype=t.float32, requires_grad=True)))
        v_loss = self.criterion_v(v_eval, v_target)
        self.optimizer_v.zero_grad()
        v_loss.backward(retain_graph=True)
        self.optimizer_v.step()

        

    def buffer_init(self):
        num = 21
        state = t.zeros([num*num, 2])
        for i in range(num):
            for j in range(num):
                state[i*num+j, :] = t.tensor([1-0.1*i, 1-0.1*j])



