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
        self.state = [0, 0]
        self.action = [0]
        self.data_buffer = []

    def step(self, u):
        x1 = self.state[0]
        x2 = self.state[1]
        state = [x1, x2]

        self.action = u
        state_ = np.array([0.1*x2 + x1, -0.49*np.sin(x1) - 0.1*0.2*x2 + x2]).T + np.array([0, 0.1]).T.dot(u)
        reward = self.get_reward()
        self.state = state_
        self.data_buffer.append(np.hstack([state, u, state_, reward]))
        return np.hstack([state, u, state_, reward])

    def reset(self):
        self.state = [0, 0]
        self.action = [0]
        self.data_buffer = []

    @staticmethod
    def get_reward(x, u):
        Q = np.eye(2)
        R = 0.2 * np.eye(1)
        reward = x.T.dot(Q).dot(x) + u.T.dot(R).dot(u)
        return reward


class LocalADP(object):
    def __init__(self):
        self.evn = Evn()

        learning_rate = 0.1

        self.critic_eval = Model(input_dim=self.evn.state_dim, output_dim=1)
        self.critic_target = Model(input_dim=self.evn.state_dim, output_dim=1)
        self.criterion_v = t.nn.MSELoss(reduction='mean')
        optimizer_v = t.optim.Adam(self.critic_eval.parameters(), lr=learning_rate)

        self.action_net = Model(input_dim=self.evn.state_dim, output_dim=self.evn.action_dim)
        criterion_a = t.nn.MSELoss(reduction='mean')
        optimizer_a = t.optim.Adam(self.action_net.parameters(), lr=learning_rate)



    def batch_sample(self):
        pass
