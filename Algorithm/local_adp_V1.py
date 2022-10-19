import numpy as np
import torch as t
from torch.autograd import Variable
from Common import *
import matplotlib.pyplot as plt


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
        self.R_inver = t.tensor([5])
        self.g_x_t = t.tensor([0, 0.1])

    def step(self, u):
        u = t.tensor([u], dtype=t.float32, requires_grad=True)
        x1 = self.state[0]
        x2 = self.state[1]
        s = self.state
        s_ = t.tensor(([0.1 * x2 + x1, -0.49 * t.sin(x1) - 0.1 * 0.2 * x2 + x2])) + t.tensor([0, 0.1])*u
        reward = self.get_reward(self.state, u)
        self.state = s_
        return s, u, s_, reward

    @staticmethod
    def get_reward(x, u):
        u = t.tensor([u], dtype=t.float32, requires_grad=True)
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

        learning_rate = 0.01

        self.critic_eval = Model(input_dim=self.evn.state_dim, output_dim=1)
        self.critic_target = Model(input_dim=self.evn.state_dim, output_dim=1)
        self.criterion_v = t.nn.MSELoss(reduction='mean')
        self.optimizer_v = t.optim.Adam(self.critic_eval.parameters(), lr=learning_rate)

        self.action_net = Model(input_dim=self.evn.state_dim, output_dim=self.evn.action_dim)
        self.criterion_a = t.nn.MSELoss(reduction='mean')
        self.optimizer_a = t.optim.Adam(self.action_net.parameters(), lr=learning_rate)

        self.a_loss = []
        self.v_loss = []
        self.s_init = []

        self.step_count = 0
        self.tau = 0.1

        # self.update_freq = 60
        self.batch_size = 32

        self.replay_buffer = ReplayBuffer(max_size=21*21,
                                          batch_size=self.batch_size,
                                          state_dim=self.evn.state_dim,
                                          action_dim=self.evn.action_dim)
        self.buffer_init()

    def train(self, train_index: float):
        self.step_count += 1
        # 1st Step: get data #
        s, a, r, s_ = self.replay_buffer.sample_buffer(is_reward_ascent=False)
        # print(s, a, s_, r)
        s = t.tensor(s, dtype=t.float32, requires_grad=True)
        a = t.tensor(a, dtype=t.float32, requires_grad=True)
        s_ = t.tensor(s_, dtype=t.float32, requires_grad=True)
        r = t.tensor(r, dtype=t.float32, requires_grad=True)
        # 1st Step: get data #

        # calculate a target #
        v_eval = self.critic_eval(s)
        v_eval.backward(t.rand(s.shape[0], 1), retain_graph=True)
        v_grad = s.grad
        a_target = -0.5 * self.evn.R_inver * self.evn.g_x_t * v_grad
        # calculate a target #

        # update a net #
        a_loss = self.criterion_a(a, a_target)
        self.optimizer_a.zero_grad()
        a_loss.backward()
        self.optimizer_a.step()
        self.a_loss.append(float(a_loss.mean(dim=0)))
        # update a net #

        # 2nd Step: calculate V target#
        rate = self.update_rate(train_index)
        v = self.critic_target(s)
        v_ = self.critic_target(s_)
        v_target = (1 - rate) * v + rate * (r + v_)
        # 2nd Step: calculate V target#

        # update v net #
        v_loss = self.criterion_v(v_eval, v_target)
        self.optimizer_v.zero_grad()
        v_loss.backward(retain_graph=True)
        self.optimizer_v.step()
        self.v_loss.append(float(v_loss.mean(dim=0)))
        # update v net #

        self.update_network_parameter()

    def update_network_parameter(self):
        for target_par, par in zip(self.critic_target.parameters(), self.critic_eval.parameters()):
            target_par.data.copy_(target_par.data * (1.0 - self.tau) + par.data * self.tau)

    @staticmethod
    def update_rate(index):
        rate = 0.39 * np.sin(index + 1) + 0.6
        return rate

    def buffer_init(self):
        self.step_count = 0
        num = 21
        for i in range(num):
            for j in range(num):
                s = t.tensor([1-0.1*i, 1-0.1*j])
                u = t.tensor([0], dtype=t.float32, requires_grad=True)
                self.evn.state = s
                s, u, s_, r = self.evn.step(u=u)
                s = s.detach().numpy()
                u = u.detach().numpy()
                s_ = s_.detach().numpy()
                r = r.detach().numpy()
                self.replay_buffer.store_transition(state=s, action=u, state_=s_, reward=r)
                self.s_init.append(s)

    def buffer_update(self):
        for i in range(len(self.s_init)):
            s = t.tensor(self.s_init[i])
            u = self.action_net(s)
            self.evn.state = s
            s, u, s_, r = self.evn.step(u=u)
            s = s.detach().numpy()
            u = u.detach().numpy()
            s_ = s_.detach().numpy()
            r = r.detach().numpy()
            self.replay_buffer.store_transition(state=s, action=u, state_=s_, reward=r)


adp = LocalADP()
for i in range(100):
    print('the %d step'%i)
    for j in range(10):
        adp.train(i)
    adp.buffer_update()
print(adp.v_loss)
# print(adp.a_loss)

fig1 = plt.figure(1)
plt.plot(adp.v_loss)
plt.show()