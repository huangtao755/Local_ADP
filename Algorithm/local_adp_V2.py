import numpy as np
import torch as t
import copy
from torch.autograd import Variable
from Common import *
import matplotlib.pyplot as plt

from Environment import Evn

t.manual_seed(2)


class Model(t.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.lay1 = t.nn.Linear(input_dim, 32, bias=True)
        self.lay1.weight.data.normal_(0, 0)
        # self.lay2 = t.nn.Linear(10, 10, bias=True)
        # self.lay2.weight.data.normal_(0, 0)
        self.lay3 = t.nn.Linear(32, output_dim, bias=False)
        self.lay3.weight.data.normal_(0, 0)

    def forward(self, x):
        layer1 = self.lay1(x)
        layer1 = t.nn.functional.elu(layer1, alpha=1)
        # layer2 = self.lay2(layer1)
        # layer2 = t.nn.functional.elu(layer2, alpha=1)
        output = self.lay3(layer1)
        return output


class LocalADP(object):
    def __init__(self):
        self.evn = Evn.Evn()

        learning_rate = 0.005
        learning_rate_a = 0.01

        self.critic_eval = Model(input_dim=self.evn.state_dim, output_dim=1)
        self.critic_target = Model(input_dim=self.evn.state_dim, output_dim=1)
        self.criterion_v = t.nn.MSELoss(reduction='mean')
        self.optimizer_v = t.optim.Adam(self.critic_eval.parameters(), lr=learning_rate)

        self.action_net = Model(input_dim=self.evn.state_dim, output_dim=self.evn.action_dim)
        self.criterion_a = t.nn.MSELoss(reduction='mean')
        self.optimizer_a = t.optim.Adam(self.action_net.parameters(), lr=learning_rate_a)

        self.u_loss = []
        self.v_loss = []
        self.s_init = []

        self.step_count = 0
        self.tau = 0.1

        # self.update_freq = 60
        self.batch_size = 25

        self.replay_buffer = ReplayBuffer(max_size=21*21,
                                          batch_size=self.batch_size,
                                          state_dim=self.evn.state_dim,
                                          action_dim=self.evn.action_dim)
        self.buffer_init()

    def train(self, train_index: float):
        self.step_count += 1

        # 1st Step: get data #
        s, a, r, s_ = self.replay_buffer.sample_buffer(is_reward_ascent=False)
        s = t.tensor(s, dtype=t.float32, requires_grad=True)
        a = t.tensor(a, dtype=t.float32, requires_grad=True)
        s_ = t.tensor(s_, dtype=t.float32, requires_grad=True)
        r = t.tensor(r, dtype=t.float32, requires_grad=True)
        # 1st Step: get data #

        for i in range(10):
            # calculate a target #
            v_eval = self.critic_target(s_)
            v_eval.backward(t.rand(s.shape[0], 1), retain_graph=True)
            v_grad = s_.grad
            # print('v_grad', v_grad)
            a_target = -0.5 * self.evn.R_inver * self.evn.g_x_t * v_grad
            # print(self.evn.g_x_t * v_grad)
            # calculate a target #

            # update a net #
            a_loss = self.criterion_a(a, a_target)
            self.optimizer_a.zero_grad()
            a_loss.backward()
            self.optimizer_a.step()
            self.u_loss.append(float(a_loss.mean(dim=0)))
        # update a net #

        for i in range(1000):
            # 2nd Step: calculate V target#
            rate = self.update_rate(train_index)
            v = self.critic_target(s)
            v_ = self.critic_target(s_)
            v_target = (1 - rate) * v + rate * (r + v_)
            # 2nd Step: calculate V target#

            # update v net #
            v_eval = self.critic_eval(s)
            v_loss = self.criterion_v(v_eval, v_target)
            self.optimizer_v.zero_grad()
            v_loss.backward()
            self.optimizer_v.step()

            self.v_loss.append(float(v_loss.mean(dim=0)))
            # update v net #

            self.network_parameter_update()

    def v_net_update(self):
        # 1st Step: get data #
        s, u, r, s_ = self.replay_buffer.sample_buffer(is_reward_ascent=False)
        s = t.tensor(s, dtype=t.float32, requires_grad=True)
        a = t.tensor(u, dtype=t.float32, requires_grad=True)
        s_ = t.tensor(s_, dtype=t.float32, requires_grad=True)
        r = t.tensor(r, dtype=t.float32, requires_grad=True)
        # 1st Step: get data #

        # 2nd Step: calculate V target#
        v_next = self.critic_eval(s_)
        v_target = r + 0.99 * v_next
        v_eval = self.critic_eval(s)
        # 2nd Step: calculate V target#

        # 3rd Step: calculate V loss #
        v_loss = self.criterion_v(v_eval, v_target)
        self.optimizer_v.zero_grad()
        v_loss.backward()
        self.optimizer_v.step()
        # 3rd Step: calculate V loss #
        # self.network_parameter_update()
        self.v_loss.append(v_loss.detach().numpy())

    def a_net_update(self):
        s, u, r, s_ = self.replay_buffer.sample_buffer(is_reward_ascent=False)
        s = t.tensor(s, dtype=t.float32, requires_grad=True)
        u = self.action_net(s)
        print(u[3])

        s_ = t.zeros(s.shape)
        for index in range(len(s)):
            self.evn.state = s[index]
            state, a, state_, r = self.evn.step(u=u[index])
            s_[index, :] = state_

        v_next_eval = self.critic_eval(s_)
        u_statr = - self.evn.R_inver * self.evn.g_x_t * v_next_eval / 2

        u_loss = self.criterion_a(u, u_statr)
        self.optimizer_a.zero_grad()
        u_loss.backward()
        self.optimizer_a.step()
        self.u_loss.append(u_loss.detach().numpy())

    def network_parameter_update(self):
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

    def simulate(self):
        self.evn.state = t.tensor([-1, -1], dtype=t.float32)
        state = []
        for i in range(100):
            u = self.action_net(self.evn.state)
            s, u, s_, r = self.evn.step(u)
            state.append(self.evn.state.detach().numpy())
        return state


adp = LocalADP()
for i in range(1):
    for j in range(5000):
        adp.a_net_update()
        if adp.u_loss[-1] < 0.01:
            break
    for j in range(20000):
        adp.v_net_update()
        if adp.v_loss[-1] < 0.01:
            break
v = []
for i in iter(adp.s_init):
    v.append(adp.critic_eval(t.tensor(i)).detach().numpy())

fig1 = plt.figure(1)
plt.plot(v)
r = adp.replay_buffer.r_mem
fig2 = plt.figure(2)
plt.plot(r)
fig3 = plt.figure(3)
plt.plot(adp.v_loss)
fig4 = plt.figure(4)
plt.plot(adp.u_loss)
plt.show()
