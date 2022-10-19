import random

import numpy as np
import torch as t
from torch.autograd import Variable

M = 1/3
l = 2/3

x = np.array([1, -1]).T


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


def evn(x, uk):
    uk = float(uk)
    [x1, x2] = x
    x1 = float(x1)
    x2 = float(x2)
    state_ = np.array([0.1*x2 + x1, -0.49*np.sin(x1) - 0.1*0.2*x2 + x2]).T + np.array([0, 0.1]).T.dot(uk)
    return state_


def reward(x, u):
    # P = np.array([[10.3, -7.16], [-7.16, 5.74]])
    # U0 = x.T.dot(P).dot(x)
    x = np.array(x)
    u = np.array(u)
    Q = np.eye(2)
    R = 0.2 * np.eye(1)
    U = x.T.dot(Q).dot(x) + u.T.dot(R).dot(u)
    return U


def reward_init(x):
    x = np.array(x)
    P = np.array([[0, 0], [0, 0]])
    U0 = x.T.dot(P).dot(x)
    return U0

def rate_a(i):
    return 0.39*np.sin(1+i) + 0.6


def learn():
    state_dim = 2
    learning_rate = 0.01
    v_eval = Model(input_dim=state_dim, output_dim=1)
    v_target = Model(input_dim=state_dim, output_dim=1)
    criterion_v = t.nn.MSELoss(reduction='mean')
    optimizer_v = t.optim.Adam(v_eval.parameters(),
                                        lr=learning_rate)
    u_eval = Model(input_dim=state_dim, output_dim=1)
    critic_target = Model(input_dim=state_dim, output_dim=1)
    criterion_u = t.nn.MSELoss(reduction='mean')
    optimizer_u = t.optim.Adam(u_eval.parameters(),
                                        lr=learning_rate)

    batch = 1
    update_targetNet_freq = 10

    x = []

    for i in range(21):
        for j in range(21):
            x.append([1 - i*0.1, 1 - j*0.1])
    data_size = len(x) - 1

    V0 = []
    for i in iter(x):
        V0.append(reward_init(i))

    x_ = []
    for i in iter(x):
        x_.append(evn(i, 0))
    print(V0)
    print(x_)

    v1 = []
    for i in iter(x):
        print(i)
        v1.append(reward(i, 0))
    print(v1)

    V1 = []
    rate = rate_a(1)
    for i in range(len(v1)):
        V1.append(v1[i]*rate + V0[i]*(1-rate))

    for j in range(5000):

        index = np.random.randint(0, data_size, 32)
        state = []
        V_s = []

        for i in iter(index):
            state.append(x[i])
            V_s.append(V0[i])

        state = t.tensor(state, dtype=t.float32, requires_grad=True)
        V_eval = v_eval(Variable(state))

        u = u_eval(Variable(state))

        state_ = []

        for i in range(len(state)):
            state_.append(evn(state[i], u[i]))
        state_ = t.tensor(state, dtype=t.float32, requires_grad=True)

        V_next = v_target(Variable(state_))
        rate = rate_a(j)

        v1 = []
        V_target = []

        for i in range(len(state)):
            r = reward(state[i], u[i])
            v1.append(r + V_next[i])
            V_target.append((1-rate)*v_eval[i]+rate*v1[i])

        # TD_err = v1 - V_eval

        u_loss = criterion_u(v1, V_eval)
        optimizer_u.zero_grad()
        u_loss.backward(retain_graph=True)
        optimizer_u.step()

        v_loss = criterion_v(V_eval, V_target)
        optimizer_v.zero_grad()
        v_loss.backward(retain_graph=True)
        optimizer_v.step()

        print(v_loss)


t = np.arange(0, 10000, 1)
a = 0.9 ** (40000/(-200-t))
print(a)
