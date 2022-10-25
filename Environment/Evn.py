import numpy as np
import torch as t
from torch.autograd import Variable
from Common import *
import matplotlib.pyplot as plt


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