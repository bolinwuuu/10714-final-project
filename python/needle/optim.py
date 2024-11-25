"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p.grad is None:
                continue
            if id(p) not in self.u:
                self.u[id(p)] = ndl.init.zeros_like(p.grad)
            grad = p.grad
            if self.weight_decay > 0:
                grad = grad + ndl.ops.mul_scalar(p.data, self.weight_decay)
            self.u[id(p)] = ndl.ops.mul_scalar(self.u[id(p)], self.momentum) + \
                              ndl.ops.mul_scalar(grad, 1 - self.momentum)
            p.data = p.data - ndl.ops.mul_scalar(self.u[id(p)], self.lr)
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        
        for p in self.params:
            if p.grad is None:
                continue
            if id(p) not in self.m:
                self.m[id(p)] = ndl.init.zeros_like(p.grad)
            if id(p) not in self.v:
                self.v[id(p)] = ndl.init.zeros_like(p.grad)

            grad = p.grad
            if self.weight_decay > 0:
                grad = ndl.add(grad, ndl.ops.mul_scalar(p.data, self.weight_decay))

            self.m[id(p)] = ndl.ops.mul_scalar(self.m[id(p)].detach(), self.beta1) + \
                              ndl.ops.mul_scalar(grad.detach(), 1 - self.beta1)
            self.v[id(p)] = ndl.ops.mul_scalar(self.v[id(p)].detach(), self.beta2) + \
                              ndl.ops.mul_scalar(ndl.ops.power_scalar(grad.detach(), 2), 1 - self.beta2)

            m_correct = ndl.ops.divide_scalar(self.m[id(p)], 1 - self.beta1**self.t)
            v_correct = ndl.ops.divide_scalar(self.v[id(p)], 1 - self.beta2**self.t)

            denom = ndl.ops.add_scalar(ndl.ops.power_scalar(v_correct, 0.5), self.eps)
            decrease = ndl.ops.mul_scalar(ndl.ops.divide(m_correct, denom), self.lr)
            p.data = p.data - decrease.detach()
        ### END YOUR SOLUTION