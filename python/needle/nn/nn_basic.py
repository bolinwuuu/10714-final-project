"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, shape=(in_features, out_features), device=device, dtype=dtype))
        self.bias = Parameter(ops.transpose(init.kaiming_uniform(out_features, 1, shape=(out_features, 1), device=device, dtype=dtype))) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = ops.matmul(X, self.weight)
        if self.bias:
            b = ops.broadcast_to(self.bias, y.shape)
            # print(f'{X.shape=}, {self.weight.shape=}, {y.shape=}, {self.bias.shape=}, {b.shape=}')
            y = ops.add(y, b)
        return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        flat_dim = 1
        for s in X.shape[1:]:
            flat_dim *= s
        return ops.reshape(X, (batch_size, flat_dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        res = ops.relu(x)
        return res
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = x
        for module in self.modules:
            y = module(y)
        return y
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        lse = ops.logsumexp(logits, axes=(1,))
        N, K = logits.shape[0], logits.shape[1]
        Y = init.one_hot(K, y, device=logits.device)
        z_y = ops.summation(Y * logits / N)
        res = (ops.summation(lse) / logits.shape[0]) - z_y

        return res
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        self.training = True
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            N = x.shape[0]
            x_mean_batch = (x.sum((0,)) / N)
            x_mean = x_mean_batch.reshape((1, x.shape[1])).broadcast_to(x.shape)
            x_centered = x - x_mean
            x_var_batch = ((x_centered**2).sum((0,)) / N)
            x_var = (x_var_batch + self.eps).reshape((1, x.shape[1])).broadcast_to(x.shape)
            x_norm = x_centered / (x_var**0.5)

            w = self.weight.reshape((1, x.shape[1])).broadcast_to(x_norm.shape)
            b = self.bias.reshape((1, x.shape[1])).broadcast_to(x_norm.shape)
            y = ops.add(ops.multiply(w, x_norm), b)
            self.running_mean = ops.add(ops.mul_scalar(self.running_mean, 1 - self.momentum), ops.mul_scalar(x_mean_batch.detach(), self.momentum))
            self.running_var = ops.add(ops.mul_scalar(self.running_var, 1 - self.momentum), ops.mul_scalar(x_var_batch.detach(), self.momentum))

            return y
        else:
            running_mean_reshaped = ops.broadcast_to(ops.reshape(self.running_mean, (1, x.shape[1])), x.shape)
            running_var_reshaped = ops.broadcast_to(ops.reshape(ops.add_scalar(self.running_var, self.eps), (1, x.shape[1])), x.shape)
            x_centered = x - running_mean_reshaped
            x_norm = ops.divide(x_centered, ops.power_scalar(running_var_reshaped, 0.5))

            return x_norm
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x_mean = ops.summation(x, axes=(1,)) / self.dim
        # print(f'{x_mean.shape=}')
        # total_size = x_mean.shape[0]
        shape = x_mean.shape  # Get the shape of the tensor
        total_size = 1
        for dim in shape:
            total_size *= dim
        x_mean = ops.reshape(x_mean, (total_size, 1))
        # print(f'{x_mean.shape=}, {x.shape=}')
        x_mean = ops.broadcast_to(x_mean, x.shape)
        x_centered = x - x_mean
        x_var = ops.add_scalar(ops.divide_scalar(ops.summation(ops.power_scalar(x_centered, 2), axes=(1,)), self.dim), self.eps)
        x_var = ops.broadcast_to(ops.reshape(x_var, (total_size, 1)), x.shape)
        x_norm = ops.divide(x_centered, ops.power_scalar(x_var, 0.5))
        
        w = ops.broadcast_to(ops.reshape(self.weight, (1, self.weight.shape[0])), x_norm.shape)
        b = ops.broadcast_to(ops.reshape(self.bias, (1, self.bias.shape[0])), x_norm.shape)
        y = ops.add(ops.multiply(w, x_norm), b)
        return y
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=1-self.p, device=x.device, dtype=x.dtype)
            x = ops.divide_scalar(ops.multiply(x, mask), 1 - self.p)
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.add(x, self.fn(x))
        ### END YOUR SOLUTION
