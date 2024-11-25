from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # z_max = array_api.max(Z, axis=(1,), keepdims=True)
        z_max = Z.max(axis=(1,), keepdims=True)
        log_sum = array_api.log(array_api.sum(array_api.exp(Z - z_max), axis=(1,)))
        lse = log_sum + z_max.reshape(log_sum.shape)
        lse = array_api.broadcast_to(lse.reshape((-1, 1)), Z.shape)
        # print(f'{Z.shape=}, {lse.shape=}')
        return Z - lse
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        lse_z = logsumexp(Z, axes=(1,))
        softmax_z = exp(Z - broadcast_to(reshape(lse_z, (-1, 1)), Z.shape))
        out_grad_sum = summation(out_grad, axes=(1,))
        out_grad_sum = broadcast_to(reshape(out_grad_sum, (-1, 1)), Z.shape)
        term1 = broadcast_to(out_grad, Z.shape)
        term2 = multiply(softmax_z, out_grad_sum)
        res = term1 - term2
        return res

        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        z_max = Z.max(axis=self.axes, keepdims=True)
        log_sum = array_api.log(array_api.sum(array_api.exp(Z - z_max.broadcast_to(Z.shape)), axis=self.axes))
        res = log_sum + z_max.reshape(log_sum.shape)
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        axes = None
        if self.axes is not None:
            if isinstance(self.axes, tuple):
                axes = self.axes
            else:
                axes = (self.axes,)
        else:
            axes = tuple(range(len(Z.shape)))

        shape_add_axes = list(out_grad.shape)
        for ax in axes:
            shape_add_axes.insert(ax, 1)

        out_grad_reshaped = broadcast_to(reshape(out_grad, shape_add_axes), Z.shape)
        lse_z = broadcast_to(reshape(logsumexp(Z, axes=self.axes), shape_add_axes), Z.shape)
        lse_grad = exp(Z - lse_z)
        return multiply(out_grad_reshaped, lse_grad)
        ### END YOUR SOLUTION

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

