"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**b
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return multiply(multiply(out_grad, b), power(a, b-1)), multiply(multiply(out_grad, power(a, b)), log(a))
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return multiply(mul_scalar(out_grad, self.scalar), power_scalar(a, self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # res = (a / b).astype(a.dtype)
        res = a / b
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return divide(out_grad, b), divide(multiply(negate(out_grad), a), power_scalar(b, 2))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # res = (a / self.scalar).astype(a.dtype)
        res = a / self.scalar
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide_scalar(out_grad, self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes if axes else (-1, -2)

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # axis1, axis2 = self.axes
        # return array_api.swapaxes(a, axis1, axis2)

        num_dims = len(a.shape)
        
        axis1, axis2 = self.axes
        if axis1 < 0:
            axis1 += num_dims
        if axis2 < 0:
            axis2 += num_dims

        permute_order = list(range(num_dims))
        permute_order[axis1], permute_order[axis2] = permute_order[axis2], permute_order[axis1]

        return a.permute(permute_order)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        axis1, axis2 = self.axes
        return transpose(out_grad, axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        res = array_api.reshape(a.compact(), self.shape)
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        res = reshape(out_grad, a.shape)
        return res
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        res = array_api.broadcast_to(a, self.shape)
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ## BEGIN YOUR SOLUTION
        a = node.inputs[0]
        if a.shape == self.shape:
            return out_grad

        shrink_axes = [i for i in range(len(self.shape))]
        for i, (ori, cur) in enumerate(zip(reversed(a.shape), reversed(self.shape))):
            if ori == cur:
                shrink_axes[len(self.shape) - i - 1] = -1
        shrink_axes = tuple(filter(lambda x: x >= 0, shrink_axes))
        assert len(shrink_axes) > 0

        sum_grad = out_grad.sum(shrink_axes)
        res = sum_grad.reshape(a.shape)
        return res

        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if isinstance(self.axes, (list, tuple)) and len(self.axes) > 1:
            sum_axes = reversed(sorted(self.axes))
            for ax in sum_axes:
                a = a.sum(ax)
            return a

        res = a.sum(self.axes)
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        axes = None
        if self.axes is not None:
            if isinstance(self.axes, tuple):
                axes = self.axes
            else:
                axes = (self.axes,)
        else:
            axes = range(len(a.shape))

        new_shape = list(a.shape)
        for ax in axes:
            new_shape[ax] = 1
        reshaped_grad = reshape(out_grad, new_shape)
        res = broadcast_to(reshaped_grad, a.shape)
        
        return res
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # res = array_api.matmul(a, b)
        res = a @ b
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a = matmul(out_grad, transpose(b))
        grad_b = matmul(transpose(a), out_grad)
        if len(grad_a.shape) != len(a.shape):
            axes_a = tuple(range(len(grad_a.shape) - len(a.shape)))
            grad_a = summation(grad_a, axes=axes_a)
        
        if len(grad_b.shape) != len(b.shape):
            axes_b = tuple(range(len(grad_b.shape) - len(b.shape)))
            grad_b = summation(grad_b, axes=axes_b)
        
        return grad_a, grad_b
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -1 * a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return divide(out_grad, a)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return multiply(out_grad, exp(a))
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        res = array_api.maximum(a, 0)
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out = node.realize_cached_data()
        return out_grad * Tensor(out > 0, device=out_grad.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)



class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        ea = array_api.exp(a)
        ena = array_api.exp(-a)
        res = (ea - ena) / (ea + ena)
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # a = node.inputs[0]
        # grad_input = add_scalar(-power_scalar(tanh(a), 2), 1)
        # res = multiply(grad_input, out_grad)
        # return res
        res = out_grad * (-tanh(node.inputs[0])**2 + 1)
        return res
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # return array_api.stack(args, axis=self.axis)

        shape = args[0].shape
        new_shape = list(shape)
        new_shape.insert(self.axis, len(args))
        res = array_api.empty(new_shape, device=args[0].device)
        slices = [slice(0, s) for s in new_shape]
        for i, arr in enumerate(args):
            slices[self.axis] = slice(i, i + 1)
            res[tuple(slices)] = arr
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        res = split(out_grad, axis=self.axis)
        return res
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        A_shape = list(A.shape)
        new_shape = A_shape[:self.axis]+ A_shape[self.axis+1:]
        res = []
        for i in range(A.shape[self.axis]):
            slices = [slice(None)] * len(A_shape)
            slices[self.axis] = slice(i, i+1)
            sub_tensor = A[tuple(slices)].compact().reshape(new_shape)
            res.append(sub_tensor)
        res = tuple(res)
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        res = stack(out_grad, axis=self.axis)
        return res
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        res = a.flip(self.axes)
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        res = flip(out_grad, self.axes)
        return res
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape) 

        for axis in self.axes:
            if axis < len(a.shape):
                new_shape[axis] = new_shape[axis] * (self.dilation + 1)

        new_shape = tuple(new_shape)
        res = array_api.full(new_shape, 0, device=a.device)
        slices = [slice(None)] * len(res.shape)

        for axis in self.axes:
            if axis < len(a.shape):
              slices[axis] = slice(0, res.shape[axis], self.dilation + 1)

        res[tuple(slices)] = a
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        res = undilate(out_grad, self.axes, self.dilation)
        return res
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        slices = [slice(None)] * len(a.shape)
        for axis in self.axes:
            if axis < len(a.shape):
                slices[axis] = slice(0, a.shape[axis], self.dilation + 1)
                
        res = a[tuple(slices)].compact()
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        res = dilate(out_grad, self.axes, self.dilation)
        return res
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        
        
        Ho, Wo = (H - K) // self.stride + 1, (W - K) // self.stride + 1
        inner_dim = K * K * C_in
        outer_dim = N*Ho*Wo

        new_shape = (N, Ho, Wo, K, K, C_in)
        new_strides = (Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)
        
        A = A.as_strided(shape=new_shape, strides=new_strides).compact().reshape((outer_dim, inner_dim))
        out = A @ B.compact().reshape((K * K * C_in, C_out))
        out = out.compact().reshape((N, Ho, Wo, C_out))
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        K, _, _, _ = B.shape

        out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride-1)

        B_T = transpose(flip(B, axes=(0, 1)), (2, 3))
        A_grad = conv(out_grad, B_T, padding=K-self.padding-1)

        A_T = transpose(A, axes=(3, 0))
        out_grad_T = transpose(transpose(out_grad, (0, 1)), (1, 2))
        B_grad = conv(A_T, out_grad_T, padding=self.padding)
        B_grad = transpose(transpose(B_grad, (0, 1)), (1, 2))

        return A_grad, B_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


