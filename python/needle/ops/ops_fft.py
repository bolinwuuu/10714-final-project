from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
from needle import ops

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 
import math

class FFT(TensorOp):
    def compute(self, a: NDArray):
        N = a.shape[0]

        if N <= 1:
            return a
        
        res = array_api.full(a.shape, 0, dtype=a.dtype, device=a.device)
        a_temp = split(a, 0)
        even = fft(a_temp[::2])
        odd = fft(a_temp[1::2])

        terms = []
        for n in range(N // 2):
            terms.append(array_api.exp(-2j * math.pi * n / N) * odd[n])

        for k in range(N // 2):
            res[k] = even[k] + terms[k]
            res[k + N // 2] = even[k] - terms[k]
        
        return res
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        return ifft(out_grad)

def fft(a):
    return FFT()(a)

class IFFT(TensorOp):
    def compute(self, a: NDArray):
        N = a.shape[0]

        if N <= 1:
            return a
        
        res = array_api.full(a.shape, 0, dtype=a.dtype, device=a.device)
        a_temp = split(a, 0)
        even = ifft(a_temp[::2])
        odd = ifft(a_temp[1::2])

        terms = []
        for n in range(N // 2):
            terms.append(array_api.exp(2j * math.pi * n / N) * odd[n])

        for k in range(N // 2):
            res[k] = even[k] + terms[k]
            res[k + N // 2] = even[k] - terms[k]
        
        return res

    def gradient(self, out_grad: Tensor, node: Tensor):
        return fft(out_grad)

def ifft(a):
    return IFFT()(a)


class ConvFFT(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):

        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape

        padded_h = H + 2 * self.padding
        padded_w = W + 2 * self.padding
        A_padded = self._pad_tensor(A, self.padding, self.padding)

        padded_kh = padded_h + K - 1
        padded_kw = padded_w + K - 1
        B_padded = self._pad_tensor(B, padded_kh - K, padded_kw - K)

        A_fft = fft(A_padded)
        B_fft = fft(B_padded)

        output_fft = A_fft * B_fft

        output_spatial = ifft(output_fft)

        Ho, Wo = (H - K + 2 * self.padding) // self.stride + 1, (W - K + 2 * self.padding) // self.stride + 1
        output_cropped = output_spatial[:, :, :Ho, :Wo]

        return output_cropped

    def gradient(self, out_grad, node):

        A, B = node.inputs
        K, _, _, _ = B.shape

        out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1)

        B_T = transpose(flip(B, axes=(0, 1)), (2, 3))
        A_grad = conv_fft(out_grad, B_T, padding=K - self.padding - 1)

        A_T = transpose(A, axes=(3, 0))
        out_grad_T = transpose(transpose(out_grad, (0, 1)), (1, 2))
        B_grad = conv_fft(A_T, out_grad_T, padding=self.padding)
        B_grad = transpose(transpose(B_grad, (0, 1)), (1, 2))

        return A_grad, B_grad

    def _pad_tensor(self, x: Tensor, pad_h: int, pad_w: int) -> Tensor:

        batch_size, channels, height, width = x.shape

        top_pad = init.zeros(batch_size, channels, pad_h, width, device=x.device, dtype=x.dtype)
        bottom_pad = init.zeros(batch_size, channels, pad_h, width, device=x.device, dtype=x.dtype)
        left_pad = init.zeros(batch_size, channels, height + 2 * pad_h, pad_w, device=x.device, dtype=x.dtype)
        right_pad = init.zeros(batch_size, channels, height + 2 * pad_h, pad_w, device=x.device, dtype=x.dtype)

        x_padded = ops.stack([top_pad, x, bottom_pad], axis=2)
        x_padded = ops.stack([left_pad, x_padded, right_pad], axis=3)

        return x_padded

def conv_fft(a, b, stride=1, padding=1):
    return ConvFFT(stride, padding)(a, b)