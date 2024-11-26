from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
from needle import ops

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 
import math
from .ops_tuple import *


class FFT(TensorOp):
    def compute(self, a: NDArray):
        print("entering FFT!!!!!!!!!!!!!!!!")
        # N = a.shape[len(a.shape) - 1]  # Size along the last axis (signal length)

        # # Base case: if the signal length is 1, return the input as-is
        # if N <= 1:
        #     return a
        
        # # Initialize the result tensor
        # res = array_api.full(a.shape, 0, dtype=a.dtype, device=a.device)
        
        # # Split the tensor into even and odd indexed elements along the last axis
        # a_temp = split(a, len(a.shape) - 1)  # Split along the last axis (dimension corresponding to signal length)
        # even = fft(a_temp[::2])  # FFT of even indexed elements
        # odd = fft(a_temp[1::2])  # FFT of odd indexed elements

        # # Calculate the FFT terms
        # terms = []
        # for n in range(N // 2):
        #     # Compute the twiddle factor W_N^n
        #     twiddle = array_api.exp(-2j * math.pi * n / N) * odd[n]
        #     terms.append(twiddle)

        # # Combine the even and odd parts using the FFT formula
        # for k in range(N // 2):
        #     res[k] = even[k] + terms[k]
        #     res[k + N // 2] = even[k] - terms[k]
        
        # return res

        print(f'{a.shape=}')
        N = a.shape[-1]  # Size along the last axis (signal length)
        print(f'{N=}')
        # Base case: if the signal length is 1, return the input as-is
        if N <= 1:
            return a
        print("didn't return")
        # Initialize the result tensor
        res = array_api.full(a.shape, 0, dtype=a.dtype, device=a.device)
        
        print("here")
        # Split the tensor into even and odd indexed elements along the last axis
        ax = len(a.shape) - 1
        print(f'{ax=}')

        even_slices = [slice(None)] * len(a.shape)
        odd_slices = [slice(None)] * len(a.shape)
        even_slices[-1] = slice(0, N, 2)  # Take every second element along the last axis
        odd_slices[-1] = slice(1, N, 2)   # Take every second element starting from 1 along the last axis
        print(f'{even_slices=}')
        print(f'{odd_slices=}')

        even_subarr = a[tuple(even_slices)].compact()
        odd_subarr = a[tuple(odd_slices)].compact()
        print(f'{even_subarr=}')
        print(f'{odd_subarr=}')
        even = fft(even_subarr)  # FFT of even-indexed elements
        odd = fft(odd_subarr)
        # evens = a_temp[::2]
        # odds = a_temp[1::2]
        print(f'{even=}')
        print(f'{odd=}')
        # even = fft(evens)  # FFT of even indexed elements
        # odd = fft(odds)  # FFT of odd indexed elements
        print("???????????????????????????")
        # print(f'{a_temp[::2]=}')

        # Calculate the FFT terms
        terms = []
        for n in range(N // 2):
            # Compute the twiddle factor W_N^n
            twiddle = array_api.exp(-2j * math.pi * n / N) * odd[n]
            terms.append(twiddle)

        # Use slicing to assign values to the result tensor
        for k in range(N // 2):
            res_slices = [slice(None)] * len(res.shape)  # Create a full slice (for all dimensions)
            res_slices[-1] = slice(k, k + 1)  # Slice on the last axis for k
            res[tuple(res_slices)] = even[k] + terms[k]  # Update corresponding slice

            res_slices[-1] = slice(k + N // 2, k + N // 2 + 1)  # Slice for the second part of the FFT
            res[tuple(res_slices)] = even[k] - terms[k]  # Update corresponding slice
        
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
        A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape

        fft_h = H + K - 1
        fft_w = W + K - 1

        B_padded = B.pad(((0, fft_h - K), (0, fft_w - K), (0, 0), (0, 0)))

        A_fft = fft(A)
        B_fft = fft(B_padded)

        output_fft = A_fft * B_fft

        output_spatial = ifft(output_fft)

        Ho, Wo = (H - K + 2 * self.padding) // self.stride + 1, (W - K + 2 * self.padding) // self.stride + 1
        output_cropped = output_spatial[:, :Ho * self.stride:self.stride, :Wo * self.stride:self.stride, :]

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

def conv_fft(a, b, stride=1, padding=1):
    return ConvFFT(stride, padding)(a, b)