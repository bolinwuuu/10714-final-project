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
        return array_api.fft(a)
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        return ifft(out_grad)

def fft(a):
    return FFT()(a)

class IFFT(TensorOp):
    def compute(self, a: NDArray):
        return array_api.ifft(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return fft(out_grad)

def ifft(a):
    return IFFT()(a)


class ConvFFT(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    # def compute(self, A, B):
    #     # print(f'{})
    #     # A = Tensor(A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))), requires_grad=True)
    #     A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
    #     N, H, W, C_in = A.shape
    #     K, _, _, C_out = B.shape

    #     fft_h = H + K - 1
    #     fft_w = W + K - 1

    #     # B_padded = Tensor(B.pad(((0, fft_h - K), (0, fft_w - K), (0, 0), (0, 0))), requires_grad=True)
    #     B_padded = B.pad(((0, fft_h - K), (0, fft_w - K), (0, 0), (0, 0)))

    #     A_fft = A.fft()
    #     B_fft = B_padded.fft()

    #     index_slice_real = [slice(None)] * len(A_fft.shape)
    #     index_slice_imag = [slice(None)] * len(A_fft.shape)
    #     index_slice_real[0] = slice(0, 1)  # First channel for real part
    #     index_slice_imag[0] = slice(1, 2)  # Second channel for imaginary part

    #     real_A = A_fft[tuple(index_slice_real)].reshape(tuple(list(A_fft.shape)[1:]))
    #     imag_A = A_fft[tuple(index_slice_imag)].reshape(tuple(list(A_fft.shape)[1:]))

    #     index_slice_real = [slice(None)] * len(B_fft.shape)
    #     index_slice_imag = [slice(None)] * len(B_fft.shape)
    #     index_slice_real[0] = slice(0, 1)  # First channel for real part
    #     index_slice_imag[0] = slice(1, 2)  # Second channel for imaginary part

    #     real_B = B_fft[tuple(index_slice_real)].reshape(tuple(list(B_fft.shape)[1:]))
    #     imag_B = B_fft[tuple(index_slice_imag)].reshape(tuple(list(B_fft.shape)[1:]))
        
    #     # real_A, imag_A = A_fft[0], A_fft[1]
    #     # real_B, imag_B = B_fft[0], B_fft[1]

    #     print(f'{A.shape=}, {A_fft.shape=}, {real_A.shape=}, {imag_A.shape=}')
    #     print(f'{B.shape=}, {B_fft.shape=}, {real_B.shape=}, {imag_B.shape=}')
    #     real = real_A * real_B - imag_A * imag_B
    #     imag = real_A * imag_B + imag_A * real_B

    #     output_fft = full((2, *real.shape), 0, dtype=self.dtype, device=self.device)
    #     index_slice = [slice(None)] * len(output_fft.shape)
    #     index_slice[0] = slice(0, 1, 1)
    #     output_fft[tuple(index_slice)] = real
    #     index_slice[0] = slice(1, 2, 1)
    #     output_fft[tuple(index_slice)] = imag

    #     output_spatial = output_fft.ifft()

    #     Ho, Wo = (H - K + 2 * self.padding) // self.stride + 1, (W - K + 2 * self.padding) // self.stride + 1
    #     output_cropped = output_spatial[:, :Ho * self.stride:self.stride, :Wo * self.stride:self.stride, :]

    #     return output_cropped

    def compute(self, A, B):
        # Pad input
        A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        
        # Compute FFT padding sizes
        fft_h = H + K - 1
        fft_w = W + K - 1
        
        # Pad kernel
        B_padded = B.pad(((0, fft_h - K), (0, fft_w - K), (0, 0), (0, 0)))
        
        # Compute FFT
        A_fft = A.fft()
        B_fft = B_padded.fft()
        
        # Prepare slice objects
        index_slice_real = [slice(None)] * len(A_fft.shape)
        index_slice_imag = [slice(None)] * len(A_fft.shape)
        
        # Extract real part for A
        index_slice_real[0] = slice(0, 1)
        real_A = A_fft[tuple(index_slice_real)].reshape(tuple(list(A_fft.shape)[1:]))
        
        # Extract imaginary part for A
        index_slice_imag[0] = slice(1, 2)
        imag_A = A_fft[tuple(index_slice_imag)].reshape(tuple(list(A_fft.shape)[1:]))
        
        # Reset slice objects for B
        index_slice_real = [slice(None)] * len(B_fft.shape)
        index_slice_imag = [slice(None)] * len(B_fft.shape)
        
        # Extract real part for B
        index_slice_real[0] = slice(0, 1)
        real_B = B_fft[tuple(index_slice_real)].reshape(tuple(list(B_fft.shape)[1:]))
        
        # Extract imaginary part for B
        index_slice_imag[0] = slice(1, 2)
        imag_B = B_fft[tuple(index_slice_imag)].reshape(tuple(list(B_fft.shape)[1:]))
        
        # Complex multiplication
        # real = real_A * real_B.transpose((2, 3)) - imag_A * imag_B.transpose((2, 3))
        # imag = real_A * imag_B.transpose((2, 3)) + imag_A * real_B.transpose((2, 3))

        # real = real_A * real_B.permute((0, 1, 3, 2)) - imag_A * imag_B.permute((0, 1, 3, 2))
        # imag = real_A * imag_B.permute((0, 1, 3, 2)) + imag_A * real_B.permute((0, 1, 3, 2))

        real_A_expanded = real_A.reshape((N, H, W, C_in, 1))
        imag_A_expanded = imag_A.reshape((N, H, W, C_in, 1))

        print(f'{real_A.shape=}')
        print(f'{real_B.shape=}')
        print(f'{imag_A.shape=}')
        print(f'{imag_B.shape=}')
        print(f'{real_A_expanded.shape=}')
        print(f'{imag_A_expanded.shape=}')
        
        real = real_A_expanded * real_B - imag_A_expanded * imag_B
        imag = real_A_expanded * imag_B + imag_A_expanded * real_B
        
        # Reconstruct FFT output
        output_fft = full((2, *real.shape), 0, dtype=self.dtype, device=self.device)
        
        # Use slices to set real and imaginary parts
        index_slice = [slice(None)] * len(output_fft.shape)
        index_slice[0] = slice(0, 1)
        output_fft[tuple(index_slice)] = real
        
        index_slice[0] = slice(1, 2)
        output_fft[tuple(index_slice)] = imag
        
        # Inverse FFT
        output_spatial = output_fft.ifft()
        
        # Crop and stride
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