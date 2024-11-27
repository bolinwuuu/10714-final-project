import sys

sys.path.append("./python")
sys.path.append("./apps")
from simple_ml import *
import numdifftools as nd

import numpy as np
import mugrade
import needle as ndl


def test_fft_forward():
    """
    Test the forward computation of FFT.
    """
    input_tensor = ndl.init.rand(*(2, 3, 8, 64), low=0.0, high=1.0)
    np_input = input_tensor.numpy()

    np_fft_result = np.fft.fft(np_input)
    res = ndl.fft(input_tensor)

    real_result, imag_result = ndl.split(res, 0)

    real_result = real_result.numpy()
    imag_result = imag_result.numpy()

    fft_result = real_result + 1j * imag_result

    # np_fft_result = np.fft.fft(np_input)

    np.testing.assert_allclose(
        fft_result,
        np_fft_result,
        rtol=1e-5,
        atol=1e-7,
        err_msg="The ndl FFT implementation does not match NumPy FFT results."
    )


def test_ifft_forward():
    """
    Test the forward computation of IFFT.
    """
    # Example input for FFT (real and imaginary parts)
#     input_tensor = ndl.Tensor([
#     [
#         [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
#         [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0], [-7.0, -8.0, -9.0]],
#     ],
#     [
#         [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]],
#         [[-10.0, -20.0, -30.0], [-40.0, -50.0, -60.0], [-70.0, -80.0, -90.0]],
#     ],
# ])
    input_tensor = ndl.init.rand(*(2, 3, 64, 64), low=0.0, high=1.0)

    # Compute FFT to get the real and imaginary parts
    np_input = input_tensor.numpy()
    np_fft_result = np.fft.fft(np_input)  # Expected result using NumPy FFT
    np_ifft_result = np.fft.ifft(np_fft_result)  # Expected result using NumPy IFFT

    # Compute FFT using ndl
    fft_result = ndl.fft(input_tensor)

    # Compute IFFT using ndl (Inverse FFT)
    ifft_result = ndl.ifft(fft_result)
    # ifft_result = ifft_result.numpy()

    ifft_result_real, ifft_result_imag = ndl.split(ifft_result, 0)

    print(f'{ifft_result_real.shape=}, {ifft_result_imag.shape=}')
    ifft_result_real = ifft_result_real.numpy()
    ifft_result_imag = ifft_result_imag.numpy()

    # Combine real and imaginary parts of the IFFT result
    ifft_result = ifft_result_real + 1j * ifft_result_imag

    
    diff = np.sum(ifft_result_real - np.real(np_ifft_result))
    print(f'{diff=}')
    # Check if IFFT result is close to the original input (after normalization)
    np.testing.assert_allclose(
        ifft_result_real,
        np.real(np_ifft_result),
        rtol=1e-4,
        atol=1e-7,
        err_msg="The ndl IFFT implementation does not match NumPy IFFT results."
    )



def test_fft_ifft_inverse():
    """
    Test that applying FFT followed by IFFT returns the original tensor.
    """
    input_tensor = ndl.Tensor([[1.0, 2.0, 3.0, 4.0]])  # Example input
    np_input = input_tensor.numpy()  # Convert to NumPy array for reference

    # Compute FFT and then IFFT using ndl
    fft_result = ndl.fft(input_tensor)
    ifft_result = ndl.ifft(fft_result).numpy()

    # IFFT(FFT(x)) should equal x
    np.testing.assert_allclose(ifft_result, np_input, rtol=1e-5, atol=1e-7)
