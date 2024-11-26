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
    input_tensor = ndl.Tensor([[1.0, 2.0, 3.0, 4.0]])  # Example input
    np_input = input_tensor.numpy()  # Convert to NumPy array for reference

    # Compute FFT using ndl
    fft_result = ndl.fft(input_tensor).numpy()

    # Compute FFT using NumPy
    np_fft_result = np.fft.fft(np_input)

    # Assert correctness
    np.testing.assert_allclose(fft_result, np_fft_result, rtol=1e-5, atol=1e-7)


def test_ifft_forward():
    """
    Test the forward computation of IFFT.
    """
    input_tensor = ndl.Tensor([[1.0 + 2.0j, 2.0 + 1.0j, 3.0 - 1.0j, 4.0 - 2.0j]])  # Example input
    np_input = input_tensor.numpy()  # Convert to NumPy array for reference

    # Compute IFFT using ndl
    ifft_result = ndl.ifft(input_tensor).numpy()

    # Compute IFFT using NumPy
    np_ifft_result = np.fft.ifft(np_input)

    # Assert correctness
    np.testing.assert_allclose(ifft_result, np_ifft_result, rtol=1e-5, atol=1e-7)


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
