import numpy as np


def ifft2_np(x):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x.astype(np.complex64)), norm='ortho')).astype(np.complex64)


def fft2_np(x):
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x.astype(np.complex64)), norm='ortho')).astype(np.complex64)