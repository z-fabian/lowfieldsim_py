import numpy as np


def ifft2_np(x):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x.astype(np.complex64), axes=[-2, -1]), norm='ortho'), axes=[-2, -1]).astype(np.complex64)


def fft2_np(x):
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x.astype(np.complex64), axes=[-2, -1]), norm='ortho'), axes=[-2, -1]).astype(np.complex64)


def center_crop(x, shape):
    assert 0 < shape[0] <= x.shape[-2]
    assert 0 < shape[1] <= x.shape[-1]
    w_from = (x.shape[-2] - shape[0]) // 2
    h_from = (x.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return x[..., w_from:w_to, h_from:h_to]


def kspace_to_target_vol(x):
    dim = len(x.shape)
    assert 3 <= dim <= 4
    if dim == 3:        # Single-coil measurements: insert dummy coil
        x = x[:, None, :, :]
    s = x.shape[0]
    target = []
    for s_i in range(s):
        im_i = ifft2_np(x[s_i, ...])
        im_i = center_crop(im_i, (320, 320))
        target_i = np.sqrt(np.sum(np.square(np.abs(im_i)), axis=0)).astype(np.float32)
        target.append(target_i)
    return np.stack(target, axis=0)
