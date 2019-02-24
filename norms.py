def sobolev_norm(input, s=1, c=5):
    signal_ndim = 2

    #fourier transform of input -- [real, imaginary]
    real_input = input
    imaginary_input = torch.zeros_like(input)
    fourier_transform = torch.fft(torch.stack((real_input, imaginary_input), -1),
                      signal_ndim=signal_ndim)

    #compution the scale \xi
    N, M = fourier_transform.shape[2], fourier_transform.shape[2]

    ns = torch.arange(0, N).type(torch.DoubleTensor) / N
    ms = torch.arange(0, M).type(torch.DoubleTensor) / M

    xi_x, xi_y = torch.meshgrid([ms, ns])
    squared_xi = xi_x[None, None, :, :] ** 2 +\
                 xi_y[None, None, :, :] ** 2
    scaled_xi = (1 + c * squared_xi) ** (s * 0.5)

    #the derivative in Sobolev norm is replaced by multiplication of \xi and fourier transform
    derivative = torch.stack([scaled_xi, scaled_xi], -1) * fourier_transform

    #final inverse fourier transform
    output = torch.ifft(derivative, signal_ndim=signal_ndim)

    #we only need the real part as an answer
    output = output[..., 0]

    return output

def lp_norm(input, p=None):
    input = input.view(input.size(0), -1).type(torch.DoubleTensor)

    #in order to find stable norm the normalization is performed
    #\|x\| = alpha * \|(x / alpha)\|
    #we will also try to avoid zero elements in alpha
    epsilon = 1e-5

    alpha, _ = torch.max((torch.abs(input) + epsilon), dim=1)
    output = alpha * torch.norm(input / alpha[:, None], p=p, dim=1)
    return output
