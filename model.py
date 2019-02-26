import torch
import torch.nn as nn


class Upsample(nn.Module):
    def __init__(self, scale):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, x):
        row = torch.cat([x] * self.scale, dim=-1)
        return torch.cat([row] * self.scale, dim=-2)


class ResBlock(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, kernel_size=3, sampling="no", normalize=False):
        super(ResBlock, self).__init__()

        same_padding = kernel_size // 2

        skip = []
        modified = []

        skip.append(nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)))
        if sampling == "upsampling":
            upsampler = Upsample(scale=2)
            skip.append(upsampler)
            modified.append(upsampler)
        elif sampling == "downsampling":
            downsampler = nn.AvgPool2d(kernel_size=2)
            skip.append(downsampler)
            modified.append(downsampler)
        else:
            assert sampling == "no", "sampling must be in {no, upsampling, downsampling}"

        for _ in range(2):
            modified.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=same_padding))
            if normalize:
                modified.append(nn.BatchNorm2d(out_channels, momentum=0.9))
            modified.append(nn.ReLU())
            modified.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=same_padding))
            if normalize:
                modified.append(nn.BatchNorm2d(out_channels, momentum=0.9))
            modified.append(nn.ReLU())

        self.skip = nn.Sequential(*skip)
        self.modified = nn.Sequential(*modified)

    def forward(self, x):
        return self.skip(x) + self.modified(x)


class Generator(nn.Module):
    def __init__(self, noise_size=128):
        super(Generator, self).__init__()

        self.noise_size = noise_size
        self.linear = nn.Linear(noise_size, noise_size * 4 * 4)
        self.generator = nn.Sequential(
            ResBlock(sampling="upsampling", normalize=True, in_channels=noise_size, out_channels=noise_size),
            ResBlock(sampling="upsampling", normalize=True, in_channels=noise_size, out_channels=noise_size),
            ResBlock(sampling="upsampling", normalize=True, in_channels=noise_size, out_channels=noise_size),
            nn.Conv2d(noise_size, 3, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x):
        image = self.linear(x).view((-1, self.noise_size, 4, 4))
        return self.generator(image)


class Discriminator(nn.Module):
    def __init__(self, noise_size):
        super(Discriminator, self).__init__()

        self.noise_size = noise_size
        self.discriminator = nn.Sequential(
            ResBlock(sampling="downsampling", in_channels=3, out_channels=noise_size, normalize=False),
            ResBlock(sampling="downsampling", in_channels=noise_size, out_channels=noise_size, normalize=False),
            ResBlock(sampling="no", in_channels=noise_size, out_channels=noise_size, normalize=False),
            ResBlock(sampling="no", in_channels=noise_size, out_channels=noise_size, normalize=False),
        )

        self.linear = nn.Linear(noise_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        discriminated = self.discriminator(x).view((batch_size, self.noise_size, -1))
        mean_discriminated = torch.mean(discriminated, dim=-1)
        return self.linear(mean_discriminated)