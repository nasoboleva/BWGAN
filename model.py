import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, kernel_size=3, sampling="no", normalize=False):
        super(ResBlock, self).__init__()

        same_padding = kernel_size // 2
        kernel_size = (kernel_size, kernel_size)

        skip = []
        modified = []

        skip.append(nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)))
        if sampling == "upsampling":
            upsampler = nn.Upsample(scale=2)
            skip.append(upsampler)
            modified.append(upsampler)
        elif sampling == "downsampling":
            downsampler = nn.AvgPool2d(kernel_size=2)
            skip.append(downsampler)
            modified.append(downsampler)
        else:
            assert sampling == "no", "sampling must be in {no, upsampling, downsampling}"

        modify_block = []
        modify_block.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=same_padding))
        if normalize:
            modify_block.append(nn.BatchNorm2d(out_channels, momentum=0.9))
        modify_block.append(nn.ReLU())
        modify_block.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=same_padding))
        if normalize:
            modify_block.append(nn.BatchNorm2d(out_channels, momentum=0.9))
        modify_block.append(nn.ReLU())
        modified = modified + modify_block + modify_block

        self.skip = nn.Sequential(*skip)
        self.modified = nn.Sequential(*modified)

    def forward(self, x):
        return self.skip(x) + self.modified(x)


class Generator(nn.Module):
    def __init__(self, noise_size=128):
        super(Generator, self).__init__()

        self.linear = nn.Linear(noise_size, noise_size * 4 * 4)
        self.generator = nn.Sequential(
            ResBlock(sampling="upsampling", normalize=True),
            ResBlock(sampling="upsampling", normalize=True),
            ResBlock(sampling="upsampling", normalize=True),
            nn.Conv2d(128, 3, kernel_size=(1, 1)),
            nn.Tanh(),
        )

    def forward(self, x):
        image = self.linear(x).view((-1, 128, 4, 4))
        return self.generator(image)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            ResBlock(in_channels=4, sampling="downsampling"),
            ResBlock(sampling="downsampling"),
            ResBlock(),
            ResBlock(),
        )

        self.linear = nn.Linear(128, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        discriminated = self.discriminator(x).view(batch_size, 128, -1)
        mean_discriminated = torch.mean(discriminated, dim=-1)
        return self.linear(mean_discriminated)