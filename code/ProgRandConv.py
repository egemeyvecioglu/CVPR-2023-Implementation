import numpy as np
import torch
import torch.nn.functional as F
import torchvision.ops as ops
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.distributions

class RandConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, batch_size=64, device='cuda', sigma_d=0.2, input_dim=32):
        super(RandConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.device = device
        self.batch_size = batch_size
        self.sigma_d = sigma_d

        # For contrast diversification
        self.sigma_beta = 0.5
        self.sigma_alpha = 0.5

        # Sampled from N(n, sigma**2)
        self.gamma = torch.normal(mean=0, std=0.25, size=()).to(device)
        self.beta = torch.normal(mean=0, std=0.25, size=()).to(device)

        self.offset_tensor = 0
        self.sigma_w = 1.0 / np.sqrt(kernel_size**2 * in_channels)
        # init weight with 0 mean and sigma_w variance
        self.weight = torch.nn.Parameter(torch.normal(mean=0, std=self.sigma_w, size=(out_channels, in_channels, kernel_size, kernel_size)).to(device))

        b_g = 1.0  # from the paper
        epsilon = 1e-2
        sigma_g = np.random.uniform(epsilon, b_g)
        gaussian_filter = torch.tensor([[np.exp(-(+(i-1)**2 + (j-1)**2) / (2 * sigma_g**2)) for i in range(kernel_size)] for j in range(kernel_size)], dtype=torch.float, device=device)
        mask = torch.eye(kernel_size, dtype=torch.float, device=device)
        # gaussian_filter *= mask
        self.weight = torch.nn.Parameter(self.weight * gaussian_filter)
        output_dim = (input_dim + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        offset_sampler = torch.distributions.uniform.Uniform(torch.tensor([0.01], device=self.device), torch.tensor([0.2], device=self.device))
        offset_sampler_value = offset_sampler.sample()

        offset = torch.distributions.normal.Normal(torch.tensor([0.0], device=self.device), torch.tensor([offset_sampler_value**2], device=self.device))
        offset_value = offset.sample(torch.Size([self.batch_size, 2*self.kernel_size*self.kernel_size, input_dim, input_dim])).squeeze(dim=-1)
        self.offset_tensor = offset_value

    def forward(self, x):
        deform_out = ops.deform_conv2d(x, self.offset_tensor, self.weight, padding=self.padding, dilation=self.dilation, stride=self.stride)

        standardized = (deform_out - deform_out.mean(dim=(2, 3), keepdim=True)) / (deform_out.std(dim=(2, 3), keepdim=True) + 1e-8)
        affined = standardized * self.gamma + self.beta
        out = F.tanh(affined)
        return out

class ProgRandConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, l_max=10, device='cuda', batch_size=64, sigma_d=0.2, input_dim=32):
        super(ProgRandConvBlock, self).__init__()
        self.n_layers = np.random.randint(1, l_max+1)
        self.lyr = RandConvBlock(in_channels, out_channels, kernel_size, device=device, batch_size=batch_size, sigma_d=sigma_d, input_dim=input_dim).to(device)

    def forward(self, x):
        for _ in range(self.n_layers):
            x = self.lyr(x)
        return x
