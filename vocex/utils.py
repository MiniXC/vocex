from torch import nn

class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class Transpose(nn.Module):
    def __init__(self, module, dims=(1, 2)):
        super().__init__()
        self.module = module
        self.dims = dims

    def forward(self, x):
        return self.module(x.transpose(*self.dims)).transpose(*self.dims)