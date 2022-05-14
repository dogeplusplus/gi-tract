import torch
import typing as t
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules import Module


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: t.Tuple[int, int] = (7, 7),
        activation: Module = nn.GELU,
    ):
        super(DoubleConv, self).__init__()
        self.norm = nn.BatchNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size, padding="same")
        self.act1 = activation()
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size, padding="same")
        self.act2 = activation()

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)

        return x


class UNet(nn.Module):
    def __init__(
        self,
        filters: t.List[int],
        in_dim: int,
        out_dim: int,
        kernel_size: t.Tuple[int, int],
        activation: nn.Module = nn.GELU,
        final_activation: nn.Module = nn.Softmax(dim=1),
    ):
        super(UNet, self).__init__()

        # Split filters into respective segments
        bottom_filters = filters[-1]
        down_filters = [in_dim] + filters
        up_filters = filters[::-1] + [out_dim]

        self.kernel_size = kernel_size

        down_pairs = zip(down_filters[:-1], down_filters[1:])
        up_pairs = zip(up_filters[:-1], up_filters[1:])
        self.down = nn.ModuleList([
            DoubleConv(cin, cout, kernel_size, activation) for (cin, cout) in down_pairs
        ])

        # Double the in channels due to concatenation
        self.up = nn.ModuleList([
            DoubleConv(2 * cin, cout, kernel_size, activation) for (cin, cout) in up_pairs
        ])
        self.bottom = DoubleConv(down_filters[-1], bottom_filters, kernel_size, activation)

        self.final = nn.Conv2d(out_dim, out_dim, kernel_size=(1, 1))
        self.final_activation = final_activation

    def forward(self, x):
        down_stack = []
        for layer in self.down:
            x = layer(x)
            down_stack.insert(0, x)
            x = F.max_pool2d(x, (2, 2))

        x = self.bottom(x)

        for down, layer in zip(down_stack, self.up):
            x = F.interpolate(x, scale_factor=(2, 2), mode="nearest")
            x = torch.cat([x, down], dim=1)
            x = layer(x)

        x = self.final(x)
        x = self.final_activation(x)

        return x
