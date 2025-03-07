# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this code are derived from the original repository at:
# https://github.com/MIC-DKFZ/MedNeXt
# and are used under the terms of the Apache License, Version 2.0.

from __future__ import annotations

import torch
import torch.nn as nn

all = ["MedNeXtBlock", "MedNeXtDownBlock", "MedNeXtUpBlock", "MedNeXtOutBlock"]


class MedNeXtBlock(nn.Module):
    """
    MedNeXtBlock class for the MedNeXt model.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (int): Expansion ratio for the block. Defaults to 4.
        kernel_size (int): Kernel size for convolutions. Defaults to 7.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int,
        kernel_size: int,
    ):

        super().__init__()

        # First convolution layer with DepthWise Convolutions
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels,
        )

        # Normalization Layer.
        self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)

        # Second convolution (Expansion) layer with Conv2D 1x1
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=expansion_ratio * in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer with Conv2D 1x1
        self.conv3 = nn.Conv2d(
            in_channels=expansion_ratio * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def _common_forward(self, x):
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        x1 = self.conv3(x1)
        return x1

    def forward(self, x):
        """
        Forward pass of the MedNeXtBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        x1 = self._common_forward(x)

        x1 = x + x1

        return x1


class MedNeXtDownBlock(MedNeXtBlock):
    """
    MedNeXtDownBlock class for downsampling in the MedNeXt model.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (int): Expansion ratio for the block. Defaults to 4.
        kernel_size (int): Kernel size for convolutions. Defaults to 7.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int = 4,
        kernel_size: int = 7,
    ):

        super().__init__(
            in_channels,
            out_channels,
            expansion_ratio,
            kernel_size,
        )

        self.res_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=2,
        )

        # Overwrite the first convolution layer with stride 2
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x):
        """
        Forward pass of the MedNeXtDownBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x1 = self._common_forward(x)

        res = self.res_conv(x)
        x1 = x1 + res

        return x1


class MedNeXtUpBlock(MedNeXtBlock):
    """
    Optimized MedNeXtUpBlock class that uses Upsample + Conv instead of ConvTranspose2d for
    better computational efficiency.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (int): Expansion ratio for the block. Defaults to 4.
        kernel_size (int): Kernel size for convolutions. Defaults to 7.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int = 4,
        kernel_size: int = 7,
    ):
        super().__init__(
            in_channels,
            out_channels,
            expansion_ratio,
            kernel_size,
        )

        # Upsampling operation
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Residual path with 1x1 conv
        self.res_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x, target_size=None):
        """
        Forward pass of the MedNeXtUpBlock with dimension matching.

        Args:
            x (torch.Tensor): Input tensor.
            target_size (tuple, optional): Target spatial size for output tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Apply block operations
        x1 = self._common_forward(x)

        # Apply upsampling to main path
        x1 = self.upsample(x1)

        # Apply upsampling to residual path
        res = self.upsample(x)
        res = self.res_conv(res)

        # Apply padding to match original implementation behavior
        x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0))
        res = torch.nn.functional.pad(res, (1, 0, 1, 0))

        # Ensure dimensions match target size if provided
        if target_size is not None:
            if x1.shape[2:] != target_size:
                x1 = torch.nn.functional.interpolate(
                    x1, size=target_size, mode="nearest"
                )
            if res.shape[2:] != target_size:
                res = torch.nn.functional.interpolate(
                    res, size=target_size, mode="nearest"
                )

        # Residual connection
        x1 = x1 + res

        return x1


class MedNeXtOutBlock(nn.Module):
    """
    Optimized MedNeXtOutBlock class using standard Conv2d.

    Args:
        in_channels (int): Number of input channels.
        n_classes (int): Number of output classes.
    """

    def __init__(self, in_channels, n_classes):
        super().__init__()

        # Use standard Conv2d instead of ConvTranspose2d
        self.conv_out = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        """
        Forward pass of the MedNeXtOutBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv_out(x)
