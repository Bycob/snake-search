from typing import Optional

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, n_channels: list[int], kernels: list[int]):
        super().__init__()

        assert (
            len(n_channels) == len(kernels) + 1
        ), "The number of layers is not consistent."

        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=n_channels[layer_id],
                        out_channels=n_channels[layer_id + 1],
                        kernel_size=kernels[layer_id],
                        padding=kernels[layer_id] // 2,
                    ),
                    nn.GELU(),
                    nn.GroupNorm(num_groups=1, num_channels=n_channels[layer_id + 1]),
                )
                for layer_id in range(len(kernels))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the image.

        ---
        Args:
            x: The image to encode.
                Shape of [batch_size, num_channels, height, width].

        ---
        Returns:
            The encoded image.
                Shape of [batch_size, num_channels, height, width].
        """
        for layer in self.conv_layers:
            x = layer(x)
        return x


class GRUModel(nn.Module):
    def __init__(
        self,
        n_channels: list[int],
        kernels: list[int],
        embedding_size: int,
        gru_hidden_size: int,
        gru_num_layers: int,
    ):
        super().__init__()

        self.cnn_encoder = CNNEncoder(n_channels, kernels)
        self.project = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LazyLinear(embedding_size),
            nn.LayerNorm(embedding_size),
        )

    def forward(
        self, x: torch.Tensor, memory: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """

        ---
        Args:
            x: The batch of patches of images.
                Shape of [batch_size, num_channels, patch_size, patch_size].
            memory: The memory of the previous step.
                Shape of [gru_n_layers, batch_size, gru_hidden_size].

        ---
        Returns:
            action: The action to take.
                Shape of [batch_size, n_actions].
            memory: The memory of the current step.
                Shape of [gru_n_layers, batch_size, gru_hidden_size].
        """
        pass
