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


class GRUPolicy(nn.Module):
    def __init__(
        self,
        n_channels: list[int],
        kernels: list[int],
        embedding_size: int,
        gru_hidden_size: int,
        gru_num_layers: int,
        n_actions: int,
    ):
        super().__init__()

        self.cnn_encoder = CNNEncoder(n_channels, kernels)
        self.project = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LazyLinear(embedding_size),
            nn.LayerNorm(embedding_size),
        )
        self.gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
        )
        self.action_head = nn.Linear(gru_hidden_size, n_actions)

    def forward(
        self, x: torch.Tensor, memory: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Predict the action to take for each patch in the batch.
        It is a one-step prediction, the memory is used to remember about
        the previous encountered patches.

        ---
        Args:
            x: The batch of patches of images.
                Shape of [batch_size, num_channels, patch_size, patch_size].
            memory: The memory of the previous step.
                Shape of [gru_num_layers, batch_size, gru_hidden_size].

        ---
        Returns:
            action: The action to take.
                Shape of [batch_size, n_actions].
            memory: The memory of the current step.
                Shape of [gru_num_layers, batch_size, gru_hidden_size].
        """
        # Project the image to [batch_size, embedding_size].
        x = self.cnn_encoder(x)
        x = self.project(x)

        # Run the GRU.
        x = x.unsqueeze(0)  # Add a fictive time dimension.
        x, memory = self.gru(x, memory)
        x = x.squeeze(0)  # Remove the fictive dimension.

        # Compute the action.
        x = self.action_head(x)
        return x, memory
