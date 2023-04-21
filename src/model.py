from typing import Optional

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, n_channels: list[int], kernels: list[int], maxpools: list[bool]):
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
                    nn.MaxPool2d(kernel_size=2, stride=2)
                    if maxpools[layer_id]
                    else nn.Identity(),
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
        maxpools: list[bool],
        embedding_size: int,
        n_layers_mlp: int,
        gru_hidden_size: int,
        gru_num_layers: int,
        maximum_steps: int,
    ):
        super().__init__()

        self.steps_encoder = nn.Embedding(maximum_steps, embedding_size)
        self.direction_encoder = nn.Embedding(2, embedding_size)
        self.project_encoded_actions = nn.Linear(4 * embedding_size, embedding_size)
        self.cnn_encoder = CNNEncoder(n_channels, kernels, maxpools)
        self.project = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LazyLinear(embedding_size),
            nn.GELU(),
            nn.LayerNorm(embedding_size),
        )
        self.mlp = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(embedding_size, embedding_size),
                    nn.GELU(),
                    nn.LayerNorm(embedding_size),
                )
                for _ in range(n_layers_mlp)
            ]
        )
        self.gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
        )
        self.heads = nn.ModuleDict(
            {
                "delta_x": nn.Linear(gru_hidden_size, maximum_steps),
                "delta_y": nn.Linear(gru_hidden_size, maximum_steps),
                "direction_x": nn.Linear(gru_hidden_size, 1),
                "direction_y": nn.Linear(gru_hidden_size, 1),
            }
        )

    def encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        delta_x = actions[:, 0]
        delta_y = actions[:, 1]
        direction_x = actions[:, 2]
        direction_y = actions[:, 3]

        delta_x = self.steps_encoder(delta_x)
        delta_y = self.steps_encoder(delta_y)
        direction_x = self.direction_encoder(direction_x)
        direction_y = self.direction_encoder(direction_y)

        encoded_actions = torch.concat(
            (delta_x, delta_y, direction_x, direction_y), dim=1
        )
        encoded_actions = self.project_encoded_actions(encoded_actions)
        return encoded_actions

    def forward(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Predict the action to take for each patch in the batch.
        It is a one-step prediction, the memory is used to remember about
        the previous encountered patches.

        ---
        Args:
            x: The batch of patches of images.
                Shape of [batch_size, num_channels, patch_size, patch_size].
            actions: The previous action taken.
                Shape of [batch_size, 4].
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

        # Add the action embeddings.
        x = x + self.encode_actions(actions)
        x = self.mlp(x)

        # Run the GRU.
        x = x.unsqueeze(0)  # Add a fictive time dimension.
        x, memory = self.gru(x, memory)
        x = x.squeeze(0)  # Remove the fictive dimension.

        # Compute the actions.
        predicted_actions = {key: self.heads[key](x) for key in self.heads.keys()}
        return predicted_actions, memory
