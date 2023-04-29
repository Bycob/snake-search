from typing import Optional

import einops
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding2D


class ViTEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        hidden_size: int,
        n_tokens: int,
        n_layers: int,
        image_size: int,
        output_size: int,
    ):
        super().__init__()
        assert (
            image_size % n_tokens == 0
        ), "The image size is not a multiple of n_tokens."
        kernel_size = image_size // n_tokens

        self.project = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                stride=kernel_size,
            ),
            nn.GELU(),
            nn.LayerNorm([hidden_size, n_tokens, n_tokens]),
        )
        self.positional_encoding = PositionalEncoding2D(hidden_size)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=2 * hidden_size,
                dropout=0.1,
            ),
            num_layers=n_layers,
        )
        self.project = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the image.

        ---
        Args:
            x: The image to encode.
                Shape of [batch_size, num_channels, height, width].

        ---
        Returns:
            The encoded image.
                Shape of [batch_size, output_size].
        """
        x = self.project(x)
        x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.positional_encoding(x) + x
        x = einops.rearrange(x, "b h w c -> b (h w) c")
        x = self.encoder(x)
        x = x.mean(dim=1)  # Average over the tokens.
        x = self.project(x)
        return x


class ActionEncoder(nn.Module):
    def __init__(self, jump_size: int, embedding_size: int):
        super().__init__()

        self.jumps_encoder = nn.Embedding(2 * jump_size + 1, embedding_size)
        self.project_encoded_actions = nn.Linear(2 * embedding_size, embedding_size)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode the actions.

        ---
        Args:
            actions: The actions to encode.
                Shape of [batch_size, 2].

        ---
        Returns:
            The encoded actions.
                Shape of [batch_size, embedding_size].
        """
        delta_x = actions[:, 0]
        delta_y = actions[:, 1]

        delta_x = self.jumps_encoder(delta_x)
        delta_y = self.jumps_encoder(delta_y)

        encoded_actions = torch.concat((delta_x, delta_y), dim=1)
        encoded_actions = self.project_encoded_actions(encoded_actions)
        return encoded_actions


class GRUPolicy(nn.Module):
    def __init__(
        self,
        n_channels: int,
        patch_size: int,
        vit_num_tokens: int,
        vit_hidden_size: int,
        vit_num_layers: int,
        gru_hidden_size: int,
        gru_num_layers: int,
        jump_size: int,
    ):
        super().__init__()
        self.jump_size = jump_size

        self.action_encoder = ActionEncoder(jump_size, gru_hidden_size)
        self.vit_encoder = ViTEncoder(
            n_channels=n_channels,
            hidden_size=vit_hidden_size,
            n_tokens=vit_num_tokens,
            n_layers=vit_num_layers,
            image_size=patch_size,
            output_size=gru_hidden_size,
        )
        self.gru = nn.GRU(
            input_size=gru_hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
        )
        self.heads = nn.ModuleDict(
            {
                "jumps_x": nn.Linear(gru_hidden_size, 2 * jump_size + 1),
                "jumps_y": nn.Linear(gru_hidden_size, 2 * jump_size + 1),
            }
        )

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
                Shape of [batch_size, 2].
            memory: The memory of the previous step.
                Shape of [gru_num_layers, batch_size, gru_hidden_size].

        ---
        Returns:
            action: The action to take.
                Shape of [batch_size, n_actions].
            memory: The memory of the current step.
                Shape of [gru_num_layers, batch_size, gru_hidden_size].
        """
        # Project the image to [batch_size, gru_hidden_size].
        x = self.vit_encoder(x)

        # Add the action embeddings.
        x = x + self.action_encoder(actions)

        # Run the GRU.
        x = x.unsqueeze(0)  # Add a fictive time dimension.
        x, memory = self.gru(x, memory)
        x = x.squeeze(0)  # Remove the fictive dimension.

        # Compute the actions.
        predicted_actions = {key: self.heads[key](x) for key in self.heads.keys()}
        return predicted_actions, memory
