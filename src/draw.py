import numpy as np
import torch
from PIL import Image, ImageDraw


def draw_grid(draw: ImageDraw.ImageDraw, patch_size: int):
    width, height = draw.im.size

    for x in range(0, width, patch_size):
        draw.line((x, 0, x, height), fill="white", width=1)
    for y in range(0, height, patch_size):
        draw.line((0, y, width, y), fill="white", width=1)


def draw_bbox(draw: ImageDraw.ImageDraw, bbox: torch.Tensor, color: str = "green"):
    bbox = bbox.cpu().numpy()
    draw.rectangle(
        (bbox[0], bbox[1], bbox[2], bbox[3]),
        outline=color,
        width=3,
    )


def draw_positions(
    image: torch.Tensor, positions: torch.Tensor, patch_size: int
) -> torch.Tensor:
    min_range = 0.3
    positions = positions * patch_size  # To pixel coordinates.
    for position_idx, position in enumerate(positions):
        coeff = position_idx / len(positions)  # In range [0, 1].
        coeff = min_range + (1 - min_range) * coeff  # In range [min_range, 1].
        red_mask = torch.zeros(
            (patch_size, patch_size, 3), dtype=torch.uint8, device=image.device
        )
        red_mask[:, :, 0] = 255 * coeff

        patch = image[
            position[0] : position[0] + patch_size,
            position[1] : position[1] + patch_size,
        ]
        merged_patch = (red_mask * 0.3 + patch * 0.7).type(torch.uint8)

        image[
            position[0] : position[0] + patch_size,
            position[1] : position[1] + patch_size,
        ] = merged_patch

    return image


def draw_image_prediction(
    image: torch.Tensor,
    positions: torch.Tensor,
    bboxes: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    # Torch operations.
    image = image.permute(1, 2, 0)
    image = draw_positions(image, positions, patch_size)

    # PIL operations.
    pil_image = Image.fromarray(image.cpu().numpy())
    draw = ImageDraw.Draw(pil_image)

    draw_grid(draw, patch_size)

    for bbox in bboxes:
        draw_bbox(draw, bbox, color="green")

    # To torch.
    image = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1)
    image = image.type(torch.uint8)

    return image
