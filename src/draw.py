import imageio
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
    image: torch.Tensor,
    positions: torch.Tensor,
    patch_size: int,
    min_range: float = 0.3,
    max_range: float = 1.0,
) -> torch.Tensor:
    assert 0 <= min_range <= max_range <= 1

    positions = positions * patch_size  # To pixel coordinates.
    for position_idx, position in enumerate(positions):
        coeff = position_idx / len(positions)  # In range [0, 1].
        coeff = min_range + (max_range - min_range) * coeff  # In range [min_range, 1].
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


@torch.no_grad()
def draw_image_prediction(
    image: torch.Tensor,
    positions: torch.Tensor,
    bboxes: torch.Tensor,
    patch_size: int,
    image_width: int = 500,
) -> torch.Tensor:
    image = image.clone()

    # Torch operations.
    image = image.permute(1, 2, 0)
    image = draw_positions(image, positions, patch_size)

    # PIL operations.
    pil_image = Image.fromarray(image.cpu().numpy())
    draw = ImageDraw.Draw(pil_image)

    draw_grid(draw, patch_size)

    for bbox in bboxes:
        draw_bbox(draw, bbox, color="green")

    # Resize image.
    image_height = int(image_width * image.shape[0] / image.shape[1])
    pil_image = pil_image.resize((image_width, image_height), resample=Image.BILINEAR)

    # To torch.
    image = torch.from_numpy(np.array(pil_image))
    image = image.permute(2, 0, 1)
    image = image.type(torch.uint8)

    return image


@torch.no_grad()
def draw_gif_prediction(
    image: torch.Tensor,
    positions: torch.Tensor,
    bboxes: torch.Tensor,
    patch_size: int,
    image_width: int = 500,
):
    def draw_image(
        image: torch.Tensor,
        previous_positions: torch.Tensor,
        current_position: torch.Tensor,
        bboxes: torch.Tensor,
        patch_size: int,
        image_width: int,
    ) -> np.ndarray:
        image = image.clone()

        # Torch operations.
        image = draw_positions(
            image,
            previous_positions,
            patch_size,
            min_range=0.1,
            max_range=0.1,
        )
        image = draw_positions(
            image,
            current_position.unsqueeze(0),
            patch_size,
            min_range=0.9,
            max_range=0.9,
        )

        # PIL operations.
        pil_image = Image.fromarray(image.cpu().numpy())
        draw = ImageDraw.Draw(pil_image)

        draw_grid(draw, patch_size)

        for bbox in bboxes:
            draw_bbox(draw, bbox, color="green")

        # Resize image.
        image_height = int(image_width * image.shape[0] / image.shape[1])
        pil_image = pil_image.resize(
            (image_width, image_height), resample=Image.BILINEAR
        )

        return np.array(pil_image)

    image = image.permute(1, 2, 0)

    gif = []
    for i in range(positions.shape[0]):
        previous_positions = positions[:i]
        current_position = positions[i]

        frame = draw_image(
            image,
            previous_positions,
            current_position,
            bboxes,
            patch_size,
            image_width,
        )
        gif.append(frame)

    imageio.mimwrite("trajectory.gif", gif, duration=500)
