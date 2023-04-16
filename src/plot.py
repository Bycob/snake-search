from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from .env import BBox, Position


def plot_image(
    axe: plt.Axes, image: np.ndarray, patch_size: int, cmap: Optional[str] = "gray"
):
    # Increase lightness of the image.
    image = image * 0.8 + 0.2
    axe.imshow(image, cmap=cmap, vmin=0, vmax=1)
    axe.set_xticks(np.arange(0, image.shape[1], patch_size))
    axe.set_yticks(np.arange(0, image.shape[0], patch_size))
    axe.grid(visible=True, color="white")


def plot_bbox(axe: plt.Axes, bbox: BBox, color: str = "green"):
    up_left = bbox.up_left
    bottom_right = bbox.bottom_right

    axe.plot(
        [up_left.x, up_left.x],
        [up_left.y, bottom_right.y],
        color=color,
        alpha=0.6,
    )
    axe.plot(
        [up_left.x, bottom_right.x],
        [bottom_right.y, bottom_right.y],
        color=color,
        alpha=0.6,
    )
    axe.plot(
        [bottom_right.x, bottom_right.x],
        [bottom_right.y, up_left.y],
        color=color,
        alpha=0.6,
    )
    axe.plot(
        [bottom_right.x, up_left.x],
        [up_left.y, up_left.y],
        color=color,
        alpha=0.6,
    )


def plot_patches(
    axe: plt.Axes,
    patches: np.ndarray,
    positions: list[Position],
    height: int,
    width: int,
):
    patch_h, patch_w, n_channels = patches[0].shape
    image = np.zeros((height, width, n_channels))
    for patch, position in zip(patches, positions):
        image[
            position.y : position.y + patch_h, position.x : position.x + patch_w
        ] = patch
    axe.imshow(image, vmin=0, vmax=1, alpha=0.3)


def plot_trajectory(
    image: torch.Tensor,
    positions: torch.Tensor,
    patch_size: int,
    bboxes: Optional[list[BBox]] = None,
) -> torch.Tensor:
    """Plot the model predictions onto the image.
    The patches that the model visited are plotted in a progressive red scale.

    ---
    Args:
        image: Original image.
            Shape of [n_channels, height, width].
        positions: List of positions visited by the model.
            Positions are in patch coordinates.
            Shape of [n_patches, (y, x)].
        patch_size: Size of the patches.
        bboxes: List of all bounding boxes of the image.

    ---
    Returns:
        image_prediction: Image with the patches in progressive red scale.
            Shape of [n_channels, height, width].
    """
    figure = plt.figure()
    axe = figure.gca()

    # To numpy.
    image = image.cpu().numpy()
    positions = positions.cpu().numpy()

    # To matplotlib shape.
    image = image.transpose(1, 2, 0)

    # Parse positions.
    parsed_positions = [
        Position(p[0] * patch_size, p[1] * patch_size) for p in positions
    ]

    # To progressive gray scale markers.
    markers = np.ones(
        (len(parsed_positions), patch_size, patch_size, image.shape[2]),
        dtype=np.float32,
    )
    min_range = 0.3
    for marker_id, marker in enumerate(markers):
        coeff = marker_id / len(markers)  # In range [0, 1].
        coeff = min_range + coeff * (1 - min_range)  # In range [min_range, 1].
        markers[marker_id] = marker * coeff

    # To red scale.
    markers[:, :, :, 1] = 0
    markers[:, :, :, 2] = 0

    plot_image(axe, image, patch_size, None)
    plot_patches(axe, markers, parsed_positions, image.shape[0], image.shape[1])

    if bboxes is not None:
        for bbox in bboxes:
            plot_bbox(axe, bbox, color="blue")

    # Get figure data to numpy array.
    canvas = figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    image = data.reshape(canvas.get_width_height()[::-1] + (3,))
    figure.clear()
    plt.close(figure)

    # To valid visdom image tensor.
    image = image / 255  # To [0, 1] range.
    image = torch.FloatTensor(image)
    image = image.permute(2, 0, 1)  # Right dimensions for visdom.

    return image
