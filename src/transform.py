"""Apply flips, rotations and translations to a 2D image.
It is implemented so that it can work on a uint8 torch tensor
on GPU. Sadly, it seems like common libraries such as `kornia`
did not implement this feature.

It takes into account the bounding boxes and transforms them as well.
"""
import torch
from einops import repeat
from kornia.geometry.boxes import Boxes


def scramble(images: torch.Tensor, boxes: Boxes) -> tuple[torch.Tensor, Boxes]:
    # Scramble the images and boxes.
    perm = torch.randperm(images.shape[0])
    images = images[perm]
    boxes = boxes[perm]
    return images, boxes


def horizontal_flip(images: torch.Tensor, boxes: Boxes) -> tuple[torch.Tensor, Boxes]:
    """Flip an image and its bounding boxes horizontally.

    ---
    Args:
        images: A batch of images.
            Shape of [batch_size, n_channels, height, width].
        boxes: The corresponding bounding boxes as kornia Boxes.

    ---
    Returns:
        The flipped images and bounding boxes.
    """
    # Flip images.
    flipped_images = torch.flip(images, dims=[3])

    # Flip boxes.
    batch_size, _, _, width = images.shape
    transform = torch.LongTensor(
        [
            [-1, 0, width],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    transform = repeat(transform, "i j -> b i j", b=batch_size)
    transform = transform.to(boxes.device)
    transform = transform.to(boxes.dtype)
    flipped_boxes = boxes.transform_boxes(transform)
    return flipped_images, flipped_boxes


def vertical_flip(images: torch.Tensor, boxes: Boxes) -> tuple[torch.Tensor, Boxes]:
    """Flip an image and its bounding boxes vertically.

    ---
    Args:
        images: A batch of images.
            Shape of [batch_size, n_channels, height, width].
        boxes: The corresponding bounding boxes as kornia Boxes.

    ---
    Returns:
        The flipped images and bounding boxes.
    """
    # Flip images.
    flipped_images = torch.flip(images, dims=[2])

    # Flip boxes.
    batch_size, _, height, _ = images.shape
    transform = torch.LongTensor(
        [
            [1, 0, 0],
            [0, -1, height],
            [0, 0, 1],
        ]
    )
    transform = repeat(transform, "i j -> b i j", b=batch_size)
    transform = transform.to(boxes.device)
    transform = transform.to(boxes.dtype)
    flipped_boxes = boxes.transform_boxes(transform)
    return flipped_images, flipped_boxes


def random_horizontal_flip(
    images: torch.Tensor, boxes: Boxes, p: float = 0.5
) -> tuple[torch.Tensor, Boxes]:
    """Randomly flip horizontally images and their bounding boxes.

    ---
    Args:
        images: A batch of images.
            Shape of [batch_size, n_channels, height, width].
        boxes: The corresponding bounding boxes as kornia Boxes.
        p: The probability of flipping each image.
            In practice, it is the ratio of flipped images.

    ---
    Returns:
        The randomly flipped images and bounding boxes.
    """
    images, boxes = scramble(images, boxes)
    tot_flips = int(images.shape[0] * p)
    flipped_images, flipped_boxes = horizontal_flip(
        images[:tot_flips], boxes[:tot_flips]
    )
    images[:tot_flips] = flipped_images
    boxes[:tot_flips] = flipped_boxes

    return images, boxes


def random_vertical_flip(
    images: torch.Tensor, boxes: Boxes, p: float = 0.5
) -> tuple[torch.Tensor, Boxes]:
    """Randomly flip vertically images and their bounding boxes.

    ---
    Args:
        images: A batch of images.
            Shape of [batch_size, n_channels, height, width].
        boxes: The corresponding bounding boxes as kornia Boxes.
        p: The probability of flipping each image.
            In practice, it is the ratio of flipped images.

    ---
    Returns:
        The randomly flipped images and bounding boxes.
    """
    images, boxes = scramble(images, boxes)
    tot_flips = int(images.shape[0] * p)
    flipped_images, flipped_boxes = vertical_flip(images[:tot_flips], boxes[:tot_flips])
    images[:tot_flips] = flipped_images
    boxes[:tot_flips] = flipped_boxes
    return images, boxes


def random_translate(
    images: torch.Tensor,
    boxes: Boxes,
    delta_x: float,
    delta_y: float,
) -> tuple[torch.Tensor, Boxes]:
    """Randomly translate images and their bounding boxes.
    Note: The images will warp around the edges, but not the boxes!

    ---
    Args:
        images: A batch of images.
            Shape of [batch_size, n_channels, height, width].
        boxes: The corresponding bounding boxes as kornia Boxes.
        delta_x: The maximum translation along the x-axis,
            between 0.0 and 1.0.
        delta_y: The maximum translation along the y-axis,
            between 0.0 and 1.0.

    ---
    Returns:
        The randomly translated images and bounding boxes.
    """
    # Randomly translate images.
    batch_size, _, height, width = images.shape
    translations_x = torch.randint(
        low=-int(delta_x * width),
        high=int(delta_x * width),
        size=(batch_size,),
        dtype=torch.int32,
    )
    translations_y = torch.randint(
        low=-int(delta_y * height),
        high=int(delta_y * height),
        size=(batch_size,),
        dtype=torch.int32,
    )
    translations = torch.stack([translations_x, translations_y], dim=1)

    # Translate images.
    for image_id, (image, translation) in enumerate(zip(images, translations)):
        image = torch.roll(image, shifts=translation[0].item(), dims=2)
        image = torch.roll(image, shifts=translation[1].item(), dims=1)
        images[image_id] = image

    # Translate boxes.
    translations = translations.to(boxes.device)
    translations = translations.to(boxes.dtype)
    boxes = boxes.translate(translations)

    return images, boxes
