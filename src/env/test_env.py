"""Test the batched env functionalities.

To run the tests, use the following command:
    `python3 -m pytest --import-mode importlib src/`
"""
import pytest
import torch

from ..dataset import NeedleDataset
from .env import NeedleEnv


def test_parse_bboxes():
    batch_size = 10
    width, height = 500, 500
    patch_size = 100
    max_ep_len = 10
    images = [torch.randn(3, height, width) for _ in range(batch_size)]

    # Simple case: only one bbox per image, in a single patch.
    bboxes = [torch.LongTensor([[0, 0, 20, 30]]) for _ in range(batch_size)]
    batch = [(image, bbox) for image, bbox in zip(images, bboxes)]
    batch_images, batch_bboxes = NeedleDataset.collate_fn(batch, patch_size)
    env = NeedleEnv(batch_images, batch_bboxes, patch_size, max_ep_len)
    bboxes, bbox_masks = env.parse_bboxes(bboxes)

    assert bboxes.shape == torch.Size(
        [batch_size, env.n_vertical_patches, env.n_horizontal_patches, 1, 4]
    )
    assert bbox_masks.shape == torch.Size(
        [batch_size, env.n_vertical_patches, env.n_horizontal_patches, 1]
    )

    # The bboxes should all be located in the first patch only.
    assert torch.all(bbox_masks[:, 0, 0, 0] == 1)
    assert torch.all(bbox_masks[:, 1:, :, :] == 0)
    assert torch.all(bbox_masks[:, :, 1:, :] == 0)

    # The bboxes should remain unchanged.
    assert torch.all(bboxes[:, 0, 0, 0] == torch.FloatTensor([0, 0, 20, 30]))

    # Harder case: One bbox per image, in multiple patches.
    bboxes = [torch.LongTensor([[10, 5, 120, 130]]) for _ in range(batch_size)]
    batch = [(image, bbox) for image, bbox in zip(images, bboxes)]
    batch_images, batch_bboxes = NeedleDataset.collate_fn(batch, patch_size)
    env = NeedleEnv(batch_images, batch_bboxes, patch_size, max_ep_len)
    bboxes, bbox_masks = env.parse_bboxes(bboxes)

    # Make sure the bbox is well located across the patches.
    assert torch.all(bbox_masks[:, 0, 0, 0] == 1)
    assert torch.all(bbox_masks[:, 1, 0, 0] == 1)
    assert torch.all(bbox_masks[:, 0, 1, 0] == 1)
    assert torch.all(bbox_masks[:, 1, 1, 0] == 1)
    assert torch.all(bbox_masks[:, 2:, :, :] == 0)
    assert torch.all(bbox_masks[:, :, 2:, :] == 0)

    # Make sure the bbox is cut to remain in the patches.
    assert torch.all(bboxes[:, 0, 0, 0] == torch.FloatTensor([10, 5, 99, 99]))
    assert torch.all(bboxes[:, 1, 0, 0] == torch.FloatTensor([10, 0, 99, 30]))
    assert torch.all(bboxes[:, 0, 1, 0] == torch.FloatTensor([0, 5, 20, 99]))
    assert torch.all(bboxes[:, 1, 1, 0] == torch.FloatTensor([0, 0, 20, 30]))


@pytest.mark.parametrize(
    "batch_size, width, height, patch_size",
    [
        (10, 500, 500, 100),
        (15, 600, 400, 20),
    ],
)
def test_patches(batch_size: int, width: int, height: int, patch_size: int):
    max_ep_len = 10
    images = [torch.randn(3, height, width) for _ in range(batch_size)]
    bboxes = [torch.LongTensor([[0, 0, 20, 30]]) for _ in range(batch_size)]
    batch = [(image, bbox) for image, bbox in zip(images, bboxes)]
    batch_images, batch_bboxes = NeedleDataset.collate_fn(batch, patch_size)
    env = NeedleEnv(batch_images, batch_bboxes, patch_size, max_ep_len)

    positions = torch.zeros((batch_size, 2), dtype=torch.long)
    positions[:, 0] = torch.randint(
        low=0, high=env.n_vertical_patches, size=(batch_size,)
    )
    positions[:, 1] = torch.randint(
        low=0, high=env.n_horizontal_patches, size=(batch_size,)
    )
    env.positions = positions

    target_patches = list()
    for batch_id, position in enumerate(positions):
        patch = batch_images[
            batch_id,
            :,
            position[0] * patch_size : (position[0] + 1) * patch_size,
            position[1] * patch_size : (position[1] + 1) * patch_size,
        ]
        target_patches.append(patch)

    target_patches = torch.stack(target_patches, dim=0)

    assert torch.all(env.patches == target_patches)


@pytest.mark.parametrize(
    "batch_size, width, height, patch_size",
    [
        (10, 500, 500, 100),
        (15, 600, 400, 20),
    ],
)
def test_movements(batch_size: int, width: int, height: int, patch_size: int):
    max_ep_len = 10
    images = [torch.randn(3, height, width) for _ in range(batch_size)]
    bboxes = [torch.LongTensor([[0, 0, 20, 30]]) for _ in range(batch_size)]
    batch = [(image, bbox) for image, bbox in zip(images, bboxes)]
    batch_images, batch_bboxes = NeedleDataset.collate_fn(batch, patch_size)
    env = NeedleEnv(batch_images, batch_bboxes, patch_size, max_ep_len)
    positions = torch.zeros((batch_size, 2), dtype=torch.long)
    positions[:, 0] = torch.randint(
        low=0, high=env.n_vertical_patches, size=(batch_size,)
    )
    positions[:, 1] = torch.randint(
        low=0, high=env.n_horizontal_patches, size=(batch_size,)
    )
    env.positions = positions

    for _ in range(10):
        movements = torch.randint(low=-5, high=5, size=(batch_size, 2))
        env.step(movements)

        for batch_id, move in enumerate(movements):
            positions[batch_id] += move
            positions[batch_id][0] = positions[batch_id][0] % env.n_vertical_patches
            positions[batch_id][1] = positions[batch_id][1] % env.n_horizontal_patches

        assert torch.all(env.positions == positions)


def test_tiles_reached():
    batch_size = 2
    height, width = 30, 40
    patch_size = 10
    max_ep_len = 10
    images = [torch.randn(3, height, width) for _ in range(batch_size)]
    bboxes = [torch.LongTensor([[0, 0, 20, 29]]) for _ in range(batch_size)]
    batch = [(image, bbox) for image, bbox in zip(images, bboxes)]
    batch_images, batch_bboxes = NeedleDataset.collate_fn(batch, patch_size)
    env = NeedleEnv(batch_images, batch_bboxes, patch_size, max_ep_len)

    env.positions = torch.LongTensor(
        [
            [0, 0],
            [1, 2],
        ]
    )

    tiles_reached = torch.BoolTensor(
        [
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
        ]
    )
    assert torch.all(env.tiles_reached == tiles_reached)


def test_convert_bboxes_to_masks():
    batch_size = 10
    width, height = 500, 500
    patch_size = 100
    max_ep_len = 10
    images = [torch.randn(3, height, width) for _ in range(batch_size)]

    # Simple case: only one bbox per image, in a single patch.
    bboxes = [torch.LongTensor([[0, 0, 20, 30]]) for _ in range(batch_size)]
    batch = [(image, bbox) for image, bbox in zip(images, bboxes)]
    batch_images, batch_bboxes = NeedleDataset.collate_fn(batch, patch_size)
    env = NeedleEnv(batch_images, batch_bboxes, patch_size, max_ep_len)

    masks = env.convert_bboxes_to_masks(batch_bboxes)
    _, parsed_masks = env.parse_bboxes(bboxes)
    parsed_masks = parsed_masks.max(dim=-1).values
    assert torch.all(masks == parsed_masks)

    # Harder case: One bbox per image, in multiple patches.
    bboxes = [torch.LongTensor([[10, 5, 120, 130]]) for _ in range(batch_size)]
    batch = [(image, bbox) for image, bbox in zip(images, bboxes)]
    batch_images, batch_bboxes = NeedleDataset.collate_fn(batch, patch_size)
    env = NeedleEnv(batch_images, batch_bboxes, patch_size, max_ep_len)

    masks = env.convert_bboxes_to_masks(batch_bboxes)
    _, parsed_masks = env.parse_bboxes(bboxes)
    parsed_masks = parsed_masks.max(dim=-1).values
    assert torch.all(masks == parsed_masks)

    # Add padding to the bounding boxes, which should have no impact.
    bboxes = [torch.LongTensor([[10, 5, 120, 130]]) for _ in range(batch_size)]
    bboxes[0] = torch.LongTensor([[10, 5, 120, 130], [0, 5, 4, 10]])
    batch = [(image, bbox) for image, bbox in zip(images, bboxes)]
    batch_images, batch_bboxes = NeedleDataset.collate_fn(batch, patch_size)
    env = NeedleEnv(batch_images, batch_bboxes, patch_size, max_ep_len)

    masks = env.convert_bboxes_to_masks(batch_bboxes)
    _, parsed_masks = env.parse_bboxes(bboxes)
    parsed_masks = parsed_masks.max(dim=-1).values
    assert torch.all(masks == parsed_masks)
