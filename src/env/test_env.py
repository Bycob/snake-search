"""Test the batched env functionalities.

To run the tests, use the following command:
    `python3 -m pytest --import-mode importlib src/`
"""
import pytest
import torch
from torch import Tensor

from .env import Action, BBox, NeedleEnv, Position


@pytest.mark.parametrize(
    "actions, movements",
    [
        (
            torch.LongTensor([Action.UP.value, Action.DOWN.value]),
            torch.LongTensor([[-1, 0], [1, 0]]),
        ),
        (
            torch.LongTensor([Action.LEFT.value, Action.RIGHT.value]),
            torch.LongTensor([[0, -1], [0, 1]]),
        ),
        (
            torch.LongTensor([Action.LEFT_UP.value, Action.RIGHT_DOWN.value]),
            torch.LongTensor([[-1, -1], [1, 1]]),
        ),
        (
            torch.LongTensor([Action.LEFT_DOWN.value, Action.RIGHT_UP.value]),
            torch.LongTensor([[1, -1], [-1, 1]]),
        ),
    ],
)
def test_parse_actions(actions: Tensor, movements: Tensor):
    assert torch.all(NeedleEnv.parse_actions(actions) == movements)
    assert torch.all(NeedleEnv.parse_movements(movements) == actions)


def test_parse_bboxes():
    batch_size = 10
    width, height = 500, 500
    patch_size = 100
    max_ep_len = 10
    images = torch.randn(batch_size, 3, height, width)

    # Simple case: only one bbox per image, in a single patch.
    bboxes = [
        [
            BBox(
                up_left=Position(x=0, y=0),
                bottom_right=Position(x=20, y=30),
            )
        ]
        for _ in range(batch_size)
    ]
    env = NeedleEnv(images, bboxes, patch_size, max_ep_len)

    assert env.bboxes.shape == torch.Size(
        [batch_size, env.n_vertical_patches, env.n_horizontal_patches, 1, 4]
    )
    assert env.bbox_masks.shape == torch.Size(
        [batch_size, env.n_vertical_patches, env.n_horizontal_patches, 1]
    )

    # The bboxes should all be located in the first patch only.
    assert torch.all(env.bbox_masks[:, 0, 0, 0] == 1)
    assert torch.all(env.bbox_masks[:, 1:, :, :] == 0)
    assert torch.all(env.bbox_masks[:, :, 1:, :] == 0)

    # The bboxes should remain unchanged.
    assert torch.all(env.bboxes[:, 0, 0, 0] == torch.FloatTensor([0, 0, 20, 30]))

    # Harder case: One bbox per image, in multiple patches.
    bboxes = [
        [
            BBox(
                up_left=Position(x=10, y=5),
                bottom_right=Position(x=120, y=130),
            )
        ]
        for _ in range(batch_size)
    ]
    env = NeedleEnv(images, bboxes, patch_size, max_ep_len)

    # Make sure the bbox is well located across the patches.
    assert torch.all(env.bbox_masks[:, 0, 0, 0] == 1)
    assert torch.all(env.bbox_masks[:, 1, 0, 0] == 1)
    assert torch.all(env.bbox_masks[:, 0, 1, 0] == 1)
    assert torch.all(env.bbox_masks[:, 1, 1, 0] == 1)
    assert torch.all(env.bbox_masks[:, 2:, :, :] == 0)
    assert torch.all(env.bbox_masks[:, :, 2:, :] == 0)

    # Make sure the bbox is cut to remain in the patches.
    assert torch.all(env.bboxes[:, 0, 0, 0] == torch.FloatTensor([10, 5, 99, 99]))
    assert torch.all(env.bboxes[:, 1, 0, 0] == torch.FloatTensor([10, 0, 99, 30]))
    assert torch.all(env.bboxes[:, 0, 1, 0] == torch.FloatTensor([0, 5, 20, 99]))
    assert torch.all(env.bboxes[:, 1, 1, 0] == torch.FloatTensor([0, 0, 20, 30]))


@pytest.mark.parametrize(
    "batch_size, width, height, patch_size",
    [
        (10, 500, 500, 100),
        (15, 600, 400, 20),
    ],
)
def test_patches(batch_size: int, width: int, height: int, patch_size: int):
    max_ep_len = 10
    images = torch.randn(batch_size, 3, height, width)
    bboxes = [
        [
            BBox(
                up_left=Position(x=0, y=0),
                bottom_right=Position(x=20, y=30),
            )
        ]
        for _ in range(batch_size)
    ]
    env = NeedleEnv(images, bboxes, patch_size, max_ep_len)

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
        patch = images[
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
    images = torch.randn(batch_size, 3, height, width)
    bboxes = [
        [
            BBox(
                up_left=Position(x=0, y=0),
                bottom_right=Position(x=20, y=30),
            )
        ]
        for _ in range(batch_size)
    ]
    env = NeedleEnv(images, bboxes, patch_size, max_ep_len)
    positions = torch.zeros((batch_size, 2), dtype=torch.long)
    positions[:, 0] = torch.randint(
        low=0, high=env.n_vertical_patches, size=(batch_size,)
    )
    positions[:, 1] = torch.randint(
        low=0, high=env.n_horizontal_patches, size=(batch_size,)
    )
    env.positions = positions

    for _ in range(10):
        actions = torch.randint(low=0, high=len(Action), size=(batch_size,))
        env.step(actions)

        movements = NeedleEnv.parse_actions(actions)
        for batch_id, move in enumerate(movements):
            positions[batch_id] += move
            positions[batch_id][0] = max(
                0, min(env.n_vertical_patches - 1, positions[batch_id][0].item())
            )
            positions[batch_id][1] = max(
                0, min(env.n_horizontal_patches - 1, positions[batch_id][1].item())
            )

        assert torch.all(env.positions == positions)


def test_tiles_reached():
    batch_size = 2
    height, width = 30, 40
    patch_size = 10
    max_ep_len = 10
    images = torch.randn(batch_size, 3, height, width)
    bboxes = [
        [
            BBox(
                up_left=Position(x=0, y=0),
                bottom_right=Position(x=20, y=29),
            )
        ]
        for _ in range(batch_size)
    ]
    env = NeedleEnv(images, bboxes, patch_size, max_ep_len)

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


def test_closest_bbox_coord_and_best_actions():
    batch_size = 3
    height, width = 30, 40
    patch_size = 10
    max_ep_len = 10
    images = torch.randn(batch_size, 3, height, width)
    bboxes = [
        [
            BBox(
                up_left=Position(x=0, y=0),
                bottom_right=Position(x=5, y=22),
            ),
        ]
        for _ in range(batch_size)
    ]
    env = NeedleEnv(images, bboxes, patch_size, max_ep_len)

    # With no visited patches.
    env.positions = torch.LongTensor([[0, 1], [1, 2], [2, 3]])
    env.visited_patches = torch.zeros_like(env.visited_patches, dtype=torch.bool)
    closest_bbox_coord = torch.LongTensor(
        [
            [0, 0],
            [1, 0],
            [2, 0],
        ]
    )
    assert torch.all(env.closest_bbox_coord == closest_bbox_coord)
    best_actions = torch.LongTensor(
        [Action.LEFT.value, Action.LEFT.value, Action.LEFT.value]
    )
    assert torch.all(env.best_actions == best_actions)

    # With some visited patches.
    env.visited_patches = torch.zeros_like(env.visited_patches, dtype=torch.bool)
    env.visited_patches[0, 0, 0] = True
    env.visited_patches[2, 2, 0] = True
    closest_bbox_coord = torch.LongTensor(
        [
            [1, 0],
            [1, 0],
            [1, 0],
        ]
    )
    assert torch.all(env.closest_bbox_coord == closest_bbox_coord)
    best_actions = torch.LongTensor(
        [Action.LEFT_DOWN.value, Action.LEFT.value, Action.LEFT_UP.value]
    )
    assert torch.all(env.best_actions == best_actions)
