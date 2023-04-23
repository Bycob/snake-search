from enum import Enum

import einops
import gymnasium as gym
import torch
from kornia.geometry.boxes import Boxes
from torch import Tensor

# Define the available actions.
_ACTIONS = [
    "START",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "LEFT_UP",
    "RIGHT_UP",
    "LEFT_DOWN",
    "RIGHT_DOWN",
    "STOP",
]
Action = Enum("Action", _ACTIONS, start=0)

MOVES = [
    Action.LEFT,
    Action.RIGHT,
    Action.UP,
    Action.DOWN,
    Action.LEFT_UP,
    Action.RIGHT_UP,
    Action.LEFT_DOWN,
    Action.RIGHT_DOWN,
]


class NeedleEnv(gym.Env):
    def __init__(
        self,
        images: Tensor,
        bboxes: Boxes,
        patch_size: int,
        max_ep_len: int,
    ):
        """Creates a batched environment for the needle problem.

        ---
        Args:
            images: A batch of images, as uint8 RGB (saves memory).
                Shape of [batch_size, n_channels, height, width].
            bboxes: The bounding boxes of the batch.
                List of length `batch_size`, where each element is
                a tensor of shape [n_bboxes, 4].
        """
        assert images.shape[0] == bboxes.shape[0]
        assert len(images.shape) == 4

        self.images = images
        self.patch_size = patch_size
        self.max_ep_len = max_ep_len

        # Save the batch dimensions.
        self.batch_size, self.n_channels, self.height, self.width = images.shape

        # Make sure that the images are divisible by the patch size.
        assert self.height % self.patch_size == 0
        assert self.width % self.patch_size == 0

        # Number of patches in the images.
        self.n_vertical_patches = self.height // self.patch_size
        self.n_horizontal_patches = self.width // self.patch_size

        # The device is the same as the images.
        self.device = self.images.device

        # Observation and action spaces.
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.batch_size, self.n_channels, self.patch_size, self.patch_size),
        )
        # Vertical and horizontal movements along with the diagonals.
        self.action_space = gym.spaces.Discrete(len(Action))

        # Bounding boxes of the images.
        self.bbox_masks = self.convert_bboxes_to_masks(bboxes)
        self.bboxes = bboxes

        # Initialize some variables for the environment.
        self.init_env_variables()

    def init_env_variables(self):
        # Positions of the agents in the images.
        self.positions = torch.zeros(
            (self.batch_size, 2),
            dtype=torch.long,
            device=self.device,
        )

        # Visited patches by the agents.
        self.visited_patches = torch.zeros(
            (self.batch_size, self.n_vertical_patches, self.n_horizontal_patches),
            dtype=torch.bool,
            device=self.device,
        )

        # Number of steps taken by the agents.
        self.steps = torch.zeros(
            (self.batch_size,),
            dtype=torch.long,
            device=self.device,
        )

        self.terminated = torch.zeros(
            (self.batch_size,),
            dtype=torch.bool,
            device=self.device,
        )

    def reset(self) -> tuple[Tensor, dict]:
        """Reset the environment variables.
        Randomly initialize the positions of the agents.

        ---
        Returns:
            patches: The patches where are the agents.
                Shape of [batch_size, n_channels, patch_size, patch_size].
            infos: Additional infos.
        """
        self.init_env_variables()
        self.positions[:, 0] = torch.randint(
            low=0, high=self.n_vertical_patches, size=(self.batch_size,)
        )
        self.positions[:, 1] = torch.randint(
            low=0, high=self.n_horizontal_patches, size=(self.batch_size,)
        )
        self.visited_patches = self.visited_patches | self.tiles_reached
        self.terminated = self.scores == self.max_scores

        percentages = self.scores / self.bbox_masks.sum(dim=(1, 2))
        infos = {
            "positions": self.positions,
            "delta": torch.zeros_like(
                percentages, dtype=torch.float, device=self.device
            ),
            "just_finished": torch.zeros_like(
                percentages, dtype=torch.bool, device=self.device
            ),
            "percentages": percentages,
        }
        patches = self.patches / 255
        return patches, infos

    @torch.no_grad()
    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, dict]:
        """Apply the actions, compute the rewards and some contextual information
        and return them.

        ---
        Args:
            actions: The actions to apply.
                Shape of [batch_size,].
        ---
        Returns:
            patches: The patches where are the agents.
                Shape of [batch_size, n_channels, patch_size, patch_size].
            rewards: The reward of the agents.
                Shape of [batch_size,].
            terminated: Whether the environments are terminated (won).
                Shape of [batch_size,].
            truncated: Whether the environments are truncated (max steps reached).
                Shape of [batch_size,].
            infos: Additional infos.
        """
        previous_scores = self.scores

        # Apply the actions.
        movements = self.parse_actions(actions)
        self.apply_movements(movements)
        self.visited_patches = self.visited_patches | self.tiles_reached
        self.steps += 1

        # Compute the rewards and terminaisons.
        new_scores = self.scores
        delta_rewards = new_scores - previous_scores
        self.terminated |= new_scores == self.max_scores
        truncated = self.steps >= self.max_ep_len

        # Give a bonus reward for finishing the episode.
        finishing_reward = self.max_ep_len - self.steps
        just_finished = self.terminated & (delta_rewards != 0)
        finishing_reward.masked_fill_(~just_finished, 0)

        rewards = delta_rewards + 0 * finishing_reward
        rewards = rewards / self.max_scores

        percentages = new_scores / self.max_scores

        infos = {
            "positions": self.positions,
            "delta": delta_rewards,
            "just_finished": just_finished,
            "percentages": percentages,
        }

        patches = self.patches / 255

        return patches, rewards, self.terminated, truncated, infos

    def apply_movements(self, movements: Tensor):
        """Apply the movements to the agents.
        Make sure that the agents don't move outside the images.

        ---
        Args:
            movements: The movements to apply.
                Shape of [batch_size, 2].
        """
        self.positions = self.positions + movements

        # Clamp the positions to the image boundaries.
        self.positions[:, 0] = torch.clamp(
            self.positions[:, 0], min=0, max=self.n_vertical_patches - 1
        )
        self.positions[:, 1] = torch.clamp(
            self.positions[:, 1], min=0, max=self.n_horizontal_patches - 1
        )

    @property
    def tiles_reached(self) -> Tensor:
        """Compute the boolean masks marking each agent's position
        in the images.

        ---
        Returns:
            The boolean masks.
                Shape of [batch_size, n_vertical_patches, n_horizontal_patches].
        """
        # Positions in the form [batch_id, tile_id], where `tile_id` is
        # a one-dimensional index of the flattened two-dimensional coordinates.
        one_dim_positions = (
            self.positions[:, 0] * self.n_horizontal_patches + self.positions[:, 1]
        )
        # Positions in the form [batch_id x tile_id,].
        # Those positions are absolute indices in the batch of patches.
        tiles_per_images = self.n_horizontal_patches * self.n_vertical_patches
        offsets = torch.arange(
            start=0,
            end=self.batch_size * tiles_per_images,
            step=tiles_per_images,
            device=self.device,
        )
        flattened_positions = one_dim_positions + offsets

        # Build the masks.
        masks = torch.zeros(
            (self.batch_size, self.n_vertical_patches, self.n_horizontal_patches),
            dtype=torch.bool,
            device=self.device,
        )
        # Use the absolute positions to index inside the masks.
        masks.flatten()[flattened_positions] = True

        return masks

    @property
    def patches(self) -> Tensor:
        """Fetch the patches of the images that the agents have reached.

        ---
        Returns:
            The batch of patches reached.
                Shape of [batch_size, n_channels, patch_size, patch_size].
        """
        # Compute the indices of the pixels of the first patch.
        row_indices = torch.arange(
            start=0,
            end=self.patch_size,
            device=self.device,
        )
        offsets = torch.arange(
            start=0,
            end=self.patch_size * self.width,
            step=self.width,
            device=self.device,
        )
        # This adds the offset to each row index, making it
        # a [patch_size, patch_size] tensor of indices.
        pixel_indices = row_indices.unsqueeze(0) + offsets.unsqueeze(1)

        # Add a starting offset to the indices depending on the
        # agent's position in the images.
        pixel_indices = einops.rearrange(pixel_indices, "h w -> (h w)")
        offsets = (
            self.positions[:, 0] * (self.width * self.patch_size)
            + self.positions[:, 1] * self.patch_size
        )
        # Add the offset of the first pixel index of each agent to the global
        # patch indices, making it a [batch_size, patch_size x patch_size] tensor.
        pixel_indices = pixel_indices.unsqueeze(0) + offsets.unsqueeze(1)

        # Add the channels dimension.
        pixel_indices = einops.repeat(pixel_indices, "b p -> b c p", c=self.n_channels)

        # Finally gather the pixels.
        images = einops.rearrange(self.images, "b c h w -> b c (h w)")
        patches = torch.gather(images, dim=2, index=pixel_indices)
        patches = einops.rearrange(patches, "b c (h w) -> b c h w", h=self.patch_size)
        return patches

    @property
    def scores(self) -> Tensor:
        """Compute the score of the agents.
        They win one point for each patch visited that contains
        a bounding box.

        ---
        Returns:
            The score of the agents.
                Shape of [batch_size,].
        """
        # Logical "OR" on the `n_bboxes` dimension.
        visited_bboxes = self.bbox_masks & self.visited_patches
        scores = visited_bboxes.sum(dim=(1, 2))
        return scores

    @property
    def max_scores(self) -> Tensor:
        """Compute the maximum possible score of each agent."""
        return self.bbox_masks.sum(dim=(1, 2))

    @property
    def closest_bbox_coord(self) -> Tensor:
        """Find the coordinates of the closest non-visited bbox
        for each agent.

        ---
        Returns:
            The closest bbox coordinates.
                Shape of [batch_size, 2].
        """
        visited_bboxes = self.bbox_masks & self.visited_patches
        nonvisited_bboxes = self.bbox_masks & ~visited_bboxes

        # Build a map of coordinates to compute the distances.
        # It is of shape [n_vertical_patches, n_horizontal_patches, 2].
        y = torch.arange(start=0, end=self.n_vertical_patches, device=self.device)
        x = torch.arange(start=0, end=self.n_horizontal_patches, device=self.device)
        coordinates = torch.cartesian_prod(y, x)
        coordinates = einops.rearrange(
            coordinates, "(h w) c -> h w c", h=self.n_vertical_patches
        )
        # coordinates = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)

        # Compute the absolute distances between the agents and the bboxes.
        # It is of shape [batch_size, n_vertical_patches x n_horizontal_patches].
        coordinates = einops.rearrange(coordinates, "(b h) w c -> b (h w) c", b=1)
        positions = einops.rearrange(self.positions, "b (p c) -> b p c", p=1)
        distances = torch.cdist(coordinates.float(), positions.float(), p=1)
        distances = distances.squeeze(dim=-1)

        # Ignore distances that are not about non-visited bboxes.
        nonvisited_bboxes = einops.rearrange(nonvisited_bboxes, "b h w -> b (h w)")
        distances.masked_fill_(~nonvisited_bboxes, float("+inf"))

        # Find the closest non-visited bbox.
        # It is of shape [batch_size, 2].
        closest_bbox_patch_id = torch.argmin(distances, dim=-1)
        closest_bbox_patch_coord = torch.stack(
            [
                closest_bbox_patch_id // self.n_horizontal_patches,
                closest_bbox_patch_id % self.n_horizontal_patches,
            ],
            dim=1,
        )
        return closest_bbox_patch_coord

    @property
    def best_actions(self) -> Tensor:
        """Find the actions that move the agents towards
        their closest non-visited bbox patch.

        ---
        Returns:
            The best actions.
                Shape of [batch_size,].
        """
        directions = self.closest_bbox_coord - self.positions
        movements = torch.sign(directions)
        actions = NeedleEnv.parse_movements(movements)
        return actions

    def init_sample(self) -> dict[str, Tensor]:
        sample = {
            "patches": torch.zeros(
                (
                    self.max_ep_len,
                    self.batch_size,
                    self.n_channels,
                    self.patch_size,
                    self.patch_size,
                ),
                dtype=torch.float,
                device=self.device,
            ),
            "actions_taken": torch.zeros(
                (self.max_ep_len, self.batch_size), dtype=torch.long, device=self.device
            ),
            "best_actions": torch.zeros(
                (self.max_ep_len, self.batch_size), dtype=torch.long, device=self.device
            ),
            "positions": torch.zeros(
                (self.max_ep_len, self.batch_size, 2),
                dtype=torch.long,
                device=self.device,
            ),
            "rewards": torch.zeros(
                (self.max_ep_len, self.batch_size),
                dtype=torch.float,
                device=self.device,
            ),
            "masks": torch.zeros(
                (self.max_ep_len, self.batch_size), dtype=torch.bool, device=self.device
            ),
        }
        return sample

    def generate_sample(self) -> dict[str, Tensor]:
        """Simulate a full episode and return the results.

        ---
        Returns:
            A dictionary containing the following keys:
                * patches: The patches visited by the agents.
                    Shape of [batch_size, max_ep_len, n_channels, patch_size, patch_size].
                * positions: The positions of the agents.
                    Shape of [batch_size, max_ep_len, 2].
                * actions_taken: The actions taken by the agents at step $i - 1$ to
                    reach the patches at step $i$.
                    Shape of [batch_size, max_ep_len].
                * best_actions: The best actions to take at step $i$ when facing the
                    patches at step $i$.
                    Shape of [batch_size, max_ep_len].
                * rewards: The rewards obtained by the agents at step $i$, after taking
                    its actions.
                    Shape of [batch_size, max_ep_len].
                * masks: The masks indicating whether the episode is terminated or not.
                    It indicates the padding of the tensors.
                    Shape of [batch_size, max_ep_len].
        """
        sample = self.init_sample()
        patches, infos = self.reset()
        best_actions = (
            torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
            * Action.START.value
        )
        terminated = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        for step_id in range(self.max_ep_len):
            # Save the current state.
            sample["patches"][step_id] = patches
            sample["positions"][step_id] = infos["positions"]
            sample["actions_taken"][step_id] = best_actions
            sample["masks"][step_id] = ~terminated

            # Play the best action.
            best_actions = self.best_actions
            patches, rewards, terminated, _, infos = self.step(best_actions)

            # Save the results of the taken action.
            sample["best_actions"][step_id] = best_actions
            sample["rewards"][step_id] = rewards

        # Swap the dimensions `batch_size` and `max_ep_len`.
        for key, value in sample.items():
            sample[key] = value.transpose(0, 1)

        return sample

    def convert_bboxes_to_masks(self, bboxes: Boxes) -> Tensor:
        """Convert the bounding boxes to masks.

        ---
        Args:
            bboxes: The bounding boxes of the batch, in kornia format.

        ---
        Returns:
            masks: The masks of the bounding boxes, to know which patch
                contains at least a bounding box.
                Shape of [batch_size, n_vertical_patches, n_horizontal_patches].
        """
        # padded_bboxes = bboxes.to_tensor()
        masks = bboxes.to_mask(self.height, self.width)
        masks = masks.max(dim=1).values  # Merge the channels.

        # Logical OR of the masks, to reduce to the patch dimensions.
        masks = torch.nn.functional.max_pool2d(masks, self.patch_size)
        return masks.bool()

    def parse_bboxes(self, bboxes: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Return the bounding boxes of the images as a tensor.
        Each bounding box of an image is given an id, which will serve as an
        index in the dimension `n_bboxes` of the tensors.

        This implementation is slow and not necessary.
        It keeps the original bboxes in the patches, which can be useful
        if you want to train a detector. Since the goal of the agent is only
        to find the patches where there are bounding boxes, it is not necessary.

        ---
        Args:
            bboxes: The bounding boxes of the batch.
                List of length `batch_size`, where each element is
                a tensor of shape [n_bboxes, 4].

        ---
        Returns:
            bboxes: The bounding boxes.
                Shape of [batch_size, n_vertical_patches, n_horizontal_patches, n_bboxes, 4].
            masks: The masks of the bounding boxes, to remove the padding
                at the `n_bboxes` dimension.
                Shape of [batch_size, n_vertical_patches, n_horizontal_patches, n_bboxes].
        """
        n_bboxes = max([bboxes_.shape[0] for bboxes_ in bboxes])
        tensor_bboxes = torch.zeros(
            (
                self.batch_size,
                self.n_vertical_patches,
                self.n_horizontal_patches,
                n_bboxes,
                4,
            ),
            dtype=torch.long,
            device=self.device,
        )
        masks = torch.zeros(
            (
                self.batch_size,
                self.n_vertical_patches,
                self.n_horizontal_patches,
                n_bboxes,
            ),
            dtype=torch.bool,
            device=self.device,
        )

        def place_bbox_recursive(
            bbox: Tensor,
            bbox_id: int,
            bboxes: Tensor,
            masks: Tensor,
            patch_size: int,
        ):
            """Place the bounding box in the tensor.
            If the bounding box is too big to fit in a patch, it will be split
            across the patches, by recursively calling this function.
            """
            # Compute the coordinates of the bounding box inside the patch.
            x1 = bbox[0] % patch_size
            y1 = bbox[1] % patch_size
            x2 = x1 + (bbox[2] - bbox[0])
            y2 = y1 + (bbox[3] - bbox[1])

            # Compute the coordinates of the patch.
            patch_x = bbox[0] // patch_size
            patch_y = bbox[1] // patch_size

            # Make sure the bounding box does not go outside the patch.
            x2_clampled = torch.clamp(x2, max=patch_size - 1)
            y2_clampled = torch.clamp(y2, max=patch_size - 1)

            # Save the bounding box.
            bboxes[patch_y, patch_x, bbox_id] = torch.LongTensor(
                [x1, y1, x2_clampled, y2_clampled]
            ).to(bboxes.device)
            masks[patch_y, patch_x, bbox_id] = True

            # Recursively place the bounding box in the other patches,
            # if the bounding box cross the borders of the current patch.
            if x2 - x2_clampled > 0:
                # The bounding box cross the right border of the patch.
                n_bbox = torch.LongTensor(
                    [
                        (patch_x + 1) * patch_size,
                        bbox[1],
                        bbox[2],
                        patch_y * patch_size + y2_clampled,
                    ]
                )
                place_bbox_recursive(n_bbox, bbox_id, bboxes, masks, patch_size)

            if y2 - y2_clampled > 0:
                # The bounding box cross the bottom border of the patch.
                n_bbox = torch.LongTensor(
                    [
                        bbox[0],
                        (patch_y + 1) * patch_size,
                        patch_x * patch_size + x2_clampled,
                        bbox[3],
                    ]
                )
                place_bbox_recursive(n_bbox, bbox_id, bboxes, masks, patch_size)

            if (x2 - x2_clampled > 0) and (y2 - y2_clampled > 0):
                # The bounding box cross the bottom-right corner of the patch.
                n_bbox = torch.LongTensor(
                    [
                        (patch_x + 1) * patch_size,
                        (patch_y + 1) * patch_size,
                        bbox[2],
                        bbox[3],
                    ]
                )
                place_bbox_recursive(n_bbox, bbox_id, bboxes, masks, patch_size)

        for batch_id, bboxes_ in enumerate(bboxes):
            for bbox_id, bbox in enumerate(bboxes_):
                place_bbox_recursive(
                    bbox,
                    bbox_id,
                    tensor_bboxes[batch_id],
                    masks[batch_id],
                    self.patch_size,
                )

        return tensor_bboxes, masks

    @staticmethod
    def parse_actions(actions: Tensor) -> Tensor:
        """Translate the action ids to actual position movements.

        ---
        Args:
            actions: The actions to apply.
                Shape of [batch_size,].
        ---
        Returns:
            The movements to apply to the agents encoded as tuples `(delta_y, delta_x)`.
                Shape of [batch_size, 2].
        """
        device = actions.device
        movements = torch.zeros(
            (actions.shape[0], 2),
            dtype=torch.long,
            device=device,
        )

        # Masks.
        up = actions == Action.UP.value
        down = actions == Action.DOWN.value
        left = actions == Action.LEFT.value
        right = actions == Action.RIGHT.value
        left_up = actions == Action.LEFT_UP.value
        right_up = actions == Action.RIGHT_UP.value
        left_down = actions == Action.LEFT_DOWN.value
        right_down = actions == Action.RIGHT_DOWN.value

        # Up.
        movements[up | left_up | right_up, 0] = -1
        # Down.
        movements[down | left_down | right_down, 0] = 1
        # Right.
        movements[right | right_up | right_down, 1] = 1
        # Left.
        movements[left | left_up | left_down, 1] = -1

        return movements

    @staticmethod
    def parse_movements(movements: Tensor) -> Tensor:
        """Translate the movements to action ids.

        ---
        Args:
            movements: The movements to apply.
                Shape of [batch_size, 2].

        ---
        Returns:
            The corresponding action ids.
                Shape of [batch_size,].
        """
        actions = torch.zeros(
            (movements.shape[0],),
            dtype=torch.long,
            device=movements.device,
        )

        # Masks.
        up = movements[:, 0] == -1
        down = movements[:, 0] == 1
        left = movements[:, 1] == -1
        right = movements[:, 1] == 1
        no_vertical = movements[:, 0] == 0
        no_horizontal = movements[:, 1] == 0

        # Horizontal and vertical movements.
        actions[up & no_horizontal] = Action.UP.value
        actions[down & no_horizontal] = Action.DOWN.value
        actions[left & no_vertical] = Action.LEFT.value
        actions[right & no_vertical] = Action.RIGHT.value

        # Diagonal movements.
        actions[up & left] = Action.LEFT_UP.value
        actions[up & right] = Action.RIGHT_UP.value
        actions[down & left] = Action.LEFT_DOWN.value
        actions[down & right] = Action.RIGHT_DOWN.value

        return actions
