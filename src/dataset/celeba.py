from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.transforms import PILToTensor
from typing import Tuple


class CelebADataset(Dataset):
    """Load the specified split of the CelebA dataset.
    It only select the right eye landmark and discard the rest.

    The images are of shape [3, 218, 178].
    """

    def __init__(self, split: str, root: Path):
        self.dataset = CelebA(
            root=str(root),
            split=split,
            target_type="landmarks",
            download=True,
        )
        self.transform = PILToTensor()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load an image and its left eye bounding box.

        ---
        Args:
            index: index of the image to load.

        ---
        Returns:
            image: The torch image tensor.
                Shape of [num_channels, height, width].
            bboxes: The landmarks as bounding boxes.
                Shape of [n_bboxes, 4].
        """
        image, landmarks = self.dataset[index]
        image = self.transform(image)
        landmarks = landmarks.reshape(-1, 2).long()
        # We only load the left eye landmark.
        left_eye = landmarks[:1,]
        bboxes = CelebADataset.landmarks_to_bbox(left_eye, image)
        return image, bboxes

    @staticmethod
    def landmarks_to_bbox(landmarks: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Transform landmarks to bounding boxes.
        It does so by taking the landmark as the center of a square and
        adding a margin of 5% of the image size.

        ---
        Args:
            landmarks: The left eye landmark (for example) as tuples `(x, y)`.
                Shape of [n_landmarks, 2].
            image: The torch image tensor.
                Shape of [num_channels, height, width].

        ---
        Returns:
            The bounding boxes as tuples `(x1, y1, x2, y2)`.
                Shape of [n_landmarks, 4].
        """
        width = image.shape[2]
        bbox_size = int(0.05 * width)
        bboxes = torch.concat((landmarks, landmarks), dim=1)
        bboxes[:, :2] -= bbox_size // 2
        bboxes[:, 2:] += bbox_size // 2
        return bboxes

    def __len__(self):
        return len(self.dataset)
