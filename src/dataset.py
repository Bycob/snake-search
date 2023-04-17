from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.transforms import ToTensor


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
        self.transform = ToTensor()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
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


class NeedleDataset(Dataset):
    """Use the CelebADataset and generate the bbox usable by the NeedleEnv.
    Since the environment handle a batch of images, we do not instantiate here.
    """

    def __init__(self, celeb_dataset: CelebADataset):
        self.celeb_dataset = celeb_dataset

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load the celebA image and landmarks and returns
        the image and its landmarks.

        ---
        Args:
            index: index of the image to load.

        ---
        Returns:
            image: The torch image tensor.
                Shape of [num_channels, height, width].
            bboxes: The list of bounding boxes.
                Shape of [n_bboxes, 4].
        """
        image, bboxes = self.celeb_dataset[index]
        return image, bboxes

    def __len__(self):
        return len(self.celeb_dataset)

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor]], patch_size: int
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Collate the batch of images and bboxes.
        The images are stacked in a tensor and the bboxes are stacked in a list.

        The images can be of varying sizes. They are padded with zeros to match
        the biggest image. They are also padded to a multiple of `patch_size`.

        ---
        Args:
            batch: The batch of images and bboxes.
                List of tuple (image, bboxes).
            patch_size: The size of the patches, for padding.

        ---
        Returns:
            images: The batch of images.
                Shape of [batch_size, num_channels, height, width].
            bounding_boxes: The list of bounding boxes.
                List of length `batch_size`, where each element is
                a tensor of shape [n_bboxes, 4].
        """
        images, bboxes = [], []
        for image, bbox in batch:
            images.append(image)
            bboxes.append(bbox)

        # Find the max height and width.
        max_height, max_width = 0, 0
        for image in images:
            height, width = image.shape[1:]
            max_height = max(max_height, height)
            max_width = max(max_width, width)

        # Add padding to match a multiple of patch_size.
        delta_h = patch_size - max_height % patch_size
        delta_w = patch_size - max_width % patch_size
        final_height = max_height + delta_h
        final_width = max_width + delta_w

        # Pad the images.
        for i, image in enumerate(images):
            height, width = image.shape[1:]
            pad_h = final_height - height
            pad_w = final_width - width
            images[i] = torch.nn.functional.pad(
                image, (0, pad_w, 0, pad_h), mode="constant", value=0
            )

        images = torch.stack(images, dim=0)
        return images, bboxes
