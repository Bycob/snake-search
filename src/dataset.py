import torch
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.transforms import ToTensor

from .env import BBox, NeedleEnv, Position


class CelebADataset(Dataset):
    """Load the specified split of the CelebA dataset.
    It only select the right eye landmark and discard the rest.

    The images are of shape [3, 218, 178].
    """

    def __init__(self, split: str):
        self.dataset = CelebA(
            root="./data",
            split=split,
            target_type="landmarks",
            download=True,
        )
        self.transform = ToTensor()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load an image and its right eye landmark.

        ---
        Args:
            index: index of the image to load.

        ---
        Returns:
            image: The torch image tensor.
                Shape of [num_channels, height, width].
            right_eye: The right eye landmark (tuple (x, y)).
                Shape of [2,].
        """
        image, landmarks = self.dataset[index]
        image = self.transform(image)
        right_eye = landmarks[0:2]
        return image, right_eye

    @staticmethod
    def landmark_to_bbox(landmark: torch.Tensor, image: torch.Tensor) -> BBox:
        """Transform a landmark to a bounding box.
        It does so by taking the landmark as the center of a square and
        adding a margin of 5% of the image size.

        ---
        Args:
            landmark: The right eye landmark (tuple (x, y)).
                Shape of [2,].
            image: The torch image tensor.
                Shape of [num_channels, height, width].

        ---
        Returns:
            The bounding box.
        """
        width = image.shape[2]
        bbox_size = int(0.05 * width)
        return BBox(
            up_left=Position(
                x=landmark[0] - bbox_size // 2,
                y=landmark[1] - bbox_size // 2,
            ),
            bottom_right=Position(
                x=landmark[0] + bbox_size // 2,
                y=landmark[1] + bbox_size // 2,
            ),
        )

    def __len__(self):
        return len(self.dataset)


class NeedleDataset(Dataset):
    """Use the CelebADataset and generate the bbox usable by the NeedleEnv.
    Since the environment handle a batch of images, we do not instantiate here.
    """

    def __init__(self, celeb_dataset: CelebADataset):
        self.celeb_dataset = celeb_dataset

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[BBox]]:
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
        """
        image, landmark = self.celeb_dataset[index]
        bbox = self.celeb_dataset.landmark_to_bbox(landmark, image)
        # We only load one bbox, at least for now.
        bboxes = [bbox]
        return image, bboxes


def init_datasets() -> tuple[NeedleDataset, NeedleDataset]:
    """Initialize the train and test datasets.

    ---
    Returns:
        train_dataset: The train dataset.
        test_dataset: The test dataset.
    """
    celeb_dataset = CelebADataset("train")
    train_dataset = NeedleDataset(celeb_dataset)

    celeb_dataset = CelebADataset("test")
    test_dataset = NeedleDataset(celeb_dataset)

    return train_dataset, test_dataset
