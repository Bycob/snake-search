import torch
from kornia.geometry.boxes import Boxes
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize


class NeedleDataset(Dataset):
    """Use the CelebADataset and generate the bbox usable by the NeedleEnv.
    Since the environment handle a batch of images, we do not instantiate here.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

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
        image, bboxes = self.dataset[index]
        return image, bboxes

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def padded_collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor]], patch_size: int
    ) -> tuple[torch.Tensor, Boxes]:
        """Collate the batch of images and bboxes.
        The images are stacked in a tensor and the bboxes are stacked
        in a kornia boxes object.

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
            bounding_boxes: The padded bounding boxes, kornia format.
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
        delta_h = delta_h if delta_h != patch_size else 0
        delta_w = delta_w if delta_w != patch_size else 0
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
        bboxes = Boxes.from_tensor(bboxes, mode="xyxy")
        return images, bboxes

    @staticmethod
    def resize_collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor]], patch_size: int
    ) -> tuple[torch.Tensor, Boxes]:
        """Collate the batch of images and bboxes.
        The images are stacked in a tensor and the bboxes are stacked
        in a kornia boxes object.

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
            bounding_boxes: The padded bounding boxes, kornia format.
        """
        images, bboxes = [], []

        # Resize the images and scale the bounding boxes accordingly.
        for image, bbox in batch:
            _, height, width = image.shape
            delta_h = patch_size - height % patch_size
            delta_w = patch_size - width % patch_size
            delta_h = delta_h if delta_h != patch_size else 0
            delta_w = delta_w if delta_w != patch_size else 0

            # Resize the image.
            image = resize(image, [height + delta_h, width + delta_w], antialias=True)

            # Rescale the bounding boxes.
            # They are of shape [n_bboxes, 4], with format "xyxy".
            for i in [0, 2]:
                bbox[:, i] = (bbox[:, i] * (width + delta_w) / width).long()

            for i in [1, 3]:
                bbox[:, i] = (bbox[:, i] * (height + delta_h) / height).long()

            images.append(image)
            bboxes.append(bbox)

        # Find the max height and width.
        max_height, max_width = 0, 0
        for image in images:
            height, width = image.shape[1:]
            max_height = max(max_height, height)
            max_width = max(max_width, width)

        # Pad the images.
        for i, image in enumerate(images):
            height, width = image.shape[1:]
            pad_h = max_height - height
            pad_w = max_width - width
            images[i] = torch.nn.functional.pad(
                image, (0, pad_w, 0, pad_h), mode="constant", value=0
            )

        images = torch.stack(images, dim=0)
        bboxes = Boxes.from_tensor(bboxes, mode="xyxy")
        return images, bboxes
