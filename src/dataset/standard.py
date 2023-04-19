"""Standard detection dataset.

The expected format is:
    - A list of images paths.
    - A corresponding list of annotations paths.
    - Each annotation is of the form:
        - class_id, x1, y1, x2, y2
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset, random_split
from torchvision.io import ImageReadMode, read_image


class StandardDataset(Dataset):
    def __init__(self, images: list[Path], annotations: list[Path]):
        self.images = images
        self.annotations = annotations

        assert len(self.images) == len(self.annotations)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Load image from disk.
        image = read_image(str(self.images[index]), ImageReadMode.RGB)
        bboxes = StandardDataset.read_bbox(self.annotations[index])
        return image, bboxes

    def __len__(self):
        return len(self.images)

    @staticmethod
    def read_bbox(bbox_path: Path) -> torch.Tensor:
        """Read bounding boxes from the file.
        Ignores the class information.
        """
        bboxes = []
        with open(bbox_path, "r") as f:
            for line in f:
                coordinates = line.strip().split()[1:]
                coordinates = [int(float(coord)) for coord in coordinates]
                bboxes.append(coordinates)
        bboxes = torch.as_tensor(bboxes, dtype=torch.long)
        return bboxes

    @classmethod
    def load_from_file(cls, file: Path):
        """Read the images and annotations paths from the given file
        and return them.
        """
        images, annotations = [], []
        with open(file, "r") as f:
            for line in f:
                image_path, annotation_path = line.strip().split()
                images.append(Path(image_path))
                annotations.append(Path(annotation_path))

        return cls(images, annotations)

    @classmethod
    def load_from_dir(cls, dir_path: Path) -> tuple:
        """Return a train and test dataset loaded from the directory.

        It tests if there's a `train.txt` and `test.txt` present in the directory.
        If so, it loads the images and annotations paths from those files.

        Otherwise, it expects the presence of a `all.txt` file. If so, the images
        and annotations paths are loaded from that file, and do a 80/20 split to create
        the two datasets.
        """
        if (dir_path / "train.txt").exists() and (dir_path / "test.txt").exists():
            train_dataset = cls.load_from_file(dir_path / "train.txt")
            test_dataset = cls.load_from_file(dir_path / "test.txt")
            return train_dataset, test_dataset

        if (dir_path / "all.txt").exists():
            dataset = cls.load_from_file(dir_path / "all.txt")
            train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
            return train_dataset, test_dataset

        raise FileNotFoundError("No dataset found in the given directory.")
