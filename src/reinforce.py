from typing import Any

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import NeedleDataset
from .env import NeedleEnv
from .model import GRUPolicy


class Reinforce:
    def __init__(
        self,
        model: GRUPolicy,
        optimizer: Optimizer,
        train_loader: DataLoader,
        test_loader: DataLoader,
        patch_size: int,
        max_ep_len: int,
        n_iterations: int,
        device: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.patch_size = patch_size
        self.max_ep_len = max_ep_len
        self.n_iterations = n_iterations
        self.device = device

    def rollout(self, env: NeedleEnv):
        pass

    def launch_training(self, group: str, config: dict[str, Any]):
        print(f"Launching on device {self.device}")
        self.model.to(self.device)

        for _ in tqdm(range(self.n_iterations)):
            for images, bboxes in self.train_loader:
                images = images.to(self.device)
                env = NeedleEnv(images, bboxes, self.patch_size, self.max_ep_len)
