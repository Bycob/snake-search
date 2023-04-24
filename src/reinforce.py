from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import kornia.augmentation as aug
import torch
import wandb
from kornia.geometry.boxes import Boxes
from torch.distributions import Categorical
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .draw import draw_image_prediction
from .env import NeedleEnv
from .model import GRUPolicy
from .transform import random_horizontal_flip


class Reinforce:
    """The REINFORCE algorithm.
    It used to train the model, on a batched environment.
    """

    def __init__(
        self,
        model: GRUPolicy,
        optimizer: Optimizer,
        train_loader: DataLoader,
        test_loader: DataLoader,
        patch_size: int,
        max_ep_len: int,
        n_iterations: int,
        log_every: int,
        plot_every: int,
        device: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.patch_size = patch_size
        self.max_ep_len = max_ep_len
        self.n_iterations = n_iterations
        self.log_every = log_every
        self.plot_every = plot_every
        self.device = device

        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.augmentations = aug.AugmentationSequential(
            aug.RandomHorizontalFlip(p=0.5),
            aug.RandomAffine(degrees=5, translate=(1 / 4, 1 / 3), p=0.5),
            data_keys=["input", "bbox"],
        )

    @torch.no_grad()
    def augment_batch(
        self, images: torch.Tensor, bboxes: Boxes
    ) -> tuple[torch.Tensor, Boxes]:
        """Apply augmentations to a batch of images and bboxes."""
        # image_dtype = images.dtype
        # images = images.to(bboxes.dtype)
        # images, bboxes = self.augmentations(images, bboxes)
        # images = images.to(image_dtype)
        # return images, bboxes
        images, bboxes = random_horizontal_flip(images, bboxes, p=0.5)
        return images, bboxes

    def sample_from_logits(
        self, logits: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = torch.zeros(
            (logits["jumps_x"].shape[0], 2),
            dtype=torch.long,
            device=self.device,
        )
        logprobs = torch.zeros(
            (logits["jumps_x"].shape[0], 2),
            dtype=torch.float,
            device=self.device,
        )
        for action_id, action_name in enumerate(["jumps_x", "jumps_y"]):
            categorical = Categorical(logits=logits[action_name])
            sampled_actions = categorical.sample()
            actions[:, action_id] = sampled_actions
            logprobs[:, action_id] = categorical.log_prob(sampled_actions)

        return actions, logprobs

    def rollout(self, env: NeedleEnv) -> dict[str, torch.Tensor]:
        """Do a rollout on the given environment.
        Returns the rewards, returns and logprobs of the rollout.
        """
        rewards = torch.zeros(
            (env.batch_size, env.max_ep_len),
            device=self.device,
        )
        logprobs = torch.zeros(
            (env.batch_size, env.max_ep_len),
            device=self.device,
        )
        masks = torch.zeros(
            (env.batch_size, env.max_ep_len),
            dtype=torch.bool,
            device=self.device,
        )
        percentages = torch.zeros(
            (env.batch_size,),
            device=self.device,
        )

        memory = None
        actions = torch.zeros((env.batch_size, 2), dtype=torch.long, device=self.device)
        patches, _ = env.reset()

        for step_id in range(env.max_ep_len):
            logits, memory = self.model(patches, actions, memory)
            print(logits)
            actions, logprobs_ = self.sample_from_logits(logits)
            patches, step_rewards, terminated, truncated, infos = env.step(
                actions - self.model.jump_size
            )

            rewards[:, step_id] = step_rewards
            logprobs[:, step_id] = logprobs_.sum(dim=1)
            masks[:, step_id] = ~terminated
            percentages = infos["percentages"]

            if torch.all(terminated | truncated):
                # All environments are done.
                break

        # The last terminated state is not counted in the masks,
        # so we need to shift the masks by 1 to make sure we include id.
        masks = torch.roll(masks, shifts=1, dims=(1,))
        masks[:, 0] = True

        rewards = torch.flip(rewards, dims=(1,))
        masks = torch.flip(masks, dims=(1,))
        cumulated_returns = torch.cumsum(rewards * masks, dim=1)
        cumulated_returns = torch.flip(cumulated_returns, dims=(1,))
        rewards = torch.flip(rewards, dims=(1,))
        masks = torch.flip(masks, dims=(1,))

        return {
            "rewards": rewards,
            "returns": cumulated_returns,
            "logprobs": logprobs,
            "masks": masks,
            "percentages": percentages,
        }

    def compute_metrics(
        self, rollout: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Compute the metrics of the given rollout.

        ---
        Args:
            rollout: A dictionary containing the rewards, returns and logprobs.

        ---
        Returns:
            The metrics, containing the loss.
        """
        metrics = dict()
        returns = rollout["returns"]
        masks = rollout["masks"]

        advantages = (returns - returns.mean(dim=0, keepdim=True)) / (
            returns.std(dim=0, keepdim=True) + 1e-8
        )
        metrics["loss"] = (
            -(rollout["logprobs"] * advantages * masks).sum() / masks.sum()
        )
        metrics["returns"] = (rollout["rewards"] * masks).sum(dim=1).mean()
        metrics["percentages"] = rollout["percentages"].mean()
        metrics["episode-length"] = masks.sum(dim=1).float().mean()
        return metrics

    def launch_training(self, group: str, config: dict[str, Any], mode: str = "online"):
        """Train the model using REINFORCE.
        The logs are sent to Weights & Biases.

        ---
        Args:
            group: The name of the group of experiments.
            config: A dictionary of hyperparameters.
            mode: The mode of the Weights & Biases run.
        """
        print(f"Launching REINFORCE on device {self.device}")
        print(f"Training dataset size: {len(self.train_loader.dataset):,}")
        print(f"Test dataset size: {len(self.test_loader.dataset):,}")

        self.model.to(self.device)

        train_iter = iter(self.train_loader)
        test_iter = iter(self.test_loader)

        with wandb.init(
            project="snake-search",
            entity="pierrotlc",
            group=group,
            config=config,
            mode=mode,
        ) as run:
            # Log gradients and model parameters.
            run.watch(self.model)

            for step_id in tqdm(range(self.n_iterations)):
                self.model.train()
                images, bboxes = next(train_iter)
                images, bboxes = images.to(self.device), bboxes.to(self.device)
                images, bboxes = self.augment_batch(images, bboxes)
                env = NeedleEnv(images, bboxes, self.patch_size, self.max_ep_len)

                rollout = self.rollout(env)
                metrics = self.compute_metrics(rollout)

                self.optimizer.zero_grad()
                metrics["loss"].backward()
                self.optimizer.step()

                metrics = dict()
                if step_id % self.log_every == 0:
                    for iter_loader, name in [
                        (train_iter, "train"),
                        (test_iter, "test"),
                    ]:
                        loader_metrics = self.test_model(iter_loader, 2)
                        for key, value in loader_metrics.items():
                            metrics[f"{name}/{key}"] = value

                if step_id % self.plot_every == 0:
                    # Log the trajectories on a batch of images.
                    plots = self.get_predictions(train_iter, augment=True)
                    metrics["trajectories/train"] = wandb.Image(plots / 255)
                    plots = self.get_predictions(test_iter, augment=False)
                    metrics["trajectories/test"] = wandb.Image(plots / 255)

                    self.checkpoint(step_id)

                if metrics:
                    run.log(metrics)

    @torch.no_grad()
    def test_model(
        self, iter_loader: DataLoader, n_iters: int
    ) -> dict[str, torch.Tensor]:
        """Test the model on the given loader, returns the computed metrics."""
        self.model.eval()
        all_metrics = defaultdict(list)
        for _ in range(n_iters):
            images, bboxes = next(iter_loader)
            images, bboxes = images.to(self.device), bboxes.to(self.device)

            env = NeedleEnv(images, bboxes, self.patch_size, self.max_ep_len)

            rollout = self.rollout(env)
            metrics = self.compute_metrics(rollout)

            for key, value in metrics.items():
                all_metrics[key].append(value.cpu())

        return {key: torch.stack(values).mean() for key, values in all_metrics.items()}

    def get_predictions(
        self,
        iter_loader: Iterator,
        n_predictions: int = 16,
        augment: bool = False,
    ) -> torch.Tensor:
        """Sample from the loader and make predictions on the sampled images."""
        images, bboxes = next(iter_loader)
        images, bboxes = images.to(self.device), bboxes.to(self.device)
        images = images[:n_predictions]
        bboxes = bboxes[:n_predictions]
        if augment:
            images, bboxes = self.augment_batch(images, bboxes)
        env = NeedleEnv(images, bboxes, self.patch_size, self.max_ep_len)
        plots = self.predict(env)
        return plots

    @torch.no_grad()
    def predict(self, env: NeedleEnv) -> torch.Tensor:
        """Evaluates the model on a batch of images.
        Return a plot of its trajectories on all images.

        ---
        Args:
            env: The environment to evaluate the model on.

        ---
        Returns:
            The images of all predicted trajectories of the model.
                Shape of [batch_size, 3, height, width].
        """
        self.model.eval()
        memory = None
        actions = torch.zeros((env.batch_size, 2), dtype=torch.long, device=self.device)
        patches, infos = env.reset()

        positions = torch.zeros(
            (env.batch_size, env.max_ep_len + 1, 2), dtype=torch.long, device=env.device
        )
        positions[:, 0] = infos["positions"]

        masks = torch.zeros(
            (env.batch_size, env.max_ep_len + 1),
            dtype=torch.bool,
            device=self.device,
        )
        masks[:, 0] = True

        for step_id in range(env.max_ep_len):
            logits, memory = self.model(patches, actions, memory)
            # actions = logits.argmax(dim=1)  # Greedy policy.
            actions, _ = self.sample_from_logits(logits)

            patches, _, terminated, _, infos = env.step(actions - self.model.jump_size)
            positions[:, step_id + 1] = infos["positions"]
            masks[:, step_id + 1] = ~terminated

        # The last terminated state is not counted in the masks,
        # so we need to shift the masks by 1 to make sure we include id.
        masks = torch.roll(masks, shifts=1, dims=(1,))
        masks[:, 0] = True

        images = [
            draw_image_prediction(image, pos[mask], bboxes, env.patch_size)
            for image, pos, mask, bboxes in zip(
                env.images, positions, masks, env.bboxes.to_tensor()
            )
        ]
        images = torch.stack(images, dim=0)
        return images

    def checkpoint(self, step_id: int):
        """Save the model's parameters."""
        state_dicts = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state_dicts, self.checkpoint_dir / f"{step_id % 20}.pt")
