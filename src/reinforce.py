from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Tuple, List, Dict

import einops
import torch
import wandb
from kornia.geometry.boxes import Boxes
from torch.distributions import Categorical
from torch.nn.utils import clip_grad
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .draw import draw_gif_prediction, draw_image_prediction
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
        n_glimps_levels: int,
        entropy_weight: float,
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
        self.n_glimps_levels = n_glimps_levels
        self.entropy_weight = entropy_weight
        self.n_iterations = n_iterations
        self.log_every = log_every
        self.plot_every = plot_every
        self.device = device

        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

    @torch.no_grad()
    def augment_batch(
        self, images: torch.Tensor, bboxes: Boxes
    ) -> Tuple[torch.Tensor, Boxes]:
        """Apply augmentations to a batch of images and bboxes."""
        images, bboxes = random_horizontal_flip(images, bboxes, p=0.5)
        return images, bboxes

    def sample_from_logits(
        self, logits: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        entropies = torch.zeros(
            (logits["jumps_x"].shape[0], 2),
            dtype=torch.float,
            device=self.device,
        )
        for action_id, action_name in enumerate(["jumps_x", "jumps_y"]):
            categorical = Categorical(logits=logits[action_name])
            sampled_actions = categorical.sample()
            actions[:, action_id] = sampled_actions
            logprobs[:, action_id] = categorical.log_prob(sampled_actions)
            entropies[:, action_id] = categorical.entropy()

        return actions, logprobs, entropies

    def rollout(self, env: NeedleEnv) -> Dict[str, torch.Tensor]:
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
        entropies = torch.zeros(
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
            patches = einops.rearrange(patches, "b g c h w -> b (g c) h w")
            logits, memory = self.model(patches, actions, memory)
            actions, logprobs_, entropies_ = self.sample_from_logits(logits)
            patches, step_rewards, terminated, truncated, infos = env.step(
                actions - self.model.jump_size
            )

            rewards[:, step_id] = step_rewards
            logprobs[:, step_id] = logprobs_.sum(dim=1)
            entropies[:, step_id] = entropies_.sum(dim=1)
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
            "entropies": entropies,
            "masks": masks,
            "percentages": percentages,
        }

    def compute_metrics(
        self, rollout: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
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

        #advantages = (returns - returns.mean(dim=0, keepdim=True)) / (
        #    returns.std(dim=0, keepdim=True) + 1e-8
        #)
        advantages = returns
        metrics["action-loss"] = (
            -(rollout["logprobs"] * advantages * masks).sum() / masks.sum()
        )
        metrics["entropy-loss"] = -(rollout["entropies"] * masks).sum() / masks.sum()
        metrics["loss"] = (
            metrics["action-loss"] + self.entropy_weight * metrics["entropy-loss"]
        )
        metrics["returns"] = (rollout["rewards"] * masks).sum(dim=1).mean()
        metrics["percentages"] = rollout["percentages"].mean()
        metrics["episode-length"] = masks.sum(dim=1).float().mean()
        return metrics

    def launch_training(self, group: str, config: Dict[str, Any], mode: str = "online"):
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
            project="needle",
            entity="bycob_gh",
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
                env = NeedleEnv(
                    images,
                    bboxes,
                    self.patch_size,
                    self.max_ep_len,
                    self.n_glimps_levels,
                )

                rollout = self.rollout(env)
                metrics = self.compute_metrics(rollout)

                self.optimizer.zero_grad()
                metrics["loss"].backward()
                clip_grad.clip_grad_value_(self.model.parameters(), 1)
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

                    env = self.load_env(train_iter, augment=True)
                    positions, masks = self.predict(env)
                    plots = self.plot_trajectories(env, positions, masks)
                    metrics["trajectories/train-images"] = wandb.Image(plots / 255)
                    gifs = self.animate_trajectories(env, positions[:1], masks)
                    self.tensor_to_gif(gifs[0], "train.gif")

                    env = self.load_env(test_iter, augment=False)
                    positions, masks = self.predict(env)
                    plots = self.plot_trajectories(env, positions, masks)
                    metrics["trajectories/test-images"] = wandb.Image(plots / 255)
                    gifs = self.animate_trajectories(env, positions[:1], masks)
                    self.tensor_to_gif(gifs[0], "test.gif")

                    self.checkpoint(step_id)

                if metrics:
                    run.log(metrics)

    @torch.no_grad()
    def test_model(
        self, iter_loader: DataLoader, n_iters: int
    ) -> Dict[str, torch.Tensor]:
        """Test the model on the given loader, returns the computed metrics."""
        self.model.eval()
        all_metrics = defaultdict(list)
        for _ in range(n_iters):
            images, bboxes = next(iter_loader)
            images, bboxes = images.to(self.device), bboxes.to(self.device)

            env = NeedleEnv(
                images, bboxes, self.patch_size, self.max_ep_len, self.n_glimps_levels
            )

            rollout = self.rollout(env)
            metrics = self.compute_metrics(rollout)

            for key, value in metrics.items():
                all_metrics[key].append(value.cpu())

        return {key: torch.stack(values).mean() for key, values in all_metrics.items()}

    def load_env(
        self,
        iter_loader: Iterator,
        n_predictions: int = 16,
        augment: bool = False,
    ) -> NeedleEnv:
        """Sample from the loader and make predictions on the sampled images."""
        images, bboxes = next(iter_loader)
        images, bboxes = images.to(self.device), bboxes.to(self.device)
        images = images[:n_predictions]
        bboxes = bboxes[:n_predictions]
        if augment:
            images, bboxes = self.augment_batch(images, bboxes)
        env = NeedleEnv(
            images, bboxes, self.patch_size, self.max_ep_len, self.n_glimps_levels
        )
        return env

    @torch.no_grad()
    def predict(self, env: NeedleEnv) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the model on a batch of images.
        Return a plot of its trajectories on all images.

        ---
        Args:
            env: The environment to evaluate the model on.

        ---
        Returns:
            positions: The positions visited by the model.
                Shape of [batch_size, max_ep_len + 1, 2].
            masks: The masks of the visited positions.
                Shape of [batch_size, max_ep_len + 1].
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
            patches = einops.rearrange(patches, "b g c h w -> b (g c) h w")
            logits, memory = self.model(patches, actions, memory)
            # actions = logits.argmax(dim=1)  # Greedy policy.
            actions, _, _ = self.sample_from_logits(logits)

            patches, _, terminated, _, infos = env.step(actions - self.model.jump_size)
            positions[:, step_id + 1] = infos["positions"]
            masks[:, step_id + 1] = ~terminated

        # The last terminated state is not counted in the masks,
        # so we need to shift the masks by 1 to make sure we include id.
        masks = torch.roll(masks, shifts=1, dims=(1,))
        masks[:, 0] = True

        return positions, masks

    def plot_trajectories(
        self, env: NeedleEnv, positions: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """Plot the trajectories of the model on a batch of images.

        ---
        Args:
            env: The environment used to generate the trajectories.
            positions: The positions visited by the model.
                Shape of [batch_size, max_ep_len + 1, 2].
            masks: The masks of the visited positions.
                Shape of [batch_size, max_ep_len + 1].
        ---
        Returns:
            The images of all predicted trajectories of the model.
                Shape of [batch_size, 3, height, width].
        """
        images = [
            draw_image_prediction(image, pos[mask], bboxes, env.patch_size)
            for image, pos, mask, bboxes in zip(
                env.images[:, 0], positions, masks, env.bboxes.to_tensor()
            )
        ]
        images = torch.stack(images, dim=0)

        return images

    def animate_trajectories(
        self, env: NeedleEnv, positions: torch.Tensor, masks: torch.Tensor
    ) -> List[torch.Tensor]:
        """Make a gif from the trajectories of the model on a batch of images.

        ---
        Args:
            env: The environment used to generate the trajectories.
            positions: The positions visited by the model.
                Shape of [batch_size, max_ep_len + 1, 2].
            masks: The masks of the visited positions.
                Shape of [batch_size, max_ep_len + 1].
        ---
        Returns:
            The gifs generated from the given trajectories.
                List of tensors of shape [max_ep_len + 1, 3, height, width].
        """
        gifs = [
            draw_gif_prediction(image, pos[mask], bboxes, env.patch_size)
            for image, pos, mask, bboxes in zip(
                env.images[:, 0], positions, masks, env.bboxes.to_tensor()
            )
        ]
        return gifs

    @staticmethod
    def tensor_to_gif(frames: torch.Tensor, filename: str):
        """Build a gif from a tensor of images.

        ---
        Args:
            frames: The tensor of images.
                Shape of [n_frames, 3, height, width].
            filename: The name of the gif file.
        """
        import imageio

        frames = frames.permute(0, 2, 3, 1).cpu().numpy()
        imageio.mimsave(filename, frames, duration=1000 / 3)

    def checkpoint(self, step_id: int):
        """Save the model's parameters."""
        state_dicts = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state_dicts, self.checkpoint_dir / f"{step_id % 20}.pt")
