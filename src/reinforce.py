from typing import Any

import torch
from torch.distributions import Categorical
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

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

    def rollout(self, env: NeedleEnv) -> dict[str, torch.Tensor]:
        rewards = torch.zeros(
            env.batch_size,
            env.max_ep_len,
            device=self.device,
        )
        logprobs = torch.zeros(
            env.batch_size,
            env.max_ep_len,
            device=self.device,
        )

        memory = None
        patches, _ = env.reset()

        for step_id in range(env.max_ep_len):
            logits, memory = self.model(patches, memory)
            categorical = Categorical(logits=logits)
            actions = categorical.sample()

            patches, step_rewards, _, _, _ = env.step(actions)

            rewards[:, step_id] = step_rewards
            logprobs[:, step_id] = categorical.log_prob(actions)

        rewards = torch.flip(rewards, dims=(1,))
        cumulated_returns = torch.cumsum(rewards, dim=1)
        cumulated_returns = torch.flip(cumulated_returns, dims=(1,))
        rewards = torch.flip(rewards, dims=(1,))

        return {
            "rewards": rewards,
            "returns": cumulated_returns,
            "logprobs": logprobs,
        }

    def compute_metrics(
        self, rollout: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        metrics = dict()
        returns = rollout["returns"]

        advantages = (returns - returns.mean(dim=0, keepdim=True)) / (
            returns.std(dim=0, keepdim=True) + 1e-8
        )
        metrics["loss"] = -(rollout["logprobs"] * advantages).sum(dim=1).mean()
        metrics["returns"] = rollout["rewards"].sum(dim=1).mean()
        return metrics

    def launch_training(self, group: str, config: dict[str, Any]):
        print(f"Launching REINFORCE on device {self.device}")
        self.model.to(self.device)

        train_iter = iter(self.train_loader)

        for _ in tqdm(range(self.n_iterations)):
            images, bboxes = next(train_iter)
            images = images.to(self.device)
            env = NeedleEnv(images, bboxes, self.patch_size, self.max_ep_len)

            rollout = self.rollout(env)
            metrics = self.compute_metrics(rollout)

            self.optimizer.zero_grad()
            metrics["loss"].backward()
            self.optimizer.step()
