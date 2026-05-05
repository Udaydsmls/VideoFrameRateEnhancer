from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from CreatingModel import build_model
from CreatingModel.Losses import L1SSIMLoss
from ImageOperations.Dataset import FrameTripletDataset, build_triplets
from utilities.Checkpoints import (
    Checkpoint,
    checkpoint_path,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from utilities.Config import Config
from utilities.Devices import resolve_device


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float | None


class Trainer:
    """Train a frame interpolator end to end."""

    def __init__(self, config: Config, model_kwargs: dict | None = None) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.model: nn.Module = build_model(config.architecture, **(model_kwargs or {})).to(self.device)
        self.criterion = L1SSIMLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self._start_epoch = 0

    @classmethod
    def from_checkpoint(
        cls,
        config: Config,
        checkpoint: Path,
        model_kwargs: dict | None = None,
    ) -> "Trainer":
        ckpt = load_checkpoint(checkpoint, map_location="cpu")
        if ckpt.architecture != config.architecture:
            raise ValueError(
                f"Checkpoint architecture '{ckpt.architecture}' does not match "
                f"config architecture '{config.architecture}'."
            )
        trainer = cls(config, model_kwargs=model_kwargs)
        trainer.model.load_state_dict(ckpt.state_dict)
        trainer._start_epoch = ckpt.epoch
        return trainer

    def fit(self) -> list[EpochMetrics]:
        train_loader, val_loader = self._build_loaders()
        history: list[EpochMetrics] = []
        for offset in range(self.config.num_epochs):
            epoch = self._start_epoch + offset + 1
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False) if val_loader else None
            metrics = EpochMetrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
            history.append(metrics)
            self._log(metrics)
            self._save(epoch)
        return history

    def _build_loaders(self) -> tuple[DataLoader, DataLoader | None]:
        triplets = build_triplets(self.config.frames)
        if not triplets:
            raise RuntimeError(
                f"No frame triplets found under {self.config.frames}. "
                "Run the data flow first."
            )
        dataset = FrameTripletDataset(triplets, image_size=self.config.image_size)

        val_size = int(round(len(dataset) * self.config.validation_split))
        if val_size > 0 and len(dataset) - val_size > 0:
            train_set, val_set = random_split(
                dataset,
                [len(dataset) - val_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
        else:
            train_set, val_set = dataset, None

        train_loader = DataLoader(
            train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )
        val_loader = (
            DataLoader(
                val_set,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.device.type == "cuda",
            )
            if val_set is not None
            else None
        )
        return train_loader, val_loader

    def _run_epoch(self, loader: DataLoader | None, *, train: bool) -> float:
        if loader is None:
            return 0.0
        self.model.train(train)
        total = 0.0
        count = 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for prev, nxt, mid in loader:
                prev = prev.to(self.device, non_blocking=True)
                nxt = nxt.to(self.device, non_blocking=True)
                mid = mid.to(self.device, non_blocking=True)

                pred = self.model(prev, nxt)
                loss = self.criterion(pred, mid)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

                total += loss.item() * prev.shape[0]
                count += prev.shape[0]
        return total / max(count, 1)

    def _save(self, epoch: int) -> None:
        path = checkpoint_path(self.config.checkpoints, self.config.architecture, epoch)
        save_checkpoint(
            Checkpoint(
                architecture=self.config.architecture,
                state_dict=self.model.state_dict(),
                epoch=epoch,
                metadata={"learning_rate": self.config.learning_rate},
            ),
            path,
        )

    def _log(self, metrics: EpochMetrics) -> None:
        if metrics.val_loss is None:
            print(f"epoch {metrics.epoch:>4d} | train {metrics.train_loss:.4f}")
        else:
            print(
                f"epoch {metrics.epoch:>4d} | train {metrics.train_loss:.4f} | "
                f"val {metrics.val_loss:.4f}"
            )


def train_model(config: Config, *, resume: bool = False, model_kwargs: dict | None = None) -> list[EpochMetrics]:
    """Convenience entry point that trains from scratch or resumes a checkpoint."""
    config.ensure_directories()
    if resume:
        latest = find_latest_checkpoint(config.checkpoints, config.architecture)
        if latest is None:
            print("No checkpoint to resume from; starting fresh.")
            trainer = Trainer(config, model_kwargs=model_kwargs)
        else:
            print(f"Resuming from {latest}")
            trainer = Trainer.from_checkpoint(config, latest, model_kwargs=model_kwargs)
    else:
        trainer = Trainer(config, model_kwargs=model_kwargs)
    return trainer.fit()
