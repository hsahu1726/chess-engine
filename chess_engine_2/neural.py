from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import chess
import torch
from torch import nn
from torch.utils.data import Dataset

from chess_engine_2.data.dataset import INPUT_PLANES, board_to_planes
from chess_engine_2.encoding import POLICY_SIZE, move_to_policy_index


@dataclass(frozen=True)
class TrainingMetrics:
    samples: int
    batches: int
    policy_loss: float
    value_loss: float
    total_loss: float


@dataclass(frozen=True)
class MovePrediction:
    move_uci: str
    policy_index: int
    score: float


class ChessJsonlDataset(Dataset):
    def __init__(self, path: Path, max_samples: int | None = None):
        self.samples = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                self.samples.append(json.loads(line))
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        board = chess.Board(sample["fen"])
        planes = torch.tensor(board_to_planes(board), dtype=torch.float32)
        policy = torch.tensor(int(sample["policy_index"]), dtype=torch.long)
        value = torch.tensor(float(sample["value"]), dtype=torch.float32)
        return planes, policy, value


class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, POLICY_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return policy_logits, value


def train_one_epoch(
    model: PolicyValueNet,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    value_loss_weight: float = 1.0,
) -> TrainingMetrics:
    model.train()
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    total_samples = 0
    total_batches = 0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    total_loss_sum = 0.0

    for planes, policy_targets, value_targets in loader:
        planes = planes.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device)

        optimizer.zero_grad()
        policy_logits, value_predictions = model(planes)
        policy_loss = policy_loss_fn(policy_logits, policy_targets)
        value_loss = value_loss_fn(value_predictions, value_targets)
        loss = policy_loss + value_loss_weight * value_loss
        loss.backward()
        optimizer.step()

        batch_size = planes.shape[0]
        total_samples += batch_size
        total_batches += 1
        policy_loss_sum += policy_loss.item() * batch_size
        value_loss_sum += value_loss.item() * batch_size
        total_loss_sum += loss.item() * batch_size

    return TrainingMetrics(
        samples=total_samples,
        batches=total_batches,
        policy_loss=policy_loss_sum / total_samples,
        value_loss=value_loss_sum / total_samples,
        total_loss=total_loss_sum / total_samples,
    )


def save_checkpoint(path: Path, model: PolicyValueNet, metrics: list[TrainingMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "metrics": [metric.__dict__ for metric in metrics],
        },
        path,
    )


def load_checkpoint(path: Path, model: PolicyValueNet, device: torch.device | None = None) -> list[TrainingMetrics]:
    checkpoint = torch.load(path, map_location=device or torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state"])
    return [TrainingMetrics(**metric) for metric in checkpoint.get("metrics", [])]


def predict_legal_moves(
    model: PolicyValueNet,
    board: chess.Board,
    device: torch.device,
    top_n: int = 5,
) -> list[MovePrediction]:
    model.eval()
    planes = torch.tensor(board_to_planes(board), dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        policy_logits, _ = model(planes)
        legal_predictions = []
        for move in board.legal_moves:
            policy_index = move_to_policy_index(move, board)
            legal_predictions.append(
                MovePrediction(
                    move_uci=move.uci(),
                    policy_index=policy_index,
                    score=float(policy_logits[0, policy_index].item()),
                )
            )

    legal_predictions.sort(key=lambda prediction: prediction.score, reverse=True)
    return legal_predictions[:top_n]
