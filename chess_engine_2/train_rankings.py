from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import torch
from torch import nn
from torch.nn import functional as F

from chess_engine_2.data.dataset import board_to_planes
from chess_engine_2.neural import PolicyValueNet, load_checkpoint, save_checkpoint
from chess_engine_2.train import configure_value_head_only
from chess_engine_2.value_ranking import pairwise_ordering_accuracy, spearman_correlation


@dataclass(frozen=True)
class RankingEpochMetrics:
    groups: int
    children: int
    total_loss: float
    regression_loss: float
    ranking_loss: float
    mean_spearman: float
    mean_pairwise_accuracy: float


def load_groups(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def pairwise_ranking_loss(
    parent_predictions: torch.Tensor,
    search_scores: torch.Tensor,
    minimum_difference: float = 20.0,
    temperature: float = 4.0,
) -> torch.Tensor:
    score_differences = search_scores[:, None] - search_scores[None, :]
    prediction_differences = parent_predictions[:, None] - parent_predictions[None, :]
    mask = torch.triu(score_differences.abs() >= minimum_difference, diagonal=1)
    if not mask.any():
        return parent_predictions.sum() * 0.0
    labels = torch.sign(score_differences[mask])
    return F.softplus(-labels * prediction_differences[mask] * temperature).mean()


def group_tensors(group: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    children = group["children"]
    planes = torch.tensor(
        [board_to_planes(chess.Board(child["child_fen"])) for child in children],
        dtype=torch.float32,
        device=device,
    )
    targets = torch.tensor(
        [float(child["child_value_target"]) for child in children],
        dtype=torch.float32,
        device=device,
    )
    scores = torch.tensor(
        [float(child["search_score"]) for child in children],
        dtype=torch.float32,
        device=device,
    )
    return planes, targets, scores


def run_epoch(
    model: PolicyValueNet,
    groups: list[dict],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    ranking_weight: float,
    minimum_difference: float,
    temperature: float,
) -> RankingEpochMetrics:
    training = optimizer is not None
    model.train(training)
    model.trunk.eval()
    model.policy_head.eval()
    mse = nn.MSELoss()
    totals = {"loss": 0.0, "regression": 0.0, "ranking": 0.0, "children": 0}
    correlations = []
    pairwise_accuracies = []

    for group in groups:
        planes, targets, scores = group_tensors(group, device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            _, child_predictions = model(planes)
            regression_loss = mse(child_predictions, targets)
            parent_predictions = -child_predictions
            ranking_loss = pairwise_ranking_loss(
                parent_predictions,
                scores,
                minimum_difference,
                temperature,
            )
            loss = regression_loss + ranking_weight * ranking_loss
            if training:
                loss.backward()
                optimizer.step()

        child_count = len(group["children"])
        totals["loss"] += float(loss.item()) * child_count
        totals["regression"] += float(regression_loss.item()) * child_count
        totals["ranking"] += float(ranking_loss.item()) * child_count
        totals["children"] += child_count
        predictions = parent_predictions.detach().cpu().tolist()
        truth = scores.detach().cpu().tolist()
        correlations.append(spearman_correlation(truth, predictions))
        pairwise_accuracies.append(pairwise_ordering_accuracy(truth, predictions))

    children = totals["children"]
    return RankingEpochMetrics(
        groups=len(groups),
        children=children,
        total_loss=totals["loss"] / children,
        regression_loss=totals["regression"] / children,
        ranking_loss=totals["ranking"] / children,
        mean_spearman=sum(correlations) / len(correlations),
        mean_pairwise_accuracy=sum(pairwise_accuracies) / len(pairwise_accuracies),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a value head on search-distilled sibling rankings.")
    parser.add_argument("groups", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/policy_value_rank_distilled.pt"))
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--minimum-difference", type=float, default=20.0)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--validation-groups", type=Path)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    groups = load_groups(args.groups)
    if args.max_groups is not None:
        groups = groups[: max(2, args.max_groups)]
    if args.validation_groups is not None:
        validation_groups = load_groups(args.validation_groups)
        training_groups = groups
        random.shuffle(training_groups)
    else:
        random.shuffle(groups)
        validation_count = max(1, int(len(groups) * max(0.0, min(0.5, args.validation_split))))
        validation_groups = groups[:validation_count]
        training_groups = groups[validation_count:]
    if not training_groups:
        raise ValueError("ranking dataset needs at least two groups")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyValueNet(channels=max(1, args.channels)).to(device)
    load_checkpoint(args.initial_checkpoint, model, device)
    configure_value_head_only(model)
    optimizer = torch.optim.AdamW(model.value_head.parameters(), lr=args.learning_rate)
    training_history = []
    validation_history = []
    print(
        f"device: {device}, train groups: {len(training_groups)}, "
        f"validation groups: {len(validation_groups)}",
        flush=True,
    )

    for epoch in range(1, max(1, args.epochs) + 1):
        random.shuffle(training_groups)
        training = run_epoch(
            model,
            training_groups,
            device,
            optimizer,
            max(0.0, args.ranking_weight),
            max(0.0, args.minimum_difference),
            max(0.01, args.temperature),
        )
        validation = run_epoch(
            model,
            validation_groups,
            device,
            None,
            max(0.0, args.ranking_weight),
            max(0.0, args.minimum_difference),
            max(0.01, args.temperature),
        )
        training_history.append(asdict(training))
        validation_history.append(asdict(validation))
        print(
            f"epoch {epoch}: train loss={training.total_loss:.4f} "
            f"rho={training.mean_spearman:.3f} pair={training.mean_pairwise_accuracy:.3f}; "
            f"validation loss={validation.total_loss:.4f} "
            f"rho={validation.mean_spearman:.3f} pair={validation.mean_pairwise_accuracy:.3f}",
            flush=True,
        )

    save_checkpoint(
        args.checkpoint,
        model,
        [],
        metadata={
            "training_type": "search_distilled_sibling_ranking",
            "dataset": str(args.groups),
            "initial_checkpoint": str(args.initial_checkpoint),
            "channels": max(1, args.channels),
            "epochs": max(1, args.epochs),
            "learning_rate": args.learning_rate,
            "ranking_weight": args.ranking_weight,
            "minimum_difference": args.minimum_difference,
            "temperature": args.temperature,
            "validation_dataset": str(args.validation_groups) if args.validation_groups is not None else None,
            "training_history": training_history,
            "validation_history": validation_history,
        },
    )
    print(f"checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
