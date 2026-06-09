from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chess_engine_2.neural import ChessJsonlDataset, PolicyValueNet, save_checkpoint, train_one_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the first policy/value neural network.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/policy_value.pt"))
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ChessJsonlDataset(args.jsonl, args.max_samples)
    if len(dataset) == 0:
        raise ValueError("dataset contains no samples")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)

    model = PolicyValueNet(channels=args.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    metrics = []
    print(f"device: {device}")
    print(f"samples: {len(dataset)}")
    for epoch in range(1, args.epochs + 1):
        metric = train_one_epoch(model, loader, optimizer, device)
        metrics.append(metric)
        print(
            f"epoch {epoch}: total_loss={metric.total_loss:.4f} "
            f"policy_loss={metric.policy_loss:.4f} value_loss={metric.value_loss:.4f}"
        )

    save_checkpoint(args.checkpoint, model, metrics)
    print(f"checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
