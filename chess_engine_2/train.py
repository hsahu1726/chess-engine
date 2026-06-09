from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import chess
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

from chess_engine_2.data.dataset import INPUT_PLANES, board_to_planes
from chess_engine_2.neural import ChessJsonlDataset, PolicyValueNet, evaluate_model, save_checkpoint, train_one_epoch


class CachedTensorDataset(Dataset):
    def __init__(self, planes: torch.Tensor, policies: torch.Tensor, values: torch.Tensor):
        self.planes = planes
        self.policies = policies
        self.values = values

    def __len__(self) -> int:
        return int(self.policies.shape[0])

    def __getitem__(self, index: int):
        return self.planes[index].float(), self.policies[index].long(), self.values[index].float()


def build_tensor_cache(jsonl: Path, cache_path: Path, max_samples: int | None) -> CachedTensorDataset:
    sample_count = 0
    with jsonl.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                sample_count += 1
                if max_samples is not None and sample_count >= max_samples:
                    break

    if sample_count == 0:
        raise ValueError("dataset contains no samples")

    planes = torch.empty((sample_count, INPUT_PLANES, 8, 8), dtype=torch.uint8)
    policies = torch.empty(sample_count, dtype=torch.long)
    values = torch.empty(sample_count, dtype=torch.float32)

    with jsonl.open("r", encoding="utf-8") as stream:
        sample_index = 0
        for line in stream:
            if not line.strip():
                continue
            sample = json.loads(line)
            board = chess.Board(sample["fen"])
            planes[sample_index] = torch.tensor(board_to_planes(board), dtype=torch.uint8)
            policies[sample_index] = int(sample["policy_index"])
            values[sample_index] = float(sample["value"])
            sample_index += 1
            if sample_index % 50000 == 0:
                print(f"cached samples: {sample_index}", flush=True)
            if sample_index >= sample_count:
                break

    dataset = CachedTensorDataset(planes, policies, values)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source": str(jsonl),
            "max_samples": max_samples,
            "planes": dataset.planes,
            "policies": dataset.policies,
            "values": dataset.values,
        },
        cache_path,
    )
    return dataset


def load_tensor_cache(jsonl: Path, cache_path: Path, max_samples: int | None, rebuild: bool) -> CachedTensorDataset:
    if not rebuild and cache_path.exists():
        cache = torch.load(cache_path, map_location=torch.device("cpu"))
        if cache.get("source") == str(jsonl) and cache.get("max_samples") == max_samples:
            return CachedTensorDataset(cache["planes"], cache["policies"], cache["values"])
        print("tensor cache metadata mismatch; rebuilding", flush=True)

    print(f"building tensor cache: {cache_path}", flush=True)
    return build_tensor_cache(jsonl, cache_path, max_samples)


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
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tensor-cache", type=Path)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.tensor_cache is not None:
        dataset = load_tensor_cache(args.jsonl, args.tensor_cache, args.max_samples, args.rebuild_cache)
    else:
        dataset = ChessJsonlDataset(args.jsonl, args.max_samples)
    if len(dataset) == 0:
        raise ValueError("dataset contains no samples")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    validation_size = int(len(dataset) * max(0.0, min(0.5, args.validation_split)))
    train_size = len(dataset) - validation_size
    if validation_size:
        train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size], generator=generator)
    else:
        train_dataset = dataset
        validation_dataset = None

    pin_memory = device.type == "cuda"
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        if validation_dataset is not None
        else None
    )

    model = PolicyValueNet(channels=args.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    metrics = []
    validation_metrics = []
    print(f"device: {device}", flush=True)
    print(f"samples: {len(dataset)}", flush=True)
    print(f"train samples: {len(train_dataset)}", flush=True)
    print(f"validation samples: {len(validation_dataset) if validation_dataset is not None else 0}", flush=True)
    print(f"num workers: {args.num_workers}", flush=True)
    for epoch in range(1, args.epochs + 1):
        metric = train_one_epoch(model, loader, optimizer, device)
        metrics.append(metric)
        print(
            f"epoch {epoch}: total_loss={metric.total_loss:.4f} "
            f"policy_loss={metric.policy_loss:.4f} value_loss={metric.value_loss:.4f}",
            flush=True,
        )
        if validation_loader is not None:
            validation_metric = evaluate_model(model, validation_loader, device)
            validation_metrics.append(validation_metric)
            print(
                f"validation {epoch}: total_loss={validation_metric.total_loss:.4f} "
                f"policy_loss={validation_metric.policy_loss:.4f} value_loss={validation_metric.value_loss:.4f} "
                f"top1={validation_metric.policy_top1:.3f} top5={validation_metric.policy_top5:.3f}",
                flush=True,
            )

    save_checkpoint(
        args.checkpoint,
        model,
        metrics,
        metadata={
            "dataset": str(args.jsonl),
            "samples": len(dataset),
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset) if validation_dataset is not None else 0,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "channels": args.channels,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "validation_split": args.validation_split,
            "num_workers": args.num_workers,
        },
        validation_metrics=validation_metrics,
    )
    print(f"checkpoint: {args.checkpoint}", flush=True)


if __name__ == "__main__":
    main()
