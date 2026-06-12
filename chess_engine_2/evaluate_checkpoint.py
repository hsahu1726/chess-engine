from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chess_engine_2.neural import VALUE_TARGETS, ChessJsonlDataset, PolicyValueNet, evaluate_model, load_checkpoint
from chess_engine_2.train import load_tensor_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a policy/value checkpoint on a JSONL dataset.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tensor-cache", type=Path)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--value-target",
        choices=VALUE_TARGETS,
        default="value",
    )
    parser.add_argument("--result-weight", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.tensor_cache is not None:
        dataset = load_tensor_cache(
            args.jsonl,
            args.tensor_cache,
            args.max_samples,
            args.rebuild_cache,
            args.value_target,
            args.result_weight,
        )
    else:
        dataset = ChessJsonlDataset(args.jsonl, args.max_samples, args.value_target, args.result_weight)
    if len(dataset) == 0:
        raise ValueError("dataset contains no samples")

    loader = DataLoader(
        dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=device.type == "cuda",
    )
    model = PolicyValueNet(channels=max(1, args.channels)).to(device)
    load_checkpoint(args.checkpoint, model, device)
    metrics = evaluate_model(model, loader, device)
    result = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.jsonl),
        "device": str(device),
        "channels": max(1, args.channels),
        "batch_size": max(1, args.batch_size),
        "max_samples": args.max_samples,
        "value_target": args.value_target,
        "result_weight": args.result_weight,
        "metrics": asdict(metrics),
    }

    print(f"checkpoint: {args.checkpoint}")
    print(f"dataset: {args.jsonl}")
    print(f"device: {device}")
    print(f"samples: {metrics.samples}")
    print(
        f"total_loss={metrics.total_loss:.4f} policy_loss={metrics.policy_loss:.4f} "
        f"value_loss={metrics.value_loss:.4f}"
    )
    print(f"top1={metrics.policy_top1:.3f} top5={metrics.policy_top5:.3f}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"output: {args.output}")


if __name__ == "__main__":
    main()
