from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import chess
import torch

from chess_engine_2.data.dataset import board_to_planes
from chess_engine_2.neural import PolicyValueNet, load_checkpoint
from chess_engine_2.train_rankings import load_groups
from chess_engine_2.value_ranking import (
    MoveRanking,
    PositionRanking,
    average_ranks,
    pairwise_ordering_accuracy,
    spearman_correlation,
    summarize,
)


def evaluate_group(model: PolicyValueNet, device: torch.device, group: dict) -> PositionRanking:
    root = chess.Board(group["root_fen"])
    children = group["children"]
    planes = torch.tensor(
        [board_to_planes(chess.Board(child["child_fen"])) for child in children],
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        _, child_values = model(planes)
    neural_values = [-float(value) for value in child_values.detach().cpu().tolist()]
    search_scores = [int(child["search_score"]) for child in children]
    moves = [chess.Move.from_uci(child["move_uci"]) for child in children]
    search_ranks = average_ranks(search_scores)
    neural_ranks = average_ranks(neural_values)
    best_score = max(search_scores)
    best_indices = {index for index, score in enumerate(search_scores) if score == best_score}
    neural_order = sorted(range(len(children)), key=neural_values.__getitem__, reverse=True)
    rows = [
        MoveRanking(move.uci(), search_scores[index], neural_values[index], search_ranks[index], neural_ranks[index])
        for index, move in enumerate(moves)
    ]
    rows.sort(key=lambda row: row.search_score, reverse=True)
    return PositionRanking(
        fen=root.fen(),
        legal_moves=len(children),
        spearman=spearman_correlation(search_scores, neural_values),
        pairwise_accuracy=pairwise_ordering_accuracy(search_scores, neural_values),
        top1_correct=neural_order[0] in best_indices,
        top3_correct=bool(best_indices.intersection(neural_order[:3])),
        neural_top1_regret_cp=best_score - search_scores[neural_order[0]],
        neural_top3_regret_cp=best_score - max(search_scores[index] for index in neural_order[:3]),
        search_score_spread=max(search_scores) - min(search_scores),
        neural_value_spread=max(neural_values) - min(neural_values),
        moves=rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on cached sibling search rankings.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("groups", type=Path)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyValueNet(channels=max(1, args.channels)).to(device)
    load_checkpoint(args.checkpoint, model, device)
    rows = [evaluate_group(model, device, group) for group in load_groups(args.groups)]
    summary = summarize(rows)
    result = {
        "checkpoint": str(args.checkpoint),
        "groups": str(args.groups),
        "device": str(device),
        "summary": summary,
        "positions": [asdict(row) for row in rows],
    }
    print(json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"output: {args.output}")


if __name__ == "__main__":
    main()
