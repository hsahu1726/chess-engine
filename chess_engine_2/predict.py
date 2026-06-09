from __future__ import annotations

import argparse
from pathlib import Path

import chess
import torch

from chess_engine_2.neural import PolicyValueNet, load_checkpoint, predict_legal_moves


def main() -> None:
    parser = argparse.ArgumentParser(description="Show top legal moves from a trained policy/value checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/policy_value_phase7.pt"))
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyValueNet(channels=args.channels).to(device)
    metrics = load_checkpoint(args.checkpoint, model, device)
    board = chess.Board(args.fen)

    print(f"checkpoint: {args.checkpoint}")
    if metrics:
        print(f"epochs: {len(metrics)}")
        print(f"last total loss: {metrics[-1].total_loss:.4f}")
    for prediction in predict_legal_moves(model, board, device, args.top):
        print(f"{prediction.move_uci} policy={prediction.policy_index} score={prediction.score:.4f}")


if __name__ == "__main__":
    main()
