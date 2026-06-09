import json

import chess
import torch
from torch.utils.data import DataLoader

from chess_engine_2.encoding import POLICY_SIZE, move_to_policy_index
from chess_engine_2.neural import (
    ChessJsonlDataset,
    MovePrediction,
    PolicyValueNet,
    load_checkpoint,
    predict_legal_moves,
    save_checkpoint,
    train_one_epoch,
)


def write_sample(path, board: chess.Board, move: chess.Move, value: float = 1.0) -> None:
    path.write_text(
        json.dumps(
            {
                "fen": board.fen(),
                "move_uci": move.uci(),
                "policy_index": move_to_policy_index(move, board),
                "value": value,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_chess_jsonl_dataset_returns_tensors(tmp_path) -> None:
    path = tmp_path / "samples.jsonl"
    write_sample(path, chess.Board(), chess.Move.from_uci("e2e4"))

    dataset = ChessJsonlDataset(path)
    planes, policy, value = dataset[0]

    assert len(dataset) == 1
    assert planes.shape == (18, 8, 8)
    assert policy.item() == 877
    assert value.item() == 1.0


def test_policy_value_net_output_shapes() -> None:
    model = PolicyValueNet(channels=8)
    policy_logits, value = model(torch.zeros((2, 18, 8, 8), dtype=torch.float32))

    assert policy_logits.shape == (2, POLICY_SIZE)
    assert value.shape == (2,)


def test_train_one_epoch_and_checkpoint_round_trip(tmp_path) -> None:
    path = tmp_path / "samples.jsonl"
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    rows = []
    for _ in range(4):
        rows.append(
            json.dumps(
                {
                    "fen": board.fen(),
                    "move_uci": move.uci(),
                    "policy_index": move_to_policy_index(move, board),
                    "value": 1.0,
                }
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    dataset = ChessJsonlDataset(path)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = PolicyValueNet(channels=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    metrics = [train_one_epoch(model, loader, optimizer, torch.device("cpu"))]
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(checkpoint_path, model, metrics)

    reloaded = PolicyValueNet(channels=8)
    loaded_metrics = load_checkpoint(checkpoint_path, reloaded)

    assert checkpoint_path.exists()
    assert loaded_metrics[0].samples == 4
    assert metrics[0].total_loss > 0


def test_predict_legal_moves_returns_ranked_legal_moves() -> None:
    model = PolicyValueNet(channels=8)
    board = chess.Board()
    predictions = predict_legal_moves(model, board, torch.device("cpu"), top_n=3)

    assert len(predictions) == 3
    assert all(isinstance(prediction, MovePrediction) for prediction in predictions)
    assert all(chess.Move.from_uci(prediction.move_uci) in board.legal_moves for prediction in predictions)
    assert predictions[0].score >= predictions[-1].score
