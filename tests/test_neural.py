import json

import chess
import torch
from torch.utils.data import DataLoader

from chess_engine_2.train import configure_value_head_only, load_tensor_cache
from chess_engine_2.encoding import POLICY_SIZE, move_to_policy_index
from chess_engine_2.neural import (
    ChessJsonlDataset,
    MovePrediction,
    NeuralPolicyScorer,
    NeuralValueEvaluator,
    PolicyValueNet,
    evaluate_model,
    load_checkpoint,
    load_checkpoint_metadata,
    load_validation_metrics,
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


def test_chess_jsonl_dataset_selects_dense_value_targets(tmp_path) -> None:
    path = tmp_path / "samples.jsonl"
    board = chess.Board()
    path.write_text(
        json.dumps(
            {
                "fen": board.fen(),
                "move_uci": "e2e4",
                "policy_index": move_to_policy_index(chess.Move.from_uci("e2e4"), board),
                "value": 1.0,
                "material_value": 0.25,
                "classical_value": -0.5,
                "discounted_value": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert ChessJsonlDataset(path, value_target="material_value")[0][2].item() == 0.25
    assert ChessJsonlDataset(path, value_target="classical_value")[0][2].item() == -0.5
    assert ChessJsonlDataset(path, value_target="blend", result_weight=0.25)[0][2].item() == -0.125
    assert ChessJsonlDataset(path, value_target="discounted_value")[0][2].item() == 0.5
    assert ChessJsonlDataset(path, value_target="result_material_blend", result_weight=0.75)[0][2].item() == 0.8125
    assert ChessJsonlDataset(path, value_target="result_classical_blend", result_weight=0.75)[0][2].item() == 0.625
    assert (
        ChessJsonlDataset(path, value_target="discounted_classical_blend", result_weight=0.75)[0][2].item()
        == 0.25
    )


def test_tensor_cache_round_trip(tmp_path) -> None:
    path = tmp_path / "samples.jsonl"
    write_sample(path, chess.Board(), chess.Move.from_uci("e2e4"))
    cache_path = tmp_path / "samples.pt"

    dataset = load_tensor_cache(path, cache_path, max_samples=None, rebuild=False)
    reloaded = load_tensor_cache(path, cache_path, max_samples=None, rebuild=False)
    planes, policy, value = reloaded[0]

    assert cache_path.exists()
    assert len(dataset) == 1
    assert len(reloaded) == 1
    assert planes.shape == (18, 8, 8)
    assert planes.dtype == torch.float32
    assert policy.item() == 877
    assert value.item() == 1.0


def test_policy_value_net_output_shapes() -> None:
    model = PolicyValueNet(channels=8)
    policy_logits, value = model(torch.zeros((2, 18, 8, 8), dtype=torch.float32))

    assert policy_logits.shape == (2, POLICY_SIZE)
    assert value.shape == (2,)


def test_configure_value_head_only_freezes_policy_and_trunk() -> None:
    model = PolicyValueNet(channels=8)

    configure_value_head_only(model)

    assert all(not parameter.requires_grad for parameter in model.trunk.parameters())
    assert all(not parameter.requires_grad for parameter in model.policy_head.parameters())
    assert all(parameter.requires_grad for parameter in model.value_head.parameters())


def test_value_head_only_training_keeps_frozen_modules_in_eval_mode(tmp_path) -> None:
    path = tmp_path / "samples.jsonl"
    write_sample(path, chess.Board(), chess.Move.from_uci("e2e4"))
    model = PolicyValueNet(channels=8)
    configure_value_head_only(model)
    optimizer = torch.optim.AdamW(model.value_head.parameters(), lr=0.001)
    loader = DataLoader(ChessJsonlDataset(path), batch_size=1)

    train_one_epoch(model, loader, optimizer, torch.device("cpu"), value_head_only=True)

    assert not model.trunk.training
    assert not model.policy_head.training
    assert model.value_head.training


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
    validation_metrics = [evaluate_model(model, loader, torch.device("cpu"))]
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        metrics,
        metadata={"samples": 4, "channels": 8},
        validation_metrics=validation_metrics,
    )

    reloaded = PolicyValueNet(channels=8)
    loaded_metrics = load_checkpoint(checkpoint_path, reloaded)
    loaded_validation_metrics = load_validation_metrics(checkpoint_path)
    metadata = load_checkpoint_metadata(checkpoint_path)

    assert checkpoint_path.exists()
    assert loaded_metrics[0].samples == 4
    assert loaded_validation_metrics[0].samples == 4
    assert 0.0 <= loaded_validation_metrics[0].policy_top1 <= 1.0
    assert 0.0 <= loaded_validation_metrics[0].policy_top5 <= 1.0
    assert metadata["samples"] == 4
    assert metrics[0].total_loss > 0


def test_predict_legal_moves_returns_ranked_legal_moves() -> None:
    model = PolicyValueNet(channels=8)
    board = chess.Board()
    predictions = predict_legal_moves(model, board, torch.device("cpu"), top_n=3)

    assert len(predictions) == 3
    assert all(isinstance(prediction, MovePrediction) for prediction in predictions)
    assert all(chess.Move.from_uci(prediction.move_uci) in board.legal_moves for prediction in predictions)
    assert predictions[0].score >= predictions[-1].score


def test_neural_policy_scorer_scores_legal_moves() -> None:
    scorer = NeuralPolicyScorer(PolicyValueNet(channels=8), torch.device("cpu"))
    board = chess.Board()
    scores = scorer.score_moves(board)

    assert set(scores) == set(board.legal_moves)


def test_neural_value_evaluator_returns_centipawn_score() -> None:
    evaluator = NeuralValueEvaluator(PolicyValueNet(channels=8), torch.device("cpu"), scale=1000)
    score = evaluator.evaluate(chess.Board())

    assert isinstance(score, int)
    assert -1000 <= score <= 1000
