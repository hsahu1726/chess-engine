import torch

from chess_engine_2.train_rankings import pairwise_ranking_loss


def test_pairwise_ranking_loss_rewards_correct_ordering() -> None:
    scores = torch.tensor([300.0, 100.0, -200.0])
    correct = pairwise_ranking_loss(torch.tensor([0.8, 0.2, -0.5]), scores)
    reversed_order = pairwise_ranking_loss(torch.tensor([-0.5, 0.2, 0.8]), scores)

    assert correct < reversed_order


def test_pairwise_ranking_loss_ignores_near_ties() -> None:
    predictions = torch.tensor([0.1, 0.2], requires_grad=True)
    loss = pairwise_ranking_loss(predictions, torch.tensor([10.0, 15.0]), minimum_difference=20.0)

    assert loss.item() == 0.0
    loss.backward()
    assert predictions.grad is not None
