from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import chess
import torch

from chess_engine_2.data.dataset import board_to_planes
from chess_engine_2.encoding import move_to_policy_index
from chess_engine_2.neural import PolicyValueNet, load_checkpoint


class PolicyValueFn(Protocol):
    def __call__(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        ...


@dataclass
class MCTSNode:
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[chess.Move, "MCTSNode"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    @property
    def expanded(self) -> bool:
        return bool(self.children)


@dataclass(frozen=True)
class MCTSResult:
    move: chess.Move | None
    simulations: int
    root_visits: int
    root_value: float


class NeuralPolicyValue:
    def __init__(self, model: PolicyValueNet, device: torch.device):
        self.model = model
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        channels: int = 32,
        device: torch.device | None = None,
    ) -> "NeuralPolicyValue":
        resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PolicyValueNet(channels=channels).to(resolved_device)
        load_checkpoint(checkpoint, model, resolved_device)
        model.eval()
        return cls(model, resolved_device)

    def __call__(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}, terminal_value(board)

        planes = torch.tensor(board_to_planes(board), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value = self.model(planes)

        logits = torch.tensor(
            [float(policy_logits[0, move_to_policy_index(move, board)].item()) for move in legal_moves],
            dtype=torch.float32,
        )
        probabilities = torch.softmax(logits, dim=0).tolist()
        return dict(zip(legal_moves, probabilities)), float(value[0].item())


class MCTSEngine:
    def __init__(
        self,
        policy_value_fn: PolicyValueFn,
        simulations: int = 100,
        cpuct: float = 1.5,
    ):
        self.policy_value_fn = policy_value_fn
        self.simulations = simulations
        self.cpuct = cpuct

    def search(self, board: chess.Board) -> MCTSResult:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return MCTSResult(None, 0, 0, terminal_value(board))

        root = MCTSNode()
        self.expand(root, board)

        completed = 0
        for _ in range(max(1, self.simulations)):
            simulation_board = board.copy(stack=False)
            node = root
            path = [node]

            while node.expanded and not simulation_board.is_game_over(claim_draw=True):
                move, node = self.select_child(node)
                simulation_board.push(move)
                path.append(node)

            if simulation_board.is_game_over(claim_draw=True):
                value = terminal_value(simulation_board)
            else:
                value = self.expand(node, simulation_board)

            self.backup(path, value)
            completed += 1

        move = self.best_move(root)
        return MCTSResult(move, completed, root.visits, root.value)

    def expand(self, node: MCTSNode, board: chess.Board) -> float:
        priors, value = self.policy_value_fn(board)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return terminal_value(board)

        if not priors:
            prior = 1.0 / len(legal_moves)
            priors = {move: prior for move in legal_moves}

        total_prior = sum(max(0.0, priors.get(move, 0.0)) for move in legal_moves)
        if total_prior <= 0:
            normalized = {move: 1.0 / len(legal_moves) for move in legal_moves}
        else:
            normalized = {move: max(0.0, priors.get(move, 0.0)) / total_prior for move in legal_moves}

        node.children = {move: MCTSNode(prior=normalized[move]) for move in legal_moves}
        return max(-1.0, min(1.0, value))

    def select_child(self, node: MCTSNode) -> tuple[chess.Move, MCTSNode]:
        parent_visits = max(1, node.visits)

        def score(item: tuple[chess.Move, MCTSNode]) -> float:
            _, child = item
            q_value = -child.value
            exploration = self.cpuct * child.prior * math.sqrt(parent_visits) / (1 + child.visits)
            return q_value + exploration

        return max(node.children.items(), key=score)

    def backup(self, path: list[MCTSNode], value: float) -> None:
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

    def best_move(self, root: MCTSNode) -> chess.Move | None:
        if not root.children:
            return None
        return max(root.children.items(), key=lambda item: item[1].visits)[0]


def terminal_value(board: chess.Board) -> float:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0
