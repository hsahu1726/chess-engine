from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable, Protocol

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
class RootMoveStat:
    move_uci: str
    prior: float
    visits: int
    q_value: float
    exploration: float
    puct_score: float


@dataclass(frozen=True)
class MCTSResult:
    move: chess.Move | None
    simulations: int
    root_visits: int
    root_value: float
    network_evaluations: int = 0
    cache_hits: int = 0
    root_moves: tuple[RootMoveStat, ...] = ()
    leaf_value_mean: float = 0.0
    leaf_value_stddev: float = 0.0
    leaf_value_min: float = 0.0
    leaf_value_max: float = 0.0


class NeuralPolicyValue:
    def __init__(self, model: PolicyValueNet, device: torch.device, cache_size: int = 100_000):
        self.model = model
        self.device = device
        self.cache_size = max(0, cache_size)
        self.cache: OrderedDict[Hashable, tuple[dict[chess.Move, float], float]] = OrderedDict()
        self.network_evaluations = 0
        self.cache_hits = 0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        channels: int = 32,
        device: torch.device | None = None,
        cache_size: int = 100_000,
    ) -> "NeuralPolicyValue":
        resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PolicyValueNet(channels=channels).to(resolved_device)
        load_checkpoint(checkpoint, model, resolved_device)
        model.eval()
        return cls(model, resolved_device, cache_size)

    def __call__(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}, terminal_value(board)

        key = None
        if self.cache_size:
            key = position_cache_key(board)
            cached = self.cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                self.cache.move_to_end(key)
                return cached

        planes = torch.tensor(board_to_planes(board), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value = self.model(planes)

        policy_indices = torch.tensor(
            [move_to_policy_index(move, board) for move in legal_moves],
            dtype=torch.long,
            device=self.device,
        )
        legal_logits = policy_logits[0].index_select(0, policy_indices)
        probabilities = torch.softmax(legal_logits, dim=0).detach().cpu().tolist()
        result = dict(zip(legal_moves, probabilities)), float(value[0].item())
        self.network_evaluations += 1
        if key is not None:
            self.cache[key] = result
            self.cache.move_to_end(key)
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return result

    @property
    def cache_misses(self) -> int:
        return self.network_evaluations

    @property
    def cache_hit_percent(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total * 100 if total else 0.0


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
        evaluations_before = getattr(self.policy_value_fn, "network_evaluations", 0)
        cache_hits_before = getattr(self.policy_value_fn, "cache_hits", 0)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return MCTSResult(None, 0, 0, terminal_value(board), 0, 0)

        root = MCTSNode()
        self.expand(root, board)

        completed = 0
        leaf_values = []
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

            leaf_values.append(value)
            self.backup(path, value)
            completed += 1

        move = self.best_move(root)
        leaf_mean = sum(leaf_values) / len(leaf_values)
        leaf_variance = sum((value - leaf_mean) ** 2 for value in leaf_values) / len(leaf_values)
        return MCTSResult(
            move,
            completed,
            root.visits,
            root.value,
            getattr(self.policy_value_fn, "network_evaluations", 0) - evaluations_before,
            getattr(self.policy_value_fn, "cache_hits", 0) - cache_hits_before,
            self.root_move_stats(root),
            leaf_mean,
            math.sqrt(leaf_variance),
            min(leaf_values),
            max(leaf_values),
        )

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
        def score(item: tuple[chess.Move, MCTSNode]) -> float:
            _, child = item
            return self.child_score(node, child)[2]

        return max(node.children.items(), key=score)

    def child_score(self, parent: MCTSNode, child: MCTSNode) -> tuple[float, float, float]:
        q_value = -child.value
        exploration = self.cpuct * child.prior * math.sqrt(max(1, parent.visits)) / (1 + child.visits)
        return q_value, exploration, q_value + exploration

    def root_move_stats(self, root: MCTSNode) -> tuple[RootMoveStat, ...]:
        stats = []
        for move, child in root.children.items():
            q_value, exploration, score = self.child_score(root, child)
            stats.append(
                RootMoveStat(
                    move_uci=move.uci(),
                    prior=child.prior,
                    visits=child.visits,
                    q_value=q_value,
                    exploration=exploration,
                    puct_score=score,
                )
            )
        stats.sort(key=lambda stat: (stat.visits, stat.prior), reverse=True)
        return tuple(stats)

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


def position_cache_key(board: chess.Board) -> tuple[int | bool | None, ...]:
    return (
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.occupied_co[chess.WHITE],
        board.occupied_co[chess.BLACK],
        board.promoted,
        board.turn,
        board.castling_rights,
        board.ep_square,
    )
