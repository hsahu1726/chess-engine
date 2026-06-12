from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import chess

from chess_engine_2 import __version__
from chess_engine_2.engine import SearchEngine


@dataclass
class UciSession:
    input_stream: TextIO = sys.stdin
    output_stream: TextIO = sys.stdout
    board: chess.Board = field(default_factory=chess.Board)
    engine: SearchEngine = field(default_factory=SearchEngine)
    mcts_policy_value: object | None = None
    mcts_simulations: int = 100
    mcts_cpuct: float = 1.5

    def write(self, line: str) -> None:
        self.output_stream.write(f"{line}\n")
        self.output_stream.flush()

    def run(self) -> None:
        for raw_line in self.input_stream:
            line = raw_line.strip()
            if not line:
                continue
            if self.handle_command(line):
                break

    def handle_command(self, line: str) -> bool:
        command, *rest = line.split(maxsplit=1)
        args = rest[0] if rest else ""

        if command == "uci":
            self.write(f"id name Chess Engine 2 {__version__}")
            self.write("id author Codex")
            self.write("uciok")
        elif command == "isready":
            self.write("readyok")
        elif command == "ucinewgame":
            self.board.reset()
        elif command == "position":
            self.set_position(args)
        elif command == "go":
            self.go(args)
        elif command == "stop":
            self.go("")
        elif command == "quit":
            return True

        return False

    def set_position(self, args: str) -> None:
        parts = args.split()
        if not parts:
            return

        move_index = None
        if "moves" in parts:
            move_index = parts.index("moves")
            position_parts = parts[:move_index]
            move_parts = parts[move_index + 1 :]
        else:
            position_parts = parts
            move_parts = []

        if position_parts[0] == "startpos":
            self.board = chess.Board()
        elif position_parts[0] == "fen":
            fen_parts = position_parts[1:]
            if len(fen_parts) == 6:
                self.board = chess.Board(" ".join(fen_parts))
            else:
                return

        for move_text in move_parts:
            try:
                move = chess.Move.from_uci(move_text)
            except ValueError:
                break
            if move not in self.board.legal_moves:
                break
            self.board.push(move)

    def go(self, args: str) -> None:
        if self.mcts_policy_value is not None:
            self.go_mcts()
            return

        depth = parse_go_depth(args)
        movetime_ms = parse_go_movetime(args)
        if movetime_ms is None:
            movetime_ms = choose_movetime_from_clock(args, self.board.turn)
        results = self.engine.iterative_search(self.board, depth, movetime_ms)
        result = results[-1] if results else self.engine.search(self.board, depth)
        move = result.move
        if move is None:
            self.write("bestmove 0000")
        else:
            for completed_result in results:
                pv = " ".join(move.uci() for move in completed_result.pv)
                pv_text = f" pv {pv}" if pv else ""
                self.write(
                    f"info depth {completed_result.depth} "
                    f"score cp {completed_result.score} "
                    f"nodes {completed_result.nodes}"
                    f"{pv_text}"
                )
            self.write(f"bestmove {move.uci()}")

    def go_mcts(self) -> None:
        from chess_engine_2.mcts import MCTSEngine

        engine = MCTSEngine(
            self.mcts_policy_value,
            simulations=max(1, self.mcts_simulations),
            cpuct=max(0.01, self.mcts_cpuct),
        )
        result = engine.search(self.board)
        if result.move is None:
            self.write("bestmove 0000")
            return

        self.write(
            f"info string mcts simulations {result.simulations} "
            f"root_value {result.root_value:.3f} "
            f"network_evaluations {result.network_evaluations} cache_hits {result.cache_hits}"
        )
        self.write(f"bestmove {result.move.uci()}")


def parse_go_depth(args: str) -> int | None:
    parts = args.split()
    if "depth" not in parts:
        return None

    depth_index = parts.index("depth") + 1
    if depth_index >= len(parts):
        return None

    try:
        depth = int(parts[depth_index])
    except ValueError:
        return None

    return max(1, depth)


def parse_go_movetime(args: str) -> int | None:
    movetime = parse_go_int(args, "movetime")
    if movetime is None:
        return None

    return max(1, movetime)


def choose_movetime_from_clock(args: str, turn: chess.Color) -> int | None:
    remaining_keyword = "wtime" if turn == chess.WHITE else "btime"
    increment_keyword = "winc" if turn == chess.WHITE else "binc"

    remaining_ms = parse_go_int(args, remaining_keyword)
    if remaining_ms is None:
        return None

    increment_ms = parse_go_int(args, increment_keyword) or 0
    return allocate_movetime(remaining_ms, increment_ms)


def allocate_movetime(remaining_ms: int, increment_ms: int = 0) -> int:
    if remaining_ms <= 0:
        return 1

    safety_buffer = min(500, max(50, remaining_ms // 20))
    usable_time = max(1, remaining_ms - safety_buffer)
    base_time = usable_time // 30
    increment_time = (increment_ms * 3) // 4
    allocated = max(20, base_time + increment_time)
    max_spend = max(1, usable_time // 5)

    return max(1, min(allocated, max_spend))


def parse_go_int(args: str, keyword: str) -> int | None:
    parts = args.split()
    if keyword not in parts:
        return None

    value_index = parts.index(keyword) + 1
    if value_index >= len(parts):
        return None

    try:
        return int(parts[value_index])
    except ValueError:
        return None


def build_engine(
    neural_checkpoint: Path | None = None,
    neural_channels: int = 32,
    neural_ordering: str = "root",
    neural_min_depth: int = 2,
    value_checkpoint: Path | None = None,
    evaluation_mode: str = "classical",
    neural_value_weight: float = 0.2,
    neural_value_scale: int = 1000,
) -> SearchEngine:
    engine = SearchEngine(
        policy_ordering_mode=neural_ordering,
        policy_ordering_min_depth=max(1, neural_min_depth),
        evaluation_mode=evaluation_mode,
        neural_value_weight=neural_value_weight,
    )
    if neural_checkpoint is not None:
        from chess_engine_2.neural import NeuralPolicyScorer

        engine.policy_scorer = NeuralPolicyScorer.from_checkpoint(neural_checkpoint, max(1, neural_channels))
    if value_checkpoint is not None:
        from chess_engine_2.neural import NeuralValueEvaluator

        engine.value_evaluator = NeuralValueEvaluator.from_checkpoint(
            value_checkpoint,
            max(1, neural_channels),
            scale=max(1, neural_value_scale),
        )
    return engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chess Engine 2 as a UCI engine.")
    parser.add_argument("--search-backend", choices=["alphabeta", "mcts"], default="alphabeta")
    parser.add_argument("--neural-checkpoint", type=Path)
    parser.add_argument("--neural-channels", type=int, default=32)
    parser.add_argument("--neural-ordering", choices=["root", "depth", "all"], default="root")
    parser.add_argument("--neural-min-depth", type=int, default=2)
    parser.add_argument("--value-checkpoint", type=Path)
    parser.add_argument("--evaluation-mode", choices=["classical", "neural", "blend"], default="classical")
    parser.add_argument("--neural-value-weight", type=float, default=0.2)
    parser.add_argument("--neural-value-scale", type=int, default=1000)
    parser.add_argument("--mcts-simulations", type=int, default=100)
    parser.add_argument("--mcts-cpuct", type=float, default=1.5)
    parser.add_argument("--mcts-cache-size", type=int, default=100_000)
    args = parser.parse_args()

    mcts_policy_value = None
    if args.search_backend == "mcts":
        if args.neural_checkpoint is None:
            raise ValueError("MCTS UCI backend requires --neural-checkpoint")
        from chess_engine_2.mcts import NeuralPolicyValue

        mcts_policy_value = NeuralPolicyValue.from_checkpoint(
            args.neural_checkpoint,
            max(1, args.neural_channels),
            cache_size=max(0, args.mcts_cache_size),
        )

    UciSession(
        engine=build_engine(
            args.neural_checkpoint,
            args.neural_channels,
            args.neural_ordering,
            args.neural_min_depth,
            args.value_checkpoint,
            args.evaluation_mode,
            args.neural_value_weight,
            args.neural_value_scale,
        ),
        mcts_policy_value=mcts_policy_value,
        mcts_simulations=max(1, args.mcts_simulations),
        mcts_cpuct=args.mcts_cpuct,
    ).run()


if __name__ == "__main__":
    main()
