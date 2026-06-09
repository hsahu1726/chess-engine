from __future__ import annotations

import sys
from dataclasses import dataclass, field
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


def main() -> None:
    UciSession().run()


if __name__ == "__main__":
    main()
