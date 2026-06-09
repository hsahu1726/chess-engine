from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import chess
import chess.pgn

from chess_engine_2.engine import RandomEngine, SearchEngine


class Player(Protocol):
    name: str

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        ...


@dataclass
class PlayerStats:
    moves: int = 0
    total_depth: int = 0
    total_nodes: int = 0
    total_main_nodes: int = 0
    total_quiescence_nodes: int = 0
    total_seconds: float = 0.0
    total_evaluations: int = 0
    total_mobility_evaluations: int = 0

    @property
    def average_depth(self) -> float:
        return self.total_depth / self.moves if self.moves else 0.0

    @property
    def average_nodes(self) -> float:
        return self.total_nodes / self.moves if self.moves else 0.0

    @property
    def nodes_per_second(self) -> float:
        return self.total_nodes / self.total_seconds if self.total_seconds > 0 else 0.0

    @property
    def average_main_nodes(self) -> float:
        return self.total_main_nodes / self.moves if self.moves else 0.0

    @property
    def average_quiescence_nodes(self) -> float:
        return self.total_quiescence_nodes / self.moves if self.moves else 0.0

    @property
    def average_evaluations(self) -> float:
        return self.total_evaluations / self.moves if self.moves else 0.0

    @property
    def average_mobility_evaluations(self) -> float:
        return self.total_mobility_evaluations / self.moves if self.moves else 0.0

    def record(
        self,
        depth: int,
        nodes: int,
        seconds: float,
        evaluations: int = 0,
        mobility_evaluations: int = 0,
        main_nodes: int = 0,
        quiescence_nodes: int = 0,
    ) -> None:
        self.moves += 1
        self.total_depth += depth
        self.total_nodes += nodes
        self.total_main_nodes += main_nodes
        self.total_quiescence_nodes += quiescence_nodes
        self.total_seconds += seconds
        self.total_evaluations += evaluations
        self.total_mobility_evaluations += mobility_evaluations


@dataclass
class SearchPlayer:
    name: str
    depth: int
    movetime_ms: int | None = None
    use_mobility: bool = True
    quiescence_depth: int = 6
    time_check_interval: int = 1024
    stats: PlayerStats = field(default_factory=PlayerStats)
    engine: SearchEngine = field(init=False)

    def __post_init__(self) -> None:
        self.reset_for_new_game()

    def reset_for_new_game(self) -> None:
        self.engine = SearchEngine(
            max_depth=self.depth,
            max_quiescence_depth=self.quiescence_depth,
            use_mobility=self.use_mobility,
            time_check_interval=self.time_check_interval,
        )

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        self.engine.max_depth = self.depth
        self.engine.max_quiescence_depth = self.quiescence_depth
        self.engine.use_mobility = self.use_mobility
        self.engine.time_check_interval = self.time_check_interval
        start = time.perf_counter()
        if self.movetime_ms is None:
            result = self.engine.search(board, self.depth)
        else:
            results = self.engine.iterative_search(board, self.depth, self.movetime_ms)
            result = results[-1] if results else self.engine.search(board, 1)

        self.stats.record(
            result.depth,
            result.nodes,
            time.perf_counter() - start,
            self.engine.evaluate_calls,
            self.engine.mobility_calls,
            result.main_nodes,
            result.quiescence_nodes,
        )
        return result.move


@dataclass
class RandomPlayer:
    name: str = "random"
    engine: RandomEngine = field(default_factory=RandomEngine)
    stats: PlayerStats = field(default_factory=PlayerStats)

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        start = time.perf_counter()
        move = self.engine.choose_move(board)
        self.stats.record(0, 0, time.perf_counter() - start)
        return move


@dataclass
class GameResult:
    white: str
    black: str
    result: str
    plies: int
    termination: str
    pgn: str


@dataclass
class MatchResult:
    games: list[GameResult]
    player_a: str
    player_b: str
    player_a_stats: PlayerStats = field(default_factory=PlayerStats)
    player_b_stats: PlayerStats = field(default_factory=PlayerStats)

    @property
    def a_wins(self) -> int:
        return sum(game_points_for(self.player_a, game) == 1.0 for game in self.games)

    @property
    def b_wins(self) -> int:
        return sum(game_points_for(self.player_b, game) == 1.0 for game in self.games)

    @property
    def draws(self) -> int:
        return sum(game.result == "1/2-1/2" for game in self.games)

    @property
    def a_score(self) -> float:
        return sum(game_points_for(self.player_a, game) for game in self.games)

    @property
    def b_score(self) -> float:
        return sum(game_points_for(self.player_b, game) for game in self.games)

    @property
    def average_plies(self) -> float:
        if not self.games:
            return 0.0
        return sum(game.plies for game in self.games) / len(self.games)

    def summary(self) -> str:
        total = len(self.games)
        a_percent = (self.a_score / total * 100) if total else 0.0
        return "\n".join(
            [
                f"{self.player_a} vs {self.player_b}",
                f"games: {total}",
                f"{self.player_a} wins: {self.a_wins}",
                f"{self.player_b} wins: {self.b_wins}",
                f"draws: {self.draws}",
                f"{self.player_a} score: {self.a_score:.1f}/{total} ({a_percent:.1f}%)",
                f"average plies: {self.average_plies:.1f}",
                (
                    f"{self.player_a} avg depth/nodes/nps: "
                    f"{self.player_a_stats.average_depth:.2f} / "
                    f"{self.player_a_stats.average_nodes:.0f} / "
                    f"{self.player_a_stats.nodes_per_second:.0f}"
                ),
                (
                    f"{self.player_a} avg main/q nodes: "
                    f"{self.player_a_stats.average_main_nodes:.0f} / "
                    f"{self.player_a_stats.average_quiescence_nodes:.0f}"
                ),
                (
                    f"{self.player_a} avg evals/mobility evals: "
                    f"{self.player_a_stats.average_evaluations:.0f} / "
                    f"{self.player_a_stats.average_mobility_evaluations:.0f}"
                ),
                (
                    f"{self.player_b} avg depth/nodes/nps: "
                    f"{self.player_b_stats.average_depth:.2f} / "
                    f"{self.player_b_stats.average_nodes:.0f} / "
                    f"{self.player_b_stats.nodes_per_second:.0f}"
                ),
                (
                    f"{self.player_b} avg main/q nodes: "
                    f"{self.player_b_stats.average_main_nodes:.0f} / "
                    f"{self.player_b_stats.average_quiescence_nodes:.0f}"
                ),
                (
                    f"{self.player_b} avg evals/mobility evals: "
                    f"{self.player_b_stats.average_evaluations:.0f} / "
                    f"{self.player_b_stats.average_mobility_evaluations:.0f}"
                ),
                f"terminations: {format_termination_counts(self.termination_counts)}",
            ]
        )

    @property
    def termination_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for game in self.games:
            counts[game.termination] = counts.get(game.termination, 0) + 1
        return counts


def play_game(
    white: Player,
    black: Player,
    max_plies: int = 200,
    record_pgn: bool = True,
    opening_plies: int = 0,
) -> GameResult:
    reset_player_for_new_game(white)
    reset_player_for_new_game(black)
    board = chess.Board()
    game = chess.pgn.Game() if record_pgn else None
    node = game
    if game is not None:
        game.headers["White"] = white.name
        game.headers["Black"] = black.name
    players = {chess.WHITE: white, chess.BLACK: black}

    for _ in range(opening_plies):
        legal_moves = list(board.legal_moves)
        if not legal_moves or board.is_game_over(claim_draw=True):
            break
        move = RandomEngine().choose_move(board)
        if move is None:
            break
        if node is not None:
            node = node.add_variation(move)
        board.push(move)

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        player = players[board.turn]
        move = player.choose_move(board)
        if move is None or move not in board.legal_moves:
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            if game is not None:
                game.headers["Result"] = result
                game.headers["Termination"] = "illegal move"
            return GameResult(white.name, black.name, result, board.ply(), "illegal move", str(game or ""))
        if node is not None:
            node = node.add_variation(move)
        board.push(move)

    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
        termination = board.outcome(claim_draw=True).termination.name.lower()
    else:
        result = "1/2-1/2"
        termination = "move limit"

    if game is not None:
        game.headers["Result"] = result
        game.headers["Termination"] = termination
    return GameResult(white.name, black.name, result, board.ply(), termination, str(game or ""))


def reset_player_for_new_game(player: Player) -> None:
    reset = getattr(player, "reset_for_new_game", None)
    if callable(reset):
        reset()


def play_match(
    player_a: Player,
    player_b: Player,
    games: int = 2,
    max_plies: int = 200,
    record_pgn: bool = True,
    opening_plies: int = 0,
) -> MatchResult:
    game_results = []
    for game_index in range(games):
        if game_index % 2 == 0:
            white, black = player_a, player_b
        else:
            white, black = player_b, player_a
        game_results.append(play_game(white, black, max_plies, record_pgn, opening_plies))

    return MatchResult(game_results, player_a.name, player_b.name, player_a.stats, player_b.stats)


def game_points_for(player_name: str, game: GameResult) -> float:
    if game.result == "1/2-1/2":
        return 0.5
    if game.result == "1-0":
        return 1.0 if game.white == player_name else 0.0
    if game.result == "0-1":
        return 1.0 if game.black == player_name else 0.0
    return 0.0


def format_termination_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))


def save_match_pgn(result: MatchResult, path: str | Path) -> None:
    Path(path).write_text("\n\n".join(game.pgn for game in result.games), encoding="utf-8")


def build_player(
    kind: str,
    depth: int,
    movetime_ms: int | None = None,
    use_mobility: bool = True,
    quiescence_depth: int = 6,
    time_check_interval: int = 1024,
) -> Player:
    if kind == "random":
        return RandomPlayer()
    if kind == "search":
        suffix = f"-{movetime_ms}ms" if movetime_ms is not None else ""
        mobility_suffix = "" if use_mobility else "-no-mobility"
        return SearchPlayer(
            f"search-depth-{depth}{suffix}{mobility_suffix}",
            depth,
            movetime_ms,
            use_mobility,
            quiescence_depth,
            time_check_interval,
        )
    raise ValueError(f"unknown player kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run engine-vs-engine matches.")
    parser.add_argument("--a", choices=["search", "random"], default="search")
    parser.add_argument("--b", choices=["search", "random"], default="random")
    parser.add_argument("--a-depth", type=int, default=2)
    parser.add_argument("--b-depth", type=int, default=1)
    parser.add_argument("--a-movetime", type=int)
    parser.add_argument("--b-movetime", type=int)
    parser.add_argument("--movetime", type=int, help="Apply the same per-move time limit to both search players.")
    parser.add_argument("--qdepth", type=int, default=6)
    parser.add_argument("--time-check-interval", type=int, default=1024)
    parser.add_argument("--no-mobility", action="store_true")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--opening-plies", type=int, default=0)
    parser.add_argument("--pgn", type=Path)
    args = parser.parse_args()

    a_movetime = args.a_movetime if args.a_movetime is not None else args.movetime
    b_movetime = args.b_movetime if args.b_movetime is not None else args.movetime
    player_a = build_player(
        args.a,
        max(1, args.a_depth),
        a_movetime,
        not args.no_mobility,
        max(0, args.qdepth),
        max(1, args.time_check_interval),
    )
    player_b = build_player(
        args.b,
        max(1, args.b_depth),
        b_movetime,
        not args.no_mobility,
        max(0, args.qdepth),
        max(1, args.time_check_interval),
    )
    result = play_match(
        player_a,
        player_b,
        max(1, args.games),
        max(1, args.max_plies),
        record_pgn=args.pgn is not None,
        opening_plies=max(0, args.opening_plies),
    )
    if args.pgn is not None:
        save_match_pgn(result, args.pgn)
    print(result.summary())


if __name__ == "__main__":
    main()
