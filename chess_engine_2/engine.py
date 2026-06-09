from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Protocol

import chess


CHECKMATE_SCORE = 100_000
DRAW_SCORE = 0
INFINITY = 1_000_000
EXACT_BOUND = "exact"
LOWER_BOUND = "lower"
UPPER_BOUND = "upper"
NULL_MOVE_REDUCTION = 2
ASPIRATION_WINDOW = 50
FUTILITY_MARGIN = 120
LMR_MOVE_THRESHOLD = 4
NEURAL_POLICY_ORDER_SCALE = 100
NEURAL_ORDER_ROOT = "root"
NEURAL_ORDER_DEPTH = "depth"
NEURAL_ORDER_ALL = "all"
EVALUATION_CLASSICAL = "classical"
EVALUATION_NEURAL = "neural"
EVALUATION_BLEND = "blend"

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 320,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

BISHOP_PAIR_BONUS = 35
DOUBLED_PAWN_PENALTY = 15
ISOLATED_PAWN_PENALTY = 12
PASSED_PAWN_BONUS_BY_RANK = [0, 5, 15, 30, 55, 90, 140, 0]
ROOK_OPEN_FILE_BONUS = 25
ROOK_SEMI_OPEN_FILE_BONUS = 12
KING_SHIELD_PAWN_BONUS = 8
KING_OPEN_FILE_PENALTY = 18
KING_SEMI_OPEN_FILE_PENALTY = 9
MOBILITY_WEIGHTS = {
    chess.KNIGHT: 4,
    chess.BISHOP: 4,
    chess.ROOK: 2,
    chess.QUEEN: 1,
}

PAWN_TABLE = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    50,
    50,
    50,
    50,
    50,
    50,
    50,
    50,
    10,
    10,
    20,
    30,
    30,
    20,
    10,
    10,
    5,
    5,
    10,
    25,
    25,
    10,
    5,
    5,
    0,
    0,
    0,
    20,
    20,
    0,
    0,
    0,
    5,
    -5,
    -10,
    0,
    0,
    -10,
    -5,
    5,
    5,
    10,
    10,
    -20,
    -20,
    10,
    10,
    5,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

KNIGHT_TABLE = [
    -50,
    -40,
    -30,
    -30,
    -30,
    -30,
    -40,
    -50,
    -40,
    -20,
    0,
    5,
    5,
    0,
    -20,
    -40,
    -30,
    5,
    10,
    15,
    15,
    10,
    5,
    -30,
    -30,
    0,
    15,
    20,
    20,
    15,
    0,
    -30,
    -30,
    5,
    15,
    20,
    20,
    15,
    5,
    -30,
    -30,
    0,
    10,
    15,
    15,
    10,
    0,
    -30,
    -40,
    -20,
    0,
    0,
    0,
    0,
    -20,
    -40,
    -50,
    -40,
    -30,
    -30,
    -30,
    -30,
    -40,
    -50,
]

BISHOP_TABLE = [
    -20,
    -10,
    -10,
    -10,
    -10,
    -10,
    -10,
    -20,
    -10,
    5,
    0,
    0,
    0,
    0,
    5,
    -10,
    -10,
    10,
    10,
    10,
    10,
    10,
    10,
    -10,
    -10,
    0,
    10,
    10,
    10,
    10,
    0,
    -10,
    -10,
    5,
    5,
    10,
    10,
    5,
    5,
    -10,
    -10,
    0,
    5,
    10,
    10,
    5,
    0,
    -10,
    -10,
    0,
    0,
    0,
    0,
    0,
    0,
    -10,
    -20,
    -10,
    -10,
    -10,
    -10,
    -10,
    -10,
    -20,
]

ROOK_TABLE = [
    0,
    0,
    0,
    5,
    5,
    0,
    0,
    0,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    5,
    10,
    10,
    10,
    10,
    10,
    10,
    5,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

QUEEN_TABLE = [
    -20,
    -10,
    -10,
    -5,
    -5,
    -10,
    -10,
    -20,
    -10,
    0,
    0,
    0,
    0,
    0,
    0,
    -10,
    -10,
    0,
    5,
    5,
    5,
    5,
    0,
    -10,
    -5,
    0,
    5,
    5,
    5,
    5,
    0,
    -5,
    0,
    0,
    5,
    5,
    5,
    5,
    0,
    -5,
    -10,
    5,
    5,
    5,
    5,
    5,
    0,
    -10,
    -10,
    0,
    5,
    0,
    0,
    0,
    0,
    -10,
    -20,
    -10,
    -10,
    -5,
    -5,
    -10,
    -10,
    -20,
]

KING_TABLE = [
    20,
    30,
    10,
    0,
    0,
    10,
    30,
    20,
    20,
    20,
    0,
    0,
    0,
    0,
    20,
    20,
    -10,
    -20,
    -20,
    -20,
    -20,
    -20,
    -20,
    -10,
    -20,
    -30,
    -30,
    -40,
    -40,
    -30,
    -30,
    -20,
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
]

KING_ENDGAME_TABLE = [
    -50,
    -30,
    -30,
    -30,
    -30,
    -30,
    -30,
    -50,
    -30,
    -10,
    0,
    0,
    0,
    0,
    -10,
    -30,
    -30,
    0,
    20,
    30,
    30,
    20,
    0,
    -30,
    -30,
    0,
    30,
    40,
    40,
    30,
    0,
    -30,
    -30,
    0,
    30,
    40,
    40,
    30,
    0,
    -30,
    -30,
    0,
    20,
    30,
    30,
    20,
    0,
    -30,
    -30,
    -10,
    0,
    0,
    0,
    0,
    -10,
    -30,
    -50,
    -30,
    -30,
    -30,
    -30,
    -30,
    -30,
    -50,
]

PIECE_SQUARE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_TABLE,
}


@dataclass
class RandomEngine:
    """Baseline engine that always returns a legal move.

    This gives the project a working UCI-compatible floor before search and
    neural inference are introduced.
    """

    rng: random.Random = field(default_factory=random.Random)

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        return self.rng.choice(legal_moves)


@dataclass
class SearchResult:
    move: chess.Move | None
    score: int
    nodes: int
    depth: int
    pv: list[chess.Move] = field(default_factory=list)
    main_nodes: int = 0
    quiescence_nodes: int = 0


@dataclass
class TranspositionEntry:
    depth: int
    score: int
    bound: str
    best_move: chess.Move | None


class SearchStopped(Exception):
    pass


class MovePolicyScorer(Protocol):
    def score_moves(self, board: chess.Board) -> dict[chess.Move, float]:
        ...


class ValueEvaluator(Protocol):
    def evaluate(self, board: chess.Board) -> int:
        ...


@dataclass
class SearchEngine:
    max_depth: int = 4
    max_quiescence_depth: int = 6
    use_mobility: bool = True
    nodes: int = 0
    main_nodes: int = 0
    quiescence_nodes: int = 0
    evaluate_calls: int = 0
    mobility_calls: int = 0
    policy_ordering_calls: int = 0
    neural_value_calls: int = 0
    neural_value_cache_hits: int = 0
    time_check_interval: int = 1024
    transposition_table: dict[tuple[object, int], TranspositionEntry] = field(default_factory=dict)
    killer_moves: dict[int, list[chess.Move]] = field(default_factory=dict)
    history_scores: dict[tuple[int, int, int | None], int] = field(default_factory=dict)
    policy_scorer: MovePolicyScorer | None = None
    policy_ordering_mode: str = NEURAL_ORDER_ROOT
    policy_ordering_min_depth: int = 2
    value_evaluator: ValueEvaluator | None = None
    evaluation_mode: str = EVALUATION_CLASSICAL
    neural_value_weight: float = 0.2
    neural_value_cache: dict[tuple[object, int], int] = field(default_factory=dict)
    deadline: float | None = None

    def choose_move(self, board: chess.Board, depth: int | None = None) -> chess.Move | None:
        return self.search(board, depth).move

    def search(self, board: chess.Board, depth: int | None = None) -> SearchResult:
        self.reset_search_counters()
        search_depth = self.max_depth if depth is None else max(1, depth)
        return self._search_depth(board, search_depth)

    def iterative_search(
        self,
        board: chess.Board,
        depth: int | None = None,
        movetime_ms: int | None = None,
    ) -> list[SearchResult]:
        self.reset_search_counters()
        max_depth = self.max_depth if depth is None else max(1, depth)
        results = []
        best_move = None
        previous_score = None
        self.deadline = None
        if movetime_ms is not None:
            self.deadline = time.perf_counter() + max(1, movetime_ms) / 1000

        try:
            for current_depth in range(1, max_depth + 1):
                result = self._search_depth(board, current_depth, best_move, previous_score)
                results.append(result)
                best_move = result.move
                previous_score = result.score
        except SearchStopped:
            if not results:
                results.append(self._fallback_result(board))
        finally:
            self.deadline = None

        return results

    def _search_depth(
        self,
        board: chess.Board,
        search_depth: int,
        preferred_move: chess.Move | None = None,
        previous_score: int | None = None,
    ) -> SearchResult:
        if previous_score is None or search_depth <= 1:
            return self._search_depth_window(board, search_depth, -INFINITY, INFINITY, preferred_move)

        alpha = previous_score - ASPIRATION_WINDOW
        beta = previous_score + ASPIRATION_WINDOW
        result = self._search_depth_window(board, search_depth, alpha, beta, preferred_move)
        if result.score <= alpha or result.score >= beta:
            result = self._search_depth_window(board, search_depth, -INFINITY, INFINITY, preferred_move)

        return result

    def _search_depth_window(
        self,
        board: chess.Board,
        search_depth: int,
        alpha: int,
        beta: int,
        preferred_move: chess.Move | None = None,
    ) -> SearchResult:
        best_move = None
        root_alpha = alpha
        root_beta = beta
        best_score = -INFINITY

        for move in ordered_moves(board, preferred_move, policy_scores=self._root_policy_scores(board)):
            board.push(move)
            try:
                score = -self._negamax(board, search_depth - 1, -root_beta, -root_alpha, ply=1, allow_null_move=True)
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            root_alpha = max(root_alpha, score)

        if best_move is None:
            best_score = self.evaluate(board)

        self.transposition_table[self._cache_key(board)] = TranspositionEntry(
            search_depth,
            best_score,
            EXACT_BOUND,
            best_move,
        )

        return SearchResult(
            best_move,
            best_score,
            self.nodes,
            search_depth,
            self.principal_variation(board),
            self.main_nodes,
            self.quiescence_nodes,
        )

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        allow_null_move: bool,
    ) -> int:
        self.nodes += 1
        self.main_nodes += 1
        self._maybe_check_deadline()
        original_alpha = alpha
        cache_key = self._cache_key(board)
        entry = self.transposition_table.get(cache_key)
        if entry is not None and entry.depth >= depth:
            if entry.bound == EXACT_BOUND:
                return entry.score
            if entry.bound == LOWER_BOUND:
                alpha = max(alpha, entry.score)
            elif entry.bound == UPPER_BOUND:
                beta = min(beta, entry.score)
            if alpha >= beta:
                return entry.score

        if board.is_checkmate():
            return -CHECKMATE_SCORE - depth
        if board.is_stalemate() or board.is_insufficient_material():
            return DRAW_SCORE
        if depth == 0:
            return self._quiescence(board, self.max_quiescence_depth, alpha, beta)

        if self._can_try_null_move(board, depth, allow_null_move):
            board.push(chess.Move.null())
            try:
                score = -self._negamax(
                    board,
                    depth - 1 - NULL_MOVE_REDUCTION,
                    -beta,
                    -beta + 1,
                    ply + 1,
                    allow_null_move=False,
                )
            finally:
                board.pop()
            if score >= beta:
                return beta

        best_score = -INFINITY
        best_move = None
        moves = ordered_moves(
            board,
            entry.best_move if entry is not None else None,
            self.killer_moves.get(ply, []),
            self.history_scores,
            self._tree_policy_scores(board, depth),
        )
        static_eval = self.evaluate(board) if depth == 1 and not board.is_check() else None

        for move_index, move in enumerate(moves):
            if self._can_futility_prune(board, move, depth, alpha, static_eval):
                continue

            reduction = self._late_move_reduction(board, move, depth, move_index)
            board.push(move)
            try:
                if reduction:
                    score = -self._negamax(
                        board,
                        depth - 1 - reduction,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        allow_null_move=True,
                    )
                    if score > alpha:
                        score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, allow_null_move=True)
                else:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, allow_null_move=True)
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                if is_quiet_move(board, move):
                    self._record_quiet_cutoff(move, depth, ply)
                break

        if best_score <= original_alpha:
            bound = UPPER_BOUND
        elif best_score >= beta:
            bound = LOWER_BOUND
        else:
            bound = EXACT_BOUND
        self.transposition_table[cache_key] = TranspositionEntry(depth, best_score, bound, best_move)

        return best_score

    def _quiescence(self, board: chess.Board, depth: int, alpha: int, beta: int) -> int:
        self.nodes += 1
        self.quiescence_nodes += 1
        self._maybe_check_deadline()

        if board.is_checkmate():
            return -CHECKMATE_SCORE
        if board.is_stalemate() or board.is_insufficient_material():
            return DRAW_SCORE

        stand_pat = self.evaluate(board)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

        if depth == 0:
            return alpha

        for move in tactical_moves(board):
            board.push(move)
            try:
                score = -self._quiescence(board, depth - 1, -beta, -alpha)
            finally:
                board.pop()

            if score >= beta:
                return beta
            alpha = max(alpha, score)

        return alpha

    def _record_quiet_cutoff(self, move: chess.Move, depth: int, ply: int) -> None:
        killers = self.killer_moves.setdefault(ply, [])
        if move not in killers:
            killers.insert(0, move)
            del killers[2:]

        history_key = move_history_key(move)
        self.history_scores[history_key] = self.history_scores.get(history_key, 0) + depth * depth

    def _can_try_null_move(self, board: chess.Board, depth: int, allow_null_move: bool) -> bool:
        return (
            allow_null_move
            and depth > NULL_MOVE_REDUCTION
            and not board.is_check()
            and has_non_pawn_material(board, board.turn)
        )

    def _can_futility_prune(
        self,
        board: chess.Board,
        move: chess.Move,
        depth: int,
        alpha: int,
        static_eval: int | None,
    ) -> bool:
        return (
            depth == 1
            and static_eval is not None
            and static_eval + FUTILITY_MARGIN <= alpha
            and is_quiet_move(board, move)
            and not board.gives_check(move)
        )

    def _late_move_reduction(self, board: chess.Board, move: chess.Move, depth: int, move_index: int) -> int:
        if (
            depth >= 3
            and move_index >= LMR_MOVE_THRESHOLD
            and is_quiet_move(board, move)
            and not board.is_check()
            and not board.gives_check(move)
        ):
            return 1

        return 0

    def _cache_key(self, board: chess.Board) -> tuple[object, int]:
        if hasattr(board, "transposition_key"):
            position_key = board.transposition_key()
        elif hasattr(board, "_transposition_key"):
            position_key = board._transposition_key()
        else:
            position_key = " ".join(board.fen().split()[:4])

        return (position_key, board.halfmove_clock)

    def _check_deadline(self) -> None:
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise SearchStopped()

    def _maybe_check_deadline(self) -> None:
        if self.deadline is not None and self.nodes % max(1, self.time_check_interval) == 0:
            self._check_deadline()

    def _fallback_result(self, board: chess.Board) -> SearchResult:
        move = next(iter(board.legal_moves), None)
        return SearchResult(
            move,
            self.evaluate(board),
            self.nodes,
            0,
            [move] if move is not None else [],
            self.main_nodes,
            self.quiescence_nodes,
        )

    def _root_policy_scores(self, board: chess.Board) -> dict[chess.Move, float] | None:
        if self.policy_ordering_mode in {NEURAL_ORDER_ROOT, NEURAL_ORDER_DEPTH, NEURAL_ORDER_ALL}:
            return self._policy_scores(board)
        return None

    def _tree_policy_scores(self, board: chess.Board, depth: int) -> dict[chess.Move, float] | None:
        if self.policy_ordering_mode == NEURAL_ORDER_ALL:
            return self._policy_scores(board)
        if self.policy_ordering_mode == NEURAL_ORDER_DEPTH and depth >= self.policy_ordering_min_depth:
            return self._policy_scores(board)
        return None

    def _policy_scores(self, board: chess.Board) -> dict[chess.Move, float] | None:
        if self.policy_scorer is None:
            return None
        self.policy_ordering_calls += 1
        return self.policy_scorer.score_moves(board)

    def reset_search_counters(self) -> None:
        self.nodes = 0
        self.main_nodes = 0
        self.quiescence_nodes = 0
        self.evaluate_calls = 0
        self.mobility_calls = 0
        self.policy_ordering_calls = 0
        self.neural_value_calls = 0
        self.neural_value_cache_hits = 0

    def evaluate(self, board: chess.Board) -> int:
        self.evaluate_calls += 1
        if self.use_mobility and self.evaluation_mode != EVALUATION_NEURAL:
            self.mobility_calls += 1
        if self.value_evaluator is None or self.evaluation_mode == EVALUATION_CLASSICAL:
            return evaluate(board, use_mobility=self.use_mobility)

        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
            return evaluate(board, use_mobility=self.use_mobility)

        neural_score = self._neural_value_score(board)
        if self.evaluation_mode == EVALUATION_NEURAL:
            return neural_score

        classical_score = evaluate(board, use_mobility=self.use_mobility)
        weight = min(1.0, max(0.0, self.neural_value_weight))
        return int((1.0 - weight) * classical_score + weight * neural_score)

    def _neural_value_score(self, board: chess.Board) -> int:
        key = self._cache_key(board)
        if key in self.neural_value_cache:
            self.neural_value_cache_hits += 1
            return self.neural_value_cache[key]

        self.neural_value_calls += 1
        score = self.value_evaluator.evaluate(board)
        self.neural_value_cache[key] = score
        return score

    def principal_variation(self, board: chess.Board, max_length: int | None = None) -> list[chess.Move]:
        max_ply = self.max_depth if max_length is None else max_length
        pv = []
        seen_keys = set()

        for _ in range(max_ply):
            key = self._cache_key(board)
            if key in seen_keys:
                break
            seen_keys.add(key)

            entry = self.transposition_table.get(key)
            if entry is None or entry.best_move is None or entry.best_move not in board.legal_moves:
                break

            pv.append(entry.best_move)
            board.push(entry.best_move)

        for _ in pv:
            board.pop()

        return pv


def evaluate(board: chess.Board, use_mobility: bool = True) -> int:
    """Evaluate from the side-to-move perspective in centipawns."""

    if board.is_checkmate():
        return -CHECKMATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return DRAW_SCORE

    score = 0
    phase = game_phase(board)
    for square, piece in board.piece_map().items():
        piece_score = PIECE_VALUES[piece.piece_type]
        piece_score += piece_square_bonus(piece, square, phase)

        if piece.color == chess.WHITE:
            score += piece_score
        else:
            score -= piece_score

    score += bishop_pair_score(board)
    score += pawn_structure_score(board)
    score += rook_file_score(board)
    score += king_safety_score(board, phase)
    if use_mobility:
        score += mobility_score(board)

    return score if board.turn == chess.WHITE else -score


def piece_square_bonus(piece: chess.Piece, square: chess.Square, phase: int) -> int:
    table_square = square if piece.color == chess.WHITE else chess.square_mirror(square)

    if piece.piece_type == chess.KING:
        middle_game_score = KING_TABLE[table_square]
        endgame_score = KING_ENDGAME_TABLE[table_square]
        return (middle_game_score * phase + endgame_score * (24 - phase)) // 24

    table = PIECE_SQUARE_TABLES.get(piece.piece_type)
    if table is None:
        return 0

    return table[table_square]


def game_phase(board: chess.Board) -> int:
    phase = 0
    phase += len(board.pieces(chess.KNIGHT, chess.WHITE) | board.pieces(chess.KNIGHT, chess.BLACK))
    phase += len(board.pieces(chess.BISHOP, chess.WHITE) | board.pieces(chess.BISHOP, chess.BLACK))
    phase += 2 * len(board.pieces(chess.ROOK, chess.WHITE) | board.pieces(chess.ROOK, chess.BLACK))
    phase += 4 * len(board.pieces(chess.QUEEN, chess.WHITE) | board.pieces(chess.QUEEN, chess.BLACK))
    return min(24, phase)


def bishop_pair_score(board: chess.Board) -> int:
    score = 0
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += BISHOP_PAIR_BONUS
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= BISHOP_PAIR_BONUS
    return score


def pawn_structure_score(board: chess.Board) -> int:
    return color_pawn_structure_score(board, chess.WHITE) - color_pawn_structure_score(board, chess.BLACK)


def color_pawn_structure_score(board: chess.Board, color: chess.Color) -> int:
    score = 0
    pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)

    for file_index in range(8):
        file_pawns = [square for square in pawns if chess.square_file(square) == file_index]
        if len(file_pawns) > 1:
            score -= DOUBLED_PAWN_PENALTY * (len(file_pawns) - 1)

    for square in pawns:
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        adjacent_files = [file for file in (file_index - 1, file_index + 1) if 0 <= file <= 7]

        if not any(chess.square_file(pawn) in adjacent_files for pawn in pawns):
            score -= ISOLATED_PAWN_PENALTY

        if is_passed_pawn(square, color, enemy_pawns):
            progress_rank = rank_index if color == chess.WHITE else 7 - rank_index
            score += PASSED_PAWN_BONUS_BY_RANK[progress_rank]

    return score


def is_passed_pawn(square: chess.Square, color: chess.Color, enemy_pawns: chess.SquareSet) -> bool:
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    relevant_files = [file for file in (file_index - 1, file_index, file_index + 1) if 0 <= file <= 7]

    for enemy_square in enemy_pawns:
        enemy_file = chess.square_file(enemy_square)
        enemy_rank = chess.square_rank(enemy_square)
        if enemy_file not in relevant_files:
            continue
        if color == chess.WHITE and enemy_rank > rank_index:
            return False
        if color == chess.BLACK and enemy_rank < rank_index:
            return False

    return True


def rook_file_score(board: chess.Board) -> int:
    return color_rook_file_score(board, chess.WHITE) - color_rook_file_score(board, chess.BLACK)


def color_rook_file_score(board: chess.Board, color: chess.Color) -> int:
    score = 0
    own_pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)

    for rook_square in board.pieces(chess.ROOK, color):
        file_index = chess.square_file(rook_square)
        has_own_pawn = any(chess.square_file(square) == file_index for square in own_pawns)
        has_enemy_pawn = any(chess.square_file(square) == file_index for square in enemy_pawns)

        if not has_own_pawn and not has_enemy_pawn:
            score += ROOK_OPEN_FILE_BONUS
        elif not has_own_pawn:
            score += ROOK_SEMI_OPEN_FILE_BONUS

    return score


def king_safety_score(board: chess.Board, phase: int) -> int:
    if phase <= 8:
        return 0

    white_score = color_king_safety_score(board, chess.WHITE)
    black_score = color_king_safety_score(board, chess.BLACK)
    return ((white_score - black_score) * phase) // 24


def color_king_safety_score(board: chess.Board, color: chess.Color) -> int:
    king_square = board.king(color)
    if king_square is None:
        return 0

    score = 0
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)
    own_pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)
    shield_rank = king_rank + 1 if color == chess.WHITE else king_rank - 1

    for file_index in nearby_files(king_file):
        if 0 <= shield_rank <= 7 and chess.square(file_index, shield_rank) in own_pawns:
            score += KING_SHIELD_PAWN_BONUS

        has_own_pawn = any(chess.square_file(square) == file_index for square in own_pawns)
        has_enemy_pawn = any(chess.square_file(square) == file_index for square in enemy_pawns)
        if not has_own_pawn and not has_enemy_pawn:
            score -= KING_OPEN_FILE_PENALTY
        elif not has_own_pawn:
            score -= KING_SEMI_OPEN_FILE_PENALTY

    return score


def nearby_files(file_index: int) -> list[int]:
    return [file for file in (file_index - 1, file_index, file_index + 1) if 0 <= file <= 7]


def mobility_score(board: chess.Board) -> int:
    return color_mobility_score(board, chess.WHITE) - color_mobility_score(board, chess.BLACK)


def color_mobility_score(board: chess.Board, color: chess.Color) -> int:
    score = 0
    own_occupied = board.occupied_co[color]

    for piece_type, weight in MOBILITY_WEIGHTS.items():
        for square in board.pieces(piece_type, color):
            reachable = board.attacks(square) & ~own_occupied
            score += weight * len(reachable)

    return score


def ordered_moves(
    board: chess.Board,
    preferred_move: chess.Move | None = None,
    killer_moves: list[chess.Move] | None = None,
    history_scores: dict[tuple[int, int, int | None], int] | None = None,
    policy_scores: dict[chess.Move, float] | None = None,
) -> list[chess.Move]:
    return sorted(
        board.legal_moves,
        key=lambda move: move_order_score(board, move, preferred_move, killer_moves, history_scores, policy_scores),
        reverse=True,
    )


def tactical_moves(board: chess.Board) -> list[chess.Move]:
    moves = [
        move
        for move in board.legal_moves
        if board.is_capture(move) or move.promotion is not None
    ]
    return sorted(moves, key=lambda move: move_order_score(board, move), reverse=True)


def move_order_score(
    board: chess.Board,
    move: chess.Move,
    preferred_move: chess.Move | None = None,
    killer_moves: list[chess.Move] | None = None,
    history_scores: dict[tuple[int, int, int | None], int] | None = None,
    policy_scores: dict[chess.Move, float] | None = None,
) -> int:
    score = 0

    if preferred_move == move:
        score += 10_000

    if policy_scores is not None:
        score += int(policy_scores.get(move, 0.0) * NEURAL_POLICY_ORDER_SCALE)

    if killer_moves is not None and move in killer_moves:
        score += 2_000

    if history_scores is not None:
        score += history_scores.get(move_history_key(move), 0)

    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim is not None and attacker is not None:
            score += 10 * PIECE_VALUES[victim.piece_type] - PIECE_VALUES[attacker.piece_type]
        else:
            score += 100

    if move.promotion is not None:
        score += PIECE_VALUES[move.promotion]

    if board.gives_check(move):
        score += 50

    return score


def is_quiet_move(board: chess.Board, move: chess.Move) -> bool:
    return not board.is_capture(move) and move.promotion is None


def move_history_key(move: chess.Move) -> tuple[int, int, int | None]:
    return (move.from_square, move.to_square, move.promotion)


def has_non_pawn_material(board: chess.Board, color: chess.Color) -> bool:
    return any(
        board.pieces(piece_type, color)
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )


def perft(board: chess.Board, depth: int) -> int:
    """Count legal leaf nodes from a position to validate move generation."""

    if depth < 0:
        raise ValueError("depth must be non-negative")
    if depth == 0:
        return 1
    if depth == 1:
        return board.legal_moves.count()

    nodes = 0
    for move in board.legal_moves:
        board.push(move)
        nodes += perft(board, depth - 1)
        board.pop()
    return nodes
