from chess_engine_2.benchmark import BenchmarkRow
from chess_engine_2.tune import QDepthTuningResult, best_qdepth, format_qdepth_results


def make_result(qdepth: int, score_percent: float, avg_depth: float, q_nodes: float) -> QDepthTuningResult:
    return QDepthTuningResult(
        qdepth,
        BenchmarkRow(
            depth=3,
            games=10,
            wins=0,
            losses=0,
            draws=10,
            score=score_percent / 10,
            score_percent=score_percent,
            average_plies=40,
            elapsed_seconds=1,
            a_average_depth=avg_depth,
            a_average_quiescence_nodes=q_nodes,
        ),
    )


def test_best_qdepth_prefers_score_then_depth_then_fewer_qnodes() -> None:
    results = [
        make_result(2, 50, 2.0, 100),
        make_result(4, 60, 1.5, 300),
        make_result(6, 60, 1.5, 500),
    ]

    assert best_qdepth(results).qdepth == 4


def test_format_qdepth_results_includes_recommendation() -> None:
    output = format_qdepth_results([make_result(4, 60, 1.5, 300)])

    assert "recommended qdepth: 4" in output

