# Chess Engine

An original Python chess engine and neural-search research project.

The project began as a legal UCI engine and grew into a hybrid system containing:

- A classical negamax alpha-beta engine
- Handcrafted chess evaluation
- A PyTorch policy/value network
- Neural move ordering for alpha-beta
- Pure neural-policy play
- PUCT Monte Carlo Tree Search
- Public Lichess data ingestion and GPU training
- Search-distilled sibling move-ranking experiments

The current research question is no longer whether the components work. They do.
The open problem is how to convert improved neural move ranking into stronger
MCTS play.

## Current Status

| Area | Status |
| --- | --- |
| UCI engine | Working |
| Alpha-beta search | Working |
| Time controls | Working |
| Match runner and adjudication | Working |
| PGN and Lichess data pipeline | Working |
| CUDA policy/value training | Working |
| Neural alpha-beta ordering | Working |
| Pure neural player | Working |
| PUCT MCTS | Working |
| PUCT and value diagnostics | Working |
| Search-distilled ranking training | Working |
| Self-play training | Not implemented |
| C++/bitboard rewrite | Intentionally deferred |

The full test suite currently contains **130 passing tests**.

## Architecture

```text
                         +----------------------+
                         |  Lichess PGN data    |
                         +----------+-----------+
                                    |
                         filtering / JSONL export
                                    |
                         +----------v-----------+
                         | Policy/value network |
                         +-----+------------+---+
                               |            |
                         policy priors   position value
                               |            |
             +-----------------+            +-----------------+
             |                                                |
    alpha-beta move ordering                          PUCT / MCTS search
             |                                                |
    +--------v---------+                             +--------v---------+
    | Classical engine |                             | Neural tree      |
    | and evaluation   |                             | search           |
    +------------------+                             +------------------+

    Classical alpha-beta also produces sibling rankings used to distill
    stronger move-discrimination targets back into the neural value head.
```

## Classical Engine

The classical backend includes:

- Negamax with alpha-beta pruning
- Iterative deepening
- Aspiration windows
- Quiescence search
- Transposition table
- Principal variation reporting
- Killer moves and history heuristic
- MVV-LVA capture ordering
- Null-move pruning
- Late-move reductions
- Futility pruning
- Fixed-depth and time-controlled search
- Safe fallback to the last completed iteration

The handcrafted evaluator includes material, piece-square tables, pawn
structure, mobility, rook activity, king safety, bishop-pair bonuses, and
tapered king evaluation.

## Neural Engine

The PyTorch network consumes `18 x 8 x 8` board planes and produces:

- A policy over **4,672** AlphaZero-style move indices
- A scalar value in `[-1, 1]`

The 4,672 policy outputs come from:

```text
64 source squares x 73 move planes
```

The network can be used for:

- Alpha-beta move ordering
- Neural-only move selection
- Neural or blended alpha-beta evaluation
- Policy priors and leaf values in PUCT MCTS
- Search-distilled sibling ranking

## Installation

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

For CUDA training, install a CUDA-enabled PyTorch build appropriate for the
machine, then verify it:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Run the UCI Engine

Start the classical engine:

```powershell
python -m chess_engine_2.uci
```

Example UCI session:

```text
uci
isready
position startpos moves e2e4 e7e5
go depth 4
go movetime 1000
go wtime 60000 btime 60000 winc 1000 binc 1000
quit
```

Use neural policy ordering with alpha-beta:

```powershell
python -m chess_engine_2.uci `
  --neural-checkpoint models/policy_value_2023_2000plus_10000.pt `
  --neural-ordering root
```

Use neural or blended leaf evaluation:

```powershell
python -m chess_engine_2.uci `
  --value-checkpoint models/policy_value_2023_2000plus_10000.pt `
  --evaluation-mode blend `
  --neural-value-weight 0.2
```

Run the MCTS backend:

```powershell
python -m chess_engine_2.uci `
  --search-backend mcts `
  --neural-checkpoint models/policy_value_2023_2000plus_10000.pt `
  --mcts-simulations 100 `
  --mcts-cpuct 1.5
```

Model checkpoints are generated locally and are not stored in Git.

## Tests

```powershell
python -m pytest -q
```

The tests cover search, UCI behavior, move encoding, data processing, neural
training, matches, MCTS, caching, PUCT diagnostics, value ranking, and
search-distillation training.

## Match Play

Run alpha-beta against the random baseline:

```powershell
python -m chess_engine_2.match `
  --a search --a-depth 2 `
  --b random `
  --games 10 `
  --max-plies 200 `
  --pgn match.pgn
```

Compare two search depths with adjudication:

```powershell
python -m chess_engine_2.match `
  --a search --a-depth 3 `
  --b search --b-depth 2 `
  --games 100 `
  --max-plies 200 `
  --opening-plies 4 `
  --adjudicate
```

The match runner reports wins, losses, draws, average plies, search statistics,
and termination reasons such as checkmate, repetition, stalemate, insufficient
material, adjudication, or move limit.

## Benchmarking

Run a depth ladder:

```powershell
python -m chess_engine_2.benchmark `
  --depths 2 3 `
  --opponent search `
  --opponent-depth 2 `
  --games-list 10 20 50 100 `
  --opening-plies 4 `
  --max-plies 200 `
  --adjudicate `
  --stream `
  --csv benchmark.csv
```

Benchmark MCTS simulation counts:

```powershell
python -m chess_engine_2.mcts_benchmark `
  models/policy_value_2023_2000plus_10000.pt `
  --simulations 64 128 256 512 `
  --games 2 `
  --opponent-depth 1 `
  --opening-plies 4 `
  --max-plies 200 `
  --adjudicate
```

Small game counts in this repository are treated as smoke tests, not reliable
Elo measurements.

## Data Pipeline

### Local Lichess Archive

Download a monthly archive:

```powershell
python -m chess_engine_2.data.download `
  --month 2013-02 `
  --output-dir data/raw
```

Export PGN positions to JSONL:

```powershell
python -m chess_engine_2.data.pgn `
  data/raw/lichess_db_standard_rated_2013-02.pgn.zst `
  --max-games 1000 `
  --output data/processed/lichess_2013-02_1000.jsonl
```

Validate the result:

```powershell
python -m chess_engine_2.data.dataset `
  data/processed/lichess_2013-02_1000.jsonl
```

### Stream High-Elo Games

Large Lichess archives can be filtered while streaming, without downloading the
entire monthly file:

```powershell
python -m chess_engine_2.data.remote_pgn `
  --month 2023-01 `
  --min-elo 2000 `
  --max-output-games 1000 `
  --output data/processed/lichess_2023-01_2000plus_valid.jsonl

python -m chess_engine_2.data.remote_pgn `
  --month 2023-01 `
  --min-elo 2000 `
  --skip-output-games 1000 `
  --max-output-games 10000 `
  --output data/processed/lichess_2023-01_2000plus_train.jsonl
```

The main high-Elo experiment used:

- 10,000 training games
- 758,917 training positions
- 1,000 separate validation games
- 75,185 validation positions
- Both players rated at least 2000

## Standard Neural Training

Train with validation and a reusable tensor cache:

```powershell
python -m chess_engine_2.train `
  data/processed/lichess_2023-01_2000plus_train.jsonl `
  --epochs 3 `
  --batch-size 256 `
  --channels 32 `
  --validation-split 0.1 `
  --num-workers 2 `
  --tensor-cache data/processed/high_elo_tensors.pt `
  --checkpoint models/policy_value_2023_2000plus_10000.pt
```

Evaluate a checkpoint:

```powershell
python -m chess_engine_2.evaluate_checkpoint `
  models/policy_value_2023_2000plus_10000.pt `
  data/processed/lichess_2023-01_2000plus_valid.jsonl `
  --channels 32 `
  --batch-size 256
```

Inspect legal policy predictions:

```powershell
python -m chess_engine_2.predict `
  --checkpoint models/policy_value_2023_2000plus_10000.pt `
  --top 5
```

## Value-Target Experiments

Dense value annotation supports material and classical alpha-beta targets:

```powershell
python -m chess_engine_2.annotate_values `
  data/processed/lichess_2023-01_2000plus_train.jsonl `
  data/processed/dense_values.jsonl `
  --depth 1 `
  --qdepth 0 `
  --no-mobility `
  --max-samples 20000
```

Available training targets include:

- `value`
- `discounted_value`
- `material_value`
- `classical_value`
- `result_material_blend`
- `result_classical_blend`
- `discounted_classical_blend`

The discounted target is:

```text
result * sqrt(current ply / game length)
```

### Value-Target Result

| Target | Holdout value loss | MCTS-64 score vs depth 1 |
| --- | ---: | ---: |
| Discounted outcome | 0.5739 | 0% |
| 70% outcome + 30% material | 0.5688 | 25% |
| 70% outcome + 30% classical | 0.5815 | 25% |
| 70% discounted + 30% classical | **0.3059** | 25% |

Lower value loss did not produce stronger MCTS. This established that target
MSE alone was not measuring the value head's usefulness for move selection.

## PUCT Diagnostics

Run a CPUCT and simulation sweep:

```powershell
python -m chess_engine_2.puct_diagnostics `
  models/policy_value_discounted_classical_valuehead_20000.pt `
  --cpuct 0.25 0.5 1.0 2.0 4.0 `
  --simulations 64 256 `
  --games 2
```

The report records:

- Policy prior `P`
- Visit count `N`
- Mean value `Q`
- Exploration bonus `U`
- Combined PUCT score
- Leaf-value distribution
- Match statistics

The diagnostics showed that MCTS does listen to the value head. Low CPUCT lets
value estimates overturn large policy priors, while high CPUCT follows the
policy more strongly. Neither CPUCT tuning nor increasing from 64 to 256
simulations solved playing strength.

## Neural Move-Ranking Study

MCTS needs useful ordering among sibling moves, not merely a low average value
error. The ranking study compares neural child values against deeper alpha-beta
scores for every legal move:

```powershell
python -m chess_engine_2.value_ranking `
  models/policy_value_discounted_classical_valuehead_20000.pt `
  data/processed/value_targets_valid_10000.jsonl `
  --positions 10 `
  --child-depth 3 `
  --qdepth 2
```

The original value model produced:

| Metric | Result |
| --- | ---: |
| Candidate moves | 363 |
| Mean Spearman correlation | -0.015 |
| Pairwise ordering accuracy | 49.6% |
| Top-1 accuracy | 0% |
| Top-3 accuracy | 10% |
| Mean top-1 regret | 413.7 cp |
| Mean top-3 regret | 232.0 cp |

The value head was fitting smooth position targets while ranking sibling moves
approximately at random. In several positions, its preferred move was six to
nine pawns worse than the alpha-beta best move.

## Search Distillation

Search distillation creates one group per root position, preserving every legal
sibling move and its alpha-beta score:

```powershell
python -m chess_engine_2.distill_rankings `
  data/processed/value_targets_train_20000.jsonl `
  data/processed/ranking_distillation_depth3_500.jsonl `
  --positions 500 `
  --child-depth 3 `
  --qdepth 2 `
  --workers 4 `
  --resume
```

Generation supports:

- Multiple worker processes
- Incremental durable writes
- Restart recovery
- Reuse of completed smaller datasets with `--seed-from`
- Deterministic nested dataset sizes

Train the value head with search-score regression and pairwise ranking loss:

```powershell
python -m chess_engine_2.train_rankings `
  data/processed/ranking_distillation_depth3_500.jsonl `
  --max-groups 500 `
  --validation-groups data/processed/ranking_holdout_depth3_10.jsonl `
  --initial-checkpoint models/policy_value_discounted_classical_valuehead_20000.pt `
  --checkpoint models/policy_value_search_ranking_depth3_500_scaled.pt `
  --epochs 30 `
  --ranking-weight 1.0
```

Evaluate against cached search rankings:

```powershell
python -m chess_engine_2.evaluate_rankings `
  models/policy_value_search_ranking_depth3_500_scaled.pt `
  data/processed/ranking_holdout_depth3_10.jsonl
```

### Distillation Scaling Result

All rungs used one fixed holdout containing 10 roots and 363 legal moves.

| Training roots | Children | Spearman | Pairwise | Top-1 | Top-3 | Top-1 regret | Top-3 regret |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | - | -0.015 | 49.6% | 0% | 10% | 413.7 cp | 232.0 cp |
| 20 | 682 | 0.155 | 55.6% | 0% | 30% | 400.3 cp | 197.3 cp |
| 50 | 1,653 | 0.236 | 58.8% | 0% | 30% | 470.3 cp | 113.3 cp |
| 100 | 3,439 | 0.260 | 59.5% | 0% | 30% | 255.6 cp | **83.5 cp** |
| 500 | 17,179 | **0.383** | **63.4%** | **10%** | **40%** | **214.7 cp** | 123.2 cp |

Search-distilled supervision is the clearest successful scaling direction in
the project. Pairwise accuracy increased from random-level `49.6%` to `63.4%`,
and mean top-1 regret nearly halved.

Matched ten-game MCTS-64 smoke tests still did not show a strength gain:

| Model | Score vs depth 1 |
| --- | ---: |
| Original value model | 20% |
| 100-root distilled model | 10% |
| 500-root distilled model | 15% |

The learning objective is improving, but its current use as an absolute MCTS
leaf value remains unresolved.

## Key Findings

1. The classical alpha-beta engine is functionally healthy.
2. Better and higher-rated data improves policy learning.
3. Neural policy ordering changes and often reduces the alpha-beta tree.
4. Better policy accuracy does not automatically produce higher match scores.
5. Lower value MSE does not imply useful MCTS guidance.
6. PUCT uses the value signal; it is not simply ignoring it.
7. The original value head ranked sibling moves approximately at random.
8. Search-distilled pairwise supervision improves sibling ranking and scales.
9. Better offline ranking has not yet become stronger MCTS play.

## Roadmap

The next work should focus on search-network interaction:

1. Separate absolute leaf value from sibling move-ranking bias.
2. Measure policy-value disagreement and calibration inside the tree.
3. Distill alpha-beta search preferences into the policy head.
4. Test hybrid selection using the original value for backup and the distilled
   model for bounded sibling ranking.
5. Scale beyond 500 distilled roots only after identifying the best way to use
   the ranking signal.
6. Introduce self-play after the neural search is competitive with shallow
   alpha-beta.
7. Consider C++ and bitboards after algorithmic strength, not before it.

## Repository Layout

```text
chess_engine_2/
  engine.py              Classical evaluation and alpha-beta search
  uci.py                 UCI protocol
  encoding.py            4,672-move policy encoding
  match.py               Match runner, adjudication, PGN, and statistics
  benchmark.py           Alpha-beta and neural benchmarks
  neural.py              Policy/value network and checkpoints
  train.py               Standard neural training
  mcts.py                PUCT search and inference cache
  mcts_benchmark.py      MCTS scaling benchmarks
  puct_diagnostics.py    Root P/N/Q/U diagnostics
  value_ranking.py       Neural-versus-search ranking analysis
  distill_rankings.py    Parallel resumable search distillation
  train_rankings.py      Regression and pairwise ranking training
  evaluate_rankings.py   Cached ranking evaluation
  annotate_values.py     Dense value annotation
  evaluate_checkpoint.py Checkpoint evaluation
  data/
    download.py          Lichess archive downloader
    remote_pgn.py        Remote filtered archive streaming
    pgn.py               PGN parsing and JSONL export
    dataset.py           Board planes and sample validation

tests/                   Automated test suite
```

## Project Scope

This is an experimental engine, not a production-strength Stockfish or Lc0
replacement. Generated datasets, checkpoints, match outputs, and benchmark
files are intentionally excluded from Git because they are large and
machine-specific.

The project deliberately remains in Python while the main bottleneck is neural
decision quality. A C++ rewrite would improve throughput, but current evidence
shows that search-network interaction is the more important problem to solve.
