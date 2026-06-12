# Chess Engine 

Python prototype for an chess engine path.

The first milestone is intentionally modest: a legal UCI engine that can be loaded
by chess GUIs and tested reliably. From there, the project can grow through
alpha-beta search, supervised policy/value training, and eventually PUCT MCTS.

## Roadmap

1. Legal move engine using `python-chess`.
2. UCI protocol loop.
3. Random legal move baseline.
4. Negamax with alpha-beta pruning. 
5. Perft tests for move-generation confidence.
6. AlphaZero-style 4672 move encoding. Done.
7. Quiescence search. Done.
8. Transposition table, iterative deepening, and time controls. Done.
9. Search pruning and move ordering heuristics. Done.
10. Richer handcrafted evaluation. Done.
11. Engine-vs-engine match runner and PGN export. Done.
12. Lichess PGN downloader and parser. Done.
13. PyTorch policy/value network. Done.
14. Neural policy move ordering for alpha-beta search. Done.
15. Neural-only move selection. Done.
16. Larger 1,000-game neural training run. Done.
17. Value network inside search. Done.
18. GPU 10,000-game training with validation and tensor caching. Done.
19. PUCT MCTS prototype. Done.
20. MCTS simulation scaling study. Done.
21. Higher-rated training data.
22. Improved value targets.
23. Residual policy/value network.
24. Self-play training loop.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run as a UCI engine

```powershell
python -m chess_engine_2.uci
```

To run the UCI engine with neural root move ordering:

```powershell
python -m chess_engine_2.uci --neural-checkpoint models/policy_value_phase7.pt --neural-ordering root
```

Try these commands:

```text
uci
isready
position startpos
go depth 4
go movetime 1000
go wtime 60000 btime 60000 winc 1000 binc 1000
quit
```

The current engine uses material, piece-square tables, pawn structure, rook file
activity, mobility, king safety, tapered king evaluation, negamax alpha-beta,
quiescence search, transposition tables, iterative deepening, UCI time controls,
null move pruning, late move reductions, futility pruning, killer moves, history
heuristics, principal-variation output, and optional neural policy move ordering.
If no depth or time control is supplied, it searches to depth 4.

## Tests

```powershell
pytest
```

## Lichess PGN Data

For a small first public dataset, use January 2013 from the Lichess standard
rated database:

```powershell
python -m chess_engine_2.data.download --month 2013-01 --output-dir data/raw
```

Parse only a few games while testing the pipeline:

```powershell
python -m chess_engine_2.data.pgn data/raw/lichess_db_standard_rated_2013-01.pgn.zst --max-games 5
```

Write streamed training samples to JSONL:

```powershell
python -m chess_engine_2.data.pgn data/raw/lichess_db_standard_rated_2013-02.pgn.zst --max-games 100 --output data/processed/lichess_2013-02_100.jsonl
```

Validate those samples and convert each FEN into neural-network input planes:

```powershell
python -m chess_engine_2.data.dataset data/processed/lichess_2013-02_100.jsonl --max-samples 1000
```

Train the first small policy/value network:

```powershell
python -m chess_engine_2.train data/processed/lichess_2013-02_100.jsonl --epochs 3 --batch-size 64 --checkpoint models/policy_value_smoke.pt
```

Train the larger 1,000-game Phase 10 checkpoint:

```powershell
python -m chess_engine_2.data.pgn data/raw/lichess_db_standard_rated_2013-02.pgn.zst --max-games 1000 --output data/processed/lichess_2013-02_1000.jsonl
python -m chess_engine_2.data.dataset data/processed/lichess_2013-02_1000.jsonl
python -m chess_engine_2.train data/processed/lichess_2013-02_1000.jsonl --epochs 3 --batch-size 256 --channels 32 --checkpoint models/policy_value_phase10_1000.pt
```

Train with validation metrics and checkpoint metadata:

```powershell
python -m chess_engine_2.train data/processed/lichess_2013-02_1000.jsonl --epochs 3 --batch-size 256 --channels 32 --checkpoint models/policy_value_phase10_1000_validated.pt --validation-split 0.1
```

Check CUDA before larger training:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Train a 10,000-game checkpoint with a reusable tensor cache:

```powershell
python -m chess_engine_2.data.pgn data/raw/lichess_db_standard_rated_2013-02.pgn.zst --max-games 10000 --output data/processed/lichess_2013-02_10000.jsonl
python -m chess_engine_2.data.dataset data/processed/lichess_2013-02_10000.jsonl
python -m chess_engine_2.train data/processed/lichess_2013-02_10000.jsonl --epochs 3 --batch-size 256 --channels 32 --checkpoint models/policy_value_phase12_10000.pt --validation-split 0.1 --num-workers 2 --tensor-cache data/processed/lichess_2013-02_10000_tensors.pt
```

Stream modern high-Elo games without downloading a full monthly archive:

```powershell
python -m chess_engine_2.data.remote_pgn --month 2023-01 --min-elo 2000 --max-output-games 1000 --output data/processed/lichess_2023-01_2000plus_1000.jsonl
python -m chess_engine_2.data.remote_pgn --month 2023-01 --min-elo 2000 --skip-output-games 1000 --max-output-games 10000 --output data/processed/lichess_2023-01_2000plus_train_10000.jsonl
```

Evaluate a checkpoint on a held-out dataset:

```powershell
python -m chess_engine_2.evaluate_checkpoint models/policy_value_phase12_10000.pt data/processed/lichess_2023-01_2000plus_1000.jsonl --channels 32 --batch-size 256
```

Add dense material and classical evaluation targets:

```powershell
python -m chess_engine_2.annotate_values data/processed/lichess_2023-01_2000plus_train_10000.jsonl data/processed/lichess_2023-01_2000plus_train_10000_material.jsonl --material-only
python -m chess_engine_2.annotate_values data/processed/lichess_2023-01_2000plus_train_10000.jsonl data/processed/dense_values_classical.jsonl --depth 1 --qdepth 0 --no-mobility --max-samples 20000
```

Train against a selected value target:

```powershell
python -m chess_engine_2.train data/processed/lichess_2023-01_2000plus_train_10000_material.jsonl --epochs 3 --batch-size 256 --channels 32 --checkpoint models/policy_value_material.pt --value-target material_value
python -m chess_engine_2.train data/processed/dense_values_classical.jsonl --epochs 5 --batch-size 256 --channels 32 --initial-checkpoint models/policy_value_2023_2000plus_10000.pt --checkpoint models/policy_value_classical.pt --value-target classical_value --value-head-only
```

Value-target experiments can also use discounted game outcomes and controlled blends:

```powershell
python -m chess_engine_2.annotate_values data/processed/dense_values_classical.jsonl data/processed/value_targets.jsonl --material-only --backfill-progress
python -m chess_engine_2.train data/processed/value_targets.jsonl --initial-checkpoint models/policy_value_2023_2000plus_10000.pt --checkpoint models/policy_value_discounted_classical.pt --value-target discounted_classical_blend --result-weight 0.7 --value-head-only
```

`discounted_value` scales the final game result by `sqrt(ply / game_plies)`, so early positions receive softer labels. Available blends are `result_material_blend`, `result_classical_blend`, and `discounted_classical_blend`. `blend` remains an alias for result plus classical evaluation.

On a controlled 20,000-position value-head experiment, discounted plus classical supervision had the best separate 10,000-position value loss:

| Value target | Holdout value loss | MCTS-64 vs depth 1 |
| --- | ---: | ---: |
| Discounted outcome | 0.5739 | 0.0% |
| 70% outcome + 30% material | 0.5688 | 25.0% |
| 70% outcome + 30% classical | 0.5815 | 25.0% |
| 70% discounted + 30% classical | 0.3059 | 25.0% |
| Original high-Elo model | - | 25.0% |

The four-game MCTS samples are smoke tests, not statistically strong matches. The useful conclusion is that cleaner value targets substantially reduced prediction error but did not yet produce measurable MCTS strength.

### PUCT diagnostics

The diagnostic runner sweeps exploration strength and records the top root moves with:

- `P`: neural policy prior
- `N`: visit count
- `Q`: mean value from the root player's perspective
- `U`: PUCT exploration bonus
- `score`: `Q + U`

It also records the mean, standard deviation, minimum, and maximum neural leaf values:

```powershell
python -m chess_engine_2.puct_diagnostics models/policy_value_discounted_classical_valuehead_20000.pt --cpuct 0.25 0.5 1.0 2.0 4.0 --simulations 64 256 --games 2
```

The first controlled sweep produced:

| CPUCT | MCTS-64 | MCTS-256 |
| ---: | ---: | ---: |
| 0.25 | 0% | 0% |
| 0.50 | 25% | 25% |
| 1.00 | 25% | 25% |
| 2.00 | 25% | 0% |
| 4.00 | 25% | 0% |

These two-game cells are diagnostic smoke tests rather than Elo estimates. Root statistics show that the value signal is active: at low CPUCT it can overturn a much larger policy prior. Higher CPUCT increasingly follows policy, while increasing from 64 to 256 simulations did not improve match results. The current evidence therefore points to decision quality and policy-value interaction, not a value signal that is simply ignored.

### Neural value move ranking

Value loss measures target fitting, but MCTS needs the value head to rank candidate moves correctly. The ranking study pushes every legal move, evaluates each child with the neural value head, and compares that ordering with a deeper alpha-beta search:

```powershell
python -m chess_engine_2.value_ranking models/policy_value_discounted_classical_valuehead_20000.pt data/processed/value_targets_valid_10000.jsonl --positions 10 --child-depth 3 --qdepth 2
```

It reports Spearman rank correlation, pairwise ordering accuracy, exact top-1/top-3 accuracy, and centipawn regret. On 10 held-out middlegame positions containing 363 legal moves, the discounted/classical model produced:

| Metric | Result |
| --- | ---: |
| Mean Spearman correlation | -0.015 |
| Pairwise ordering accuracy | 49.6% |
| Top-1 accuracy | 0% |
| Top-3 accuracy | 10% |
| Mean neural top-1 regret | 413.7 cp |
| Mean neural top-3 regret | 232.0 cp |

The same first position remained poorly correlated at child depth 3 and a fast child-depth-4 check. This explains why lower value MSE did not improve MCTS: the value head can fit smooth position targets while providing almost no useful discrimination between sibling moves.

### Search-distilled sibling supervision

The distillation pipeline preserves complete sibling groups instead of flattening them into unrelated positions:

```powershell
python -m chess_engine_2.distill_rankings data/processed/value_targets_train_20000.jsonl data/processed/ranking_distillation_depth3_20.jsonl --positions 20 --child-depth 3 --qdepth 2
python -m chess_engine_2.train_rankings data/processed/ranking_distillation_depth3_20.jsonl --initial-checkpoint models/policy_value_discounted_classical_valuehead_20000.pt --checkpoint models/policy_value_search_ranking_depth3_20.pt --epochs 30 --ranking-weight 1.0
```

Each child receives a normalized alpha-beta score from the child side's perspective. Training combines value regression with a pairwise logistic loss over sibling moves whose search scores differ by at least 20 centipawns.

The first controlled experiment used 20 training roots, 682 searched children, and the unchanged 10-position ranking holdout:

| Model | Spearman | Pairwise | Top-1 | Top-3 | Top-1 regret | Top-3 regret | MCTS-64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original value model | -0.015 | 49.6% | 0% | 10% | 413.7 cp | 232.0 cp | 25.0% |
| Search regression | 0.066 | 52.7% | 10% | 30% | 485.3 cp | 243.8 cp | 37.5% |
| Regression + 0.25 ranking | 0.088 | 53.3% | 10% | 20% | 485.3 cp | 259.1 cp | 0.0% |
| Regression + 1.0 ranking | **0.118** | **54.6%** | 0% | 20% | 457.7 cp | **206.5 cp** | 12.5% |

The four-game MCTS cells remain smoke tests. The ranking study provides the stronger conclusion: direct search supervision improves sibling ordering, and pairwise loss adds signal beyond regression, but 20 roots generalize only modestly. Scaling the number and diversity of distilled root positions is now more justified than further loss-weight tuning.

Distillation generation supports multiple CPU workers, incremental writes, restart recovery, and reuse of a smaller completed rung:

```powershell
python -m chess_engine_2.distill_rankings data/processed/value_targets_train_20000.jsonl data/processed/ranking_distillation_depth3_500.jsonl --positions 500 --child-depth 3 --qdepth 2 --workers 4 --seed-from data/processed/ranking_distillation_depth3_100.jsonl --resume
python -m chess_engine_2.train_rankings data/processed/ranking_distillation_depth3_500.jsonl --max-groups 500 --validation-groups data/processed/ranking_holdout_depth3_10.jsonl --initial-checkpoint models/policy_value_discounted_classical_valuehead_20000.pt --checkpoint models/policy_value_search_ranking_depth3_500_scaled.pt --epochs 30 --ranking-weight 1.0
python -m chess_engine_2.evaluate_rankings models/policy_value_search_ranking_depth3_500_scaled.pt data/processed/ranking_holdout_depth3_10.jsonl
```

The controlled nested scaling study used one fixed ten-root holdout:

| Training roots | Children | Spearman | Pairwise | Top-1 | Top-3 | Top-1 regret | Top-3 regret |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | - | -0.015 | 49.6% | 0% | 10% | 413.7 cp | 232.0 cp |
| 20 | 682 | 0.155 | 55.6% | 0% | 30% | 400.3 cp | 197.3 cp |
| 50 | 1,653 | 0.236 | 58.8% | 0% | 30% | 470.3 cp | 113.3 cp |
| 100 | 3,439 | 0.260 | 59.5% | 0% | 30% | 255.6 cp | **83.5 cp** |
| 500 | 17,179 | **0.383** | **63.4%** | **10%** | **40%** | **214.7 cp** | 123.2 cp |

The 500-root run confirms that search-distilled sibling ranking scales. Matched ten-game MCTS-64 smoke tests still did not improve over the original model (`15%` versus `20%`), so better ranking has not yet been converted into reliable playing strength. The remaining problem is now narrower: integrate improved move discrimination with policy priors and value calibration rather than collecting another kind of target.

Material targets reached very low value loss, while shallow classical targets
also learned successfully. Neither produced a clear MCTS strength improvement in
the first small match study, so low target loss alone is not evidence that a
value target represents win probability well enough for PUCT.

Inspect the trained policy on a position:

```powershell
python -m chess_engine_2.predict --checkpoint models/policy_value_phase7.pt --top 5
```

Use the trained policy network only for alpha-beta move ordering:

```powershell
python -m chess_engine_2.benchmark --depths 2 --games-list 2 --opponent search --opponent-depth 2 --max-plies 40 --neural-checkpoint models/policy_value_phase7.pt
```

The default neural ordering mode is root-only. You can also test deeper modes:

```powershell
python -m chess_engine_2.benchmark --depths 2 --games-list 2 --opponent search --opponent-depth 2 --max-plies 40 --neural-checkpoint models/policy_value_phase7.pt --neural-ordering all
```

Run a neural-policy-only player with no search:

```powershell
python -m chess_engine_2.match --a neural --a-neural-checkpoint models/policy_value_phase7.pt --b random --games 4 --max-plies 80
```

Run the Phase 13 PUCT/MCTS prototype:

```powershell
python -m chess_engine_2.match --a mcts --a-neural-checkpoint models/policy_value_phase12_10000.pt --b random --games 4 --max-plies 80 --mcts-simulations 100 --mcts-cpuct 1.5
```

Run the Phase 14 MCTS scaling study against depth-1 alpha-beta:

```powershell
python -m chess_engine_2.mcts_benchmark models/policy_value_phase12_10000.pt --simulations 64 128 256 512 --games 2 --opponent-depth 1 --qdepth 2 --opening-plies 4 --max-plies 200 --adjudicate
```

The scaling study resets the opening seed for every simulation count and writes
both CSV and JSON results.

Measure the MCTS neural leaf cache against an uncached baseline:

```powershell
python -m chess_engine_2.mcts_benchmark models/policy_value_phase12_10000.pt --simulations 64 128 256 --games 2 --cache-size 0 --adjudicate --csv benchmark_mcts_cache_off.csv --json benchmark_mcts_cache_off.json
python -m chess_engine_2.mcts_benchmark models/policy_value_phase12_10000.pt --simulations 64 128 256 --games 2 --cache-size 100000 --adjudicate --csv benchmark_mcts_cache_on.csv --json benchmark_mcts_cache_on.json
```

The report includes network evaluations per move, cache hits per move, and cache
hit percentage. The compact-key cache experiment observed roughly 9-15% hits.
Throughput changes were mixed, ranging from about -2% to +11%, so caching is
available but is not yet a decisive optimization.

Experiment with the value head inside search:

```powershell
python -m chess_engine_2.match --a search --a-depth 1 --a-value-checkpoint models/policy_value_phase10_1000_validated.pt --evaluation-mode blend --neural-value-weight 0.2 --b search --b-depth 1 --games 2 --max-plies 20 --qdepth 0
```

Compare classical, neural-only, and blended evaluation:

```powershell
python -m chess_engine_2.benchmark --depths 1 --games-list 4 --opponent search --opponent-depth 1 --max-plies 24 --qdepth 0 --opponent-qdepth 0
python -m chess_engine_2.benchmark --depths 1 --games-list 4 --opponent search --opponent-depth 1 --max-plies 24 --qdepth 0 --opponent-qdepth 0 --value-checkpoint models/policy_value_phase10_1000_validated.pt --evaluation-mode neural
python -m chess_engine_2.benchmark --depths 1 --games-list 4 --opponent search --opponent-depth 1 --max-plies 24 --qdepth 0 --opponent-qdepth 0 --value-checkpoint models/policy_value_phase10_1000_validated.pt --evaluation-mode blend --neural-value-weight 0.2
```

## Match Testing

```powershell
python -m chess_engine_2.match --a search --a-depth 2 --b random --games 4 --max-plies 200 --pgn match.pgn
```

Use Match Runner 2.0 adjudication to reduce artificial move-limit draws:

```powershell
python -m chess_engine_2.match --a search --a-depth 2 --b search --b-depth 1 --games 4 --max-plies 200 --adjudicate --adjudicate-eval 500 --adjudicate-eval-plies 8 --adjudicate-material 900 --adjudicate-material-plies 8 --adjudicate-min-plies 40
```

```powershell
python -m chess_engine_2.benchmark --depths 3 --opponent search --opponent-depth 2 --movetime 500 --opponent-movetime 500 --games-list 5,10 --opening-plies 4 --stream --csv benchmark.csv
```

Benchmark with adjudication:

```powershell
python -m chess_engine_2.benchmark --depths 2 --games-list 10 --opponent search --opponent-depth 1 --max-plies 200 --adjudicate --adjudicate-eval 500 --adjudicate-eval-plies 8 --adjudicate-material 900 --adjudicate-material-plies 8 --adjudicate-min-plies 40
```

Benchmark neural ordering against classical ordering:

```powershell
python -m chess_engine_2.benchmark --depths 2 --games-list 10 20 30 40 50 60 70 80 90 100 --opponent search --opponent-depth 2 --max-plies 200 --qdepth 2 --opponent-qdepth 2 --opening-plies 4 --adjudicate --adjudicate-eval 500 --adjudicate-eval-plies 8 --adjudicate-material 900 --adjudicate-material-plies 8 --adjudicate-min-plies 40 --neural-checkpoint models/policy_value_phase12_10000.pt --neural-channels 32 --neural-ordering root --stream --csv benchmark_phase12_neural_order_vs_classical_100.csv
```

```powershell
python -m chess_engine_2.tune --qdepths 2,4,6 --depth 3 --opponent-depth 2 --games 5 --movetime 500
```
