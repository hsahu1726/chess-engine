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
17. Value network inside search.
18. PUCT MCTS.
19. Self-play training loop.

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

## Match Testing

```powershell
python -m chess_engine_2.match --a search --a-depth 2 --b random --games 4 --max-plies 200 --pgn match.pgn
```

```powershell
python -m chess_engine_2.benchmark --depths 3 --opponent search --opponent-depth 2 --movetime 500 --opponent-movetime 500 --games-list 5,10 --opening-plies 4 --stream --csv benchmark.csv
```

```powershell
python -m chess_engine_2.tune --qdepths 2,4,6 --depth 3 --opponent-depth 2 --games 5 --movetime 500
```
