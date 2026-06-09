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
6. AlphaZero-style 4672 move encoding. 
7. Quiescence search. 
8. Transposition table, iterative deepening, and time controls. 
9. Search pruning and move ordering heuristics. 
10. Richer handcrafted evaluation. 
11. Engine-vs-engine match runner and PGN export. 
12. Lichess PGN downloader and parser. 
13. PyTorch policy/value network.
14. Neural-guided move selection.
15. PUCT MCTS.
16. Self-play training loop.

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
heuristics, and principal-variation output. If no depth or time control is
supplied, it searches to depth 4.

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

Inspect the trained policy on a position:

```powershell
python -m chess_engine_2.predict --checkpoint models/policy_value_phase7.pt --top 5
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
