from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path


LICHESS_STANDARD_BASE_URL = "https://database.lichess.org/standard"
DEFAULT_DATA_DIR = Path("data/raw")
DOWNLOAD_CHUNK_SIZE = 64 * 1024


def lichess_month_filename(month: str) -> str:
    normalized = normalize_month(month)
    return f"lichess_db_standard_rated_{normalized}.pgn.zst"


def lichess_month_url(month: str) -> str:
    return f"{LICHESS_STANDARD_BASE_URL}/{lichess_month_filename(month)}"


def normalize_month(month: str) -> str:
    if len(month) != 7 or month[4] != "-":
        raise ValueError("month must use YYYY-MM format, for example 2013-01")

    year, month_number = month.split("-")
    if not year.isdigit() or not month_number.isdigit():
        raise ValueError("month must use YYYY-MM format, for example 2013-01")
    if int(month_number) < 1 or int(month_number) > 12:
        raise ValueError("month must be between 01 and 12")

    return f"{int(year):04d}-{int(month_number):02d}"


def download_lichess_month(
    month: str,
    output_dir: Path = DEFAULT_DATA_DIR,
    overwrite: bool = False,
    timeout_seconds: float = 30.0,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / lichess_month_filename(month)
    partial_path = output_path.with_suffix(output_path.suffix + ".part")

    if output_path.exists() and not overwrite:
        return output_path

    if partial_path.exists():
        partial_path.unlink()

    try:
        with urllib.request.urlopen(lichess_month_url(month), timeout=timeout_seconds) as response:
            with partial_path.open("wb") as output:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
        os.replace(partial_path, output_path)
    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one Lichess monthly PGN archive.")
    parser.add_argument("--month", default="2013-01", help="Month in YYYY-MM format.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0, help="Network timeout in seconds.")
    args = parser.parse_args()

    path = download_lichess_month(args.month, args.output_dir, args.overwrite, args.timeout)
    print(path)


if __name__ == "__main__":
    main()
