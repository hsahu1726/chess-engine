from io import BytesIO

import pytest

from chess_engine_2.data import download
from chess_engine_2.data.download import download_lichess_month, lichess_month_filename, lichess_month_url, normalize_month


def test_normalize_month_accepts_valid_month() -> None:
    assert normalize_month("2013-01") == "2013-01"


def test_normalize_month_rejects_invalid_month() -> None:
    with pytest.raises(ValueError):
        normalize_month("2013-13")


def test_lichess_month_url_targets_standard_database() -> None:
    assert lichess_month_filename("2013-01") == "lichess_db_standard_rated_2013-01.pgn.zst"
    assert (
        lichess_month_url("2013-01")
        == "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"
    )


def test_download_uses_partial_file_then_final_path(tmp_path, monkeypatch) -> None:
    def fake_urlopen(url: str, timeout: float):
        assert url == lichess_month_url("2013-01")
        assert timeout == 30.0
        return BytesIO(b"pgn data")

    monkeypatch.setattr(download.urllib.request, "urlopen", fake_urlopen)

    path = download_lichess_month("2013-01", tmp_path)

    assert path.name == "lichess_db_standard_rated_2013-01.pgn.zst"
    assert path.read_bytes() == b"pgn data"
    assert not path.with_suffix(path.suffix + ".part").exists()
