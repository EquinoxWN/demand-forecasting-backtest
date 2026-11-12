"""Command line runs on synthetic data, input checks and atomic outputs."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from demand_forecasting_backtest import cli
from demand_forecasting_backtest.data import M5_SHA256, M5_URL
from demand_forecasting_backtest.files import atomic_text


def test_backtest_prints_the_table_and_writes_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "scores.json"
    assert cli.main(["backtest", "--synthetic", "--stores", "CA_1,TX_1", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "2 store(s), 24 item series, 364 days" in text
    assert "| total | seasonal_naive |" in text
    assert "| baseline |" in text
    payload = json.loads(out.read_text())
    assert payload["created_utc"].endswith("+00:00")
    assert len(payload["scores"]) == 12
    assert payload["scores"][0]["model"] == "seasonal_naive"


def test_feature_export_is_complete_and_repeatable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "store.csv.gz"
    args = [
        "features",
        "--synthetic",
        "--stores",
        "CA_1,TX_1",
        "--level",
        "store",
        "--out",
        str(out),
    ]
    assert cli.main(args) == 0
    first = out.read_bytes()
    lines = gzip.decompress(first).decode().splitlines()
    assert len(lines) == 1 + 2 * 364
    assert lines[0].startswith("store_id,state_id,date,wday")
    assert "Wrote 728 rows" in capsys.readouterr().out
    assert cli.main(args) == 0
    assert out.read_bytes() == first
    assert sorted(p.name for p in tmp_path.iterdir()) == ["store.csv.gz"]


@pytest.mark.parametrize(
    "argv",
    [
        ["backtest", "--synthetic", "--horizon", "0"],
        ["backtest", "--synthetic", "--horizon", "x"],
        ["backtest", "--synthetic", "--stores", ",,"],
        ["backtest", "--synthetic", "--stores", "CA_1;rm"],
        ["features", "--synthetic", "--level", "state", "--out", "a.csv"],
        ["backtest", "--synthetic", "--data", "x.zip"],
    ],
)
def test_bad_arguments_exit_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


def test_bad_data_and_outputs_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["backtest", "--data", str(tmp_path / "missing.zip")]) == 2
    assert "does not exist" in capsys.readouterr().err
    out = tmp_path / "f.parquet"
    assert cli.main(["features", "--synthetic", "--level", "total", "--out", str(out)]) == 2
    assert not out.exists()
    assert cli.main(["backtest", "--synthetic", "--horizon", "364"]) == 2
    assert "horizon" in capsys.readouterr().err


def test_download_command_uses_the_pinned_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, Path, str]] = []

    def fake(url: str, dest: Path, digest: str) -> Path:
        calls.append((url, dest, digest))
        return dest

    monkeypatch.setattr(cli, "download", fake)
    dest = tmp_path / "m5.zip"
    assert cli.main(["download", "--dest", str(dest)]) == 0
    assert calls == [(M5_URL, dest, M5_SHA256)]
    assert "SHA-256 verified" in capsys.readouterr().out


def test_a_failed_write_keeps_the_old_file(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("old")

    def write_then_fail() -> None:
        with atomic_text(path) as fh:
            fh.write("new partial")
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        write_then_fail()
    assert path.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["x.csv"]
