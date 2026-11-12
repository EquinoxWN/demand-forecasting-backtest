"""Loading and validating M5 files, and the verified download."""

from __future__ import annotations

import hashlib
import io
import urllib.error
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from demand_forecasting_backtest.data import (
    SALES_FILE,
    DataError,
    download,
    load_calendar,
    load_m5,
)
from demand_forecasting_backtest.synthetic import write_synthetic_m5


def test_loads_only_the_selected_stores(m5_dir: Path) -> None:
    data = load_m5(m5_dir, ["TX_1"])
    assert set(data.ids["store_id"]) == {"TX_1"}
    assert data.sales.shape == (8, 140)
    assert len(data.calendar) == 140
    assert set(data.prices["store_id"]) == {"TX_1"}
    assert (data.sales >= 0).all()


def test_reads_the_same_data_from_a_zip_as_from_a_directory(m5_dir: Path, tmp_path: Path) -> None:
    archive = tmp_path / "m5.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for f in m5_dir.iterdir():
            zf.write(f, f.name)
    a, b = load_m5(m5_dir, ["CA_1", "TX_1"]), load_m5(archive, ["CA_1", "TX_1"])
    assert (a.sales == b.sales).all()
    pd.testing.assert_frame_equal(a.ids, b.ids)
    pd.testing.assert_frame_equal(a.calendar, b.calendar)


def test_a_calendar_without_the_d_column_is_read_in_row_order(tmp_path: Path) -> None:
    src = write_synthetic_m5(tmp_path / "x", stores=("CA_1",), n_items=2, n_days=28, d_column=False)
    cal = load_calendar(src)
    assert cal["date"].iloc[0] == pd.Timestamp("2011-01-29")
    assert cal["wday"].iloc[0] == 1
    assert (cal["event_name_1"].iloc[:45] == "").all()


def _rewrite(path: Path, edit: Any) -> None:
    """Load a CSV, apply edit, write it back."""
    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    changed = edit(frame)
    (frame if changed is None else changed).to_csv(path, index=False)


def _set(column: str, value: str, row: int = 0) -> Any:
    """Edit that sets one cell."""

    def edit(frame: pd.DataFrame) -> None:
        frame.loc[row, column] = value

    return edit


@pytest.mark.parametrize(
    ("file", "edit", "message"),
    [
        (SALES_FILE, _set("d_3", "-1"), "whole numbers"),
        (SALES_FILE, _set("d_3", "1.5"), "whole numbers"),
        (SALES_FILE, _set("d_3", "abc"), "non-numeric"),
        (SALES_FILE, lambda f: f.drop(columns=["state_id"]), "lacks columns"),
        (SALES_FILE, lambda f: f.rename(columns={"d_2": "d_9999"}), "in order"),
        (SALES_FILE, lambda f: f.loc.__setitem__((1, "item_id"), f.loc[0, "item_id"]), "repeats"),
        ("calendar.csv", _set("date", "2011-02-30", 1), "calendar.csv"),
        ("calendar.csv", _set("date", "2011-03-01", 1), "consecutive"),
        ("calendar.csv", _set("d", "d_7", 1), "in order"),
        ("calendar.csv", _set("wday", "8"), "out of range"),
        ("calendar.csv", _set("snap_CA", "2"), "snap_CA"),
        ("calendar.csv", lambda f: f.drop(columns=["snap_TX"]), "snap_TX"),
        ("calendar.csv", lambda f: f.drop(columns=["wm_yr_wk"]), "lacks columns"),
        ("sell_prices.csv", _set("sell_price", "0"), "positive"),
        ("sell_prices.csv", _set("sell_price", "x"), "positive"),
        (
            "sell_prices.csv",
            lambda f: f.loc.__setitem__((1, "wm_yr_wk"), f.loc[0, "wm_yr_wk"]),
            "repeats",
        ),
    ],
)
def test_malformed_files_are_refused(m5_dir: Path, file: str, edit: Any, message: str) -> None:
    _rewrite(m5_dir / file, edit)
    with pytest.raises(DataError, match=message):
        load_m5(m5_dir, ["CA_1", "TX_1"])


def test_sales_longer_than_the_calendar_are_refused(m5_dir: Path) -> None:
    _rewrite(m5_dir / "calendar.csv", lambda f: f.drop(index=f.index[-1]))
    with pytest.raises(DataError, match="calendar only 139"):
        load_m5(m5_dir, ["CA_1"])


@pytest.mark.parametrize(("stores", "message"), [([], "no stores"), (["CA_1", "WI_9"], "WI_9")])
def test_unknown_or_no_stores_are_refused(m5_dir: Path, stores: list[str], message: str) -> None:
    with pytest.raises(DataError, match=message):
        load_m5(m5_dir, stores)


def test_missing_sources_and_files_are_data_errors(m5_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(DataError, match="does not exist"):
        load_m5(tmp_path / "nope", ["CA_1"])
    (m5_dir / "sell_prices.csv").unlink()
    with pytest.raises(DataError, match=r"missing sell_prices.csv"):
        load_m5(m5_dir, ["CA_1"])
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(DataError, match="cannot read archive"):
        load_m5(bad, ["CA_1"])
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "calendar.csv").write_text("")
    with pytest.raises(DataError, match=r"calendar\.csv"):
        load_m5(empty, ["CA_1"])


class FakeResponse(io.BytesIO):
    """A urlopen response serving fixed bytes, optionally failing midway."""

    def __init__(self, payload: bytes, fail_after: int | None = None) -> None:
        super().__init__(payload)
        self.fail_after = fail_after

    def read(self, size: int | None = -1) -> bytes:
        """Serve bytes, raising once fail_after bytes have gone out."""
        if self.fail_after is not None and self.tell() >= self.fail_after:
            raise ConnectionError("connection reset")
        return super().read(min(size or 1, 4) if self.fail_after is not None else size)


class FakeOpener:
    """Replays a list of responses or exceptions and records each call."""

    def __init__(self, *outcomes: FakeResponse | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, float]] = []

    def __call__(self, url: str, timeout: float) -> FakeResponse:
        """Return or raise the next outcome."""
        self.calls.append((url, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


PAYLOAD = b"m5 archive bytes"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
URL = "https://example.org/m5.zip"


def test_a_download_is_kept_only_when_the_checksum_matches(tmp_path: Path) -> None:
    dest = tmp_path / "m5.zip"
    opener = FakeOpener(FakeResponse(PAYLOAD))
    assert download(URL, dest, DIGEST, opener=opener) == dest
    assert dest.read_bytes() == PAYLOAD
    assert opener.calls == [(URL, 60.0)]
    assert list(tmp_path.iterdir()) == [dest]
    with pytest.raises(DataError, match="checksum mismatch"):
        download(URL, tmp_path / "other.zip", "0" * 64, opener=FakeOpener(FakeResponse(PAYLOAD)))
    assert list(tmp_path.iterdir()) == [dest]


def test_a_verified_file_is_not_downloaded_again(tmp_path: Path) -> None:
    dest = tmp_path / "m5.zip"
    dest.write_bytes(PAYLOAD)
    opener = FakeOpener()
    download(URL, dest, DIGEST, opener=opener)
    assert opener.calls == []


def test_only_https_is_allowed(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="non-HTTPS"):
        download("http://example.org/m5.zip", tmp_path / "a", DIGEST, opener=FakeOpener())


def test_the_size_cap_keeps_the_limit_and_refuses_more(tmp_path: Path) -> None:
    n = len(PAYLOAD)
    ok = download(
        URL, tmp_path / "a", DIGEST, max_bytes=n, opener=FakeOpener(FakeResponse(PAYLOAD))
    )
    assert ok.read_bytes() == PAYLOAD
    with pytest.raises(DataError, match="larger than"):
        download(
            URL, tmp_path / "b", DIGEST, max_bytes=n - 1, opener=FakeOpener(FakeResponse(PAYLOAD))
        )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a"]


def test_transient_failures_are_retried_with_backoff(tmp_path: Path) -> None:
    sleeps: list[float] = []
    opener = FakeOpener(
        urllib.error.URLError("dns"),
        FakeResponse(PAYLOAD, fail_after=8),
        FakeResponse(PAYLOAD),
    )
    dest = download(URL, tmp_path / "m5.zip", DIGEST, opener=opener, sleep=sleeps.append)
    assert dest.read_bytes() == PAYLOAD
    assert sleeps == [1.0, 2.0]
    assert len(opener.calls) == 3


def test_retries_are_bounded_and_leave_no_partial_file(tmp_path: Path) -> None:
    sleeps: list[float] = []
    opener = FakeOpener(TimeoutError("t1"), TimeoutError("t2"), FakeResponse(PAYLOAD, fail_after=4))
    with pytest.raises(DataError, match="after 3 attempts"):
        download(URL, tmp_path / "m5.zip", DIGEST, opener=opener, sleep=sleeps.append)
    assert sleeps == [1.0, 2.0]
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="attempts"):
        download(URL, tmp_path / "m5.zip", DIGEST, attempts=0, opener=FakeOpener())
