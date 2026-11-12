"""Verified M5 download and validated loading of the M5 CSV files for a set of stores."""

from __future__ import annotations

import hashlib
import io
import os
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

# Public mirror of the M5 competition files (Kaggle needs a sign-in; this copy does not).
M5_URL = "https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip"
# No checksum is published upstream: this is the SHA-256 of the archive downloaded on 2026-11-11.
M5_SHA256 = "cc704ba15d6802f8262e6ec7d4c6041e4ad6366a94365e8c84f721e450eed774"
M5_MAX_BYTES = 60_000_000
SALES_FILE = "sales_train_evaluation.csv"
CALENDAR_COLUMNS = (
    "date",
    "wm_yr_wk",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
)
ID_COLUMNS = ("item_id", "dept_id", "cat_id", "store_id", "state_id")
PRICE_COLUMNS = ("store_id", "item_id", "wm_yr_wk", "sell_price")
MAX_SALE = 32_767


class DataError(ValueError):
    """Raised when a download or an input file is missing, tampered with or malformed."""


Opener = Callable[..., Any]


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(
    url: str,
    dest: Path,
    expected_sha256: str,
    *,
    max_bytes: int = M5_MAX_BYTES,
    timeout: float = 60.0,
    attempts: int = 3,
    opener: Opener = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download over HTTPS into dest, keeping it only if its SHA-256 matches."""
    if not url.startswith("https://"):
        raise DataError(f"refusing a non-HTTPS URL: {url}")
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if dest.exists() and sha256(dest) == expected_sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Per-process partial file: two concurrent downloads never write into the same file.
    part = dest.with_name(f".{dest.name}.{os.getpid()}.part")
    for attempt in range(1, attempts + 1):
        try:
            _fetch(url, part, max_bytes, timeout, opener)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            part.unlink(missing_ok=True)
            if attempt == attempts:
                raise DataError(f"download failed after {attempts} attempts: {exc}") from exc
            sleep(2.0 ** (attempt - 1))
        except BaseException:
            part.unlink(missing_ok=True)
            raise
    actual = sha256(part)
    if actual != expected_sha256:
        part.unlink(missing_ok=True)
        raise DataError(f"checksum mismatch: expected {expected_sha256}, got {actual}")
    os.replace(part, dest)
    return dest


def _fetch(url: str, part: Path, max_bytes: int, timeout: float, opener: Opener) -> None:
    """Stream one response into part, enforcing the size cap."""
    total = 0
    with opener(url, timeout=timeout) as resp, part.open("wb") as out:
        while chunk := resp.read(1 << 20):
            total += len(chunk)
            if total > max_bytes:
                raise DataError(f"download larger than {max_bytes} bytes")
            out.write(chunk)


@contextmanager
def _open_member(source: Path, name: str) -> Iterator[IO[str]]:
    """Open a CSV from a directory or a zip archive as text."""
    if source.is_dir():
        path = source / name
        if not path.is_file():
            raise DataError(f"missing {name} in {source}")
        with path.open(encoding="utf-8", newline="") as fh:
            yield fh
        return
    try:
        archive = zipfile.ZipFile(source)
    except (OSError, zipfile.BadZipFile) as exc:
        raise DataError(f"cannot read archive {source}: {exc}") from exc
    with archive:
        if name not in archive.namelist():
            raise DataError(f"missing {name} in {source}")
        with archive.open(name) as raw, io.TextIOWrapper(raw, encoding="utf-8", newline="") as fh:
            yield fh


def _require(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    """Fail when a frame lacks any of the columns."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise DataError(f"{name} lacks columns: {', '.join(missing)}")


@dataclass(frozen=True)
class M5Data:
    """Sales of the selected stores as a series-by-day matrix, with calendar and prices."""

    ids: pd.DataFrame
    sales: npt.NDArray[np.int16]
    calendar: pd.DataFrame
    prices: pd.DataFrame


def load_calendar(source: Path) -> pd.DataFrame:
    """Read and validate calendar.csv; one row per day, d_1 first."""
    with _open_member(source, "calendar.csv") as fh:
        try:
            cal = pd.read_csv(fh, keep_default_na=False, na_values=["NA", ""], dtype=str)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
            raise DataError(f"calendar.csv: {exc}") from exc
    _require(cal, CALENDAR_COLUMNS, "calendar.csv")
    if cal.empty:
        raise DataError("calendar.csv has no rows")
    try:
        dates = pd.to_datetime(cal["date"], format="%Y-%m-%d")
        out = pd.DataFrame(
            {
                "date": dates,
                "wm_yr_wk": cal["wm_yr_wk"].astype(int),
                "wday": cal["wday"].astype(int),
                "month": cal["month"].astype(int),
                "year": cal["year"].astype(int),
            }
        )
    except (ValueError, TypeError) as exc:
        raise DataError(f"calendar.csv: {exc}") from exc
    if not (dates.diff().iloc[1:] == pd.Timedelta(days=1)).all():
        raise DataError("calendar.csv dates are not consecutive days")
    if "d" in cal.columns and list(cal["d"]) != [f"d_{i}" for i in range(1, len(cal) + 1)]:
        raise DataError("calendar.csv column d is not d_1, d_2, ... in order")
    if not out["wday"].between(1, 7).all() or not out["month"].between(1, 12).all():
        raise DataError("calendar.csv has wday or month out of range")
    if (out["wm_yr_wk"].diff().iloc[1:] < 0).any():
        raise DataError("calendar.csv wm_yr_wk goes backwards")
    for col in ("event_name_1", "event_type_1", "event_name_2", "event_type_2"):
        out[col] = cal[col].fillna("")
    for col in [c for c in cal.columns if c.startswith("snap_")]:
        snap = pd.to_numeric(cal[col], errors="coerce")
        if not snap.isin([0, 1]).all():
            raise DataError(f"calendar.csv {col} must be 0 or 1")
        out[col] = snap.astype(np.int8)
    return out


def load_sales(
    source: Path, stores: Sequence[str], n_calendar_days: int, *, file: str = SALES_FILE
) -> tuple[pd.DataFrame, npt.NDArray[np.int16]]:
    """Read the wide sales file in chunks, keeping only the given stores."""
    if not stores:
        raise DataError("no stores selected")
    wanted = set(stores)
    kept: list[pd.DataFrame] = []
    with _open_member(source, file) as fh:
        try:
            reader = pd.read_csv(fh, chunksize=2000, dtype=dict.fromkeys(ID_COLUMNS, str))
            for chunk in reader:
                _require(chunk, ID_COLUMNS, file)
                kept.append(chunk[chunk["store_id"].isin(wanted)])
        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
            raise DataError(f"{file}: {exc}") from exc
    if not kept:
        raise DataError(f"{file} has no rows")
    wide = pd.concat(kept, ignore_index=True)
    found = set(wide["store_id"])
    if missing := sorted(wanted - found):
        raise DataError(f"stores not found in {file}: {', '.join(missing)}")
    day_cols = [c for c in wide.columns if c not in ID_COLUMNS]
    if not day_cols or day_cols != [f"d_{i}" for i in range(1, len(day_cols) + 1)]:
        raise DataError(f"{file} day columns are not d_1, d_2, ... in order")
    if len(day_cols) > n_calendar_days:
        raise DataError(f"{file} has {len(day_cols)} days but the calendar only {n_calendar_days}")
    if wide.duplicated(["item_id", "store_id"]).any():
        raise DataError(f"{file} repeats an item_id and store_id pair")
    values = wide[day_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(values).any():
        raise DataError(f"{file} has missing or non-numeric sales")
    if (values < 0).any() or (values > MAX_SALE).any() or (values != np.floor(values)).any():
        raise DataError(f"{file} sales must be whole numbers from 0 to {MAX_SALE}")
    ids = wide[list(ID_COLUMNS)].reset_index(drop=True)
    return ids, values.astype(np.int16)


def load_prices(source: Path, stores: Sequence[str]) -> pd.DataFrame:
    """Read sell_prices.csv in chunks for the given stores and validate it."""
    wanted = set(stores)
    kept: list[pd.DataFrame] = []
    with _open_member(source, "sell_prices.csv") as fh:
        try:
            reader = pd.read_csv(fh, chunksize=200_000, dtype={"store_id": str, "item_id": str})
            for chunk in reader:
                _require(chunk, PRICE_COLUMNS, "sell_prices.csv")
                kept.append(chunk.loc[chunk["store_id"].isin(wanted), list(PRICE_COLUMNS)])
        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
            raise DataError(f"sell_prices.csv: {exc}") from exc
    if not kept:
        raise DataError("sell_prices.csv has no rows")
    prices = pd.concat(kept, ignore_index=True)
    price = pd.to_numeric(prices["sell_price"], errors="coerce")
    week = pd.to_numeric(prices["wm_yr_wk"], errors="coerce")
    if price.isna().any() or week.isna().any() or (price <= 0).any():
        raise DataError("sell_prices.csv needs a positive sell_price and a numeric wm_yr_wk")
    prices = prices.assign(sell_price=price.astype(float), wm_yr_wk=week.astype(int))
    if prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).any():
        raise DataError("sell_prices.csv repeats a store, item and week")
    return prices


def load_m5(source: Path, stores: Sequence[str], *, file: str = SALES_FILE) -> M5Data:
    """Load calendar, sales and prices for the stores from a directory or the M5 zip."""
    if not source.exists():
        raise DataError(f"{source} does not exist")
    calendar = load_calendar(source)
    ids, sales = load_sales(source, stores, len(calendar), file=file)
    calendar = calendar.iloc[: sales.shape[1]].reset_index(drop=True)
    for state in sorted(set(ids["state_id"])):
        if f"snap_{state}" not in calendar.columns:
            raise DataError(f"calendar.csv lacks snap_{state}")
    prices = load_prices(source, stores)
    return M5Data(ids=ids, sales=sales, calendar=calendar, prices=prices)
