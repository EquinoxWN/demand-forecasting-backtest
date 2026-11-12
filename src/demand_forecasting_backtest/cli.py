"""Command line: download M5, export features, and score the baseline against challengers."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .baseline import BASELINE, evaluate
from .data import M5_SHA256, M5_URL, DataError, M5Data, download, load_m5
from .features import LEVELS, build_panels, iter_frames
from .files import atomic_text, write_text
from .synthetic import write_synthetic_m5

DEFAULT_ARCHIVE = Path("data/m5.zip")
MAX_HORIZON = 364


def _stores(text: str) -> list[str]:
    """Parse a comma-separated store list such as CA_1,TX_2."""
    stores = [s.strip() for s in text.split(",") if s.strip()]
    if not stores or any(not s.replace("_", "").isalnum() or len(s) > 16 for s in stores):
        raise argparse.ArgumentTypeError(f"invalid store list: {text!r}")
    return stores


def _horizon(text: str) -> int:
    """Parse a forecast horizon in days."""
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from exc
    if not 1 <= value <= MAX_HORIZON:
        raise argparse.ArgumentTypeError(f"horizon must be 1 to {MAX_HORIZON} days")
    return value


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(prog="dfb", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="download the M5 archive (SHA-256 checked)")
    dl.add_argument("--dest", type=Path, default=DEFAULT_ARCHIVE)
    for name, text in (
        ("features", "export the feature table of one level"),
        ("backtest", "score the baseline and challengers on a holdout"),
    ):
        sp = sub.add_parser(name, help=text)
        src = sp.add_mutually_exclusive_group()
        src.add_argument(
            "--data",
            type=Path,
            default=DEFAULT_ARCHIVE,
            help="M5 zip or a directory with the CSV files",
        )
        src.add_argument("--synthetic", action="store_true", help="use generated M5-format data")
        sp.add_argument("--stores", type=_stores, default=["CA_1"])
        if name == "features":
            sp.add_argument("--level", choices=LEVELS, required=True)
            sp.add_argument("--out", type=Path, required=True, help=".csv or .csv.gz")
        else:
            sp.add_argument("--horizon", type=_horizon, default=28)
            sp.add_argument("--json", type=Path, help="also write the scores as JSON")
    return p


def _load(args: argparse.Namespace) -> M5Data:
    """Load real or synthetic data for the chosen stores."""
    if args.synthetic:
        with tempfile.TemporaryDirectory() as tmp:
            return load_m5(write_synthetic_m5(Path(tmp), stores=args.stores), args.stores)
    return load_m5(args.data, args.stores)


def _backtest(args: argparse.Namespace) -> None:
    """Print the holdout table and optionally write it as JSON."""
    started = time.perf_counter()
    data = _load(args)
    panels = build_panels(data)
    scores = evaluate(panels, args.horizon)
    elapsed = time.perf_counter() - started
    n, days = data.sales.shape
    print(
        f"Data: {len(set(data.ids['store_id']))} store(s), {n:,} item series, {days:,} days "
        f"({data.calendar['date'].iloc[0]:%Y-%m-%d} to {data.calendar['date'].iloc[-1]:%Y-%m-%d}); "
        f"holdout = last {args.horizon} days"
    )
    print("| Level | Model | WAPE | MASE (series) | Beats seasonal naive |")
    print("|---|---|---|---|---|")
    for s in scores:
        verdict = "baseline" if s.model == BASELINE else ("yes" if s.counts else "no")
        print(
            f"| {s.level} | {s.model} | {s.wape:.4f} | {s.mase:.4f} ({s.mase_series:,}) | {verdict} |"
        )
    print(f"Elapsed: {elapsed:.1f} s")
    if args.json:
        payload = {
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "stores": args.stores,
            "horizon": args.horizon,
            "synthetic": bool(args.synthetic),
            "scores": [asdict(s) for s in scores],
        }
        write_text(args.json, json.dumps(payload, indent=2) + "\n")


def _features(args: argparse.Namespace) -> None:
    """Write the long feature table of one level."""
    if not args.out.name.endswith((".csv", ".csv.gz")):
        raise DataError("--out must end with .csv or .csv.gz")
    panel = build_panels(_load(args))[args.level]
    rows = 0
    with atomic_text(args.out) as fh:
        for i, frame in enumerate(iter_frames(panel)):
            frame.to_csv(
                fh, index=False, header=i == 0, float_format="%.4f", date_format="%Y-%m-%d"
            )
            rows += len(frame)
    print(f"Wrote {rows:,} rows ({panel.n_series:,} series x {panel.n_days:,} days) to {args.out}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run a subcommand; exit 2 on bad input or data."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "download":
            path = download(M5_URL, args.dest, M5_SHA256)
            print(f"M5 archive ready at {path} (SHA-256 verified)")
        elif args.command == "backtest":
            _backtest(args)
        else:
            _features(args)
    except (DataError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
