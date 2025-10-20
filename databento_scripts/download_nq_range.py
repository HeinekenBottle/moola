"""Download a continuous NQ OHLCV-1m slice from Databento.

This helper wraps `Historical.timeseries.get_range` so you can pull a large
continuous-contract span (e.g., the 2020-09-01 → 2024-09-01 backfill you just
priced) straight into a DBN+zstd file.  The script leaves you with a single
`.dbn.zst` artifact that you can append to the existing
`nq_ohlc_1min.parquet`/processing pipeline once you’re ready.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import databento as db

DEFAULT_SYMBOL = "NQ.v.0"  # Continuous front-month contract by volume
DEFAULT_DATASET = "GLBX.MDP3"
DEFAULT_SCHEMA = "ohlcv-1m"
DEFAULT_START = "2020-09-01"
# Databento treats `end` as exclusive; using 2024-09-01 avoids overlap with
# the 2024-09-01+ data you already have in `nq_ohlc_1min.parquet`.
DEFAULT_END = "2024-09-01"
DEFAULT_OUTPUT = (
    Path.home() / "HopSketch" / "data" / "raw" / "nq_ohlcv_1min_2020-09_2024-08.dbn.zst"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download continuous NQ OHLCV-1m data from Databento."
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Inclusive start date.")
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        help="Exclusive end date (set to day after your intended last bar).",
    )
    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        help="Continuous contract symbol (defaults to NQ front-month).",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Databento dataset code (defaults to GLBX.MDP3).",
    )
    parser.add_argument(
        "--schema",
        default=DEFAULT_SCHEMA,
        help="Schema to request (defaults to ohlcv-1m).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Destination `.dbn.zst` file for the download.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        sys.exit("Set DATABENTO_API_KEY in your environment before running.")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        sys.exit(
            f"{output_path} already exists. "
            "Re-run with --overwrite if you really want to replace it."
        )

    client = db.Historical(api_key)
    print(
        f"Requesting {args.schema} for {args.symbol} "
        f"{args.start} → {args.end} into {output_path}..."
    )

    store = client.timeseries.get_range(
        dataset=args.dataset,
        symbols=args.symbol,
        schema=args.schema,
        stype_in="continuous",
        stype_out="instrument_id",
        start=args.start,
        end=args.end,
        path=str(output_path),
    )

    # `store` is a `DBNStore`. Even when `path` is provided (so the bytes are
    # already flushed to disk) we can still peek at high-level metadata.
    summary = {
        "schema": str(store.schema),
        "symbols": store.symbols,
        "start": store.start,
        "end": store.end,
        "nbytes": store.nbytes,
    }
    print(f"✅ Download complete: {summary}")


if __name__ == "__main__":
    main()
