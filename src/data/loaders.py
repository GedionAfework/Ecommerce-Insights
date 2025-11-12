"""
Streaming loaders for the large Amazon review JSON datasets.

The raw files are newline-delimited JSON (NDJSON). These helpers enable
chunked ingestion to avoid exhausting memory when working with 20+ GB files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"


def get_data_path(filename: str) -> Path:
    """
    Return the absolute path to a file inside the data directory.
    """
    # Allow either direct placement under data/ or within data/raw/
    candidates = [
        DATA_DIR / filename,
        RAW_DIR / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Dataset not found in expected locations. "
        f"Tried: {', '.join(str(p) for p in candidates)}"
    )


def iter_json_chunks(
    filename: str,
    *,
    chunksize: int = 10000,
    columns: Optional[Sequence[str]] = None,
    convert_dates: bool = False,
) -> Iterator[pd.DataFrame]:
    """
    Yield successive pandas DataFrame chunks from a newline-delimited JSON file.

    Parameters
    ----------
    filename:
        Name of the file within `data/`.
    chunksize:
        Number of records per chunk read into memory.
    columns:
        Optional subset of columns to retain in each chunk.
    convert_dates:
        When True, converts `unixReviewTime` into pandas datetime using second units.
    """
    path = get_data_path(filename)
    reader = pd.read_json(path, lines=True, chunksize=chunksize)
    for chunk in reader:
        if columns is not None:
            missing = set(columns) - set(chunk.columns)
            if missing:
                raise KeyError(f"Missing columns {missing} in {filename}")
            chunk = chunk.loc[:, list(columns)]
        if convert_dates and "unixReviewTime" in chunk.columns:
            chunk = chunk.assign(
                review_datetime=pd.to_datetime(chunk["unixReviewTime"], unit="s")
            )
        yield chunk


def load_sample(
    filename: str,
    *,
    n_rows: int = 5000,
    columns: Optional[Sequence[str]] = None,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Load a random sample of rows from a large JSON file by reservoir sampling.

    Parameters
    ----------
    filename:
        Name of the file within `data/`.
    n_rows:
        Approximate number of rows to return.
    columns:
        Optional subset of columns to retain.
    random_state:
        Seed for reproducible sampling.
    """
    rng = np.random.default_rng(random_state)
    sample_records: list[dict] = []
    total_seen = 0

    for chunk in iter_json_chunks(filename, columns=columns):
        for record in chunk.to_dict(orient="records"):
            total_seen += 1
            if len(sample_records) < n_rows:
                sample_records.append(record)
            else:
                idx = rng.integers(0, total_seen)
                if idx < n_rows:
                    sample_records[idx] = record

    if not sample_records:
        raise ValueError(f"No data read from {filename}")
    return pd.DataFrame(sample_records)

