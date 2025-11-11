"""
Lightweight dataset profiling utilities for the raw JSON files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Dict, Iterable, Optional

import pandas as pd

from .loaders import iter_json_chunks


@dataclass
class DatasetProfile:
    filename: str
    total_rows: int
    columns: Iterable[str]
    numeric_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)


def profile_dataset(
    filename: str,
    *,
    chunksize: int = 100000,
    numeric_columns: Optional[Iterable[str]] = None,
) -> DatasetProfile:
    """
    Compute high-level metrics (row count, numeric aggregates) for a large JSON file.
    """
    total_rows = 0
    column_names: Optional[Iterable[str]] = None
    numeric_summary: Dict[str, Dict[str, float]] = {}

    for chunk in iter_json_chunks(filename, chunksize=chunksize):
        total_rows += len(chunk)
        if column_names is None:
            column_names = list(chunk.columns)

        selected_cols = list(numeric_columns) if numeric_columns else list(
            chunk.select_dtypes("number").columns
        )

        for col in selected_cols:
            series = chunk[col].dropna()
            if series.empty:
                continue

            values = series.to_numpy()
            stats = numeric_summary.setdefault(
                col,
                {"count": 0, "mean": 0.0, "m2": 0.0, "min": inf, "max": -inf},
            )

            for x in values:
                stats["count"] += 1
                delta = x - stats["mean"]
                stats["mean"] += delta / stats["count"]
                delta2 = x - stats["mean"]
                stats["m2"] += delta * delta2
                stats["min"] = min(stats["min"], float(x))
                stats["max"] = max(stats["max"], float(x))

    collapsed_summary: Dict[str, Dict[str, float]] = {}
    for col, stats in numeric_summary.items():
        count = stats["count"]
        variance = stats["m2"] / (count - 1) if count > 1 else 0.0
        collapsed_summary[col] = {
            "mean": stats["mean"],
            "std": variance**0.5,
            "min": stats["min"],
            "max": stats["max"],
        }

    return DatasetProfile(
        filename=filename,
        total_rows=total_rows,
        columns=column_names or [],
        numeric_summary=collapsed_summary,
    )

