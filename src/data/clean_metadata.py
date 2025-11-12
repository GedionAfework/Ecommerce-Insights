"""
Cleaning pipeline for Amazon product metadata.

Usage:
    python -m src.data.clean_metadata --dataset books
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .loaders import DATA_DIR, iter_json_chunks

RAW_DATASETS = {
    "books": "meta_Books.json",
    "electronics": "meta_Electronics.json",
}

OUTPUT_DIR = DATA_DIR / "processed" / "metadata"
INTERIM_DIR = DATA_DIR / "interim"

REQUIRED_COLUMNS = {"asin", "title"}
PRICE_PATTERN = re.compile(r"[\d,.]+")


@dataclass
class MetadataStats:
    dataset: str
    total_rows: int = 0
    rows_after_required: int = 0
    rows_after_dedup: int = 0
    written_rows: int = 0
    dropped_missing_required: int = 0
    duplicate_count: int = 0
    price_missing: int = 0
    chunk_files: Iterable[Path] = field(default_factory=list)

    def to_dict(self) -> Dict[str, int]:
        return {
            "total_rows": self.total_rows,
            "rows_after_required": self.rows_after_required,
            "rows_after_dedup": self.rows_after_dedup,
            "written_rows": self.written_rows,
            "dropped_missing_required": self.dropped_missing_required,
            "duplicate_count": self.duplicate_count,
            "price_missing": self.price_missing,
            "chunk_count": len(list(self.chunk_files)),
        }


def ensure_columns_present(columns: Iterable[str], filename: str) -> None:
    missing = REQUIRED_COLUMNS - set(columns)
    if missing:
        raise ValueError(f"{filename} missing required columns: {missing}")


def parse_price(value: Optional[str | float | int]) -> Tuple[Optional[float], bool]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, True
    if isinstance(value, (int, float)):
        return float(value), False
    cleaned = value.strip()
    if not cleaned:
        return None, True
    match = PRICE_PATTERN.search(cleaned.replace("US$", "$"))
    if not match:
        return None, True
    try:
        return float(match.group(0).replace(",", "")), False
    except ValueError:
        return None, True


def extract_primary_category(categories: List[List[str]]) -> Tuple[Optional[str], int]:
    if not categories:
        return None, 0
    path = categories[0]
    if not path:
        return None, 0
    depth = len(path)
    return path[-1], depth


def clean_chunk(df: pd.DataFrame, stats: MetadataStats) -> pd.DataFrame:
    stats.total_rows += len(df)

    before_required = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    stats.rows_after_required += len(df)
    stats.dropped_missing_required += before_required - len(df)

    before_dedup = len(df)
    df = df.sort_values("title").drop_duplicates(subset=["asin"], keep="last")
    stats.rows_after_dedup += len(df)
    stats.duplicate_count += before_dedup - len(df)

    df["title"] = df["title"].astype(str).str.strip()
    if "brand" in df.columns:
        df["brand"] = df["brand"].fillna("").astype(str).str.strip()
    else:
        df["brand"] = ""

    categories = df["categories"] if "categories" in df.columns else pd.Series([[]] * len(df))
    primary_category, category_depth = zip(
        *(extract_primary_category(cat if isinstance(cat, list) else []) for cat in categories)
    )
    df["primary_category"] = primary_category
    df["category_depth"] = category_depth

    price_series = df["price"] if "price" in df.columns else pd.Series([None] * len(df))
    parsed_prices = [parse_price(val) for val in price_series]
    df["price"], missing_flags = zip(*parsed_prices)
    stats.price_missing += int(sum(flag for flag in missing_flags))
    df["price_missing"] = missing_flags

    if "feature" in df.columns:
        df["num_features"] = df["feature"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    else:
        df["num_features"] = 0

    if "imageURL" in df.columns:
        df["num_images"] = df["imageURL"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    else:
        df["num_images"] = 0

    if "description" in df.columns:
        df["description_length"] = df["description"].astype(str).str.len()
    else:
        df["description_length"] = 0

    df["brand_normalized"] = df["brand"].str.lower().replace({"": None})

    # Serialize nested/list columns to JSON strings for parquet compatibility
    def _serialize_nested(value: object) -> object:
        if isinstance(value, (list, dict)):
            try:
                return json.dumps(value)
            except (TypeError, ValueError):
                return str(value)
        return value

    for column in df.select_dtypes(include="object").columns:
        if df[column].apply(lambda x: isinstance(x, (list, dict))).any():
            df[column] = df[column].apply(_serialize_nested)

    return df.reset_index(drop=True)


def clean_metadata_dataset(
    dataset: str,
    *,
    chunksize: int = 50_000,
    limit_chunks: Optional[int] = None,
) -> MetadataStats:
    dataset = dataset.lower()
    if dataset not in RAW_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of {list(RAW_DATASETS)}.")

    raw_filename = RAW_DATASETS[dataset]
    stats = MetadataStats(dataset=dataset)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    chunk_outputs: list[Path] = []
    for idx, chunk in enumerate(iter_json_chunks(raw_filename, chunksize=chunksize)):
        ensure_columns_present(chunk.columns, raw_filename)
        cleaned = clean_chunk(chunk, stats)
        if cleaned.empty:
            continue
        output_path = OUTPUT_DIR / f"{dataset}_metadata_chunk_{idx:05d}.parquet"
        cleaned.to_parquet(output_path, index=False)
        chunk_outputs.append(output_path)
        stats.written_rows += len(cleaned)
        if limit_chunks is not None and idx + 1 >= limit_chunks:
            break

    stats.chunk_files = chunk_outputs

    summary_path = INTERIM_DIR / f"{dataset}_metadata_cleaning_stats.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(stats.to_dict(), fp, indent=2)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Amazon metadata datasets.")
    parser.add_argument(
        "--dataset",
        choices=list(RAW_DATASETS.keys()),
        required=True,
        help="Dataset to process (books or electronics).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50_000,
        help="Number of rows to load per chunk.",
    )
    parser.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="Optional cap on number of chunks to process (useful for testing).",
    )

    args = parser.parse_args()
    stats = clean_metadata_dataset(
        dataset=args.dataset,
        chunksize=args.chunksize,
        limit_chunks=args.limit_chunks,
    )
    print(json.dumps(stats.to_dict(), indent=2))


if __name__ == "__main__":
    main()

