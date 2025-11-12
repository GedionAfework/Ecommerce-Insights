"""
Chunked cleaning pipeline for Amazon review datasets.

Usage:
    python -m src.data.clean_reviews --dataset books
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .loaders import DATA_DIR, iter_json_chunks

RAW_DATASETS = {
    "books": "Books_5.json",
    "electronics": "Electronics_5.json",
}

REQUIRED_COLUMNS = {
    "asin",
    "reviewerID",
    "overall",
    "reviewText",
    "summary",
    "unixReviewTime",
}

OUTPUT_DIR = DATA_DIR / "processed" / "reviews"
INTERIM_DIR = DATA_DIR / "interim"


@dataclass
class CleaningStats:
    dataset: str
    total_rows: int = 0
    rows_after_required: int = 0
    rows_after_dedup: int = 0
    written_rows: int = 0
    dropped_missing_required: int = 0
    duplicate_count: int = 0
    text_too_short: int = 0
    language_flagged: int = 0
    chunk_files: Iterable[Path] = field(default_factory=list)

    def to_dict(self) -> Dict[str, int]:
        return {
            "total_rows": self.total_rows,
            "rows_after_required": self.rows_after_required,
            "rows_after_dedup": self.rows_after_dedup,
            "written_rows": self.written_rows,
            "dropped_missing_required": self.dropped_missing_required,
            "duplicate_count": self.duplicate_count,
            "text_too_short": self.text_too_short,
            "language_flagged": self.language_flagged,
            "chunk_count": len(list(self.chunk_files)),
        }


def ensure_columns_present(columns: Iterable[str], filename: str) -> None:
    missing = REQUIRED_COLUMNS - set(columns)
    if missing:
        raise ValueError(f"{filename} missing required columns: {missing}")


def parse_helpful(helpful_value: Optional[Iterable]) -> Tuple[Optional[int], Optional[int]]:
    if helpful_value is None or (isinstance(helpful_value, float) and np.isnan(helpful_value)):
        return None, None
    if isinstance(helpful_value, str):
        try:
            parsed = json.loads(helpful_value)
            if (
                isinstance(parsed, list)
                and len(parsed) == 2
                and all(isinstance(x, (int, float)) for x in parsed)
            ):
                return int(parsed[0]), int(parsed[1])
        except json.JSONDecodeError:
            pass
    if isinstance(helpful_value, (list, tuple)) and len(helpful_value) == 2:
        try:
            return int(helpful_value[0]), int(helpful_value[1])
        except (TypeError, ValueError):
            return None, None
    return None, None


def normalize_vote(vote: Optional[str]) -> Optional[int]:
    if vote is None or (isinstance(vote, float) and np.isnan(vote)):
        return None
    if isinstance(vote, (int, float)):
        return int(vote)
    cleaned = vote.replace(",", "").strip()
    if cleaned == "":
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def clean_chunk(df: pd.DataFrame, stats: CleaningStats) -> pd.DataFrame:
    stats.total_rows += len(df)

    before_required = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    stats.rows_after_required += len(df)
    stats.dropped_missing_required += before_required - len(df)

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["reviewerID", "asin", "unixReviewTime"])
    stats.rows_after_dedup += len(df)
    stats.duplicate_count += before_dedup - len(df)

    df = df.assign(
        review_datetime=pd.to_datetime(df["unixReviewTime"], unit="s", utc=True),
        review_length_chars=df["reviewText"].str.len(),
        review_length_words=df["reviewText"].str.split().str.len(),
        summary_length_words=df["summary"].fillna("").str.split().str.len(),
        is_verified=df.get("verified", False).fillna(False).astype(bool),
    )

    df["review_year"] = df["review_datetime"].dt.year
    df["review_month"] = df["review_datetime"].dt.month

    length_mask = df["review_length_chars"].fillna(0) >= 5
    stats.text_too_short += int((~length_mask).sum())
    df = df[length_mask]

    if "helpful" in df.columns:
        helpful_pairs = [parse_helpful(value) for value in df["helpful"]]
    else:
        helpful_pairs = [(None, None)] * len(df)
    helpful_votes = [pair[0] for pair in helpful_pairs]
    total_votes = [pair[1] for pair in helpful_pairs]
    vote_series = df["vote"] if "vote" in df.columns else pd.Series([None] * len(df))
    df = df.assign(
        helpful_votes=helpful_votes,
        total_votes=total_votes,
        vote_count=vote_series.map(normalize_vote),
    )

    df["helpfulness_ratio"] = np.where(
        (df["total_votes"].notna()) & (df["total_votes"] > 0),
        df["helpful_votes"] / df["total_votes"],
        np.nan,
    )

    rating = df["overall"].astype(float)
    df["rating_bucket"] = pd.cut(
        rating,
        bins=[-np.inf, 2.0, 3.5, np.inf],
        labels=["negative", "neutral", "positive"],
        right=True,
    )

    if "language" in df.columns:
        language_series = df["language"].fillna("").astype(str).str.lower()
        non_english = (language_series != "") & (language_series != "en")
    else:
        non_english = pd.Series([False] * len(df))
    df["non_english_flag"] = non_english
    stats.language_flagged += int(non_english.sum())

    df = df.reset_index(drop=True)
    return df


def clean_reviews_dataset(
    dataset: str,
    *,
    chunksize: int = 100_000,
    limit_chunks: Optional[int] = None,
) -> CleaningStats:
    dataset = dataset.lower()
    if dataset not in RAW_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of {list(RAW_DATASETS)}.")

    raw_filename = RAW_DATASETS[dataset]
    stats = CleaningStats(dataset=dataset)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    chunk_outputs: list[Path] = []
    for idx, chunk in enumerate(iter_json_chunks(raw_filename, chunksize=chunksize)):
        ensure_columns_present(chunk.columns, raw_filename)
        cleaned = clean_chunk(chunk, stats)
        if cleaned.empty:
            continue
        output_path = OUTPUT_DIR / f"{dataset}_reviews_chunk_{idx:05d}.parquet"
        cleaned.to_parquet(output_path, index=False)
        chunk_outputs.append(output_path)
        stats.written_rows += len(cleaned)
        if limit_chunks is not None and idx + 1 >= limit_chunks:
            break

    stats.chunk_files = chunk_outputs

    summary_path = INTERIM_DIR / f"{dataset}_reviews_cleaning_stats.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(stats.to_dict(), fp, indent=2)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Amazon review datasets.")
    parser.add_argument(
        "--dataset",
        choices=list(RAW_DATASETS.keys()),
        required=True,
        help="Dataset to process (books or electronics).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Number of rows to load per chunk.",
    )
    parser.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="Optional cap on number of chunks to process (useful for testing).",
    )

    args = parser.parse_args()
    stats = clean_reviews_dataset(
        dataset=args.dataset,
        chunksize=args.chunksize,
        limit_chunks=args.limit_chunks,
    )
    print(json.dumps(stats.to_dict(), indent=2))


if __name__ == "__main__":
    main()

