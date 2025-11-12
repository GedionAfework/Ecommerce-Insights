"""
Dataset fusion utilities to combine cleaned reviews and metadata.

Usage:
    python -m src.data.fuse_datasets --reviews books --metadata books
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
from rapidfuzz import fuzz, process

from .loaders import DATA_DIR


@dataclass
class FusionStats:
    review_dataset: str
    metadata_dataset: str
    reviews_rows: int
    metadata_rows: int
    merged_rows: int
    metadata_coverage: float
    fuzzy_matches_attempted: int
    fuzzy_matches_applied: int

    def to_dict(self) -> dict:
        return {
            "review_dataset": self.review_dataset,
            "metadata_dataset": self.metadata_dataset,
            "reviews_rows": self.reviews_rows,
            "metadata_rows": self.metadata_rows,
            "merged_rows": self.merged_rows,
            "metadata_coverage": self.metadata_coverage,
            "fuzzy_matches_attempted": self.fuzzy_matches_attempted,
            "fuzzy_matches_applied": self.fuzzy_matches_applied,
        }


def load_parquet_chunks(paths: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frames.append(pd.read_parquet(path))
    if not frames:
        raise ValueError("No parquet files provided.")
    return pd.concat(frames, ignore_index=True)


def collect_parquet_files(directory: Path, prefix: str) -> List[Path]:
    return sorted(directory.glob(f"{prefix}*.parquet"))


def fuzzy_match_titles(
    reviews: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    score_threshold: int = 90,
    limit: int = 5_000,
) -> pd.DataFrame:
    if "title" not in reviews.columns or "title" not in metadata.columns:
        return reviews

    unmatched = reviews[reviews["title"].isna()].copy()
    if unmatched.empty:
        return reviews

    metadata_titles = metadata[["asin", "title"]].dropna()
    title_map = {}
    for _, row in metadata_titles.iterrows():
        title_map[row["asin"]] = row["title"]

    fuzzy_applied = 0
    scores_attempted = 0

    for idx, review_row in unmatched.iterrows():
        if limit is not None and scores_attempted >= limit:
            break
        summary = review_row.get("summary")
        if not summary or not isinstance(summary, str):
            continue
        scores_attempted += 1
        best = process.extractOne(
            summary,
            title_map.values(),
            scorer=fuzz.token_set_ratio,
        )
        if best and best[1] >= score_threshold:
            matched_title = best[0]
            matched_asin = next(
                (asin for asin, title in title_map.items() if title == matched_title),
                None,
            )
            if matched_asin is not None:
                reviews.at[idx, "asin"] = matched_asin
                reviews.at[idx, "title"] = matched_title
                fuzzy_applied += 1

    reviews.attrs["fuzzy_attempted"] = scores_attempted
    reviews.attrs["fuzzy_applied"] = fuzzy_applied
    return reviews


def fuse_reviews_metadata(
    review_dataset: str,
    metadata_dataset: str,
    *,
    fuzzy_threshold: int = 90,
    fuzzy_limit: int = 10_000,
) -> FusionStats:
    reviews_dir = DATA_DIR / "processed" / "reviews"
    metadata_dir = DATA_DIR / "processed" / "metadata"

    review_prefix = f"{review_dataset}_reviews"
    metadata_prefix = f"{metadata_dataset}_metadata"

    review_files = collect_parquet_files(reviews_dir, review_prefix)
    metadata_files = collect_parquet_files(metadata_dir, metadata_prefix)

    if not review_files:
        raise FileNotFoundError(f"No cleaned review files found for '{review_dataset}'.")
    if not metadata_files:
        raise FileNotFoundError(f"No cleaned metadata files found for '{metadata_dataset}'.")

    reviews_df = load_parquet_chunks(review_files)
    metadata_df = load_parquet_chunks(metadata_files)

    reviews_df = fuzzy_match_titles(
        reviews_df,
        metadata_df,
        score_threshold=fuzzy_threshold,
        limit=fuzzy_limit,
    )

    merged = reviews_df.merge(metadata_df, on="asin", how="left", suffixes=("", "_meta"))

    metadata_coverage = merged["title_meta"].notna().mean()
    merged_rows = len(merged)

    output_dir = DATA_DIR / "processed" / "fused"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{review_dataset}_{metadata_dataset}_fused.parquet"
    merged.to_parquet(output_path, index=False)

    stats = FusionStats(
        review_dataset=review_dataset,
        metadata_dataset=metadata_dataset,
        reviews_rows=len(reviews_df),
        metadata_rows=len(metadata_df),
        merged_rows=merged_rows,
        metadata_coverage=float(metadata_coverage),
        fuzzy_matches_attempted=int(reviews_df.attrs.get("fuzzy_attempted", 0)),
        fuzzy_matches_applied=int(reviews_df.attrs.get("fuzzy_applied", 0)),
    )

    summary_path = DATA_DIR / "interim" / f"{review_dataset}_{metadata_dataset}_fusion_stats.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(stats.to_dict(), fp, indent=2)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse cleaned reviews and metadata datasets.")
    parser.add_argument(
        "--reviews",
        required=True,
        help="Name of cleaned review dataset (books or electronics).",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Name of cleaned metadata dataset (books or electronics).",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=int,
        default=90,
        help="Minimum RapidFuzz score to accept a fuzzy match.",
    )
    parser.add_argument(
        "--fuzzy-limit",
        type=int,
        default=10_000,
        help="Maximum number of fuzzy comparisons to attempt.",
    )

    args = parser.parse_args()
    stats = fuse_reviews_metadata(
        review_dataset=args.reviews,
        metadata_dataset=args.metadata,
        fuzzy_threshold=args.fuzzy_threshold,
        fuzzy_limit=args.fuzzy_limit,
    )
    print(json.dumps(stats.to_dict(), indent=2))


if __name__ == "__main__":
    main()

