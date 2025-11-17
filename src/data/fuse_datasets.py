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
import pyarrow as pa
import pyarrow.parquet as pq
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

    # Load metadata once (it's smaller)
    metadata_df = load_parquet_chunks(metadata_files)

    # Process review chunks incrementally to avoid memory issues
    output_dir = DATA_DIR / "processed" / "fused"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{review_dataset}_{metadata_dataset}_fused.parquet"

    total_reviews = 0
    total_merged = 0
    total_with_metadata = 0
    total_fuzzy_attempted = 0
    total_fuzzy_applied = 0

    output_chunks = []

    for idx, review_file in enumerate(review_files):
        chunk_df = pd.read_parquet(review_file)
        total_reviews += len(chunk_df)

        # Apply fuzzy matching if needed (skip for memory efficiency on large datasets)
        if fuzzy_limit > 0 and idx == 0:  # Only on first chunk to limit memory
            chunk_df = fuzzy_match_titles(
                chunk_df,
                metadata_df,
                score_threshold=fuzzy_threshold,
                limit=fuzzy_limit,
            )
            total_fuzzy_attempted += int(chunk_df.attrs.get("fuzzy_attempted", 0))
            total_fuzzy_applied += int(chunk_df.attrs.get("fuzzy_applied", 0))

        # Merge with metadata
        merged_chunk = chunk_df.merge(
            metadata_df, on="asin", how="left", suffixes=("", "_meta")
        )
        total_merged += len(merged_chunk)
        total_with_metadata += merged_chunk["title_meta"].notna().sum() if "title_meta" in merged_chunk.columns else 0

        # Normalize datetime columns for parquet compatibility
        for col in merged_chunk.columns:
            if merged_chunk[col].dtype == "object":
                # Check if column contains any datetime-like objects
                sample = merged_chunk[col].dropna()
                if len(sample) > 0:
                    has_timestamp = any(isinstance(x, pd.Timestamp) for x in sample.head(100))
                    if has_timestamp or col.lower() in ("date", "datetime", "timestamp", "time"):
                        # Convert all datetime-like objects to datetime64
                        try:
                            merged_chunk[col] = pd.to_datetime(merged_chunk[col], errors="coerce")
                        except (TypeError, ValueError):
                            # If conversion fails, convert to string
                            merged_chunk[col] = merged_chunk[col].astype(str)
            elif pd.api.types.is_datetime64_any_dtype(merged_chunk[col]):
                # Ensure datetime columns are properly typed
                merged_chunk[col] = pd.to_datetime(merged_chunk[col], errors="coerce")

        # Write chunk to temporary file
        chunk_output = output_dir / f"temp_chunk_{idx:05d}.parquet"
        merged_chunk.to_parquet(chunk_output, index=False)
        output_chunks.append(chunk_output)

    # Combine all chunks into final output incrementally to avoid memory issues
    if output_chunks:
        # Helper function to normalize a dataframe
        def normalize_chunk(df: pd.DataFrame) -> pd.DataFrame:
            # Normalize datetime columns
            for col in df.columns:
                if df[col].dtype == "object":
                    sample = df[col].dropna()
                    if len(sample) > 0:
                        has_timestamp = any(isinstance(x, pd.Timestamp) for x in sample.head(100))
                        if has_timestamp or col.lower() in ("date", "datetime", "timestamp", "time"):
                            try:
                                df[col] = pd.to_datetime(df[col], errors="coerce")
                            except (TypeError, ValueError):
                                df[col] = df[col].astype(str)
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            
            # Normalize nullable boolean columns (e.g., non_english_flag)
            for col in df.columns:
                if df[col].dtype == "object" and col in ("non_english_flag",):
                    # Convert to bool, handling nulls
                    df[col] = df[col].fillna(False).astype(bool)
            
            # Normalize mixed-type object columns to strings for parquet compatibility
            for col in df.columns:
                if df[col].dtype == "object":
                    # Check if column has mixed types
                    sample = df[col].dropna()
                    if len(sample) > 0:
                        types = set(type(x).__name__ for x in sample.head(100))
                        if len(types) > 1:  # Mixed types detected
                            # Convert everything to string for consistency
                            df[col] = df[col].astype(str).replace("nan", None)
            
            # Ensure date column is always nullable datetime for consistent schema
            if "date" in df.columns:
                # Always convert to datetime (pd.NaT for nulls maintains nullable datetime type)
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            
            # Ensure nullable string columns are consistently typed (not pure null)
            # These columns may be null in some chunks but have string data in others
            nullable_string_cols = ["vote", "details", "primary_category"]
            for col in nullable_string_cols:
                if col in df.columns:
                    # Convert to string, keeping nulls as None (creates nullable string type)
                    # Handle numeric types (like vote which might be double) by converting to string
                    if df[col].dtype in ["float64", "int64", "float32", "int32"]:
                        # Convert numeric to string, preserving nulls
                        df[col] = df[col].astype(str).replace("nan", None).replace("None", None)
                    elif df[col].dtype != "object":
                        df[col] = df[col].astype(str)
                    # Ensure nulls are preserved as None, not "nan" string
                    df[col] = df[col].replace("nan", None).replace("None", None)
                    # If all null, ensure it's still object type (nullable string)
                    if df[col].isna().all():
                        df[col] = df[col].astype(object)
            
            return df
        
        # Get and normalize first chunk to establish schema
        first_chunk = pd.read_parquet(output_chunks[0])
        first_chunk = normalize_chunk(first_chunk)
        
        # Ensure nullable string columns have at least one non-null value so PyArrow infers nullable string type
        # (not pure null type which can't be cast from string later)
        nullable_string_cols = ["vote", "details", "primary_category"]
        for col in nullable_string_cols:
            if col in first_chunk.columns and first_chunk[col].isna().all():
                # Set first value to empty string to force nullable string type
                if len(first_chunk) > 0:
                    first_chunk.loc[first_chunk.index[0], col] = ""
                else:
                    # If empty dataframe, add a row with empty string
                    first_chunk.loc[0, col] = ""
        
        # Write first chunk and establish schema
        table = pa.Table.from_pandas(first_chunk)
        schema = table.schema
        
        # Clean up the temporary empty string values we added
        for col in nullable_string_cols:
            if col in first_chunk.columns:
                # Replace empty strings with None for all-null columns
                mask = (first_chunk[col] == "")
                if mask.any():
                    first_chunk.loc[mask, col] = None
        
        # Recreate table with cleaned data
        table = pa.Table.from_pandas(first_chunk, schema=schema)
        writer = pq.ParquetWriter(output_path, schema)
        writer.write_table(table)
        
        # Process remaining chunks incrementally, aligning to schema
        for chunk_file in output_chunks[1:]:
            chunk_df = pd.read_parquet(chunk_file)
            chunk_df = normalize_chunk(chunk_df)
            
            # Align columns to match schema order and types
            chunk_df = chunk_df.reindex(columns=first_chunk.columns, fill_value=None)
            
            # Ensure types match - handle nullable string columns specially
            nullable_string_cols = ["vote", "details", "primary_category"]
            for col in first_chunk.columns:
                if col in chunk_df.columns:
                    # For nullable string columns, ensure they're object type (nullable string)
                    if col in nullable_string_cols:
                        # Handle numeric types (like vote which might be double) by converting to string
                        if chunk_df[col].dtype in ["float64", "int64", "float32", "int32"]:
                            chunk_df[col] = chunk_df[col].astype(str).replace("nan", None).replace("None", None)
                        elif chunk_df[col].dtype != "object":
                            chunk_df[col] = chunk_df[col].astype(str)
                        # Replace "nan" strings with None
                        chunk_df[col] = chunk_df[col].replace("nan", None).replace("None", None)
                        chunk_df[col] = chunk_df[col].astype(object)
                    elif first_chunk[col].dtype != chunk_df[col].dtype:
                        # For object columns, just ensure they're object type
                        if first_chunk[col].dtype == "object" or chunk_df[col].dtype == "object":
                            chunk_df[col] = chunk_df[col].astype(object)
                        else:
                            try:
                                chunk_df[col] = chunk_df[col].astype(first_chunk[col].dtype)
                            except (TypeError, ValueError):
                                # If type conversion fails, use object type
                                chunk_df[col] = chunk_df[col].astype(object)
            
            # Ensure date column matches schema type
            if "date" in chunk_df.columns and "date" in schema.names:
                chunk_df["date"] = pd.to_datetime(chunk_df["date"], errors="coerce")
            
            # Convert to table
            table = pa.Table.from_pandas(chunk_df)
            # Cast to match schema if possible
            try:
                table = table.cast(schema)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                # If casting fails, ensure at least date column matches
                if "date" in schema.names:
                    date_field = schema.field("date")
                    if table.schema.field("date").type != date_field.type:
                        # Recast date column specifically
                        table = table.set_column(
                            table.column_names.index("date"),
                            "date",
                            table.column("date").cast(date_field.type)
                        )
                # Try casting again
                try:
                    table = table.cast(schema)
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                    # If still fails, raise the error with helpful message
                    raise ValueError(
                        f"Schema mismatch for chunk {chunk_file.name}. "
                        f"Table schema: {table.schema}, Required schema: {schema}"
                    )
            writer.write_table(table)
        
        writer.close()
        
        # Clean up temp files
        for temp_file in output_chunks:
            temp_file.unlink()

    metadata_coverage = total_with_metadata / total_merged if total_merged > 0 else 0.0

    stats = FusionStats(
        review_dataset=review_dataset,
        metadata_dataset=metadata_dataset,
        reviews_rows=total_reviews,
        metadata_rows=len(metadata_df),
        merged_rows=total_merged,
        metadata_coverage=float(metadata_coverage),
        fuzzy_matches_attempted=total_fuzzy_attempted,
        fuzzy_matches_applied=total_fuzzy_applied,
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

