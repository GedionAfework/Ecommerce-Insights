# Phase 2 — Data Cleaning & Fusion Plan

## Objectives

1. Clean review datasets (`Books_5.json`, `Electronics_5.json`) and produce standardized parquet outputs with engineered features.
2. Clean metadata datasets (`meta_Books.json`, `meta_Electronics.json`) to align price, category, and descriptive fields.
3. Merge reviews with metadata on `asin`, augmenting with fuzzy title matching where necessary.

## Directory Layout

- `data/raw/` — original JSON files (immutable; do not modify in place).
- `data/interim/` — temporary parquet batches for debugging or partial outputs.
- `data/processed/` — final cleaned datasets ready for modeling (`reviews_books.parquet`, etc.).

## Cleaning Steps (Reviews)

1. Chunked load via `iter_json_chunks`.
2. Drop duplicates on `["reviewerID", "asin", "unixReviewTime"]`.
3. Remove entries with missing `overall` rating or `reviewText`.
4. Normalize timestamps (`review_datetime`), derive `review_year`, `review_month`.
5. Engineer features:
   - `review_length_words`, `review_length_chars`
   - `is_verified` cast to boolean
   - `helpful_ratio = vote / max(1, total_votes)` when vote present
6. Persist cleaned chunk to `data/interim/reviews_{category}_{part}.parquet`; consolidate after full pass.

## Cleaning Steps (Metadata)

1. Load in chunks; retain key fields (`asin`, `title`, `price`, `brand`, `categories`, `description`).
2. Standardize `price`: remove currency symbols, convert to float, handle missing via category median.
3. Normalize `categories` to top-level (e.g., first two levels) and explode for analysis.
4. Drop records without `asin` or missing `title`.
5. Compute derived features:
   - `primary_category`
   - `n_categories`
   - `description_length`
6. Save cleaned parquet files under `data/processed/metadata_{category}.parquet`.

## Merge Strategy

1. Perform left join of review dataset with corresponding metadata on `asin`.
2. For unmatched `asin`, attempt fuzzy title match (`RapidFuzz` ratio ≥ 90) using `summary` vs `title`; log matches.
3. Flag merges with missing metadata fields for downstream handling (`metadata_missing` boolean).
4. Output merged datasets to `data/processed/reviews_with_metadata_{category}.parquet`.

## Quality Checks

- Row counts before/after each transformation (logged to console and saved in audit CSV).
- Null value matrix snapshots (e.g., `missingno`, `pandas-profiling`) stored in `reports/`.
- Validate price distributions post-imputation.
- Ensure schema consistency across categories (matching column order/types).

## Next Deliverables

- `src/data/clean_reviews.py`
- `src/data/clean_metadata.py`
- `notebooks/02_data_cleaning.ipynb` (optional exploration/explanation)
- Updated `reports/data_profile.md` with summary stats from cleaned datasets.

