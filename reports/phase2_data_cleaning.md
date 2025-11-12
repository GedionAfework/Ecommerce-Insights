# Phase 2 — Data Cleaning & Fusion Plan

## 1. Directory Layout (completed)
- Raw JSON files relocated to `data/raw/`.
- `data/interim/` reserved for sampled subsets and temporary artifacts.
- `data/processed/` will store parquet outputs:
  - `data/processed/reviews/` — cleaned reviews by category.
  - `data/processed/metadata/` — cleaned metadata tables.
  - `data/processed/fused/` — final merged datasets ready for modeling.

## 2. Cleaning Checklist — Reviews
1. **Schema validation**
   - Ensure presence of `asin`, `reviewerID`, `overall`, `reviewText`, `summary`, `unixReviewTime`.
   - Drop records missing core identifiers or ratings.
2. **Deduplication**
   - Remove duplicates by `(reviewerID, asin, unixReviewTime)`.
3. **Timestamp normalization**
   - Convert `unixReviewTime` to timezone-aware datetime (`review_datetime`), derive `review_year`, `review_month`.
4. **Text quality**
   - Filter extremely short reviews (`len(reviewText.strip()) < 5`).
   - Track Unicode normalization and language detection (flag non-English for separate handling).
5. **Feature engineering**
   - Compute `review_length_tokens`, `review_length_chars`, `summary_length`.
   - Derive `rating_bucket` (positive/neutral/negative) and `is_verified`.
   - Calculate `helpfulness_ratio` = `vote_up / (vote_total)` where available.
6. **Outlier handling**
   - Flag unusual rating patterns via z-score on `overall` within category/time windows for later anomaly analysis.

## 3. Cleaning Checklist — Metadata
1. **Schema normalization**
   - Enforce string type for `asin`, `title`, `brand`.
   - Parse nested `categories` into normalized list-of-lists and create primary category field.
2. **Price handling**
   - Convert price strings to numeric USD (strip currency symbols, handle ranges).
   - Impute missing prices using median within `(category, brand)` groups; add `price_missing` indicator.
3. **Feature engineering**
   - Extract `category_path_depth`, `num_features`, `num_images`.
   - Generate textual embeddings placeholder fields for `description` (computed in later phases).
4. **Deduplication & enrichment**
   - Retain most recent metadata entry per `asin` (if duplicates).
   - Flag products with sparse metadata for downstream caution.

## 4. Dataset Fusion
1. **Primary join**
   - Merge reviews and metadata on `asin`, left join from reviews.
2. **Fuzzy reconciliation**
   - For reviews missing metadata, use RapidFuzz matching on `title` vs review `summary` or user-provided product names.
   - Record match scores and keep only matches above threshold (e.g., 90/100).
3. **Quality audits**
   - Calculate coverage metrics (percentage of reviews with metadata coverage, price availability).
   - Create anomaly report for mismatched categories or negative pricing.

## 5. Implementation Steps
1. Implement review cleaning pipeline in `src/data/clean_reviews.py` with CLI entry point (`python -m src.data.clean_reviews --dataset books`).
2. Implement metadata cleaning pipeline in `src/data/clean_metadata.py`.
3. Build fusion script `src/data/fuse_datasets.py` writing outputs to `data/processed/fused/`.
4. Update unit tests under `tests/` to cover critical transformations.
5. Document outputs and key statistics in `reports/phase2_summary.md` (to be created).

## 6. Open Questions / Assumptions
- Language filtering threshold — default to keeping all languages but mark non-English for optional exclusion.
- Handling of extremely long reviews (truncate vs keep) — postpone decision to modeling phase, but compute length metrics now.
- Decide whether to downsample Electronics for faster iteration (retain full dataset for final pipeline).

