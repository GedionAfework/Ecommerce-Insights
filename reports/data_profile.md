# Dataset Intake Notes

## Raw Files (as of 2025-11-11)

| Dataset | Path | Size (GB) | Notes |
|---------|------|-----------|-------|
| Books Reviews (5-core) | `data/Books_5.json` | 22.36 | Newline-delimited JSON |
| Electronics Reviews (5-core) | `data/Electronics_5.json` | 4.19 | Newline-delimited JSON |
| Books Metadata | `data/meta_Books.json` | 4.05 | Contains nested category arrays |
| Electronics Metadata | `data/meta_Electronics.json` | 10.99 | Sparse pricing fields |

## Immediate Action Items

- Validate schema consistency across review subsets (ensure `asin`, `reviewerID`, `overall`, `reviewText`, `unixReviewTime` present).
- Profile metadata pricing fields to estimate missingness and currency inconsistencies.
- Establish processed storage layout (`data/processed/`) for parquet outputs.
- Set chunk size defaults to 100k rows for profiling to balance throughput vs memory.

## Quick Access

Use the utilities in `src/data/`:

```python
from src.data import iter_json_chunks, profile_dataset

profile = profile_dataset("Books_5.json", chunksize=100_000)
print(profile.total_rows)
```

