# Ecommerce-Insights

Advanced data fusion, exploratory analysis, sentiment modeling, and forecasting for large-scale e-commerce datasets (UCSD Amazon Review Data, 2018).

## Repository Layout

```
Ecommerce-Insights/
├── data/                  # Raw datasets (kept out of version control)
├── notebooks/             # Jupyter notebooks for EDA and modeling
├── src/                   # Reusable Python modules and pipelines
├── reports/               # Generated reports, figures, and documentation
├── dashboard/             # Streamlit/Dash prototype assets
├── tests/                 # Automated tests for reproducibility
└── README.md
```

> Note: large JSON files live in `data/` and are ignored by Git.

## Datasets

| Dataset               | Filename               | Size (approx.) | Description |
|-----------------------|------------------------|----------------|-------------|
| Reviews - Books       | `Books_5.json`         | 22 GB          | 5-core Amazon book reviews with ratings, text, metadata. |
| Reviews - Electronics | `Electronics_5.json`   | 4.2 GB         | 5-core Amazon electronics reviews. |
| Metadata - Books      | `meta_Books.json`      | 4.0 GB         | Product metadata (pricing, categories, descriptions). |
| Metadata - Electronics| `meta_Electronics.json`| 11 GB          | Electronics product metadata. |

All files originate from the UCSD Amazon Review Data (https://nijianmo.github.io/amazon/). Keep raw files compressed when archiving; only extract into `data/` for local processing.

## Environment Setup

1. Install Python 3.10+ (conda recommended).
2. Create environment:
   ```bash
   conda create -n ecommerce-insights python=3.10
   conda activate ecommerce-insights
   ```
3. Install core packages (to be expanded as development continues):
   ```bash
   pip install pandas numpy scikit-learn jupyterlab
   ```
4. Optional: add GPU-enabled frameworks (PyTorch/TensorFlow) as modeling starts.

Dependencies will be formalized in `requirements.txt` once the core pipeline is established.

## Getting Started

1. Verify datasets are present in `data/`.
2. Open `notebooks/` and create `00_data_intake.ipynb` to experiment with chunked loaders.
3. Implement reusable loaders under `src/data/` for reading large JSON files in chunks (with sampling utilities).
4. Commit analysis-ready parquet artifacts to `data/processed/` (create as needed) rather than large JSONs.

## Project Roadmap

- **Phase 0:** Repository scaffolding, environment, data intake notes. ✅
- **Phase 1:** Data acquisition & management (chunked loaders, profiling, dataset documentation). ⏳
- **Phase 2:** Data cleaning, feature engineering, fuzzy joins.
- **Phase 3:** EDA with statistical testing and visualizations.
- **Phase 4:** Sentiment modeling (baseline → ensemble/deep models) with hyperparameter tuning.
- **Phase 5:** Advanced analytics (clustering, forecasting, causal inference).
- **Phase 6:** Reporting (40+ page report, dashboard stub, exec slides) and packaging.

Checkpoints and deliverables will be tracked alongside the project plan.
