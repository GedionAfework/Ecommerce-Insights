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

| Dataset               | Path                             | Size (approx.) | Description |
|-----------------------|----------------------------------|----------------|-------------|
| Reviews - Books       | `data/raw/Books_5.json`          | 22 GB          | 5-core Amazon book reviews with ratings, text, metadata. |
| Reviews - Electronics | `data/raw/Electronics_5.json`    | 4.2 GB         | 5-core Amazon electronics reviews. |
| Metadata - Books      | `data/raw/meta_Books.json`       | 4.0 GB         | Product metadata (pricing, categories, descriptions). |
| Metadata - Electronics| `data/raw/meta_Electronics.json` | 11 GB          | Electronics product metadata. |

All files originate from the UCSD Amazon Review Data (https://nijianmo.github.io/amazon/). Keep raw files compressed when archiving; only extract into `data/` for local processing.

## Environment Setup

1. Install Python 3.10+ (conda recommended).
2. Create environment:
   ```bash
   conda create -n ecommerce-insights python=3.10
   conda activate ecommerce-insights
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Optional: add GPU-enabled frameworks (PyTorch/TensorFlow) for deep learning models.

## Getting Started

1. Verify datasets are present in `data/`.
2. Open `notebooks/` and create `00_data_intake.ipynb` to experiment with chunked loaders.
3. Implement reusable loaders under `src/data/` for reading large JSON files in chunks (with sampling utilities).
4. Commit analysis-ready parquet artifacts to `data/processed/` (create as needed) rather than large JSONs.

## Project Roadmap

- **Phase 0:** Repository scaffolding, environment, data intake notes. ✅
- **Phase 1:** Data acquisition & management (chunked loaders, profiling, dataset documentation). ✅
- **Phase 2:** Data cleaning, feature engineering, fuzzy joins. ✅
- **Phase 3:** EDA with statistical testing and visualizations. ✅ **COMPLETE**
- **Phase 4:** Sentiment modeling (baseline → ensemble/deep models) with hyperparameter tuning. ✅ **COMPLETE**
- **Phase 5:** Advanced analytics (clustering, forecasting, causal inference). ✅ **COMPLETE**
- **Phase 6:** Reporting (40+ page report, dashboard stub, exec slides) and packaging. ✅ **COMPLETE**

## Running the Pipelines

### Phase 2: Data Cleaning and Fusion

1. **Clean Reviews Dataset:**
   ```bash
   python -m src.data.clean_reviews --dataset books
   ```

2. **Clean Metadata Dataset:**
   ```bash
   python -m src.data.clean_metadata --dataset books
   ```

3. **Fuse Datasets:**
   ```bash
   python -m src.data.fuse_datasets --reviews books --metadata books --fuzzy-threshold 92 --fuzzy-limit 20000
   ```

### Phase 3: Exploratory Data Analysis

1. **Open Jupyter Notebook:**
   ```bash
   jupyter lab notebooks/01_eda_analysis.ipynb
   ```

2. **Run all cells** to generate:
   - Statistical summaries
   - Visualizations (saved to `reports/figures/`)
   - Correlation analysis
   - Temporal trends
   - Statistical tests

### Phase 4: Sentiment Modeling

1. **Open Jupyter Notebook:**
   ```bash
   jupyter lab notebooks/02_sentiment_modeling.ipynb
   ```

2. **Run all cells** to:
   - Train baseline and advanced models
   - Perform hyperparameter tuning
   - Evaluate model performance
   - Save best model to `models/`

### Phase 5: Advanced Analytics

1. **Open Jupyter Notebook:**
   ```bash
   jupyter lab notebooks/03_advanced_analytics.ipynb
   ```

2. **Run all cells** to:
   - Perform customer and product segmentation
   - Generate time series forecasts
   - Conduct causal inference analysis

### Phase 6: Interactive Dashboard

1. **Start the Dash Dashboard:**
   ```bash
   python dashboard/app.py
   ```

2. **Access the dashboard:**
   - Open browser to `http://localhost:8050`
   - Explore EDA visualizations
   - View model performance metrics
   - Make real-time sentiment predictions
   - Analyze advanced analytics results

## Project Deliverables

### Reports
- **Comprehensive Report:** `reports/comprehensive_report.md` (40+ pages)
- **Executive Presentation:** `reports/executive_presentation.md`
- **Phase-specific Reports:** Available in `reports/` directory

### Models
- **Best Sentiment Model:** `models/best_sentiment_model_logistic_regression_(tuned).pkl`
- **TF-IDF Vectorizer:** `models/tfidf_vectorizer.pkl`
- **Model Metadata:** `models/model_metadata.json`

### Visualizations
- All figures saved to `reports/figures/`
- Includes EDA plots, model comparisons, clustering results, time series forecasts

### Dashboard
- Interactive Dash application in `dashboard/app.py`
- Real-time sentiment prediction
- Comprehensive data exploration

## Key Results

- **Model Accuracy:** 83.9% (3-class sentiment classification)
- **Dataset Analyzed:** 26.9M reviews (sampled for analysis)
- **Customer Clusters:** 4 distinct segments identified
- **Time Series Model:** ARIMA(2,1,2) for review volume forecasting
- **Key Insight:** 80% of reviews are positive, longer reviews are more helpful

## Documentation

- **Dashboard README:** `dashboard/README.md`
- **Comprehensive Report:** `reports/comprehensive_report.md`
- **Executive Presentation:** `reports/executive_presentation.md`

Checkpoints and deliverables are tracked alongside the project plan.
