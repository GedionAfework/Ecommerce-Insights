# Ecommerce Insights Dashboard

Interactive Dash dashboard for exploring e-commerce review data, sentiment analysis, and advanced analytics.

## Features

- **Overview**: Project statistics and key insights
- **EDA Analysis**: Interactive visualizations from exploratory data analysis
- **Sentiment Analysis**: Model performance metrics and visualizations
- **Advanced Analytics**: Clustering, time series, and causal inference results
- **Model Predictions**: Real-time sentiment prediction for review text

## Installation

1. Install dependencies:
```bash
pip install dash plotly pandas numpy scikit-learn joblib
```

2. Ensure model files are in the `models/` directory:
   - `best_sentiment_model_logistic_regression_(tuned).pkl`
   - `tfidf_vectorizer.pkl`
   - `model_metadata.json`

3. Ensure visualization files are in `reports/figures/`

## Running the Dashboard

```bash
python dashboard/app.py
```

The dashboard will be available at `http://localhost:8050`

## Usage

1. **Overview Tab**: View dataset statistics and model performance metrics
2. **EDA Analysis Tab**: Explore various data visualizations using the dropdown
3. **Sentiment Analysis Tab**: View model comparison, confusion matrices, and word clouds
4. **Advanced Analytics Tab**: Explore clustering results, time series forecasts, and causal analysis
5. **Model Predictions Tab**: Enter review text to get real-time sentiment predictions with confidence scores

## Notes

- The dashboard loads visualizations from pre-generated PNG files in `reports/figures/`
- For real-time data loading, modify the callbacks to load from Parquet files
- Model predictions require the trained model files to be present

