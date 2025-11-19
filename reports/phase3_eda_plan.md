# Phase 3: Exploratory Data Analysis (EDA) Plan

## Overview
Comprehensive exploratory analysis of the fused dataset to understand patterns, relationships, and prepare for sentiment modeling.

## Objectives
1. **Data Overview**: Understand the structure and quality of the fused dataset
2. **Statistical Profiling**: Compute descriptive statistics for all key variables
3. **Distribution Analysis**: Examine distributions of ratings, text lengths, prices, etc.
4. **Relationship Analysis**: Identify correlations and associations between features
5. **Temporal Analysis**: Explore trends over time (ratings, review volume)
6. **Hypothesis Testing**: Validate key hypotheses about the data
7. **Data Quality Assessment**: Identify missing data patterns and outliers

## Key Analyses

### 1. Dataset Overview
- Total rows, columns, memory usage
- Data types and schema
- Missing value patterns
- Sample rows inspection

### 2. Review Features Analysis
- **Rating Distribution**: Histogram, summary stats
- **Review Text**: Length distributions (characters, words)
- **Summary Length**: Distribution and relationship with ratings
- **Helpfulness Metrics**: Distribution of helpful votes, ratios
- **Verification Status**: Proportion of verified purchases
- **Language Detection**: Non-English review analysis

### 3. Temporal Analysis
- Review volume over time (monthly/yearly trends)
- Average rating trends over time
- Seasonal patterns
- Review activity by product category

### 4. Product/Metadata Features Analysis
- **Price Distribution**: Histograms, outliers, missing rates
- **Category Analysis**: Category depth, primary categories distribution
- **Brand Analysis**: Top brands, brand distribution
- **Feature Counts**: Number of features per product
- **Image Counts**: Distribution of product images

### 5. Relationship Analysis
- Rating vs. Review Length
- Rating vs. Helpfulness
- Price vs. Rating
- Verified vs. Non-verified ratings
- Category vs. Rating patterns
- Temporal trends in ratings

### 6. Statistical Tests
- **Normality Tests**: Shapiro-Wilk for numeric distributions
- **Correlation Analysis**: Pearson/Spearman correlations
- **Group Comparisons**: t-tests or Mann-Whitney for verified vs. non-verified
- **Chi-square Tests**: Categorical associations (e.g., category vs. rating bucket)

### 7. Visualizations
- Distribution plots (histograms, KDE plots)
- Box plots for comparisons
- Scatter plots for relationships
- Time series plots
- Correlation heatmaps
- Bar charts for categorical data

## Implementation Steps

1. **Create EDA Notebook**: `notebooks/01_eda_analysis.ipynb`
2. **Load Fused Dataset**: Use chunked reading for large dataset
3. **Generate Summary Statistics**: Automated summary report
4. **Create Visualizations**: Save figures to `reports/figures/`
5. **Statistical Tests**: Document results in notebook
6. **Generate EDA Report**: Summary document in `reports/phase3_eda_results.md`

## Deliverables
- Jupyter notebook with complete EDA analysis
- Summary statistics report
- Visualization gallery (10+ key figures)
- Statistical test results
- Data quality assessment report
- Insights and findings document

## Next Steps After EDA
- Feature engineering for sentiment modeling
- Data preparation for ML models
- Baseline model development

