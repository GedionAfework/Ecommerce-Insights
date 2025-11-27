"""
Ecommerce Insights Dashboard
A Dash application for exploring e-commerce review data, sentiment analysis, and advanced analytics.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import base64
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "fused"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Ecommerce Insights Dashboard"

# Load model metadata
try:
    with open(MODELS_DIR / "model_metadata.json", "r") as f:
        model_metadata = json.load(f)
except:
    model_metadata = {}

# App layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    # Header
    html.Div([
        html.H1("Ecommerce Insights Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
        html.P("Advanced Analytics for Amazon Review Data",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '18px'})
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'}),
    
    # Navigation
    html.Div([
        dcc.Tabs(id="tabs", value='overview', children=[
            dcc.Tab(label='Overview', value='overview'),
            dcc.Tab(label='EDA Analysis', value='eda'),
            dcc.Tab(label='Sentiment Analysis', value='sentiment'),
            dcc.Tab(label='Advanced Analytics', value='analytics'),
            dcc.Tab(label='Model Predictions', value='predictions'),
        ], style={'fontSize': '16px'})
    ], style={'margin': '20px'}),
    
    # Content
    html.Div(id='page-content', style={'padding': '20px'})
])

# Overview Page
overview_layout = html.Div([
    html.H2("Project Overview", style={'color': '#2c3e50'}),
    html.Hr(),
    
    html.Div([
        html.Div([
            html.H3("Dataset Statistics", style={'color': '#34495e'}),
            html.Div(id='dataset-stats')
        ], className='six columns', style={'padding': '20px', 'backgroundColor': '#ffffff', 
                                          'borderRadius': '5px', 'margin': '10px'}),
        
        html.Div([
            html.H3("Model Performance", style={'color': '#34495e'}),
            html.Div(id='model-stats')
        ], className='six columns', style={'padding': '20px', 'backgroundColor': '#ffffff',
                                          'borderRadius': '5px', 'margin': '10px'})
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),
    
    html.Div([
        html.H3("Key Insights", style={'color': '#34495e', 'marginTop': '30px'}),
        html.Ul([
            html.Li("Comprehensive analysis of 26+ million Amazon book reviews"),
            html.Li("Sentiment classification models with 84% accuracy"),
            html.Li("Customer and product segmentation using clustering"),
            html.Li("Time series forecasting for review trends"),
            html.Li("Causal inference analysis on review helpfulness")
        ], style={'fontSize': '16px', 'lineHeight': '2'})
    ])
])

# EDA Page
eda_layout = html.Div([
    html.H2("Exploratory Data Analysis", style={'color': '#2c3e50'}),
    html.Hr(),
    
    html.Div([
        dcc.Dropdown(
            id='eda-plot-selector',
            options=[
                {'label': 'Rating Distribution', 'value': 'rating_distribution'},
                {'label': 'Review Length Distribution', 'value': 'review_length'},
                {'label': 'Temporal Analysis', 'value': 'temporal'},
                {'label': 'Correlation Heatmap', 'value': 'correlation'},
                {'label': 'Verified vs Non-Verified', 'value': 'verified'},
                {'label': 'Yearly/Quarterly Analysis', 'value': 'yearly'},
            ],
            value='rating_distribution',
            style={'width': '100%', 'marginBottom': '20px'}
        ),
        dcc.Graph(id='eda-plot')
    ])
])

# Sentiment Analysis Page
sentiment_layout = html.Div([
    html.H2("Sentiment Analysis", style={'color': '#2c3e50'}),
    html.Hr(),
    
    html.Div([
        dcc.Dropdown(
            id='sentiment-plot-selector',
            options=[
                {'label': 'Model Comparison', 'value': 'model_comparison'},
                {'label': 'Confusion Matrix', 'value': 'confusion_matrix'},
                {'label': 'Word Clouds', 'value': 'wordclouds'},
                {'label': 'Sentiment Distribution', 'value': 'sentiment_dist'},
            ],
            value='model_comparison',
            style={'width': '100%', 'marginBottom': '20px'}
        ),
        dcc.Graph(id='sentiment-plot')
    ])
])

# Advanced Analytics Page
analytics_layout = html.Div([
    html.H2("Advanced Analytics", style={'color': '#2c3e50'}),
    html.Hr(),
    
    html.Div([
        dcc.Dropdown(
            id='analytics-plot-selector',
            options=[
                {'label': 'Customer Clustering', 'value': 'customer_clusters'},
                {'label': 'Product Clustering', 'value': 'product_clusters'},
                {'label': 'Time Series Decomposition', 'value': 'time_series'},
                {'label': 'ARIMA Forecast', 'value': 'arima'},
                {'label': 'Causal Analysis', 'value': 'causal'},
            ],
            value='customer_clusters',
            style={'width': '100%', 'marginBottom': '20px'}
        ),
        dcc.Graph(id='analytics-plot')
    ])
])

# Model Predictions Page
predictions_layout = html.Div([
    html.H2("Sentiment Prediction", style={'color': '#2c3e50'}),
    html.Hr(),
    
    html.Div([
        html.Div([
            html.Label("Enter Review Text:", style={'fontSize': '16px', 'fontWeight': 'bold'}),
            dcc.Textarea(
                id='review-input',
                value='This book is absolutely amazing! I loved every page of it.',
                style={'width': '100%', 'height': '150px', 'fontSize': '14px', 'padding': '10px'}
            ),
            html.Br(),
            html.Button('Predict Sentiment', id='predict-button', 
                       style={'marginTop': '10px', 'padding': '10px 20px', 
                             'fontSize': '16px', 'backgroundColor': '#3498db', 
                             'color': 'white', 'border': 'none', 'borderRadius': '5px',
                             'cursor': 'pointer'}),
        ], style={'marginBottom': '30px'}),
        
        html.Div(id='prediction-output', style={
            'padding': '20px', 'backgroundColor': '#ecf0f1', 
            'borderRadius': '5px', 'marginTop': '20px'
        })
    ])
])

# Callback for page routing
@app.callback(Output('page-content', 'children'),
              Input('tabs', 'value'))
def render_page(tab):
    if tab == 'overview':
        return overview_layout
    elif tab == 'eda':
        return eda_layout
    elif tab == 'sentiment':
        return sentiment_layout
    elif tab == 'analytics':
        return analytics_layout
    elif tab == 'predictions':
        return predictions_layout
    return overview_layout

# Callback for dataset stats
@app.callback(Output('dataset-stats', 'children'),
              Input('url', 'pathname'))
def update_dataset_stats(pathname):
    try:
        stats_file = REPORTS_DIR / "summary_statistics.csv"
        if stats_file.exists():
            stats_df = pd.read_csv(stats_file)
            return html.Div([
                html.P(f"Total Reviews: {stats_df.get('count', ['N/A'])[0]:,.0f}" if 'count' in stats_df.columns else "N/A"),
                html.P(f"Average Rating: {stats_df.get('mean', ['N/A'])[0]:.2f}" if 'mean' in stats_df.columns else "N/A"),
                html.P(f"Date Range: 1995 - 2018"),
            ], style={'fontSize': '16px'})
    except:
        pass
    return html.Div("Statistics loading...")

# Callback for model stats
@app.callback(Output('model-stats', 'children'),
              Input('url', 'pathname'))
def update_model_stats(pathname):
    if model_metadata:
        return html.Div([
            html.P(f"Model: {model_metadata.get('model_name', 'N/A')}"),
            html.P(f"Test Accuracy: {model_metadata.get('test_accuracy', 0):.2%}"),
            html.P(f"F1-Score (Macro): {model_metadata.get('test_f1_macro', 0):.2%}"),
            html.P(f"F1-Score (Weighted): {model_metadata.get('test_f1_weighted', 0):.2%}"),
        ], style={'fontSize': '16px'})
    return html.Div("Model statistics loading...")

def encode_image(image_path):
    """Encode image to base64 for Dash display"""
    try:
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('ascii')
            return f'data:image/png;base64,{encoded}'
    except:
        return None

# Callback for EDA plots
@app.callback(Output('eda-plot', 'figure'),
              Input('eda-plot-selector', 'value'))
def update_eda_plot(selected_plot):
    plot_map = {
        'rating_distribution': 'rating_distribution.png',
        'temporal': 'temporal_analysis.png',
        'correlation': 'correlation_heatmap.png',
        'verified': 'verified_vs_nonverified.png',
        'yearly': 'yearly_quarterly_analysis.png',
        'review_length': 'review_length_distribution.png'
    }
    
    try:
        img_filename = plot_map.get(selected_plot)
        if img_filename:
            img_path = FIGURES_DIR / img_filename
            if img_path.exists():
                encoded_image = encode_image(img_path)
                if encoded_image:
                    return go.Figure().add_layout_image(
                        dict(source=encoded_image, xref="paper", yref="paper", 
                             x=0, y=1, sizex=1, sizey=1, xanchor="left", 
                             yanchor="top", sizing="stretch", layer="below")
                    ).update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                                   height=600, title=selected_plot.replace('_', ' ').title(),
                                   margin=dict(l=0, r=0, t=50, b=0))
    except Exception as e:
        pass
    
    return go.Figure().add_annotation(
        text="Plot not available. Please ensure the figure file exists in reports/figures/",
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16)
    ).update_layout(height=600)

# Callback for sentiment plots
@app.callback(Output('sentiment-plot', 'figure'),
              Input('sentiment-plot-selector', 'value'))
def update_sentiment_plot(selected_plot):
    plot_map = {
        'model_comparison': 'final_models_comparison.png',
        'confusion_matrix': 'test_set_confusion_matrix.png',
        'wordclouds': 'wordclouds_by_sentiment.png',
        'sentiment_dist': 'rating_distribution.png'
    }
    
    try:
        img_filename = plot_map.get(selected_plot)
        if img_filename:
            img_path = FIGURES_DIR / img_filename
            if img_path.exists():
                encoded_image = encode_image(img_path)
                if encoded_image:
                    return go.Figure().add_layout_image(
                        dict(source=encoded_image, xref="paper", yref="paper",
                             x=0, y=1, sizex=1, sizey=1, xanchor="left",
                             yanchor="top", sizing="stretch", layer="below")
                    ).update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                                   height=600, title=selected_plot.replace('_', ' ').title(),
                                   margin=dict(l=0, r=0, t=50, b=0))
    except Exception as e:
        pass
    
    return go.Figure().add_annotation(
        text="Plot not available. Please ensure the figure file exists in reports/figures/",
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16)
    ).update_layout(height=600)

# Callback for analytics plots
@app.callback(Output('analytics-plot', 'figure'),
              Input('analytics-plot-selector', 'value'))
def update_analytics_plot(selected_plot):
    plot_map = {
        'customer_clusters': 'customer_clusters_analysis.png',
        'product_clusters': 'product_user_analysis.png',
        'time_series': 'time_series_decomposition.png',
        'arima': 'arima_forecast.png',
        'causal': 'causal_correlation_matrix.png'
    }
    
    try:
        img_filename = plot_map.get(selected_plot)
        if img_filename:
            img_path = FIGURES_DIR / img_filename
            if img_path.exists():
                encoded_image = encode_image(img_path)
                if encoded_image:
                    return go.Figure().add_layout_image(
                        dict(source=encoded_image, xref="paper", yref="paper",
                             x=0, y=1, sizex=1, sizey=1, xanchor="left",
                             yanchor="top", sizing="stretch", layer="below")
                    ).update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                                   height=600, title=selected_plot.replace('_', ' ').title(),
                                   margin=dict(l=0, r=0, t=50, b=0))
    except Exception as e:
        pass
    
    return go.Figure().add_annotation(
        text="Plot not available. Please ensure the figure file exists in reports/figures/",
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16)
    ).update_layout(height=600)

# Callback for sentiment prediction
@app.callback(Output('prediction-output', 'children'),
              Input('predict-button', 'n_clicks'),
              State('review-input', 'value'))
def predict_sentiment(n_clicks, review_text):
    if n_clicks is None or not review_text:
        return html.Div("Enter review text and click 'Predict Sentiment' to see results.")
    
    try:
        # Load model and vectorizer
        model_path = MODELS_DIR / "best_sentiment_model_logistic_regression_(tuned).pkl"
        vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        
        if not model_path.exists() or not vectorizer_path.exists():
            return html.Div([
                html.H4("Model files not found", style={'color': '#e74c3c'}),
                html.P("Please ensure model files are in the models/ directory.")
            ])
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # Preprocess and predict
        review_vectorized = vectorizer.transform([review_text])
        prediction = model.predict(review_vectorized)[0]
        probabilities = model.predict_proba(review_vectorized)[0]
        
        # Get class names
        classes = model.classes_
        prob_dict = dict(zip(classes, probabilities))
        
        # Determine color based on sentiment
        if prediction == 'Positive':
            color = '#27ae60'
        elif prediction == 'Neutral':
            color = '#f39c12'
        else:
            color = '#e74c3c'
        
        return html.Div([
            html.H4(f"Predicted Sentiment: {prediction}", 
                   style={'color': color, 'fontSize': '24px', 'marginBottom': '20px'}),
            html.H5("Confidence Scores:", style={'marginTop': '20px'}),
            html.Div([
                html.Div([
                    html.P(f"Positive: {prob_dict.get('Positive', 0):.2%}", 
                          style={'fontSize': '18px', 'margin': '5px'}),
                    html.Div(style={'width': f'{prob_dict.get("Positive", 0)*100}%', 
                                   'height': '20px', 'backgroundColor': '#27ae60',
                                   'borderRadius': '3px', 'margin': '5px'})
                ]),
                html.Div([
                    html.P(f"Neutral: {prob_dict.get('Neutral', 0):.2%}", 
                          style={'fontSize': '18px', 'margin': '5px'}),
                    html.Div(style={'width': f'{prob_dict.get("Neutral", 0)*100}%', 
                                   'height': '20px', 'backgroundColor': '#f39c12',
                                   'borderRadius': '3px', 'margin': '5px'})
                ]),
                html.Div([
                    html.P(f"Negative: {prob_dict.get('Negative', 0):.2%}", 
                          style={'fontSize': '18px', 'margin': '5px'}),
                    html.Div(style={'width': f'{prob_dict.get("Negative", 0)*100}%', 
                                   'height': '20px', 'backgroundColor': '#e74c3c',
                                   'borderRadius': '3px', 'margin': '5px'})
                ])
            ])
        ])
    except Exception as e:
        return html.Div([
            html.H4("Error", style={'color': '#e74c3c'}),
            html.P(str(e))
        ])

if __name__ == '__main__':
    app.run(debug=True, port=8050)

