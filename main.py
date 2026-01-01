
import feedparser
import yfinance as yf
import pandas as pd
import numpy as np
from transformers import pipeline
from datetime import datetime, timedelta
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st
from scipy import stats
import re
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Financial Sentiment Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for professional finance theme
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stMetric {
        background-color: #1e2530; 
        padding: 20px; 
        border-radius: 10px; 
        border: 2px solid #2e3650;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-label {
        font-size: 14px;
        color: #888;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .positive {color: #00ff88;}
    .negative {color: #ff4444;}
    .neutral {color: #ffaa00;}
    h1 {
        color: #ffffff;
        font-size: 42px;
        margin-bottom: 10px;
    }
    h2 {
        color: #00ff88;
        font-size: 28px;
        margin-top: 30px;
        margin-bottom: 20px;
        border-bottom: 2px solid #2e3650;
        padding-bottom: 10px;
    }
    h3 {
        color: #ffffff;
        font-size: 22px;
    }
    .word-list {
        background-color: #1e2530;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2e3650;
        margin: 10px 0;
    }
    .word-item {
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 5px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .word-positive {
        background-color: rgba(0, 255, 136, 0.1);
        border-left: 3px solid #00ff88;
    }
    .word-negative {
        background-color: rgba(255, 68, 68, 0.1);
        border-left: 3px solid #ff4444;
    }
    .explanation {
        background-color: #1e2530;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #00ff88;
        margin: 15px 0;
        color: #ccc;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize FinBERT model
@st.cache_resource
def load_sentiment_model():
    """Load FinBERT sentiment analysis model"""
    try:
        return pipeline("text-classification", model="ProsusAI/finbert")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Fetch RSS news for ticker
def fetch_news(ticker, days_back=30):
    """Fetch news from Yahoo Finance RSS feed"""
    try:
        rss_url = f'https://finance.yahoo.com/rss/headline?s={ticker}'
        feed = feedparser.parse(rss_url)
        
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for entry in feed.entries:
            try:
                pub_date = datetime(*entry.published_parsed[:6])
                if pub_date >= cutoff_date:
                    articles.append({
                        'title': entry.title,
                        'link': entry.link,
                        'published': pub_date,
                        'summary': entry.get('summary', entry.title),
                        'ticker': ticker
                    })
            except Exception as e:
                continue
        
        return articles
    except Exception as e:
        st.error(f"Error fetching news for {ticker}: {e}")
        return []

# Extract meaningful words from text
def extract_keywords(text, min_word_length=4):
    """Extract keywords from text, filtering out common words"""
    # Common stop words to exclude
    stop_words = {
        'this', 'that', 'with', 'from', 'have', 'been', 'were', 'will',
        'what', 'when', 'where', 'which', 'while', 'would', 'could', 'should',
        'their', 'there', 'these', 'those', 'than', 'then', 'them', 'they',
        'about', 'after', 'before', 'between', 'into', 'through', 'during',
        'above', 'below', 'under', 'again', 'further', 'once', 'here', 'more',
        'most', 'other', 'some', 'such', 'only', 'same', 'just', 'very',
        'says', 'said', 'also', 'however', 'still', 'years', 'year'
    }
    
    # Clean and tokenize
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    
    # Filter words
    keywords = [
        word for word in words 
        if len(word) >= min_word_length 
        and word not in stop_words
        and not word.isdigit()
    ]
    
    return keywords

# Analyze sentiment with FinBERT
def analyze_sentiment(articles, sentiment_pipeline):
    """Classify articles using FinBERT"""
    results = []
    
    progress_bar = st.progress(0)
    total = len(articles)
    
    for idx, article in enumerate(articles):
        try:
            # Analyze sentiment
            text = article['summary'][:512]  # FinBERT max length
            sentiment_result = sentiment_pipeline(text)[0]
            
            # Calculate weighted score
            text_length = len(article['summary'].split())
            length_weight = min(text_length / 100, 2.0)  # Cap at 2x
            
            # Recency weight (more recent = higher weight)
            days_old = (datetime.now() - article['published']).days
            recency_weight = 1 / (1 + days_old * 0.1)
            
            # Compute final weighted score
            base_score = sentiment_result['score']
            if sentiment_result['label'] == 'negative':
                base_score = -base_score
            elif sentiment_result['label'] == 'neutral':
                base_score = 0
            
            weighted_score = base_score * length_weight * recency_weight
            
            # Extract keywords
            keywords = extract_keywords(article['title'] + ' ' + article['summary'])
            
            results.append({
                **article,
                'sentiment': sentiment_result['label'],
                'confidence': sentiment_result['score'],
                'weighted_score': weighted_score,
                'length_weight': length_weight,
                'recency_weight': recency_weight,
                'keywords': keywords
            })
            
            progress_bar.progress((idx + 1) / total)
        except Exception as e:
            st.warning(f"Error analyzing article: {str(e)[:100]}")
            continue
    
    progress_bar.empty()
    return results

# Fetch stock prices
def fetch_stock_prices(ticker, start_date, end_date):
    """Fetch daily stock prices from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            st.warning(f"No stock data available for {ticker}")
            return pd.DataFrame()
        
        df['return'] = df['Close'].pct_change()
        df.index = df.index.date  # Convert to date only
        return df
    except Exception as e:
        st.error(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()

# Aggregate daily sentiment
def aggregate_daily_sentiment(sentiment_results):
    """Aggregate sentiment scores by day"""
    if not sentiment_results:
        return pd.DataFrame()
    
    df = pd.DataFrame(sentiment_results)
    df['date'] = pd.to_datetime(df['published']).dt.date
    
    daily_agg = df.groupby('date').agg({
        'weighted_score': 'sum',
        'sentiment': lambda x: list(x),
        'title': 'count'
    }).reset_index()
    
    daily_agg.columns = ['date', 'sentiment_score', 'sentiments', 'article_count']
    
    # Count positive and negative articles
    daily_agg['positive_count'] = daily_agg['sentiments'].apply(
        lambda x: sum(1 for s in x if s == 'positive')
    )
    daily_agg['negative_count'] = daily_agg['sentiments'].apply(
        lambda x: sum(1 for s in x if s == 'negative')
    )
    daily_agg['neutral_count'] = daily_agg['sentiments'].apply(
        lambda x: sum(1 for s in x if s == 'neutral')
    )
    
    return daily_agg

# Calculate metrics
def calculate_metrics(sentiment_results, daily_sentiment, stock_data):
    """Calculate summary metrics"""
    total_articles = len(sentiment_results)
    
    if total_articles == 0:
        return {
            'total_articles': 0,
            'avg_sentiment': 0,
            'positive_ratio': 0,
            'correlation': 0
        }
    
    positive_count = sum(1 for r in sentiment_results if r['sentiment'] == 'positive')
    negative_count = sum(1 for r in sentiment_results if r['sentiment'] == 'negative')
    avg_sentiment = np.mean([r['weighted_score'] for r in sentiment_results])
    
    # Calculate correlation with stock returns
    if not daily_sentiment.empty and not stock_data.empty:
        merged = pd.merge(
            daily_sentiment,
            stock_data[['return']],
            left_on='date',
            right_index=True,
            how='inner'
        )
        
        if len(merged) > 2 and merged['sentiment_score'].std() > 0 and merged['return'].std() > 0:
            correlation, _ = stats.pearsonr(merged['sentiment_score'], merged['return'])
        else:
            correlation = 0
    else:
        correlation = 0
    
    return {
        'total_articles': total_articles,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'avg_sentiment': avg_sentiment,
        'positive_ratio': positive_count / total_articles if total_articles > 0 else 0,
        'correlation': correlation
    }

# Extract top keywords
def get_top_keywords(sentiment_results, sentiment_type, top_n=15):
    """Extract top keywords for a specific sentiment"""
    filtered = [r for r in sentiment_results if r['sentiment'] == sentiment_type]
    
    all_keywords = []
    for result in filtered:
        all_keywords.extend(result.get('keywords', []))
    
    if not all_keywords:
        return []
    
    keyword_counts = Counter(all_keywords)
    return keyword_counts.most_common(top_n)

# Visualization functions
def plot_sentiment_trend(data_dict):
    """Interactive line chart for daily sentiment trends with annotations"""
    fig = go.Figure()
    
    for ticker, data in data_dict.items():
        daily = data['daily_sentiment']
        
        if daily.empty:
            continue
        
        fig.add_trace(go.Scatter(
            x=daily['date'],
            y=daily['sentiment_score'],
            mode='lines+markers',
            name=ticker,
            line=dict(width=3),
            marker=dict(size=8),
            hovertemplate=(
                '<b>%{fullData.name}</b><br>' +
                'Date: %{x}<br>' +
                'Sentiment Score: %{y:.3f}<br>' +
                'Articles: %{customdata}<br>' +
                '<extra></extra>'
            ),
            customdata=daily['article_count']
        ))
    
    fig.update_layout(
        title={
            'text': 'Daily Sentiment Trend Analysis',
            'font': {'size': 24, 'color': '#ffffff'}
        },
        xaxis_title='Date',
        yaxis_title='Weighted Sentiment Score',
        hovermode='x unified',
        template='plotly_dark',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117'
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig

def plot_article_counts(data_dict):
    """Stacked bar chart for positive vs negative article counts"""
    
    fig = make_subplots(
        rows=1, cols=len(data_dict),
        subplot_titles=[f'<b>{ticker}</b>' for ticker in data_dict.keys()],
        horizontal_spacing=0.05
    )
    
    for idx, (ticker, data) in enumerate(data_dict.items(), 1):
        daily = data['daily_sentiment']
        
        if daily.empty:
            continue
        
        fig.add_trace(
            go.Bar(
                name='Positive',
                x=daily['date'],
                y=daily['positive_count'],
                marker_color='#00ff88',
                showlegend=(idx == 1),
                hovertemplate='Positive: %{y}<extra></extra>'
            ),
            row=1, col=idx
        )
        fig.add_trace(
            go.Bar(
                name='Negative',
                x=daily['date'],
                y=daily['negative_count'],
                marker_color='#ff4444',
                showlegend=(idx == 1),
                hovertemplate='Negative: %{y}<extra></extra>'
            ),
            row=1, col=idx
        )
    
    fig.update_layout(
        title={
            'text': 'Daily Article Sentiment Distribution',
            'font': {'size': 24, 'color': '#ffffff'}
        },
        barmode='stack',
        template='plotly_dark',
        height=400,
        showlegend=True,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117'
    )
    
    return fig

def display_keyword_rankings(sentiment_results, ticker):
    """Display ranked lists of positive and negative keywords"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 📈 Top Positive Keywords")
        positive_keywords = get_top_keywords(sentiment_results, 'positive', top_n=15)
        
        if positive_keywords:
            st.markdown('<div class="word-list">', unsafe_allow_html=True)
            for rank, (word, count) in enumerate(positive_keywords, 1):
                st.markdown(
                    f'<div class="word-item word-positive">'
                    f'<span><b>#{rank}</b> {word.title()}</span>'
                    f'<span style="color: #00ff88; font-weight: bold;">{count} mentions</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No positive articles found")
    
    with col2:
        st.markdown(f"### 📉 Top Negative Keywords")
        negative_keywords = get_top_keywords(sentiment_results, 'negative', top_n=15)
        
        if negative_keywords:
            st.markdown('<div class="word-list">', unsafe_allow_html=True)
            for rank, (word, count) in enumerate(negative_keywords, 1):
                st.markdown(
                    f'<div class="word-item word-negative">'
                    f'<span><b>#{rank}</b> {word.title()}</span>'
                    f'<span style="color: #ff4444; font-weight: bold;">{count} mentions</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No negative articles found")

def plot_sentiment_vs_return(data_dict):
    """Scatter plot with regression line"""
    fig = go.Figure()
    
    has_data = False
    
    for ticker, data in data_dict.items():
        merged = data.get('merged_data')
        
        if merged is None or merged.empty or len(merged) < 2:
            continue
        
        # Check if we have valid data
        if merged['sentiment_score'].std() == 0 or merged['return'].std() == 0:
            continue
        
        has_data = True
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=merged['sentiment_score'],
            y=merged['return'] * 100,
            mode='markers',
            name=ticker,
            marker=dict(size=10, opacity=0.7),
            hovertemplate=(
                '<b>%{fullData.name}</b><br>' +
                'Sentiment: %{x:.3f}<br>' +
                'Return: %{y:.2f}%<br>' +
                '<extra></extra>'
            )
        ))
        
        # Regression line
        try:
            z = np.polyfit(merged['sentiment_score'], merged['return'] * 100, 1)
            p = np.poly1d(z)
            x_line = np.linspace(merged['sentiment_score'].min(), merged['sentiment_score'].max(), 100)
            
            fig.add_trace(go.Scatter(
                x=x_line,
                y=p(x_line),
                mode='lines',
                name=f'{ticker} Trend',
                line=dict(dash='dash', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        except:
            pass
    
    if not has_data:
        st.warning("Not enough data points to generate correlation scatter plot. Need at least 2 days with both sentiment and stock data.")
        return None
    
    fig.update_layout(
        title={
            'text': 'Sentiment vs Stock Return Correlation',
            'font': {'size': 24, 'color': '#ffffff'}
        },
        xaxis_title='Daily Sentiment Score',
        yaxis_title='Daily Stock Return (%)',
        template='plotly_dark',
        height=500,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117'
    )
    
    return fig

# Main Streamlit App
def main():
    st.title("📊 Financial News Sentiment Dashboard")
    st.markdown("*AI-powered sentiment analysis for stock market news*")
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Dashboard Controls")
    
    # Ticker selection
    available_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BA', 'JPM', 'V', 'WMT', 'DIS']
    selected_tickers = st.sidebar.multiselect(
        "Select Tickers",
        available_tickers,
        default=['AAPL'],
        help="Choose one or more stock tickers to analyze"
    )
    
    # Date range
    days_back = st.sidebar.slider(
        "Days to Analyze",
        min_value=7,
        max_value=90,
        value=30,
        help="Number of days of historical news to analyze"
    )
    
    # Analyze button
    analyze_button = st.sidebar.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    # Information section
    with st.sidebar.expander("ℹ️ How It Works"):
        st.markdown("""
        **Sentiment Analysis Process:**
        1. Fetches news from Yahoo Finance RSS
        2. Analyzes sentiment using FinBERT AI model
        3. Weights scores by article length & recency
        4. Correlates sentiment with stock returns
        
        **Sentiment Scores:**
        - Positive: > 0.1
        - Neutral: -0.1 to 0.1
        - Negative: < -0.1
        """)
    
    if analyze_button and selected_tickers:
        with st.spinner("🔄 Loading AI sentiment model..."):
            sentiment_pipeline = load_sentiment_model()
        
        if sentiment_pipeline is None:
            st.error("Failed to load sentiment model. Please check your internet connection and try again.")
            return
        
        data_dict = {}
        
        for ticker in selected_tickers:
            with st.spinner(f"📰 Analyzing news for {ticker}..."):
                # Fetch news
                articles = fetch_news(ticker, days_back)
                
                if not articles:
                    st.warning(f"⚠️ No recent articles found for {ticker} in the last {days_back} days")
                    continue
                
                st.success(f"✅ Found {len(articles)} articles for {ticker}")
                
                # Analyze sentiment
                sentiment_results = analyze_sentiment(articles, sentiment_pipeline)
                
                if not sentiment_results:
                    st.warning(f"⚠️ Could not analyze sentiment for {ticker}")
                    continue
                
                # Aggregate daily sentiment
                daily_sentiment = aggregate_daily_sentiment(sentiment_results)
                
                # Fetch stock prices
                start_date = datetime.now() - timedelta(days=days_back + 5)  # Extra buffer
                end_date = datetime.now()
                stock_data = fetch_stock_prices(ticker, start_date, end_date)
                
                # Merge sentiment and stock data
                merged_data = pd.DataFrame()
                if not daily_sentiment.empty and not stock_data.empty:
                    merged_data = pd.merge(
                        daily_sentiment,
                        stock_data[['return']],
                        left_on='date',
                        right_index=True,
                        how='inner'
                    )
                
                # Calculate metrics
                metrics = calculate_metrics(sentiment_results, daily_sentiment, stock_data)
                
                data_dict[ticker] = {
                    'articles': articles,
                    'sentiment_results': sentiment_results,
                    'daily_sentiment': daily_sentiment,
                    'stock_data': stock_data,
                    'merged_data': merged_data,
                    'metrics': metrics
                }
        
        if not data_dict:
            st.error("❌ No data available for selected tickers. Try different tickers or increase the date range.")
            return
        
        # Display metrics
        st.header("📊 Summary Metrics")
        st.markdown('<div class="explanation">Key performance indicators showing overall sentiment analysis results and correlation with stock performance.</div>', unsafe_allow_html=True)
        
        cols = st.columns(len(data_dict))
        
        for idx, (ticker, data) in enumerate(data_dict.items()):
            with cols[idx]:
                st.markdown(f"### {ticker}")
                metrics = data['metrics']
                
                # Total articles
                st.markdown("""
    <style>
    div[data-testid="stMetricValue"] {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)
                st.metric("📰 Total Articles", f"{metrics['total_articles']:,}")
                
                # Average sentiment with color
                avg_sent = metrics['avg_sentiment']
                sentiment_label = "🟢 Positive" if avg_sent > 0.1 else "🔴 Negative" if avg_sent < -0.1 else "🟡 Neutral"
                sentiment_color = "positive" if avg_sent > 0.1 else "negative" if avg_sent < -0.1 else "neutral"
                
                st.markdown(
                    f'<div class="metric-label">Average Sentiment</div>'
                    f'<div class="metric-value {sentiment_color}">{avg_sent:.3f}</div>'
                    f'<div style="color: #888;">{sentiment_label}</div>',
                    unsafe_allow_html=True
                )
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Positive vs negative count
                st.metric(
                    "✅ Positive Articles",
                    f"{metrics['positive_count']}",
                    f"{metrics['positive_ratio']:.1%} of total"
                )
                
                st.metric(
                    "❌ Negative Articles",
                    f"{metrics['negative_count']}",
                    f"{(1-metrics['positive_ratio']):.1%} of total"
                )
                
                # Correlation
                corr = metrics['correlation']
                corr_strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
                st.metric(
                    "📈 Sentiment-Return Correlation",
                    f"{corr:.2f}",
                    f"{corr_strength} relationship"
                )
        
        st.markdown("---")
        
        # Sentiment trend chart
        st.header("📈 Daily Sentiment Trends")
        st.markdown('<div class="explanation">This chart shows how sentiment changes over time. Positive values indicate bullish sentiment, negative values indicate bearish sentiment. Each data point represents the weighted average sentiment for all articles published that day.</div>', unsafe_allow_html=True)
        
        fig_trend = plot_sentiment_trend(data_dict)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Article counts
        st.header("📊 Article Sentiment Distribution")
        st.markdown('<div class="explanation">Stacked bar chart showing the daily volume of positive vs negative articles. Higher bars indicate more news coverage. The color distribution shows sentiment bias for each day.</div>', unsafe_allow_html=True)
        
        fig_counts = plot_article_counts(data_dict)
        st.plotly_chart(fig_counts, use_container_width=True)
        
        # Keyword rankings
        st.header("🔤 Top Keywords Analysis")
        st.markdown('<div class="explanation">Most frequently mentioned words in positive and negative articles, ranked by occurrence. These keywords reveal the main themes driving sentiment in each direction.</div>', unsafe_allow_html=True)
        
        for ticker, data in data_dict.items():
            st.subheader(f"📌 {ticker} - Keyword Rankings")
            display_keyword_rankings(data['sentiment_results'], ticker)
            st.markdown("---")
        
        # Correlation scatter plot
        st.header("🎯 Sentiment-Return Correlation Analysis")
        st.markdown('<div class="explanation">Scatter plot comparing daily sentiment scores with actual stock returns. Each point represents one day. The trend line shows the correlation strength - steeper slopes indicate stronger relationships between sentiment and price movement.</div>', unsafe_allow_html=True)
        
        fig_scatter = plot_sentiment_vs_return(data_dict)
        if fig_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Summary insights
        st.header("💡 Key Insights")
        for ticker, data in data_dict.items():
            metrics = data['metrics']
            
            insight_text = f"""
            **{ticker} Analysis Summary:**
            - Analyzed **{metrics['total_articles']}** articles over the past {days_back} days
            - Overall sentiment is **{"BULLISH 🟢" if metrics['avg_sentiment'] > 0.1 else "BEARISH 🔴" if metrics['avg_sentiment'] < -0.1 else "NEUTRAL 🟡"}** 
              with an average score of **{metrics['avg_sentiment']:.3f}**
            - **{metrics['positive_ratio']:.1%}** of articles are positive
            - Sentiment shows a **{"strong" if abs(metrics['correlation']) > 0.7 else "moderate" if abs(metrics['correlation']) > 0.4 else "weak"}** 
              correlation (**{metrics['correlation']:.2f}**) with stock returns
            """
            
            st.markdown(insight_text)
        
    else:
        # Welcome screen
        st.info("👈 **Get Started:** Select stock tickers from the sidebar and click 'Analyze Sentiment'")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Features
            
            - **Real-time News Analysis** - Latest articles from Yahoo Finance
            - **AI-Powered Sentiment** - FinBERT financial language model
            - **Smart Weighting** - Considers recency and article length
            - **Stock Correlation** - Compares sentiment with actual returns
            - **Interactive Charts** - Hover, zoom, and explore data
            - **Keyword Analysis** - Top words driving sentiment
            """)
        
        with col2:
            st.markdown("""
            ### 📖 How to Use
            
            1. **Select Tickers** - Choose stocks to analyze
            2. **Set Date Range** - Pick analysis period (7-90 days)
            3. **Click Analyze** - AI processes news articles
            4. **Review Results** - Explore visualizations and insights
            5. **Compare Tickers** - Add multiple stocks for comparison
            """)
        
        st.markdown("---")
        st.markdown("""
        ### 🔬 Methodology
        
        This dashboard uses **FinBERT**, a BERT-based model fine-tuned on financial text, to classify news sentiment.
        Each article receives a weighted score based on:
        - **Sentiment classification** (positive/negative/neutral)
        - **Article length** (longer articles weighted higher, up to 2x)
        - **Recency** (newer articles weighted higher)
        
        Daily sentiment scores are then correlated with actual stock returns to measure predictive power.
        """)

if __name__ == "__main__":
    main()