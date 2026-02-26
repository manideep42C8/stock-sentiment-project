import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
from datetime import datetime, timedelta
import nltk

# --- NLTK Setup ---
try:
    # Initialize VADER (Sentiment Analysis tool)
    sia = SentimentIntensityAnalyzer()
except LookupError:
    # Download VADER lexicon if not already downloaded
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    
# --- CONFIGURATION (Change these to real finance news sites!) ---
MOCK_NEWS_URLS = [
    'http://feeds.reuters.com/news/wealth', # Example of a reliable, static feed (use a real one!)
    'https://www.cnbc.com/finance/', 
]

# Map common tickers to search terms
TICKER_MAP = {
    'AAPL': ['Apple', 'iPhone', 'Tim Cook'],
    'TSLA': ['Tesla', 'Musk', 'Electric Vehicle'],
    'GOOGL': ['Google', 'Alphabet', 'Search'],
    'AMZN': ['Amazon', 'Cloud', 'AWS']
}

# --- 3. CORE FUNCTIONS ---

@st.cache_data(ttl=600) # Intermediate Feature: Cache the result for 10 minutes (600 seconds)
def scrape_and_analyze_data(urls):
    """Scrapes multiple sources, performs NLP, and returns the combined DataFrame."""
    all_headlines = []
    
    # 3a. Data Acquisition (requests + BeautifulSoup)
    for url in urls:
        try:
            # Send HTTP request and parse content
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for common headline tags. You may need to adjust these.
            headlines = soup.find_all(['h2', 'h3', 'a'], limit=15) 
            
            for element in headlines:
                headline_text = element.text.strip()
                if len(headline_text) > 30 and 'advertisement' not in headline_text.lower():
                    
                    # NLP Analysis and Text Cleaning
                    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', headline_text).lower().strip()
                    polarity_score = sia.polarity_scores(cleaned_text)['compound']
                    
                    # Classification Logic
                    if polarity_score >= 0.05:
                        sentiment = 'Positive'
                    elif polarity_score <= -0.05:
                        sentiment = 'Negative'
                    else:
                        sentiment = 'Neutral'
                        
                    all_headlines.append({
                        'Timestamp': datetime.now() - timedelta(minutes=len(all_headlines)*2), # Mocking time difference
                        'Headline': headline_text,
                        'Source': url.split('//')[-1].split('/')[0],
                        'Sentiment_Score': polarity_score,
                        'Sentiment_Category': sentiment,
                    })
        except Exception as e:
            # Fallback for failed scraping
            st.info(f"Using mock data as live scraping failed for {url.split('//')[-1].split('/')[0]}")
            continue
            
    # Fallback to Mock Data if ALL live data fails
    if not all_headlines:
        all_headlines = [
            {'Timestamp': datetime.now(), 'Headline': 'Apple stock surges on massive iPhone sales forecast.', 'Source': 'MockNews', 'Sentiment_Score': 0.8, 'Sentiment_Category': 'Positive'},
            {'Timestamp': datetime.now(), 'Headline': 'Tesla shares dip amid supply chain constraints.', 'Source': 'MockNews', 'Sentiment_Score': -0.7, 'Sentiment_Category': 'Negative'},
            {'Timestamp': datetime.now() - timedelta(hours=1), 'Headline': 'Google\'s new AI model shows neutral market reaction.', 'Source': 'MockNews', 'Sentiment_Score': 0.1, 'Sentiment_Category': 'Neutral'},
        ]
        
    return pd.DataFrame(all_headlines)


# --- 4. STREAMLIT APP LAYOUT ---

st.set_page_config(layout="wide", page_title="Sentiment Classifier")
st.title("📰 Real-Time Stock Headline Sentiment Classifier")

# --- User Input Sidebar ---
st.sidebar.header("Select Analysis Target")
ticker_options = sorted(list(TICKER_MAP.keys()))
selected_ticker = st.sidebar.selectbox(
    "Choose a Ticker Symbol:",
    options=ticker_options,
    index=0
)

# --- 5. DATA PROCESSING AND FILTERING ---

with st.spinner(f"Acquiring and analyzing news for {selected_ticker}..."):
    # This calls the cached function to get and process data
    df_analyzed = scrape_and_analyze_data(MOCK_NEWS_URLS)

# 5a. Filtering based on User Input (Dynamic Response)
search_keywords = TICKER_MAP.get(selected_ticker, [])
search_regex = '|'.join([re.escape(k) for k in search_keywords])
filtered_df = df_analyzed[
    df_analyzed['Headline'].str.contains(search_regex, case=False, na=False)
].copy()

# --- 6. VISUALIZATION AND REPORTING ---

st.header(f"Sentiment Report for **{selected_ticker}**")

if filtered_df.empty:
    st.warning(f"No recent headlines found for {selected_ticker} matching keywords: {search_keywords}.")
else:
    total_headlines = len(filtered_df)
    sentiment_counts = filtered_df['Sentiment_Category'].value_counts()
    
    overall_mood = sentiment_counts.index[0] if not sentiment_counts.empty else 'Neutral'
    
    # --- Metrics Cards ---
    colA, colB, colC = st.columns(3)
    colA.metric("Total Relevant Headlines", total_headlines)
    colB.metric("Most Frequent Sentiment", overall_mood)
    colC.metric("Negative Headlines (%)", f"{(sentiment_counts.get('Negative', 0) / total_headlines * 100):.1f}%")

    st.divider()
    
    col1, col2 = st.columns([2, 3])

    # 6a. Chart 1: Sentiment Distribution (Pie Chart)
    with col1:
        st.subheader("Sentiment Distribution")
        fig_pie = px.pie(
            sentiment_counts.reset_index(),
            names='Sentiment_Category',
            values='count',
            title='Last 24 Hours Sentiment Breakdown',
            color='Sentiment_Category',
            color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'orange'}
        )
        st.plotly_chart(fig_pie, use_container_width=True) # 

    # 6b. Chart 2: Polarity Trend (Line Chart - Intermediate Time-Series)
    with col2:
        st.subheader("Polarity Trend")
        # Resample data by a 1-hour window to calculate the average polarity for trend
        df_trend = filtered_df.set_index('Timestamp').resample('1H')['Sentiment_Score'].mean().reset_index()
        df_trend.columns = ['Time', 'Avg_Polarity']
        
        fig_line = px.line(
            df_trend.dropna(), 
            x='Time', 
            y='Avg_Polarity', 
            title=f'Hourly Polarity Trend for {selected_ticker}',
            markers=True
        )
        fig_line.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_line, use_container_width=True)


    # 6c. Color-Coded Headlines Table (Raw Data)
    st.subheader("Recent Analyzed Headlines")
    
    # Function for table styling
    def color_sentiment(val):
        if val == 'Positive': return 'background-color: #d4edda; color: #155724'
        elif val == 'Negative': return 'background-color: #f8d7da; color: #721c24'
        else: return 'background-color: #fff3cd; color: #856404'
    
    # Display the table
    st.dataframe(
        filtered_df[['Timestamp', 'Headline', 'Source', 'Sentiment_Category', 'Sentiment_Score']]
        .sort_values('Timestamp', ascending=False)
        .style.applymap(color_sentiment, subset=['Sentiment_Category']),
        use_container_width=True
    )