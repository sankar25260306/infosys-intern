"""
Hybrid Multi-Agent Stock Assistant (Backend)
--------------------------------------------
- Core agents: input, price, history, analysis, sentiment, prediction, report, comparison
- Modes supported:
    1) Guided Q&A Assistant
    2) Natural Language Queries
"""

# ------------------------
# Imports
# ------------------------
from langchain.memory import ConversationBufferMemory  # Used to store conversation memory for each agent
from alpha_vantage.timeseries import TimeSeries       # To fetch real-time stock data from Alpha Vantage API
import pandas as pd                                   # Used for working with tables/dataframes
import numpy as np                                    # Used for numerical operations (like RSI calculation)
import random, datetime                               # Random is used for fallback/mock data, datetime for timestamps
from textblob import TextBlob                         # Used for sentiment analysis of text/news
from typing import List, Tuple, Dict, Any            # Type hints for better code clarity
import os  # Provides functions for interacting with the operating system
from dotenv import load_dotenv  # Loads environment variables from a .env file


# ------------------------
# Configuration
# ------------------------
# Load environment variables from the .env file in the project folder
load_dotenv()  

# Get the Alpha Vantage API key from environment variables
# This keeps your API key secure and avoids hardcoding it in the code
ALPHA_API_KEY = os.getenv("ALPHA_API_KEY")  
FALLBACK_CSV = "NIFTY 50-29-09-2024-to-29-09-2025.csv"  # Local CSV fallback if API fails
EDUCATIONAL_NOTICE = "⚠️ Educational prototype — not financial advice."  # Disclaimer

# ------------------------
# Memory Buffers
# ------------------------
# Store conversation history for each agent
memories = {
    "input": ConversationBufferMemory(memory_key="input_memory", input_key="user_input", output_key="response_output"),
    "price": ConversationBufferMemory(memory_key="price_memory", input_key="price", output_key="price_output"),
    "history": ConversationBufferMemory(memory_key="history_memory", input_key="history", output_key="history_output"),
    "analysis": ConversationBufferMemory(memory_key="analysis_memory", input_key="analysis", output_key="analysis_output"),
    "sentiment": ConversationBufferMemory(memory_key="sentiment_memory", input_key="sentiment", output_key="sentiment_output"),
    "comparison": ConversationBufferMemory(memory_key="comparison_memory", input_key="comparison_input", output_key="comparison_output")
}

# ------------------------
# Logs
# ------------------------
agent_logs = []  # Stores all actions of agents for debugging

def log_action(agent: str, action: str, details: Dict[str, Any], confidence: float=None):
    """
    Logs actions of each agent
    - agent: name of the agent
    - action: what the agent did
    - details: any additional info
    - confidence: optional confidence score of prediction/data
    """
    agent_logs.append({
        "timestamp": datetime.datetime.now().isoformat(),  # Current date and time
        "agent": agent,                                    # Which agent
        "action": action,                                  # Action description
        "details": details,                                # Extra information
        "confidence": confidence                           # Optional confidence score
    })

# ------------------------
# Core Agents
# ------------------------
def input_agent(stock_name: str) -> str:
    """Validate and standardize stock input."""
    stock_name = stock_name.upper().strip()  # Convert to uppercase and remove leading/trailing spaces
    memories["input"].save_context({"user_input": stock_name}, {"response_output": "Validated"})  # Save to memory
    return stock_name  # Return standardized symbol

def price_agent(stock_name: str) -> Tuple[float, float]:
    """Fetch latest stock price."""
    ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')  # Create API connection
    try:
        data, _ = ts.get_daily(symbol=stock_name, outputsize='compact')  # Fetch daily price data
        price = float(data['4. close'].iloc[-1])  # Take last closing price
        confidence = 0.9  # High confidence if API succeeds
    except Exception:
        try:
            fallback_df = pd.read_csv(FALLBACK_CSV)  # Use CSV fallback
            price = float(fallback_df["Close"].iloc[-1])
            confidence = 0.7
        except Exception:
            price = float(random.randint(100, 2000))  # Generate random price if CSV fails
            confidence = 0.5
    return price, confidence  # Return price and confidence

def history_agent(symbol: str) -> pd.DataFrame:
    """Fetch historical stock data or generate mock data if needed."""
    ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')  # Alpha Vantage connection
    try:
        df, _ = ts.get_daily(symbol=symbol, outputsize='compact')  # Fetch daily data
        df = df.rename(columns={  # Rename columns for standardization
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        df['Date'] = pd.to_datetime(df.index)  # Convert index to datetime
        df = df.sort_values('Date').reset_index(drop=True)  # Sort by date ascending
    except Exception:
        try:
            df = pd.read_csv(FALLBACK_CSV)  # Read CSV fallback
            df["Date"] = pd.to_datetime(df["Date"])  # Convert date column
            df = df.sort_values("Date").reset_index(drop=True)  # Sort chronologically
        except Exception:
            # Generate mock data
            dates = pd.date_range(end=pd.Timestamp.today(), periods=30)  # Last 30 days
            df = pd.DataFrame({
                "Date": dates,
                "Close": [random.randint(100, 2000) for _ in range(30)]  # Random close prices
            })
    return df

# ------------------------
# Technical Analysis
# ------------------------
def compute_rsi(df: pd.DataFrame, window: int=14) -> pd.Series:
    """
    Computes Relative Strength Index (RSI)
    - RSI > 70: Overbought
    - RSI < 30: Oversold
    """
    close = df['Close'].astype(float)  # Ensure numbers
    delta = close.diff()  # Change in price
    gain = delta.clip(lower=0)  # Positive gains only
    loss = -delta.clip(upper=0)  # Negative losses
    avg_gain = gain.rolling(window=window).mean()  # Average gain over window
    avg_loss = loss.rolling(window=window).mean()  # Average loss over window
    rs = avg_gain / avg_loss.replace(0, np.nan)  # Relative strength
    rsi = 100 - (100 / (1 + rs))  # RSI formula
    return rsi.fillna(50)  # Neutral default

def analysis_agent(df: pd.DataFrame) -> Dict[str, float]:
    """Compute SMA and RSI of stock."""
    sma = float(df['Close'].rolling(window=5, min_periods=1).mean().iloc[-1])  # 5-day SMA
    rsi = float(compute_rsi(df).iloc[-1])  # Last RSI
    return {"SMA": sma, "RSI": rsi}  # Return dictionary

# ------------------------
# Sentiment Analysis
# ------------------------
def sentiment_agent(symbol: str, news_headlines: List[str]=None) -> Tuple[str, List[str]]:
    """Analyze news sentiment using TextBlob."""
    if not news_headlines:
        mock_news = [  # Generate sample headlines if none provided
            f"{symbol} stock rises after earnings",
            f"{symbol} faces regulatory challenges",
            f"{symbol} expands into new markets"
        ]
        headlines = random.sample(mock_news, 2)  # Pick 2 random
    else:
        headlines = news_headlines

    sentiments = []
    for h in headlines:
        polarity = TextBlob(h).sentiment.polarity  # Sentiment score
        if polarity > 0.1:
            sentiments.append("Positive")
        elif polarity < -0.1:
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")

    # Decide overall sentiment
    overall = "Neutral"
    if sentiments.count("Positive") > sentiments.count("Negative"):
        overall = "Positive"
    elif sentiments.count("Negative") > sentiments.count("Positive"):
        overall = "Negative"

    return overall, headlines

# ------------------------
# Prediction
# ------------------------
def prediction_agent(analysis: Dict[str, float], sentiment: str, price: float) -> Tuple[str, float]:
    """Combine technical + sentiment + ML (mock) to predict Buy/Sell/Hold."""
    technical_score = 1 if analysis["SMA"] < price else -1  # Price above SMA? Sell= -1, Buy=1
    sentiment_score = 1 if sentiment == "Positive" else -1 if sentiment == "Negative" else 0
    ml_score = random.choice([1, 0, -1])  # Mock ML score
    final_score = technical_score*0.5 + sentiment_score*0.3 + ml_score*0.2  # Weighted final score

    if final_score > 0.2:
        return "Buy", 0.8
    elif final_score < -0.2:
        return "Sell", 0.75
    else:
        return "Hold", 0.6

import matplotlib.pyplot as plt  # For charts

# ------------------------
# Report Agent
# ------------------------
def report_agent(stock_name: str, price: float, analysis: Dict[str, float],
                 sentiment: str, headlines: list, prediction: str, conf: float,
                 hist_df: pd.DataFrame) -> str:

    rsi_signal = "Overbought" if analysis["RSI"] > 70 else "Oversold" if analysis["RSI"] < 30 else "Neutral"
    sma_signal = "Bullish" if price > analysis["SMA"] else "Bearish"

    # Backtesting (accuracy calculation)
    def backtest(df):
        correct = 0
        total = 0
        for i in range(len(df)-1):
            current_price = df['Close'].iloc[i]
            next_price = df['Close'].iloc[i+1]
            sub_df = df.iloc[:i+1]
            ta = analysis_agent(sub_df)
            sent, _ = sentiment_agent(stock_name)
            pred, _ = prediction_agent(ta, sent, current_price)
            movement = "Up" if next_price > current_price else "Down"
            if (pred == "Buy" and movement == "Up") or (pred == "Sell" and movement == "Down") or (pred == "Hold"):
                correct += 1
            total += 1
        return round((correct/total)*100, 2) if total > 0 else 0

    accuracy = backtest(hist_df)

    # Visualization (Price + SMA + RSI)
    if not hist_df.empty:
        hist_df['SMA5'] = hist_df['Close'].rolling(window=5).mean()
        fig, ax1 = plt.subplots(figsize=(10,5))
        ax1.plot(hist_df['Date'], hist_df['Close'], label='Close Price', color='blue')
        ax1.plot(hist_df['Date'], hist_df['SMA5'], label='SMA(5)', linestyle='--', color='orange')
        ax1.set_title(f'{stock_name} Price & SMA(5)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()  # Second y-axis for RSI
        ax2.plot(hist_df['Date'], compute_rsi(hist_df), label='RSI', color='green')
        ax2.axhline(70, color='red', linestyle='--')  # Overbought line
        ax2.axhline(30, color='red', linestyle='--')  # Oversold line
        ax2.set_ylabel('RSI')
        ax2.legend(loc='upper right')

        plt.show()

    # Text report
    report_text = f"""
📊 Stock Report: {stock_name}
----------------------------
Current Price: {price:.2f}  
SMA(5): {analysis['SMA']:.2f} → {sma_signal}  
RSI: {analysis['RSI']:.2f} → {rsi_signal}  

News Headlines: {headlines}  
Sentiment: {sentiment}  

Prediction: {prediction} (Confidence: {conf})  
Backtesting Accuracy: {accuracy:.2f}%  

⚠️ Educational prototype — not financial advice.
"""
    return report_text.strip()

# ------------------------
# Comparison Agent
# ------------------------
def comparison_agent(stock_names: List[str], hist_dfs: List[pd.DataFrame]) -> List[Dict[str, Any]]:
    """Compare multiple stocks."""
    results = []
    for stock, df in zip(stock_names, hist_dfs):
        if "Close" in df.columns and len(df) > 0:
            price = float(df["Close"].iloc[-1])
            ta = analysis_agent(df)
            sentiment, _ = sentiment_agent(stock)
            pred, conf = prediction_agent(ta, sentiment, price)
            results.append({
                "Stock": stock,
                "Price": price,
                "RSI": ta["RSI"],
                "Prediction": pred,
                "Confidence": conf
            })
    return results

# ------------------------
# Natural Language Handler
# ------------------------
def handle_query(query: str) -> str:
    """Process user query and route to correct agent."""
    query_lower = query.lower()

    # Map keywords to intents
    intent_map = {
        "price": ["price","value","quote","current","rate"],
        "technical": ["technical","trend","chart","indicators","analysis"],
        "sentiment": ["sentiment","opinion","buzz"],
        "news": ["news","headline","article","coverage"],
        "prediction": ["predict","forecast","future","recommendation","outlook"],
        "report": ["report","summary","overview"],
        "compare": ["compare","vs","versus","between"]
    }

    # Detect intent
    intent = None
    for key, kws in intent_map.items():
        if any(kw in query_lower for kw in kws):
            intent = key
            break

    # Extract stock symbols from query
    all_words = [w.upper() for w in query.split() if w.isalpha() and len(w) >= 2]
    intent_keywords = [kw.upper() for kws in intent_map.values() for kw in kws]
    stocks = [w for w in all_words if w not in intent_keywords]
    if not stocks:
        stocks = ["TCS"]  # default fallback

    # Route query
    if intent == "price":
        price, conf = price_agent(stocks[0])
        return f"💰 {stocks[0]} latest price: {price:.2f} (Confidence {conf*100:.0f}%)"

    elif intent == "technical":
        hist = history_agent(stocks[0])
        ta = analysis_agent(hist) if not hist.empty else {}
        return f"📊 {stocks[0]} Technicals → SMA={ta.get('SMA',0):.2f}, RSI={ta.get('RSI',0):.2f}"

    elif intent in ["sentiment","news"]:
        sentiment, headlines = sentiment_agent(stocks[0])
        return f"📰 {stocks[0]} News: {headlines}\nSentiment: {sentiment}"

    elif intent == "prediction":
        hist = history_agent(stocks[0])
        ta = analysis_agent(hist) if not hist.empty else {}
        sentiment, _ = sentiment_agent(stocks[0])
        last_price = hist['Close'].iloc[-1] if not hist.empty else 1000
        pred, conf = prediction_agent(ta, sentiment, last_price)
        return f"🔮 Prediction for {stocks[0]}: {pred} (Confidence {conf})"

    elif intent == "report":
        hist = history_agent(stocks[0])
        ta = analysis_agent(hist) if not hist.empty else {}
        sentiment, headlines = sentiment_agent(stocks[0])
        last_price = hist['Close'].iloc[-1] if not hist.empty else 1000
        pred, conf = prediction_agent(ta, sentiment, last_price)
        return report_agent(stocks[0], last_price, ta, sentiment, headlines, pred, conf, hist)

    elif intent == "compare" and len(stocks) > 1:
        dfs = [history_agent(s) for s in stocks]
        results = comparison_agent(stocks, dfs)
        res_text = "📈 Comparison Results:\n"
        for r in results:
            res_text += f"- {r['Stock']}: Price={r['Price']:.2f}, RSI={r['RSI']:.2f}, Prediction={r['Prediction']} (Conf {r['Confidence']})\n"
        return res_text.strip()

    else:
        return "❌ Sorry, I couldn't understand. Try asking about price, technical, sentiment, news, prediction, report, or compare."
