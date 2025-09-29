# multi_agent_stock_app.py

# Import memory class from LangChain to store conversation/agent context
from langchain.memory import ConversationBufferMemory

# Import Alpha Vantage API for fetching stock market data
from alpha_vantage.timeseries import TimeSeries

# Import pandas for data manipulation and analysis
import pandas as pd

# Import matplotlib for plotting (if needed later)
import matplotlib.pyplot as plt

# Import random module to generate fallback/mock values
import random

# Import json for storing structured data
import json

# Import TextBlob for sentiment analysis
from textblob import TextBlob

# Import BytesIO for file-like byte streams (used in reports if needed)
from io import BytesIO

# ------------------------
# Alpha Vantage API Key
# ------------------------
ALPHA_API_KEY = "BBMI502O0ZLD50VF"  # Your API key for Alpha Vantage
FALLBACK_CSV = "NIFTY 50-29-09-2024-to-29-09-2025.csv"  # CSV fallback dataset if API fails

# ------------------------
# Scoped Memories
# ------------------------
# Each agent has its own memory to store input/output context
memories = {
    "input": ConversationBufferMemory(memory_key="input_memory", input_key="user_input", output_key="response_output"),
    "price": ConversationBufferMemory(memory_key="price_memory", input_key="price", output_key="price_output"),
    "history": ConversationBufferMemory(memory_key="history_memory", input_key="history", output_key="history_output"),
    "analysis": ConversationBufferMemory(memory_key="analysis_memory", input_key="analysis", output_key="analysis_output"),
    "sentiment": ConversationBufferMemory(memory_key="sentiment_memory", input_key="sentiment", output_key="sentiment_output")
}

# ------------------------
# Structured Logs
# ------------------------
agent_logs = []  # Stores all actions by agents for tracking/debugging

# Function to log each agent's action
def log_action(agent, action, details):
    agent_logs.append({"agent": agent, "action": action, "details": details})

# ------------------------
# Agents
# ------------------------

# Input Agent: validates stock symbol input
def input_agent(stock_name):
    stock_name = stock_name.upper().strip()  # Standardize symbol
    memories["input"].save_context({"user_input": stock_name}, {"response_output": "Validated"})  # Save in memory
    log_action("Input Agent", "validate_stock_name", {"stock_name": stock_name})  # Log action
    return stock_name

# Price Agent: fetches current price, uses fallback CSV or random if API fails
def price_agent(stock_name):
    symbol = stock_name.upper()  # Standardize symbol
    ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')  # Create Alpha Vantage object
    try:
        data, meta_data = ts.get_daily(symbol=symbol, outputsize='compact')  # Fetch daily data
        if data.empty:
            raise Exception("No data")  # Raise if empty
        price = data['4. close'].iloc[-1]  # Get latest closing price
        confidence = 0.9  # High confidence since API worked
        log_action("Price Agent", "api_price_used", {"stock_name": stock_name, "price": price})
    except Exception as e:  # Fallback if API fails
        try:
            fallback_df = pd.read_csv(FALLBACK_CSV)  # Read fallback CSV
            price = fallback_df["Close"].iloc[-1]  # Get latest price from CSV
            confidence = 0.7
            log_action("Price Agent", "fallback_csv_used", {"stock_name": stock_name, "price": price})
        except Exception:  # Final fallback: random price
            price = random.randint(100, 2000)
            confidence = 0.5
            log_action("Price Agent", "fallback_random_used", {"stock_name": stock_name, "price": price})

    memories["price"].save_context({"price": str(price)}, {"price_output": str(confidence)})  # Save in memory
    return price, confidence

# History Agent: fetches historical prices, fallback to CSV or random
def history_agent(symbol):
    ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')  # Alpha Vantage object
    try:
        df, meta = ts.get_daily(symbol=symbol, outputsize='compact')  # Fetch historical data
        df = df.rename(columns={  # Rename columns to standard format
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        df['Date'] = pd.to_datetime(df.index)  # Convert index to datetime
        df = df.sort_values('Date')  # Sort chronologically
        if df.empty:
            raise Exception("No historical data")  # Raise if empty
        log_action("Historical Data Agent", "api_history_used", {"symbol": symbol, "rows": len(df)})
    except Exception as e:
        try:
            fallback_df = pd.read_csv(FALLBACK_CSV)  # Fallback CSV
            fallback_df["Date"] = pd.to_datetime(fallback_df["Date"])
            df = fallback_df.sort_values("Date")
            log_action("Historical Data Agent", "fallback_csv_used", {"symbol": symbol, "rows": len(df)})
        except Exception:
            dates = pd.date_range(end=pd.Timestamp.today(), periods=30)  # Random fallback
            df = pd.DataFrame({"Date": dates, "Close": [random.randint(100, 2000) for _ in range(30)]})
            log_action("Historical Data Agent", "fallback_random_used", {"symbol": symbol})

    memories["history"].save_context({"history": df.to_json()}, {"history_output": "Stored"})
    return df

# Analysis Agent: calculates SMA and placeholder RSI
def analysis_agent(df):
    sma = df['Close'].rolling(window=5).mean().iloc[-1]  # 5-day simple moving average
    rsi = random.uniform(30, 70)  # Placeholder RSI
    memories["analysis"].save_context({"analysis": json.dumps({"SMA": sma, "RSI": rsi})}, {"analysis_output": str(0.8)})
    log_action("Technical Analysis Agent", "compute_indicators", {"SMA": sma, "RSI": rsi})
    return {"SMA": sma, "RSI": rsi}

# Sentiment Agent: analyzes news sentiment
def sentiment_agent(symbol, news_headlines=None):
    if not news_headlines:
        # Use mock headlines if none provided
        mock_news = [
            f"{symbol} stock rises after quarterly earnings",
            f"{symbol} faces regulatory challenges",
            f"{symbol} announces new product line"
        ]
        headlines = random.sample(mock_news, 2)
    else:
        headlines = news_headlines

    sentiments = []
    for h in headlines:
        polarity = TextBlob(h).sentiment.polarity  # Sentiment polarity (-1 to 1)
        if polarity > 0.1:
            sentiments.append("Positive")
        elif polarity < -0.1:
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")

    # Determine overall sentiment
    if sentiments.count("Positive") > sentiments.count("Negative"):
        overall_sentiment = "Positive"
    elif sentiments.count("Negative") > sentiments.count("Positive"):
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    memories["sentiment"].save_context({"sentiment": json.dumps(headlines)}, {"sentiment_output": overall_sentiment})
    log_action("News & Sentiment Agent", "analyze_sentiment", {"symbol": symbol, "headlines": headlines, "sentiment": overall_sentiment})
    return overall_sentiment, headlines

# Prediction Agent: calculates Buy/Hold/Sell
def prediction_agent(analysis, sentiment, price):
    weights = {"technical": 0.5, "sentiment": 0.3, "ml": 0.2}  # Weights for decision
    technical_score = 1 if analysis["SMA"] < price else -1
    sentiment_score = 1 if sentiment == "Positive" else -1 if sentiment == "Negative" else 0
    ml_score = random.choice([1, 0, -1])  # Placeholder ML score

    final_score = weights["technical"] * technical_score + weights["sentiment"] * sentiment_score + weights["ml"] * ml_score

    if final_score > 0.2:
        pred, conf = "Buy", round(min(1.0, 0.6 + final_score * 0.4), 2)
    elif final_score < -0.2:
        pred, conf = "Sell", round(min(1.0, 0.6 - final_score * 0.4), 2)
    else:
        pred, conf = "Hold", 0.6

    log_action("Prediction Agent", "make_prediction", {"prediction": pred, "confidence": conf})
    return pred, conf

# Report Agent: generates summary report
def report_agent(stock_name, price, analysis, sentiment, headlines, prediction, conf):
    report = f"""
Stock: {stock_name}
Current Price: {price}
Technical Analysis: SMA={analysis['SMA']:.2f}, RSI={analysis['RSI']:.2f}
News Headlines: {headlines}
Sentiment: {sentiment}
Prediction: {prediction} (Confidence: {conf})
"""
    log_action("Report Agent", "generate_report", {"stock_name": stock_name})
    return report

# ------------------------
# Workflow
# ------------------------
def run_workflow(stock_names=None):
    if not stock_names:
        stock_names = ["SBI"]  # Default stock if none provided

    results = []
    for name in stock_names:
        stock_name = input_agent(name)  # Validate input
        price, price_conf = price_agent(stock_name)  # Get price with fallback
        hist = history_agent(stock_name)  # Get historical data with fallback
        analysis = analysis_agent(hist)  # Technical analysis
        sentiment, headlines = sentiment_agent(stock_name)  # Sentiment analysis
        prediction, pred_conf = prediction_agent(analysis, sentiment, price)  # Predict Buy/Sell/Hold
        final_report = report_agent(stock_name, price, analysis, sentiment, headlines, prediction, pred_conf)  # Generate report
        results.append(final_report)

    return results, agent_logs  # Return reports and log of agent actions
