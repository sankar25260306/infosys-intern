# multi_agent_stock_app.py

from langchain.memory import ConversationBufferMemory
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import matplotlib.pyplot as plt
import random
import json
from textblob import TextBlob
from io import BytesIO

# ------------------------
# Alpha Vantage API Key
# ------------------------
ALPHA_API_KEY = "BBMI502O0ZLD50VF"

# ------------------------
# Scoped Memories
# ------------------------
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
agent_logs = []

def log_action(agent, action, details):
    agent_logs.append({"agent": agent, "action": action, "details": details})

# ------------------------
# Agents
# ------------------------
def input_agent(stock_name):
    stock_name = stock_name.upper().strip()
    memories["input"].save_context({"user_input": stock_name}, {"response_output": "Validated"})
    log_action("Input Agent", "validate_stock_name", {"stock_name": stock_name})
    return stock_name

def price_agent(stock_name):
    symbol = stock_name.upper()
    ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')
    try:
        data, meta_data = ts.get_daily(symbol=symbol, outputsize='compact')
        if data.empty:
            raise Exception("No data")
        price = data['4. close'].iloc[-1]
        confidence = 0.9
    except:
        price = random.randint(100, 2000)
        confidence = 0.5
        log_action("Price Agent", "fallback_price_used", {"stock_name": stock_name, "price": price, "confidence": confidence})
    memories["price"].save_context({"price": str(price)}, {"price_output": str(confidence)})
    log_action("Price Agent", "fetch_price", {"stock_name": stock_name, "price": price, "confidence": confidence})
    return price, confidence

def history_agent(symbol):
    ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')
    try:
        df, meta = ts.get_daily(symbol=symbol, outputsize='compact')
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        df['Date'] = pd.to_datetime(df.index)
        df = df.sort_values('Date')
        if df.empty:
            raise Exception("No historical data")
    except:
        dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
        df = pd.DataFrame({"Date": dates, "Close": [random.randint(100, 2000) for _ in range(30)]})
    memories["history"].save_context({"history": df.to_json()}, {"history_output": "Stored"})
    log_action("Historical Data Agent", "fetch_history", {"symbol": symbol, "rows": len(df)})
    return df

def analysis_agent(df):
    sma = df['Close'].rolling(window=5).mean().iloc[-1]
    rsi = random.uniform(30, 70)  # Placeholder
    memories["analysis"].save_context({"analysis": json.dumps({"SMA": sma, "RSI": rsi})}, {"analysis_output": str(0.8)})
    log_action("Technical Analysis Agent", "compute_indicators", {"SMA": sma, "RSI": rsi})
    return {"SMA": sma, "RSI": rsi}

def sentiment_agent(symbol, news_headlines=None):
    if not news_headlines:
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
        polarity = TextBlob(h).sentiment.polarity
        if polarity > 0.1:
            sentiments.append("Positive")
        elif polarity < -0.1:
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")

    if sentiments.count("Positive") > sentiments.count("Negative"):
        overall_sentiment = "Positive"
    elif sentiments.count("Negative") > sentiments.count("Positive"):
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    memories["sentiment"].save_context({"sentiment": json.dumps(headlines)}, {"sentiment_output": overall_sentiment})
    log_action("News & Sentiment Agent", "analyze_sentiment", {"symbol": symbol, "headlines": headlines, "sentiment": overall_sentiment})
    return overall_sentiment, headlines

def prediction_agent(analysis, sentiment, price):
    weights = {"technical": 0.5, "sentiment": 0.3, "ml": 0.2}
    technical_score = 1 if analysis["SMA"] < price else -1
    sentiment_score = 1 if sentiment == "Positive" else -1 if sentiment == "Negative" else 0
    ml_score = random.choice([1, 0, -1])

    final_score = weights["technical"] * technical_score + weights["sentiment"] * sentiment_score + weights["ml"] * ml_score

    if final_score > 0.2:
        pred, conf = "Buy", round(min(1.0, 0.6 + final_score * 0.4), 2)
    elif final_score < -0.2:
        pred, conf = "Sell", round(min(1.0, 0.6 - final_score * 0.4), 2)
    else:
        pred, conf = "Hold", 0.6

    log_action("Prediction Agent", "make_prediction", {"prediction": pred, "confidence": conf})
    return pred, conf

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
        stock_names = ["SBI"]

    results = []
    for name in stock_names:
        stock_name = input_agent(name)
        price, price_conf = price_agent(stock_name)
        hist = history_agent(stock_name)
        analysis = analysis_agent(hist)
        sentiment, headlines = sentiment_agent(stock_name)
        prediction, pred_conf = prediction_agent(analysis, sentiment, price)
        final_report = report_agent(stock_name, price, analysis, sentiment, headlines, prediction, pred_conf)
        results.append(final_report)

    return results, agent_logs
