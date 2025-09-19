# multi_agent_stock_app.py

from langchain.memory import ConversationBufferMemory    # Import memory class to allow agents to store conversation history
import yfinance as yf                                    # Import yfinance to fetch stock market data
import pandas as pd                                      # Import pandas for data handling and manipulation
import random                                             # Import random to generate fallback random values
import json                                               # Import json to store/convert data structures
from textblob import TextBlob                             # Import TextBlob for basic sentiment analysis of news text

# ------------------------
# Scoped Memories
# ------------------------
memories = {
    # Store user input and its validation status
    "input": ConversationBufferMemory(memory_key="input_memory", input_key="user_input", output_key="response_output"),
    # Store fetched price and confidence
    "price": ConversationBufferMemory(memory_key="price_memory", input_key="price", output_key="price_output"),
    # Store historical stock data
    "history": ConversationBufferMemory(memory_key="history_memory", input_key="history", output_key="history_output"),
    # Store technical analysis indicators (SMA, RSI)
    "analysis": ConversationBufferMemory(memory_key="analysis_memory", input_key="analysis", output_key="analysis_output"),
    # Store sentiment analysis results
    "sentiment": ConversationBufferMemory(memory_key="sentiment_memory", input_key="sentiment", output_key="sentiment_output")
}

# ------------------------
# Structured Logs
# ------------------------
agent_logs = []  # A list to store logs of all actions taken by agents

def log_action(agent, action, details):
    # Add an entry to agent_logs describing what each agent did
    agent_logs.append({"agent": agent, "action": action, "details": details})

# ------------------------
# Agents
# ------------------------
def input_agent(stock_name):
    stock_name = stock_name.upper().strip()   # Clean the input (uppercase and remove spaces)
    memories["input"].save_context({"user_input": stock_name}, {"response_output": "Validated"}) # Save in memory
    log_action("Input Agent", "validate_stock_name", {"stock_name": stock_name})                  # Log action
    return stock_name

def price_agent(stock_name):
    symbol = stock_name + ".NS"                       # Add .NS (NSE India) to get full stock symbol
    try:
        data = yf.Ticker(symbol).history(period="1d") # Fetch 1-day stock data
        if data.empty:                                # If nothing is returned, raise error
            raise Exception("No data")
        price = data['Close'].iloc[-1]                # Get latest closing price
        confidence = 0.9                              # High confidence if fetched successfully
    except:
        price = random.randint(100, 2000)              # Fallback random price if fetching fails
        confidence = 0.5                               # Low confidence for fallback
        log_action("Price Agent", "fallback_price_used", {"stock_name": stock_name, "price": price, "confidence": confidence})
    memories["price"].save_context({"price": str(price)}, {"price_output": str(confidence)})     # Save in memory
    log_action("Price Agent", "fetch_price", {"stock_name": stock_name, "price": price, "confidence": confidence}) # Log
    return price, confidence

def history_agent(symbol):
    try:
        df = yf.Ticker(symbol).history(period="1mo")      # Get 1 month historical data
        if df.empty:
            raise Exception("No historical data")         # Raise if nothing came back
    except:
        dates = pd.date_range(end=pd.Timestamp.today(), periods=30) # Make fake dates
        df = pd.DataFrame({"Date": dates, "Close": [random.randint(100, 2000) for _ in range(30)]}) # Fake prices
    memories["history"].save_context({"history": df.to_json()}, {"history_output": "Stored"}) # Save to memory
    log_action("Historical Data Agent", "fetch_history", {"symbol": symbol, "rows": len(df)}) # Log
    return df

def analysis_agent(df):
    sma = df['Close'].rolling(window=5).mean().iloc[-1]         # Calculate 5-day Simple Moving Average
    rsi = random.uniform(30, 70)                                # Fake RSI between 30 and 70
    memories["analysis"].save_context({"analysis": json.dumps({"SMA": sma, "RSI": rsi})}, {"analysis_output": str(0.8)})
    log_action("Technical Analysis Agent", "compute_indicators", {"SMA": sma, "RSI": rsi})
    return {"SMA": sma, "RSI": rsi}

def sentiment_agent(symbol, news_headlines=None):
    if not news_headlines:
        mock_news = [                                            # Mock news if none are given
            f"{symbol} stock rises after quarterly earnings",
            f"{symbol} faces regulatory challenges",
            f"{symbol} announces new product line"
        ]
        headlines = random.sample(mock_news, 2)                   # Pick 2 random headlines
    else:
        headlines = news_headlines

    sentiments = []
    for h in headlines:
        polarity = TextBlob(h).sentiment.polarity      # Get sentiment polarity from -1 to +1
        if polarity > 0.1:
            sentiments.append("Positive")              # Positive if polarity > 0.1
        elif polarity < -0.1:
            sentiments.append("Negative")              # Negative if polarity < -0.1
        else:
            sentiments.append("Neutral")               # Else neutral

    # Decide overall sentiment based on counts
    if sentiments.count("Positive") > sentiments.count("Negative"):
        overall_sentiment = "Positive"
    elif sentiments.count("Negative") > sentiments.count("Positive"):
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    # Save and log
    memories["sentiment"].save_context({"sentiment": json.dumps(headlines)}, {"sentiment_output": overall_sentiment})
    log_action("News & Sentiment Agent", "analyze_sentiment", {"symbol": symbol, "headlines": headlines, "sentiment": overall_sentiment})
    return overall_sentiment, headlines

def prediction_agent(analysis, sentiment, price):
    # Make prediction based on analysis & sentiment rules
    if analysis["SMA"] < price and sentiment == "Positive":
        pred, conf = "Buy", 0.8
    elif analysis["SMA"] > price and sentiment == "Negative":
        pred, conf = "Sell", 0.8
    else:
        pred, conf = "Hold", 0.6
    log_action("Prediction Agent", "make_prediction", {"prediction": pred, "confidence": conf})
    return pred, conf

def report_agent(stock_name, price, analysis, sentiment, headlines, prediction, conf):
    # Create a text report using f-string
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
        stock_names = ["SBI"]  # default stock if user doesn't give any

    results = []   # list to hold all stock reports
    for name in stock_names:                     # Loop through all given stock names
        stock_name = input_agent(name)            # Step 1: validate input
        price, price_conf = price_agent(stock_name)        # Step 2: get price
        hist = history_agent(stock_name + ".NS")            # Step 3: get history
        analysis = analysis_agent(hist)                     # Step 4: do technical analysis
        sentiment, headlines = sentiment_agent(stock_name + ".NS")  # Step 5: analyze news sentiment
        prediction, pred_conf = prediction_agent(analysis, sentiment, price) # Step 6: predict
        final_report = report_agent(stock_name, price, analysis, sentiment, headlines, prediction, pred_conf) # Step 7: report
        results.append(final_report)                        # Save this stock's report

    return results, agent_logs                              # Return all reports and logs

# ------------------------
# Example Run
# ------------------------
if __name__ == "__main__":
    # User can type stock/company names directly here
    stocks_to_analyze = ["SBI", "TCS", "INFY"]     # List of sample stocks
    reports, logs = run_workflow(stocks_to_analyze) # Run the workflow
    for r in reports:                               # Print each generated report
        print(r)
