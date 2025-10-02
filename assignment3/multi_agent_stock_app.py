# ------------------------
# Imports
# ------------------------
from langchain.memory import ConversationBufferMemory  # Memory buffer for storing agent input/output
from alpha_vantage.timeseries import TimeSeries         # Alpha Vantage API to fetch stock prices
import pandas as pd                                    # Data handling and manipulation library
import matplotlib.pyplot as plt                        # Plotting library for charts
import random                                         # Generate random numbers for fallbacks
import json                                           # JSON serialization for storing structured data
import datetime                                       # Handling dates and times
from textblob import TextBlob                          # Sentiment analysis on text
from io import BytesIO                                 # In-memory file handling
import numpy as np                                     # Numerical operations (arrays, math functions)

# ------------------------
# Config
# ------------------------
ALPHA_API_KEY = "BBMI502O0ZLD50VF"                    # API key for Alpha Vantage
FALLBACK_CSV = "NIFTY 50-29-09-2024-to-29-09-2025.csv"  # CSV file fallback if API fails
EDUCATIONAL_NOTICE = "This is an educational prototype, not a live trading advisor."  # Disclaimer

# ------------------------
# Memories & Scratchpad
# ------------------------
memories = {    
    "input": ConversationBufferMemory(memory_key="input_memory", input_key="user_input", output_key="response_output"),  # Stores user input
    "price": ConversationBufferMemory(memory_key="price_memory", input_key="price", output_key="price_output"),  # Stores price data
    "history": ConversationBufferMemory(memory_key="history_memory", input_key="history", output_key="history_output"),  # Stores historical data
    "analysis": ConversationBufferMemory(memory_key="analysis_memory", input_key="analysis", output_key="analysis_output"),  # Stores technical analysis
    "sentiment": ConversationBufferMemory(memory_key="sentiment_memory", input_key="sentiment", output_key="sentiment_output"),  # Stores sentiment
    "comparison": ConversationBufferMemory(memory_key="comparison_memory", input_key="comparison_input", output_key="comparison_output"),  # Stores comparisons
    "scratchpad": ConversationBufferMemory(memory_key="scratchpad_memory", input_key="scratchpad_input", output_key="scratchpad_output")  # Temporary scratchpad
}    

# ------------------------
# Structured JSON Logs
# ------------------------
agent_logs = []  # List to store structured logs of agent actions

def log_action(agent, action, details, confidence=None):    
    """Log an action by an agent with timestamp, details, and optional confidence."""
    agent_logs.append({    # Append dictionary to logs
        "timestamp": datetime.datetime.now().isoformat(),  # Current timestamp
        "agent": agent,    # Agent name
        "action": action,  # Action description
        "details": details,  # Action details
        "confidence": confidence  # Optional confidence score
    })    

# ------------------------
# Input Agent
# ------------------------
def input_agent(stock_name):    
    stock_name = stock_name.upper().strip()  # Convert to uppercase and remove whitespace
    memories["input"].save_context({"user_input": stock_name}, {"response_output": "Validated"})  # Save input to memory
    memories["scratchpad"].save_context({"stock_validated": stock_name}, {"status": "Input Agent done"})  # Save validated stock
    log_action("Input Agent", "validate_stock_name", {"stock_name": stock_name})  # Log the action
    return stock_name  # Return validated stock name

# ------------------------
# Price Agent
# ------------------------
def price_agent(stock_name):    
    ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')  # Initialize Alpha Vantage API
    try:    
        data, _ = ts.get_daily(symbol=stock_name, outputsize='compact')  # Fetch daily stock prices
        if data.empty: raise Exception("No data")  # Raise exception if API returns empty
        price = data['4. close'].iloc[-1]  # Get latest closing price
        confidence = 0.9  # High confidence from API
        log_action("Price Agent", "api_price_used", {"stock_name": stock_name, "price": price}, confidence)  # Log API usage
    except Exception:    
        try:    
            fallback_df = pd.read_csv(FALLBACK_CSV)  # Read CSV fallback
            price = fallback_df["Close"].iloc[-1]  # Last close price from CSV
            confidence = 0.7  # Medium confidence
            log_action("Price Agent", "fallback_csv_used", {"stock_name": stock_name, "price": price}, confidence)  # Log CSV usage
        except Exception:    
            price = random.randint(100, 2000)  # Random fallback price
            confidence = 0.5  # Low confidence
            log_action("Price Agent", "fallback_random_used", {"stock_name": stock_name, "price": price}, confidence)  # Log random fallback
    memories["price"].save_context({"price": str(price)}, {"price_output": str(confidence)})  # Save price to memory
    memories["scratchpad"].save_context({"latest_price": price}, {"confidence": confidence})  # Save latest price to scratchpad
    return price, confidence  # Return price and confidence

# ------------------------
# History Agent
# ------------------------
def history_agent(symbol):    
    ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')  # Initialize Alpha Vantage API
    try:    
        df, _ = ts.get_daily(symbol=symbol, outputsize='compact')  # Fetch historical prices
        df = df.rename(columns={'1. open':'Open','2. high':'High','3. low':'Low','4. close':'Close','5. volume':'Volume'})  # Rename columns
        df['Date'] = pd.to_datetime(df.index)  # Convert index to datetime
        df = df.sort_values('Date')  # Sort by date
        if df.empty: raise Exception("No historical data")  # Raise exception if empty
        log_action("Historical Data Agent", "api_history_used", {"symbol": symbol, "rows": len(df)})  # Log API success
    except Exception:    
        try:    
            fallback_df = pd.read_csv(FALLBACK_CSV)  # Read CSV fallback
            fallback_df["Date"] = pd.to_datetime(fallback_df["Date"])  # Convert Date column
            df = fallback_df.sort_values("Date")  # Sort by date
            log_action("Historical Data Agent", "fallback_csv_used", {"symbol": symbol, "rows": len(df)})  # Log CSV usage
        except Exception:    
            dates = pd.date_range(end=pd.Timestamp.today(), periods=30)  # Generate 30 random dates
            df = pd.DataFrame({"Date": dates, "Close": [random.randint(100, 2000) for _ in range(30)]})  # Random close prices
            log_action("Historical Data Agent", "fallback_random_used", {"symbol": symbol})  # Log random fallback
    memories["history"].save_context({"history": df.to_json()}, {"history_output": "Stored"})  # Save history to memory
    return df  # Return historical dataframe

# ------------------------
# Technical Analysis
# ------------------------
def compute_rsi(df, window=14):    
    delta = df['Close'].diff()  # Price changes
    gain = (delta.where(delta>0,0)).rolling(window).mean()  # Average gains
    loss = (-delta.where(delta<0,0)).rolling(window).mean()  # Average losses
    rs = gain / loss  # Relative strength
    return 100 - (100 / (1 + rs))  # RSI formula

def analysis_agent(df):    
    sma = df['Close'].rolling(window=5).mean().iloc[-1]  # Compute SMA(5)
    rsi_series = compute_rsi(df)  # Compute RSI
    rsi = rsi_series.iloc[-1] if not rsi_series.empty else 50  # Latest RSI or default 50
    memories["analysis"].save_context({"analysis": json.dumps({"SMA": sma, "RSI": rsi})}, {"analysis_output": str(0.8)})  # Save to memory
    memories["scratchpad"].save_context({"SMA": sma, "RSI": rsi}, {"status": "Technical Analysis Done"})  # Save to scratchpad
    log_action("Technical Analysis Agent", "compute_indicators", {"SMA": sma, "RSI": rsi}, 0.8)  # Log analysis
    return {"SMA": sma, "RSI": rsi}  # Return indicators

# ------------------------
# Sentiment Agent
# ------------------------
def sentiment_agent(symbol, news_headlines=None):    
    if not news_headlines:    
        mock_news = [  # Mock news headlines
            f"{symbol} stock rises after quarterly earnings",    
            f"{symbol} faces regulatory challenges",    
            f"{symbol} announces new product line"    
        ]    
        headlines = random.sample(mock_news, 2)  # Pick 2 random headlines
    else:    
        headlines = news_headlines  # Use provided headlines
    
    sentiments = []  # List to store sentiment labels
    for h in headlines:    
        polarity = TextBlob(h).sentiment.polarity  # Compute polarity
        if polarity > 0.1: sentiments.append("Positive")  # Positive sentiment
        elif polarity < -0.1: sentiments.append("Negative")  # Negative sentiment
        else: sentiments.append("Neutral")  # Neutral sentiment
    
    overall_sentiment = "Neutral"  # Default overall sentiment
    if sentiments.count("Positive") > sentiments.count("Negative"): overall_sentiment = "Positive"    
    elif sentiments.count("Negative") > sentiments.count("Positive"): overall_sentiment = "Negative"    
    
    memories["sentiment"].save_context({"sentiment": json.dumps(headlines)}, {"sentiment_output": overall_sentiment})  # Save headlines
    memories["scratchpad"].save_context({"Sentiment": overall_sentiment}, {"status": "Sentiment Analysis Done"})  # Save result
    log_action("Sentiment Agent", "analyze_sentiment", {"symbol": symbol, "headlines": headlines, "sentiment": overall_sentiment}, 0.9)  # Log sentiment
    return overall_sentiment, headlines  # Return sentiment and headlines

# ------------------------
# Prediction Agent
# ------------------------
def prediction_agent(analysis, sentiment, price, conf_tech=0.8, conf_sent=0.9, conf_ml=0.7):    
    weights = {"technical":0.5, "sentiment":0.3, "ml":0.2}  # Weight each signal
    technical_score = 1 if analysis["SMA"] < price else -1  # Buy if SMA < price
    sentiment_score = 1 if sentiment=="Positive" else -1 if sentiment=="Negative" else 0  # Map sentiment
    ml_score = random.choice([1,0,-1])  # Mock ML score
    
    final_score = technical_score*weights["technical"]*conf_tech + \  # Weighted final score
                  sentiment_score*weights["sentiment"]*conf_sent + \
                  ml_score*weights["ml"]*conf_ml
    
    if final_score>0.2: pred, conf = "Buy", round(min(1.0,0.6+final_score*0.4),2)  # Buy signal
    elif final_score<-0.2: pred, conf = "Sell", round(min(1.0,0.6-final_score*0.4),2)  # Sell signal
    else: pred, conf = "Hold", 0.6  # Hold signal
    
    log_action("Prediction Agent", "resolve_conflict",  # Log decision
               {"technical_score": technical_score, "sentiment_score": sentiment_score,    
                "ml_score": ml_score, "final_score": final_score, "prediction": pred})    
    memories["scratchpad"].save_context({"Prediction": pred, "Prediction_Confidence": conf}, {"status": "Prediction Done"})  # Save prediction
    return pred, conf  # Return prediction and confidence

# ------------------------
# Report Agent
# ------------------------
def report_agent(stock_name, price, analysis, sentiment, headlines, prediction, conf, hist_df):    
    plt.figure(figsize=(10,4))  # Set figure size
    plt.plot(hist_df['Date'], hist_df['Close'], label='Close Price')  # Plot close price
    plt.plot(hist_df['Date'], hist_df['Close'].rolling(5).mean(), label='SMA(5)')  # Plot SMA(5)
    plt.title(f"{stock_name} - Price & SMA")  # Title
    plt.xlabel("Date")  # X-axis label
    plt.ylabel("Price")  # Y-axis label
    plt.legend()  # Show legend
    plt.show()  # Display plot
    
    report = f"""  # Create text report
Stock: {stock_name}    
Current Price: {price}    
Technical Analysis: SMA={analysis['SMA']:.2f}, RSI={analysis['RSI']:.2f}    
News Headlines: {headlines}    
Sentiment: {sentiment}    
Prediction: {prediction} (Confidence: {conf})    
{EDUCATIONAL_NOTICE}    
"""    
    log_action("Report Agent", "generate_report", {"stock_name": stock_name})  # Log report generation
    return report  # Return report string

# ------------------------
# Comparison Agent
# ------------------------
def comparison_agent(stock_names):    
    comparison_results = []  # List to store stock comparisons
    scratch = memories["scratchpad"].load_memory_variables()  # Load latest scratchpad variables
    for stock in stock_names:    
        comparison_results.append({  # Append metrics for each stock
            "Stock": stock,    
            "Price": scratch.get("latest_price", None),    
            "SMA": scratch.get("SMA", None),    
            "RSI": scratch.get("RSI", None),    
            "Sentiment": scratch.get("Sentiment", None),    
            "Prediction": scratch.get("Prediction", None),    
            "Confidence": scratch.get("Prediction_Confidence", 0)    
        })    
    
    ranked = sorted(comparison_results, key=lambda x: x["Confidence"], reverse=True)  # Rank by confidence
    memories["comparison"].save_context({"comparison": json.dumps(ranked)}, {"comparison_output": "Stored"})  # Save ranked comparison
    
    print("\n=== Comparative Analysis ===")  # Print header
    for r in ranked:  # Print each stock's metrics
        print(f"{r['Stock']}: Price={r['Price']}, SMA={r['SMA']:.2f}, RSI={r['RSI']:.2f}, "    
              f"Sentiment={r['Sentiment']}, Prediction={r['Prediction']}, Confidence={r['Confidence']}")    
    return ranked  # Return ranked comparison

# ------------------------
# Workflow
# ------------------------
def run_workflow(stock_names=None):    
    if not stock_names:    
        stock_names = ["SBI"]  # Default stock if none provided
    
    results = []  # List to store reports
    for name in stock_names:    
        stock_name = input_agent(name)  # Validate stock
        price, price_conf = price_agent(stock_name)  # Get latest price
        hist = history_agent(stock_name)  # Get historical prices
        analysis = analysis_agent(hist)  # Compute SMA/RSI
        sentiment, headlines = sentiment_agent(stock_name)  # Analyze sentiment
        prediction, pred_conf = prediction_agent(analysis, sentiment, price)  # Generate prediction
        final_report = report_agent(stock_name, price, analysis, sentiment, headlines, prediction, pred_conf, hist)  # Generate report
        results.append(final_report)  # Append report
    return results, agent_logs  # Return all reports and agent logs