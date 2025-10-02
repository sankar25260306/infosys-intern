"""
multi_agent_stock_assistant.py

Final integrated multi-agent stock assistant:
- Core agents: input, price, history, analysis, sentiment, prediction, report, comparison
- Specialists: Stock Information Specialist, Stock Calculation Specialist
- Conversation router to choose specialist based on intent
- Example usage at the bottom
"""

# ------------------------
# Imports
# ------------------------
from langchain.memory import ConversationBufferMemory  # Memory buffer for storing agent input/output
from alpha_vantage.timeseries import TimeSeries         # Alpha Vantage API to fetch stock prices
import pandas as pd                                    # Data handling and manipulation library
import matplotlib.pyplot as plt                        # Plotting library for charts
import random, json, datetime                           # Random fallback, JSON, date-time handling
from textblob import TextBlob                           # Sentiment analysis
from io import BytesIO                                  # For in-memory chart handling (not fully used here)
import numpy as np                                      # Numerical computations
import mplfinance as mpf                                # For candlestick charts
from typing import List, Tuple, Dict, Any              # Type hints for clarity

# ------------------------
# Config
# ------------------------
ALPHA_API_KEY = "BBMI502O0ZLD50VF"                     # Alpha Vantage API Key
FALLBACK_CSV = "NIFTY 50-29-09-2024-to-29-09-2025.csv" # Fallback CSV if API fails
EDUCATIONAL_NOTICE = "This is an educational prototype, not a live trading advisor."  # Disclaimer

# ------------------------
# Memories
# ------------------------
# Create memory buffers for each agent to store conversation context
memories = {
    "input": ConversationBufferMemory(memory_key="input_memory", input_key="user_input", output_key="response_output"),
    "price": ConversationBufferMemory(memory_key="price_memory", input_key="price", output_key="price_output"),
    "history": ConversationBufferMemory(memory_key="history_memory", input_key="history", output_key="history_output"),
    "analysis": ConversationBufferMemory(memory_key="analysis_memory", input_key="analysis", output_key="analysis_output"),
    "sentiment": ConversationBufferMemory(memory_key="sentiment_memory", input_key="sentiment", output_key="sentiment_output"),
    "comparison": ConversationBufferMemory(memory_key="comparison_memory", input_key="comparison_input", output_key="comparison_output"),
    "scratchpad": ConversationBufferMemory(memory_key="scratchpad_memory", input_key="scratchpad_input", output_key="scratchpad_output")
}

# ------------------------
# Logs
# ------------------------
agent_logs = []  # Stores timestamped actions of all agents

def log_action(agent: str, action: str, details: Dict[str, Any], confidence: float=None):
    """Record agent actions with optional confidence level"""
    agent_logs.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "agent": agent,
        "action": action,
        "details": details,
        "confidence": confidence
    })

# ------------------------
# Agents
# ------------------------
def input_agent(stock_name: str) -> str:
    """Validate and standardize stock symbol input"""
    stock_name = stock_name.upper().strip()  # Ensure uppercase and no spaces
    memories["input"].save_context({"user_input": stock_name}, {"response_output": "Validated"})
    memories["scratchpad"].save_context({"stock_validated": stock_name}, {"status": "Input Agent done"})
    log_action("Input Agent", "validate_stock_name", {"stock_name": stock_name})
    return stock_name

def price_agent(stock_name: str) -> Tuple[float, float]:
    """Fetch latest stock price from API, fallback CSV, or random value"""
    ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')
    try:
        # Try Alpha Vantage API first
        data, _ = ts.get_daily(symbol=stock_name, outputsize='compact')
        if data.empty:
            raise Exception("No data from API")
        price = float(data['4. close'].iloc[-1])
        confidence = 0.9
        log_action("Price Agent", "api_price_used", {"stock_name": stock_name, "price": price}, confidence)
    except Exception as e:
        try:
            # Fallback CSV if API fails
            fallback_df = pd.read_csv(FALLBACK_CSV)
            price = float(fallback_df["Close"].iloc[-1])
            confidence = 0.7
            log_action("Price Agent", "fallback_csv_used", {"stock_name": stock_name, "price": price}, confidence)
        except Exception:
            # Last resort: random price
            price = float(random.randint(100, 2000))
            confidence = 0.5
            log_action("Price Agent", "fallback_random_used", {"stock_name": stock_name, "price": price}, confidence)

    memories["price"].save_context({"price": str(price)}, {"price_output": str(confidence)})
    memories["scratchpad"].save_context({"latest_price": price}, {"confidence": confidence})
    return price, confidence

def history_agent(symbol: str) -> pd.DataFrame:
    """Fetch historical OHLC data for stock, fallback to CSV or random data"""
    ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')
    try:
        df, _ = ts.get_daily(symbol=symbol, outputsize='compact')
        df = df.rename(columns={'1. open':'Open','2. high':'High','3. low':'Low','4. close':'Close','5. volume':'Volume'})
        df['Date'] = pd.to_datetime(df.index)
        df = df.sort_values('Date').reset_index(drop=True)
        if df.empty:
            raise Exception("No historical data")
        log_action("Historical Data Agent", "api_history_used", {"symbol": symbol, "rows": len(df)})
    except Exception:
        try:
            fallback_df = pd.read_csv(FALLBACK_CSV)
            fallback_df["Date"] = pd.to_datetime(fallback_df["Date"])
            df = fallback_df.sort_values("Date").reset_index(drop=True)
            log_action("Historical Data Agent", "fallback_csv_used", {"symbol": symbol, "rows": len(df)})
        except Exception:
            # Fallback: generate random data for last 30 days
            dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
            df = pd.DataFrame({"Date": dates, "Close": [random.randint(100, 2000) for _ in range(30)]})
            log_action("Historical Data Agent", "fallback_random_used", {"symbol": symbol})

    memories["history"].save_context({"history": df.to_json(orient="records", date_format="iso")}, {"history_output": "Stored"})
    return df

def compute_rsi(df: pd.DataFrame, window: int=14) -> pd.Series:
    """Compute Relative Strength Index (RSI) for the stock"""
    close = df['Close'].astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # Neutral where RSI cannot be computed
    return rsi

def analysis_agent(df: pd.DataFrame) -> Dict[str, float]:
    """Compute technical indicators like SMA and RSI"""
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column for analysis")
    sma = float(df['Close'].rolling(window=5, min_periods=1).mean().iloc[-1])
    rsi_series = compute_rsi(df)
    rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
    memories["analysis"].save_context({"analysis": json.dumps({"SMA": sma, "RSI": rsi})}, {"analysis_output": str(0.8)})
    memories["scratchpad"].save_context({"SMA": sma, "RSI": rsi}, {"status": "Technical Analysis Done"})
    log_action("Technical Analysis Agent", "compute_indicators", {"SMA": sma, "RSI": rsi}, 0.8)
    return {"SMA": sma, "RSI": rsi}

def sentiment_agent(symbol: str, news_headlines: List[str]=None) -> Tuple[str, List[str]]:
    """Analyze sentiment from news headlines"""
    if not news_headlines:
        # Mock headlines if none provided
        mock_news = [
            f"{symbol} stock rises after quarterly earnings",
            f"{symbol} faces regulatory challenges",
            f"{symbol} announces new product line",
            f"{symbol} expands into new market",
            f"{symbol} sees slowdown in demand"
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

    # Aggregate overall sentiment
    overall_sentiment = "Neutral"
    if sentiments.count("Positive") > sentiments.count("Negative"):
        overall_sentiment = "Positive"
    elif sentiments.count("Negative") > sentiments.count("Positive"):
        overall_sentiment = "Negative"

    memories["sentiment"].save_context({"sentiment": json.dumps(headlines)}, {"sentiment_output": overall_sentiment})
    memories["scratchpad"].save_context({"Sentiment": overall_sentiment}, {"status": "Sentiment Analysis Done"})
    log_action("Sentiment Agent", "analyze_sentiment", {"symbol": symbol, "headlines": headlines, "sentiment": overall_sentiment}, 0.9)
    return overall_sentiment, headlines

def prediction_agent(analysis: Dict[str, float], sentiment: str, price: float, conf_tech: float=0.8, conf_sent: float=0.9, conf_ml: float=0.7) -> Tuple[str, float]:
    """Combine technical + sentiment + ML to predict Buy/Sell/Hold"""
    weights = {"technical":0.5, "sentiment":0.3, "ml":0.2}
    technical_score = 1 if analysis["SMA"] < price else -1
    sentiment_score = 1 if sentiment=="Positive" else -1 if sentiment=="Negative" else 0
    ml_score = random.choice([1,0,-1])  # placeholder for ML model
    final_score = technical_score*weights["technical"]*conf_tech + \
                  sentiment_score*weights["sentiment"]*conf_sent + \
                  ml_score*weights["ml"]*conf_ml

    # Map final_score to prediction
    if final_score > 0.2:
        pred, conf = "Buy", round(min(1.0, 0.6 + final_score * 0.4), 2)
    elif final_score < -0.2:
        pred, conf = "Sell", round(min(1.0, 0.6 - final_score * 0.4), 2)
    else:
        pred, conf = "Hold", 0.6

    log_action("Prediction Agent", "resolve_conflict",
               {"technical_score": technical_score, "sentiment_score": sentiment_score,
                "ml_score": ml_score, "final_score": final_score, "prediction": pred})
    memories["scratchpad"].save_context({"Prediction": pred, "Prediction_Confidence": conf}, {"status": "Prediction Done"})
    return pred, conf

# ------------------------
# Report Agent (Upgraded)
# ------------------------
def report_agent(stock_name: str, price: float, analysis: Dict[str, float], sentiment: str, headlines: List[str], prediction: str, conf: float, hist_df: pd.DataFrame) -> str:
    """Generate textual + visual report for stock"""
    try:
        df_plot = hist_df.copy()
        if set(['Open','High','Low','Close']).issubset(df_plot.columns):
            df_plot = df_plot.set_index("Date")
            mpf.plot(df_plot, type="candle", mav=(5,), volume=False, show_nontrading=False, title=f"{stock_name} - Candlestick & SMA(5)")
        else:
            # fallback simple line plot
            plt.figure(figsize=(10,4))
            plt.plot(hist_df["Date"], hist_df["Close"], label="Close")
            plt.title(f"{stock_name} - Close Price")
            plt.xlabel("Date")
            plt.ylabel("Close")
            plt.legend()
            plt.tight_layout()
            plt.show()
    except Exception as e:
        log_action("Report Agent", "plotting_failed", {"error": str(e)})

    # Interpret RSI/SMA
    rsi_value = analysis.get('RSI', 50.0)
    rsi_signal = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
    sma_signal = "Bullish (Price above SMA)" if price > analysis.get('SMA', price) else "Bearish (Price below SMA)"

    # Textual report
    report = f"""
Stock: {stock_name}
Current Price: {price:.2f}
Technical Analysis:
  - SMA(5) = {analysis.get('SMA', 0):.2f} → {sma_signal}
  - RSI = {rsi_value:.2f} → {rsi_signal}
News Headlines: {headlines}
Sentiment: {sentiment}
Prediction: {prediction} (Confidence: {conf})
{EDUCATIONAL_NOTICE}
"""
    log_action("Report Agent", "generate_report", {"stock_name": stock_name, "charts": ["candlestick_or_close","SMA","RSI"]})
    return report

# ------------------------
# Comparison Agent (Upgraded)
# ------------------------
def comparison_agent(stock_names: List[str], hist_dfs: List[pd.DataFrame]) -> List[Dict[str, Any]]:
    """Compare multiple stocks and plot confidence & normalized price trends"""
    comparison_results = []
    scratch = memories["scratchpad"].load_memory_variables()
    for stock, df in zip(stock_names, hist_dfs):
        comparison_results.append({
            "Stock": stock,
            "Price": scratch.get("latest_price", None),
            "SMA": scratch.get("SMA", None),
            "RSI": scratch.get("RSI", None),
            "Sentiment": scratch.get("Sentiment", None),
            "Prediction": scratch.get("Prediction", None),
            "Confidence": scratch.get("Prediction_Confidence", 0)
        })

    # Rank by confidence
    ranked = sorted(comparison_results, key=lambda x: x["Confidence"] or 0, reverse=True)
    memories["comparison"].save_context({"comparison": json.dumps(ranked)}, {"comparison_output": "Stored"})

    # Normalized price chart
    plt.figure(figsize=(10,5))
    for stock, df in zip(stock_names, hist_dfs):
        df2 = df.copy()
        if "Close" in df2.columns and len(df2) > 0:
            df2["Norm"] = df2["Close"].astype(float) / float(df2["Close"].iloc[0]) * 100
            plt.plot(df2["Date"], df2["Norm"], label=stock)
    plt.title("Normalized Price Comparison")
    plt.xlabel("Date")
    plt.ylabel("Index (100=Start)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Confidence bar chart
    plt.figure(figsize=(8,4))
    plt.bar([r["Stock"] for r in ranked], [r["Confidence"] for r in ranked])
    plt.title("Prediction Confidence Comparison")
    plt.ylabel("Confidence")
    plt.tight_layout()
    plt.show()

    return ranked

# ------------------------
# Specialists (Conversation Flows)
# ------------------------
def stock_information_specialist(symbol: str) -> str:
    """Return current price + basic market info for a symbol."""
    stock = input_agent(symbol)
    price, price_conf = price_agent(stock)
    response = f"The current stock price of {stock} is ${price:.2f} (source confidence: {price_conf})."
    log_action("Stock Information Specialist", "respond", {"symbol": stock, "price": price, "confidence": price_conf})
    return response

def stock_calculation_specialist(buy_price: float, current_price: float, shares: int) -> str:
    """Calculate potential profit/loss"""
    pnl = (current_price - buy_price) * shares
    pnl_str = f"You have a potential {'profit' if pnl>=0 else 'loss'} of ${abs(pnl):,.2f} on {shares} shares (Buy: ${buy_price:.2f}, Current: ${current_price:.2f})."
    log_action("Stock Calculation Specialist", "pnl_calculated", {"buy": buy_price, "current": current_price, "shares": shares, "pnl": pnl})
    return pnl_str

# ------------------------
# Conversation Router (Simple)
# ------------------------
def route_conversation(intent: str, payload: Dict[str, Any]) -> str:
    """Route user intent to appropriate specialist or workflow"""
    intent = intent.lower().strip()
    if intent in ("info", "price", "quote"):
        symbol = payload.get("symbol") or payload.get("stock") or "SBI"
        return stock_information_specialist(symbol)
    elif intent in ("pnl", "profit_loss", "calculate_pnl"):
        buy = float(payload.get("buy_price", 0))
        current = float(payload.get("current_price", 0))
        shares = int(payload.get("shares", 0))
        return stock_calculation_specialist(buy, current, shares)
    elif intent in ("workflow", "full"):
        stock_names = payload.get("stock_names", ["SBI"])
        results, logs = run_workflow(stock_names)
        return "\n\n".join(results)
    else:
        return "Sorry — I didn't understand the request. Use intent 'info' or 'pnl' or 'workflow'."

# ------------------------
# Workflow (ties agents together, same as before)
# ------------------------
def run_workflow(stock_names: List[str]=None) -> Tuple[List[str], List[Dict[str,Any]]]:
    """Full multi-agent workflow: price + history + analysis + sentiment + prediction + report"""
    if not stock_names:
        stock_names = ["SBI"]

    results, hist_list = [], []
    for name in stock_names:
        stock_name = input_agent(name)
        price, price_conf = price_agent(stock_name)
        hist = history_agent(stock_name)
        analysis = analysis_agent(hist)
        sentiment, headlines = sentiment_agent(stock_name)