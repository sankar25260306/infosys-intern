# Import Streamlit for UI components
import streamlit as st

# Import pandas for handling tabular data
import pandas as pd

# Import datetime for working with dates if needed
from datetime import datetime

# Import backend functions from your multi-agent stock assistant
from multi_agent_stock_assistant import (
    handle_query,       # Handles natural language queries
    history_agent,      # Fetches historical stock data
    analysis_agent,     # Performs technical analysis (SMA, RSI)
    sentiment_agent,    # Performs sentiment analysis
    prediction_agent,   # Generates Buy/Sell/Hold predictions
    report_agent,       # Generates detailed stock report
    comparison_agent    # Compares multiple stocks
)

# ------------------------
# Streamlit UI Setup
# ------------------------
st.set_page_config(
    page_title="📊 Hybrid Multi-Agent Stock Assistant",  # Set the page title
    layout="wide"                                      # Use wide layout
)

# Main title for the app
st.title("📈 Hybrid Multi-Agent Stock Assistant")

# Caption / disclaimer
st.caption("⚠️ Educational prototype — not financial advice.")

# ------------------------
# Sidebar Settings
# ------------------------
st.sidebar.header("🔧 Settings")  # Header in sidebar

# Radio button to choose mode
mode = st.sidebar.radio(
    "Choose Mode",              # Label
    ["Natural Language Query", "Guided Q&A"],  # Options
    index=0                     # Default selection
)

# Horizontal line separator
st.sidebar.markdown("---")

# Sidebar footer / info
st.sidebar.write("Built with 💙 using Streamlit + Alpha Vantage + LangChain")

# ------------------------
# Natural Language Query Mode
# ------------------------
if mode == "Natural Language Query":
    st.subheader("💬 Ask in Natural Language")  # Section header
    
    # Input box for user's query
    query = st.text_input(
        "Type your query (e.g., 'What is the price of TCS?')"
    )

    # Run query button
    if st.button("Run Query", type="primary"):
        if query.strip():  # Ensure query is not empty
            response = handle_query(query)  # Call backend to process query
            st.write(response)  # Display response
        else:
            st.warning("Please enter a query.")  # Warning if empty

# ------------------------
# Guided Q&A Mode
# ------------------------
else:
    st.subheader("🧭 Guided Q&A Mode")  # Section header

    # Input box for stock symbol, default "TCS"
    stock = st.text_input("Enter Stock Symbol", "TCS").upper().strip()  # Convert to uppercase and remove spaces

    # Create 3 columns for buttons
    col1, col2, col3 = st.columns(3)

    # ------------------------ Column 1 ------------------------
    with col1:
        if st.button("💰 Current Price"):  # Button for price
            st.write(handle_query(f"price {stock}"))  # Fetch and display price

    # ------------------------ Column 2 ------------------------
    with col2:
        if st.button("📊 Technical Analysis"):  # Button for technicals
            df = history_agent(stock)  # Get historical data
            ta = analysis_agent(df)    # Compute SMA and RSI
            st.write(f"**SMA(5):** {ta['SMA']:.2f}")  # Show SMA
            st.write(f"**RSI:** {ta['RSI']:.2f}")     # Show RSI
            st.line_chart(df.set_index("Date")["Close"])  # Plot closing price chart

    # ------------------------ Column 3 ------------------------
    with col3:
        if st.button("📰 News & Sentiment"):  # Button for news
            sentiment, headlines = sentiment_agent(stock)  # Get sentiment and headlines
            st.write(f"**Sentiment:** {sentiment}")  # Show sentiment
            for h in headlines:                      # List headlines
                st.markdown(f"- {h}")

    # ------------------------ Second Row: Prediction & Report ------------------------
    c1, c2 = st.columns(2)

    # Column 1: Prediction
    with c1:
        if st.button("🔮 Prediction"):  # Button for prediction
            df = history_agent(stock)           # Get historical data
            ta = analysis_agent(df)             # Compute technicals
            sentiment, _ = sentiment_agent(stock)  # Get sentiment
            last_price = df['Close'].iloc[-1] if not df.empty else 1000  # Last closing price
            pred, conf = prediction_agent(ta, sentiment, last_price)     # Get prediction
            st.write(f"Prediction for {stock}: **{pred}** (Confidence: {conf})")  # Display

    # Column 2: Full Report
    with c2:
        if st.button("📑 Full Report"):  # Button for report
            df = history_agent(stock)           # Historical data
            ta = analysis_agent(df)             # Technical analysis
            sentiment, headlines = sentiment_agent(stock)  # Sentiment
            last_price = df['Close'].iloc[-1] if not df.empty else 1000  # Last price
            pred, conf = prediction_agent(ta, sentiment, last_price)     # Prediction
            report = report_agent(stock, last_price, ta, sentiment, headlines, pred, conf, df)  # Full report
            st.text_area("Stock Report", report, height=300)  # Display in a scrollable text area

    # ------------------------ Stock Comparison ------------------------
    st.markdown("---")
    st.subheader("📈 Compare Multiple Stocks")

    # Input for multiple stock symbols separated by commas
    stock_list = st.text_input("Enter symbols separated by commas", "TCS, INFY").upper().split(",")

    if st.button("Compare Stocks"):  # Button to compare
        dfs = [history_agent(s.strip()) for s in stock_list]  # Fetch historical data for all stocks
        results = comparison_agent(stock_list, dfs)           # Get comparison results
        st.dataframe(pd.DataFrame(results))                  # Display as table
