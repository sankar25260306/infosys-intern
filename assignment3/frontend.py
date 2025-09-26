# frontend.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from multi_agent_stock_app import run_workflow, agent_logs

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(page_title="📈 Multi-Agent Stock Market Research & Advisory", layout="wide")
st.title("📊 Multi-Agent Stock Market Research & Advisory Assistant")

# -------------------------
# User Input
# -------------------------
stocks_input = st.text_input("Enter stock symbols (comma-separated):", "SBI,TCS,INFY")
analyze_button = st.button("Run Analysis")

# -------------------------
# Run Workflow
# -------------------------
if analyze_button:
    stock_list = [s.strip().upper() for s in stocks_input.split(",") if s.strip()]
    reports, logs = run_workflow(stock_list)

    st.subheader("📑 Reports")
    for r in reports:
        st.text(r)

    # -------------------------
    # Visualization Section
    # -------------------------
    st.subheader("📈 Visualizations")

    for name in stock_list:
        st.markdown(f"### {name} - Price Trend (1 Month)")

        # Extract historical data from agent_logs
        hist_entry = next((l for l in logs if l["agent"] == "Historical Data Agent" and name in l["action"] or name in str(l["details"])), None)

        if hist_entry:
            try:
                df = pd.read_json(hist_entry["details"]["df_json"])
            except:
                df = None
        else:
            df = None

        if df is not None and not df.empty:
            # Candlestick chart
            fig, ax = plt.subplots(figsize=(8, 4))
            mpf.plot(df.set_index("Date"), type="candle", mav=(5, 10), volume=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning(f"No data available to visualize for {name}")

    # -------------------------
    # Logs Section
    # -------------------------
    st.subheader("🛠️ Agent Logs")
    st.json(agent_logs)