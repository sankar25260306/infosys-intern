**Multi-Agent Stock Market Research & Advisory Assistant**
**Overview**
The Multi-Agent Stock Assistant is an interactive web app that leverages a modular
multi-agent system to provide comprehensive stock market insights. It combines natural
language processing, technical analysis, sentiment evaluation, prediction, and reporting into a
seamless user experience.
The app enables users to ask free-form questions about stocks or navigate through guided
query modes with predefined analyses and comparisons.

**Objectives**

● Provide an intuitive interface for stock data exploration.
● Enable both natural language and guided interaction modes.
● Utilize multiple specialized agents for robust financial analysis:
○ Historical data retrieval
○ Technical indicators (SMA, RSI)
○ Sentiment analysis from news
○ Predictive recommendations (Buy/Sell/Hold)
○ Detailed stock reports
○ Multi-stock comparison

Support educational purposes and prototype development (not financial advice)

**Prerequisites**
**1.1 System Requirements**
● Operating System: Windows, Linux, or macOS
● Python: Version 3.9 or higher
● PIP: Python package manager (comes with Python)
● Git: Required for cloning the project repository
● Internet Connection: Required for fetching live stock data from Alpha Vantage
**1.2 Required Python Libraries**
● langchain
● Alpha_vantage
● Pandas
● Numpy
● matplotli
● streamlit
● Textblob
● python-dotenv
Install all dependencies:
pip install -r requirements.txt
**Installation and Setup**
Step 1: Get the Code
Option 1: Download the project ZIP and unzip it
Option 2: Clone the repository:

git clone https://github.com/your-username/project-repo.git cd
project-repo
Step 2: Create and Activate a Virtual Environment
● macOS / Linux: python3 -m venv venv source venv/bin/activate
● Windows:python -m venv venv.\venv\Scripts\activate
Step 3: Install Dependencies
pip install -r requirements.txt
Step 4: Configure Alpha Vantage API Key
● Obtain a free API key from Alpha Vantage.
● Create a .env file in the project root and add:
ALPHA_API_KEY="YOUR_API_KEY_HERE"
Replace "YOUR_API_KEY_HERE" with your actual key.
**Execution**
Option 1: Run the Streamlit Web App
streamlit run app.py
Opens a web interface at http://localhost:8501

Option 2: Run Backend Script Directly
python main.py
Initializes the agents and provides a console-based interface.
**Usage**
4.1 Guided Q&A Mode
● Step-by-step interaction using buttons:

○ 💰 Current Price — Fetch latest price of TCS, INFY,
RELIANCE, etc.
○ 📊 Technical Analysis — SMA, RSI, closing price chart
○ 📰 News & Sentiment — Headlines and sentiment
○ 🔮 Prediction — Buy / Sell / Hold
○ 📑 Full Report — Detailed report with technical indicators
and sentiment
○ 📈 Compare Stocks — Compare multiple Indian stocks

Example Input:
Stock Symbol: TCS
Click: Technical Analysis
Output:
● SMA(5): 385.42
● RSI: 56.78
● Closing price chart
4.2 Natural Language Query Mode
● Ask stock-related questions in plain English using Indian stock
symbols only. Examples:
○ “What is the price of INFY?”
○ “Show technical analysis for RELIANCE”
○ “Compare TCS vs HDFC”
○ “Generate full report for HDFC”

Example Input:
Compare TCS and INFY stock performance

Example Output:
📈 Comparison Results:
- TCS: Price=385.42, RSI=56.78, Prediction=Buy (Conf 0.8)

- INFY: Price=1700.65, RSI=60.12, Prediction=Hold (Conf 0.6)
