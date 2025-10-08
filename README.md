# 🧠 Multi-Agent Stock Market Research & Advisory Assistant

An interactive web app that uses a **modular multi-agent system** to deliver comprehensive **stock market insights** for **Indian stocks**. It integrates **natural language processing**, **technical analysis**, **news sentiment analysis**, **buy/sell/hold prediction**, and **comparative stock reports** — all in one seamless interface.

> ⚠️ For educational and prototype use only. Not financial advice.

---

## 🔍 Features

- 🗣️ Natural Language & Button-Based Query Modes  
- 📉 Technical Indicators (SMA, RSI)  
- 📰 News Headlines + Sentiment Analysis  
- 🔮 Stock Prediction (Buy / Sell / Hold)  
- 📑 Auto-Generated Reports  
- 📈 Multi-Stock Comparison  
- 🧩 Modular Multi-Agent Architecture  

---

## 🎯 Objectives

- Provide an intuitive interface for stock exploration  
- Enable free-form and guided user interactions  
- Leverage specialized agents for:  
  - Historical Data Retrieval  
  - Technical Indicator Computation  
  - Sentiment Analysis from Financial News  
  - Predictive Modeling  
  - Full Report Generation  
  - Stock Comparisons  

---

## 🛠️ Prerequisites

### 1. System Requirements

- OS: Windows / Linux / macOS  
- Python: 3.9+  
- PIP (Python Package Manager)  
- Git (for cloning)  
- Internet Connection (for live data via Alpha Vantage)  

### 2. Python Dependencies

Install using:

```bash
pip install -r requirements.txt
```

Libraries:

- `langchain`  
- `alpha_vantage`  
- `pandas`  
- `numpy`  
- `matplotlib`  
- `streamlit`  
- `textblob`  
- `python-dotenv`  

---

## 🚀 Installation & Setup

### Step 1: Get the Code

**Option 1: Download ZIP**  
Unzip and navigate into the project folder.

**Option 2: Clone Repository**

```bash
git clone https://github.com/your-username/project-repo.git
cd project-repo
```

---

### Step 2: Create a Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4: Configure Alpha Vantage API Key

1. Get your free API key from [https://www.alphavantage.co](https://www.alphavantage.co)  
2. Create a `.env` file in the project root:

```env
ALPHA_API_KEY="YOUR_API_KEY_HERE"
```

---

## 🧪 Execution

### Option 1: Run Streamlit Web App

```bash
streamlit run app.py
```

Opens at: `http://localhost:8501`

### Option 2: Run CLI Backend

```bash
python main.py
```

Initializes the agents and starts a console-based interface.

---

## 📘 Usage Guide

### 🧭 1. Guided Q&A Mode (Button-Based)

Interact via on-screen buttons:

- 💰 **Current Price** – Get latest stock price  
- 📊 **Technical Analysis** – View SMA, RSI, charts  
- 📰 **News & Sentiment** – News + sentiment  
- 🔮 **Prediction** – Buy / Sell / Hold  
- 📑 **Full Report** – Technical + Sentiment Report  
- 📈 **Compare Stocks** – Compare two Indian stocks  

**Example Input:**

> Stock Symbol: `TCS`  
> Click: `📊 Technical Analysis`

**Output:**
```
SMA(5): 385.42
RSI: 56.78
[Closing Price Chart]
```

---

### 💬 2. Natural Language Query Mode

Ask in plain English using **Indian stock symbols**:

**Examples:**

- “What is the price of INFY?”  
- “Show technical analysis for RELIANCE”  
- “Compare TCS vs HDFC”  
- “Generate full report for HDFC”  

**Example Input:**

> Compare TCS and INFY stock performance

**Example Output:**

```
📈 Comparison Results:
- TCS: Price = ₹385.42, RSI = 56.78, Prediction = Buy (Confidence: 0.8)
- INFY: Price = ₹1700.65, RSI = 60.12, Prediction = Hold (Confidence: 0.6)
```

---

## 📎 Repository Structure (Optional)

```
project-repo/
│
├── app.py                  # Streamlit Web App
├── main.py                 # CLI Backend
├── agents/                 # Multi-agent logic
├── utils/                  # Helper functions
├── data/                   # Local or cached data
├── .env                    # API key config
├── requirements.txt
└── README.md
```

---

## 🤝 Acknowledgements

- [Alpha Vantage API](https://www.alphavantage.co/)  
- [Streamlit](https://streamlit.io/)  
- [LangChain](https://www.langchain.com/)  

---

## 📌 License

This project is for educational and prototype use only.  
**Not intended for financial decision-making.**
