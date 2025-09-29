# Import Streamlit library for building web apps
import streamlit as st  

# Import pandas for data manipulation and analysis
import pandas as pd  

# Import matplotlib for plotting charts
import matplotlib.pyplot as plt  

# Import random for fallback/random data generation
import random  

# Import BytesIO to handle in-memory image buffers
from io import BytesIO  

# Import FPDF for generating PDF reports
from fpdf import FPDF  

# Import tempfile and os for temporary file handling
import tempfile, os  

# Import workflow function and fallback CSV from another module
from final_multi_agent_stock_with_sentiment import run_workflow, FALLBACK_CSV  

# ------------------------
# Page Config
# ------------------------

# Set Streamlit page title and layout
st.set_page_config(page_title="📈 Multi-Agent Stock Analyzer", layout="wide")  

# Display main title on the web app
st.title("📊 Multi-Agent Stock Market Research & Advisory")  

# ------------------------
# Sidebar Input
# ------------------------

# Display a sidebar header
st.sidebar.header("🔹 Input")  

# Text input widget in sidebar for company names
companies_input = st.sidebar.text_input(
    "Enter company names (comma separated, e.g., SBI, TCS, INFY):"
)  

# Sidebar button to trigger analysis
analyze_button = st.sidebar.button("Analyze")  

# Markdown instruction text displayed on main page
st.markdown(
    "👈 Enter company names in the sidebar. "
    "The system will provide reports, predictions, charts, and you can download a PDF."
)  

# ------------------------
# Helper: Add chart from BytesIO to PDF
# ------------------------

# Function to add a matplotlib chart image to PDF
def add_chart_to_pdf(pdf, img_bytes, title):  
    # Create temporary file for image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:  
        tmpfile.write(img_bytes.getvalue())  # Write bytes to temp file
        tmp_path = tmpfile.name  # Get temp file path

    # Add chart title to PDF
    pdf.set_font("Arial", 'B', 14)  
    pdf.cell(0, 8, title, ln=True)  

    # Insert image into PDF
    pdf.image(tmp_path, w=180)  
    pdf.ln(5)  # Add spacing

    # Remove temporary file
    os.remove(tmp_path)  

# ------------------------
# Analysis & Visualization
# ------------------------

# Run analysis only if analyze button is pressed
if analyze_button:  
    # Check if input is empty
    if not companies_input.strip():  
        st.sidebar.warning("⚠️ Please enter at least one company name.")  # Show warning
    else:  
        # Split input by comma and clean whitespace
        companies = [c.strip() for c in companies_input.split(",") if c.strip()]  

        # Show spinner while workflow is running
        with st.spinner("Running multi-agent analysis... ⏳"):  
            # Run multi-agent workflow and get reports and logs
            reports, logs = run_workflow(companies)  

        st.success("✅ Analysis Complete!")  # Show success message

        # Prepare PDF report
        pdf = FPDF()  
        pdf.set_auto_page_break(auto=True, margin=15)  # Auto page breaks

        # Loop through each company
        for i, company in enumerate(companies):  
            # Display report header
            st.subheader(f"📄 Report: {company.upper()}")  

            # ------------------------
            # Show Report Text
            # ------------------------
            st.code(reports[i], language="text")  # Show report as code block

            # Highlight prediction from report
            pred_line = [line for line in reports[i].splitlines() if "Prediction" in line]  
            if pred_line:  
                pred = pred_line[0].split(":")[1].strip()  # Extract prediction text
                if "Buy" in pred:  
                    st.success(pred)  # Green highlight for Buy
                elif "Sell" in pred:  
                    st.error(pred)  # Red highlight for Sell
                else:  
                    st.warning(pred)  # Yellow highlight for Hold

            # ------------------------
            # Historical Data
            # ------------------------
            try:  
                # Try reading fallback CSV
                df = pd.read_csv(FALLBACK_CSV)  
                df["Date"] = pd.to_datetime(df["Date"])  # Convert Date column
                df = df.sort_values("Date").tail(60)  # Use last 60 days
            except Exception:  
                # Random fallback data if CSV fails
                df = pd.DataFrame({  
                    "Date": pd.date_range(end=pd.Timestamp.today(), periods=30),  
                    "Close": [random.randint(100, 2000) for _ in range(30)]  
                })  

            # Calculate SMA
            df["SMA"] = df["Close"].rolling(window=5).mean()  
            # Generate placeholder RSI
            df["RSI"] = [random.uniform(30, 70) for _ in range(len(df))]  
            # Generate placeholder Sentiment
            df["Sentiment"] = [random.choice([1, 0, -1]) for _ in range(len(df))]  

            # ------------------------
            # Charts
            # ------------------------
            col1, col2, col3 = st.columns(3)  # Divide page into 3 columns

            # Close + SMA Chart
            with col1:  
                st.markdown("**Close & SMA Chart**")  
                fig, ax = plt.subplots(figsize=(5, 3))  
                ax.plot(df["Date"], df["Close"], label="Close")  
                ax.plot(df["Date"], df["SMA"], label="SMA", color="orange")  
                ax.legend()  
                ax.set_xticklabels(df["Date"].dt.strftime("%Y-%m-%d"), rotation=45)  
                img_sma = BytesIO()  
                fig.savefig(img_sma, format="png")  
                img_sma.seek(0)  
                st.pyplot(fig)  
                plt.close(fig)  

            # RSI Chart
            with col2:  
                st.markdown("**RSI Chart**")  
                fig, ax = plt.subplots(figsize=(5, 3))  
                ax.plot(df["Date"], df["RSI"], color="green")  
                ax.axhline(70, color="red", linestyle="--")  # Overbought line
                ax.axhline(30, color="red", linestyle="--")  # Oversold line
                ax.set_xticklabels(df["Date"].dt.strftime("%Y-%m-%d"), rotation=45)  
                img_rsi = BytesIO()  
                fig.savefig(img_rsi, format="png")  
                img_rsi.seek(0)  
                st.pyplot(fig)  
                plt.close(fig)  

            # Sentiment Chart
            with col3:  
                st.markdown("**Sentiment Chart**")  
                fig, ax = plt.subplots(figsize=(5, 3))  
                ax.bar(  
                    df["Date"],  
                    df["Sentiment"],  
                    color=["green" if x == 1 else "red" if x == -1 else "gray" for x in df["Sentiment"]]  
                )  
                ax.set_xticklabels(df["Date"].dt.strftime("%Y-%m-%d"), rotation=45)  
                img_sent = BytesIO()  
                fig.savefig(img_sent, format="png")  
                img_sent.seek(0)  
                st.pyplot(fig)  
                plt.close(fig)  

            # ------------------------
            # Add to PDF
            # ------------------------
            pdf.add_page()  # Start new PDF page
            pdf.set_font("Arial", 'B', 16)  # Title font
            pdf.cell(0, 10, f"Report: {company.upper()}", ln=True)  # Add title
            pdf.set_font("Arial", '', 12)  # Normal font
            pdf.multi_cell(0, 8, reports[i])  # Add report text
            pdf.ln(5)  # Add spacing

            # Add charts to PDF
            add_chart_to_pdf(pdf, img_sma, "Close & SMA Chart")  
            add_chart_to_pdf(pdf, img_rsi, "RSI Chart")  
            add_chart_to_pdf(pdf, img_sent, "Sentiment Chart")  

        # ------------------------
        # Download PDF Button
        # ------------------------
        pdf_bytes = pdf.output(dest="S").encode("latin1")  # Convert PDF to bytes
        st.download_button(  
            label="📥 Download Full Report as PDF",  
            data=pdf_bytes,  
            file_name="Stock_Analysis_Report.pdf",  
            mime="application/pdf"  
        )  

        # ------------------------
        # Show Logs
        # ------------------------
        st.subheader("📝 Agent Logs")  
        st.dataframe(pd.DataFrame(logs))  # Display logs as table

# ------------------------
# Footer
# ------------------------
st.markdown("---\n👨‍💻 Designed by: **Sankar Pandi S** | Multi-Agent Stock Analyzer")  
