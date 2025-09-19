import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
from io import BytesIO
from fpdf import FPDF
import tempfile, os
from final_multi_agent_stock_with_sentiment import run_workflow

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="Multi-Agent Stock Analyzer", layout="wide")
st.title("📈 Multi-Agent Stock Market Research & Advisory")

# ------------------------
# Sidebar Input
# ------------------------
st.sidebar.header("🔹 Input")
companies_input = st.sidebar.text_input(
    "Enter company names (comma separated, e.g., SBI, TCS, INFY):"
)
analyze_button = st.sidebar.button("Analyze")

st.markdown(
    "Enter the company names in the sidebar. "
    "The system will provide reports, predictions, charts, and you can download a PDF."
)

# ------------------------
# Helper: Add chart from BytesIO to PDF
# ------------------------
def add_chart_to_pdf(pdf, img_bytes, title):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        tmpfile.write(img_bytes.getvalue())
        tmp_path = tmpfile.name

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 8, title, ln=True)
    pdf.image(tmp_path, w=180)
    pdf.ln(5)

    os.remove(tmp_path)

# ------------------------
# Analysis & Visualization
# ------------------------
if analyze_button:
    if not companies_input.strip():
        st.sidebar.warning("Please enter at least one company name.")
    else:
        companies = [c.strip() for c in companies_input.split(",") if c.strip()]
        with st.spinner("Running multi-agent analysis... ⏳"):
            reports, logs = run_workflow(companies)
        
        st.success("✅ Analysis Complete!")

        # Prepare PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        for i, company in enumerate(companies):
            st.subheader(f"📄 Report: {company.upper()}")
            st.code(reports[i], language="text")

            # ------------------------
            # Generate mock chart data (replace with real data if available)
            # ------------------------
            df = pd.DataFrame({
                "Date": pd.date_range(end=pd.Timestamp.today(), periods=30),
                "Close": [random.randint(100, 2000) for _ in range(30)]
            })
            df['SMA'] = df['Close'].rolling(window=5).mean()
            df['RSI'] = [random.uniform(30, 70) for _ in range(30)]
            df['Sentiment'] = [random.choice([1, 0, -1]) for _ in range(30)]

            # ------------------------
            # Frontend Charts
            # ------------------------
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Close & SMA Chart**")
                plt.figure(figsize=(5,3))
                plt.plot(df['Date'], df['Close'], label='Close')
                plt.plot(df['Date'], df['SMA'], label='SMA', color='orange')
                plt.xticks(rotation=45)
                plt.legend()
                st.pyplot(plt)
                img_sma = BytesIO()
                plt.savefig(img_sma, format='png')
                plt.close()
                img_sma.seek(0)

            with col2:
                st.markdown("**RSI Chart**")
                plt.figure(figsize=(5,3))
                plt.plot(df['Date'], df['RSI'], label='RSI', color='green')
                plt.axhline(70, color='red', linestyle='--')
                plt.axhline(30, color='red', linestyle='--')
                plt.xticks(rotation=45)
                plt.legend()
                st.pyplot(plt)
                img_rsi = BytesIO()
                plt.savefig(img_rsi, format='png')
                plt.close()
                img_rsi.seek(0)

            with col3:
                st.markdown("**Sentiment Chart**")
                plt.figure(figsize=(5,3))
                plt.bar(
                    df['Date'],
                    df['Sentiment'],
                    color=['green' if x==1 else 'red' if x==-1 else 'gray' for x in df['Sentiment']]
                )
                plt.xticks(rotation=45)
                st.pyplot(plt)
                img_sent = BytesIO()
                plt.savefig(img_sent, format='png')
                plt.close()
                img_sent.seek(0)

            # ------------------------
            # Add Report + Charts to PDF
            # ------------------------
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, f"Report: {company.upper()}", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 8, reports[i])
            pdf.ln(5)

            add_chart_to_pdf(pdf, img_sma, "Close & SMA Chart")
            add_chart_to_pdf(pdf, img_rsi, "RSI Chart")
            add_chart_to_pdf(pdf, img_sent, "Sentiment Chart")

        # ------------------------
        # Download PDF Button
        # ------------------------
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        st.download_button(
            label="📥 Download Full Report as PDF",
            data=pdf_bytes,
            file_name="Stock_Analysis_Report.pdf",
            mime="application/pdf"
        )

# ------------------------
# Footer
# ------------------------
st.markdown("---\nDesigned by: Sankar Pandi S | Multi-Agent Stock Analyzer")
