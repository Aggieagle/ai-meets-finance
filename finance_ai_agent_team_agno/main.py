import pandas as pd
import streamlit as st
import yfinance as yf
from duckduckgo_search import DDGS
from transformers import pipeline


@st.cache_resource(show_spinner=False)
def get_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


def ddg_search(query: str, max_results: int = 5) -> pd.DataFrame:
    ddg = DDGS()
    results = list(ddg.text(query, max_results=max_results))
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)[["title", "body", "href"]]


def summarize_notes(text: str) -> str:
    summarizer = get_summarizer()
    return summarizer(text, max_length=200, min_length=60, do_sample=False)[0]["summary_text"]


def ticker_snapshot(ticker: str) -> pd.DataFrame:
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    key_fields = {
        "currentPrice": "Price",
        "previousClose": "Prev Close",
        "dayHigh": "Day High",
        "dayLow": "Day Low",
        "fiftyTwoWeekHigh": "52w High",
        "fiftyTwoWeekLow": "52w Low",
        "marketCap": "Market Cap",
    }
    rows = []
    for key, label in key_fields.items():
        if key in info:
            rows.append({"Metric": label, "Value": info[key]})
    return pd.DataFrame(rows)


st.set_page_config(
    page_title="Financial Analysis Team (Free)",
    page_icon="📊",
    layout="wide",
)

st.title("AI Financial Analysis Team — No API Keys Needed")
st.markdown("Get a quick research bundle combining DuckDuckGo search results, YFinance data, and a free local summarizer.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    st.header("Helper Inputs")
    suggested_ticker = st.text_input("Optional stock ticker (e.g., AAPL)")
    st.info("Uses open data sources only — no Gemini key required.")

if prompt := st.chat_input("What would you like to analyze?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Collecting open web data and summarizing..."):
            news_df = ddg_search(prompt)
            ticker_df = ticker_snapshot(suggested_ticker) if suggested_ticker else pd.DataFrame()

            report_parts = []
            if not news_df.empty:
                top_news = "\n".join(
                    f"- {row.title}: {row.body} ({row.href})" for row in news_df.itertuples()
                )
                report_parts.append(f"Recent findings:\n{top_news}")

            if not ticker_df.empty:
                metrics = ", ".join(f"{row.Metric}: {row.Value}" for row in ticker_df.itertuples())
                report_parts.append(f"Key metrics for {suggested_ticker.upper()}: {metrics}")

            combined_notes = "\n\n".join(report_parts) if report_parts else "No public data found; please try a broader query."
            summary = summarize_notes(combined_notes)

            st.markdown(summary)

            if not news_df.empty:
                st.subheader("Source Snippets")
                st.dataframe(news_df, use_container_width=True)

            if not ticker_df.empty:
                st.subheader("Ticker Snapshot")
                st.dataframe(ticker_df, use_container_width=True)

            st.session_state.messages.append({"role": "assistant", "content": summary})
