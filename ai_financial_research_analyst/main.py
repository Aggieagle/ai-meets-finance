import tempfile

import pandas as pd
import requests
import streamlit as st
from PIL import Image
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from utils.technical_analysis import analyze_stock


# Streamlit page instructions
st.set_page_config(
    page_title="Financial Analyst",
    page_icon="🕵🏻‍♂️",
    layout="wide"
)

# Function to get earnings call transcripts
def get_earnings_calls(symbol, year, quarter, api_key):
    endpoint = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{symbol}?year={year}&quarter={quarter}&apikey={api_key}"
    response = requests.get(endpoint)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Error fetching earnings call data")
        return None
    
def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary file and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling uploaded file: {e}")
        return None

def save_image_file(uploaded_file):
    """Save uploaded file to a temporary file and return the path."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # Save the Plotly figure as PNG
            uploaded_file.write_image(tmp_file.name)
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling uploaded file: {e}")
        return None

def summarize_text(text: str) -> str:
    summarizer = get_summarizer()
    words = text.split()
    summaries = []
    # Chunk long texts so we do not overwhelm the small offline model
    for i in range(0, len(words), 800):
        chunk = " ".join(words[i : i + 800])
        summary = summarizer(
            chunk,
            max_length=200,
            min_length=60,
            do_sample=False,
        )[0]["summary_text"]
        summaries.append(summary)
    return "\n\n".join(summaries)


def sentiment_breakdown(text: str) -> pd.DataFrame:
    model = get_sentiment_model()
    raw_scores = model(text[:5000])[0]
    rows = []
    for entry in raw_scores:
        rows.append({
            "label": entry["label"].replace("LABEL_", "").title(),
            "score": round(entry["score"], 3)
        })
    return pd.DataFrame(rows)


def caption_image(path: str) -> str:
    captioner = get_image_captioner()
    return captioner(Image.open(path))[0]["generated_text"]


def transcribe_audio(path: str) -> str:
    transcriber = get_transcriber()
    result = transcriber(path)
    return result["text"] if isinstance(result, dict) else result


def fetch_youtube_transcript(video_url: str) -> str:
    """Pull a transcript from YouTube without an API key."""
    video_id = video_url.split("v=")[-1].split("&")[0]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except (TranscriptsDisabled, NoTranscriptFound):
        st.error("Transcript is unavailable for this video.")
        return ""
    except Exception as e:
        st.error(f"Unable to fetch transcript: {e}")
        return ""

    return " ".join([entry["text"] for entry in transcript])


# Streamlit app title
st.title("🕵🏻‍♂️ AI-Powered Financial Research Analyst")

@st.cache_resource(show_spinner=False)
def get_summarizer():
    """Lightweight summarization pipeline that works without API keys."""
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


@st.cache_resource(show_spinner=False)
def get_sentiment_model():
    return pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment", return_all_scores=True)


@st.cache_resource(show_spinner=False)
def get_image_captioner():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


@st.cache_resource(show_spinner=False)
def get_transcriber():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")


# Sidebar for API keys
st.sidebar.header("Optional API Keys")
FMP_API_KEY = st.sidebar.text_input("Enter your FMP API Key", type="password")
st.sidebar.info("Get your FMP API key [here](https://financialmodelingprep.com). Paste a transcript below if you prefer not to use an API key.")

# Tabs for different sections
tabs = ["📞 Earnings Call Analysis", "📸 Image/Chart Analysis", "🎙️ Podcast Analysis", "🎥 Youtube Video Analysis"]
tab1, tab2, tab3, tab4 = st.tabs(tabs)

# Display content based on selected tab
with tab1:
    # Get Earnings Data
    st.subheader("Analyze Earnings Calls")

    col1, col2, col3 = st.columns(3)

    with col1:
        symbol = st.selectbox("Select Ticker", ["ASML","FIVE", "DVN", "PLTR", "TSLA", "NVDA", "MSFT", "AAPL", "META"])
    with col2:
        year = st.selectbox("Select Year", [2024, 2023, 2022])
    with col3:
        quarter = st.selectbox("Select Quarter", ["Q1", "Q2", "Q3", "Q4"])

    manual_transcript = st.text_area(
        "Or paste any transcript text (no API key required)",
        placeholder="Paste earnings call notes or any long-form text to summarize",
        height=200,
    )

    if st.button("Process Earnings Call"):
        if not manual_transcript and not FMP_API_KEY:
            st.error("Please paste a transcript or provide an FMP API key to download one.")
        else:
            with st.spinner('Processing with free local models...'):
                transcript_text = manual_transcript
                if not transcript_text and FMP_API_KEY:
                    transcript = get_earnings_calls(symbol, year, quarter, FMP_API_KEY)
                    if transcript and isinstance(transcript, list):
                        transcript_text = transcript[0].get('content', '')
                    else:
                        st.warning("⚠️ No earnings call transcript found for this company and quarter.")
                        transcript_text = ""

                if transcript_text:
                    summary = summarize_text(transcript_text)
                    st.subheader(f"{quarter} {year} Earnings Call Summary:")
                    st.write(summary)

                    st.subheader("Sentiment Snapshot")
                    st.dataframe(sentiment_breakdown(transcript_text), use_container_width=True)
                else:
                    st.info("Please provide a transcript to analyze.")


with tab2:
    st.subheader("Analyze Images&Charts")
    analysis_type = st.radio(

        "Please select",
        ["Image Analysis", "Technical Chart Analysis"],
        captions=[
            "Get a summary of the key insights",
            "Get a comprehensive technical analysis of the stock chart"
        ]                            
    )

    if analysis_type == "Image Analysis":
        image_file = st.file_uploader("Upload Image")
        if image_file is not None:
            image_path = save_uploaded_file(image_file)
            st.subheader("Image Preview")
            img = Image.open(image_path)
            st.image(img, "Uploaded image")

            if st.button("Process Image"):
                with st.spinner('Captioning with free model...'):
                    st.subheader("Image Analysis:")
                    st.write(caption_image(image_path))

    if analysis_type == "Technical Chart Analysis":
        st.subheader("Basic Settings")

        col1, col2, col3 = st.columns(3)
        with col1: 
            symbol = st.selectbox("Select Ticker", ["ASML","FIVE","DVN","PLTR", "TSLA", "NVDA", "MSFT", "ISRG", "AAPL", "META", "ETH-USD"])
            
            indicators = st.multiselect(
                "Select Technical Indicators",
                options=[
                    "SMA10", "SMA20", "SMA50", "SMA100", "SMA200",
                    "Bollinger Bands"
                ],
                default=["SMA20"],
                help="Choose multiple technical indicators to display on the chart"
            )

        with col2:
            period = st.selectbox("Select Data Range", ["1d","5d","1mo", "3mo", "6mo", "1y", "2y","5y", "ytd", "max"])
        with col3:
            interval = st.selectbox("Select Data Interval", 
                                    ["1h","1d","1wk","1mo", "3mo"]
            )

        # Get technical analysis results
        result = analyze_stock(symbol, period, interval, indicators)

        if result is not None:
            st.plotly_chart(result['figure'], use_container_width=True)

        if symbol is not None:
            if st.button("Start Analysis"):
                with st.spinner('Generating rule-based summary...'):
                    summary_lines = []
                    supports = result.get("support_levels", [])
                    resistances = result.get("resistance_levels", [])

                    if supports:
                        summary_lines.append(f"Recent support areas: {', '.join([f'${level:.2f}' for level in supports])}")
                    if resistances:
                        summary_lines.append(f"Recent resistance areas: {', '.join([f'${level:.2f}' for level in resistances])}")

                    if result.get("double_top"):
                        summary_lines.append("Pattern watch: possible double top detected.")
                    if result.get("double_bottom"):
                        summary_lines.append("Pattern watch: possible double bottom detected.")

                    if not summary_lines:
                        summary_lines.append("No major patterns detected. Review the chart above for indicator context.")

                    st.subheader("Technical Analysis Summary")
                    st.write("\n\n".join(summary_lines))

with tab3:
    st.subheader("Analyze Podcasts")
    audio_file = st.file_uploader("Upload Podcast File (MP3)", type=['mp3'])
    if audio_file is not None:
        audio_path = save_uploaded_file(audio_file)
        st.subheader("Play Podcast Episode")
        st.audio(audio_path)

        if st.button("Process Podcast"):
            with st.spinner('Transcribing and summarizing with free models...'):
                transcript = transcribe_audio(audio_path)
                st.subheader("Podcast Summary:")
                st.write(summarize_text(transcript))

                st.subheader("Sentiment Snapshot")
                st.dataframe(sentiment_breakdown(transcript), use_container_width=True)

with tab4:
    st.subheader("Analyze YT Videos")

    # Input YouTube link
    video_url = st.text_input("Enter YouTube Video URL:")

    if video_url:
        try:
            # Display video
            st.video(video_url)

        except Exception as e:
            st.error(f"An error occurred: {e}")

        if st.button("Process Video"):
            with st.spinner('Summarizing transcript with free model...'):
                transcript = fetch_youtube_transcript(video_url)
                if transcript:
                    st.subheader("Video Summary:")
                    st.write(summarize_text(transcript))

                    st.subheader("Sentiment Snapshot")
                    st.dataframe(sentiment_breakdown(transcript), use_container_width=True)
