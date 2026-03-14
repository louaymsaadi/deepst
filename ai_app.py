import os
import streamlit as st
import pandas as pd
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()




# Helper function to get dataset path
def get_dataset_path():
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the CSV file
    csv_path = os.path.join(current_dir, "data", "customer_reviews.csv")
    # csv_path = os.path.join(current_dir, "..", "..", "data", "customer_reviews.csv")
    return csv_path



# Helper function to clean text
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text



# Initialize the client once
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="Louay & Gemini Thought & Cache", page_icon="🧠", layout="wide")

st.title("🧠 Gemini 3 Flash: Thinking + Caching ---- Louay")

# --- UI Sidebar: Cache Management ---
with st.sidebar:
    st.header("1. Context Caching")
    st.write("Cache large data (docs, code) to save costs on repeated queries.")
    
    # Input for the large context you want to cache
    cache_context = st.text_area("Context to Cache:", height=150, placeholder="Paste a long document here...")
    
    col1, col2 = st.columns(2)
    if col1.button("🚀 Create Cache"):
        if cache_context:
            with st.spinner("Creating cache..."):
                try:
                    # Create the cache (Note: minimum token count applies)
                    new_cache = client.caches.create(
                        model="gemini-3.1-flash-lite-preview",
                        config=types.CreateCachedContentConfig(
                            display_name="streamlit_session_cache",
                            contents=[cache_context],
                            ttl="3600s", # 1 hour
                        )
                    )
                    st.session_state['active_cache_name'] = new_cache.name
                    st.success("Cache Created!")
                except Exception as e:
                    st.error(f"Cache Error: {e}")
        else:
            st.warning("Please enter text to cache.")

    if col2.button("🗑️ Clear Cache"):
        if 'active_cache_name' in st.session_state:
            client.caches.delete(name=st.session_state['active_cache_name'])
            del st.session_state['active_cache_name']
            st.rerun()

    if 'active_cache_name' in st.session_state:
        st.info(f"Active Cache: {st.session_state['active_cache_name']}")

    st.divider()
    st.header("2. Model Configuration")
    model_id = "gemini-3.1-flash-lite-preview" 
    temp = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    
    think_level = st.select_slider(
        "Thinking Level",
        options=["MINIMAL", "LOW", "MEDIUM", "HIGH"],
        value="MEDIUM"
    )

# --- Main UI ---
user_input = st.text_area("Your Question:", "Summarize the key points of the cached context.")

if st.button("Generate"):
    with st.spinner("Processing..."):
        try:
            # 1. Build the generation config
            gen_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=getattr(types.ThinkingLevel, think_level)
                ),
                temperature=temp,
            )

            # 2. If a cache exists, attach it to the config
            if 'active_cache_name' in st.session_state:
                gen_config.cached_content = st.session_state['active_cache_name']

            # 3. Call the API
            response = client.models.generate_content(
                model=model_id,
                contents=user_input,
                config=gen_config
            )

            # 4. Extract parts
            full_thought = ""
            final_answer = ""

            for part in response.candidates[0].content.parts:
                if part.thought:
                    full_thought += part.text
                else:
                    final_answer += part.text

            # 5. Display Results
            if full_thought:
                with st.expander("🔍 View Internal Reasoning"):
                    st.info(full_thought)

            st.subheader("Response")
            st.markdown(final_answer)

            # 6. Show usage metadata (Proof of caching)
            usage = response.usage_metadata
            st.caption(f"📊 Tokens: {usage.total_token_count} | Cached: {usage.cached_content_token_count or 0}")

        except Exception as e:
            st.error(f"An error occurred: {e}")


# Layout two buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Ingest Dataset"):
        try:
            csv_path = get_dataset_path()
            st.session_state["df"] = pd.read_csv(csv_path)
            st.success("Dataset loaded successfully!")
        except FileNotFoundError:
            st.error("Dataset not found. Please check the file path.")

with col2:
    if st.button("🧹 Parse Reviews"):
        if "df" in st.session_state:
            st.session_state["df"]["CLEANED_SUMMARY"] = st.session_state["df"]["SUMMARY"].apply(clean_text)
            st.success("Reviews parsed and cleaned!")
        else:
            st.warning("Please ingest the dataset first.")

# Display the dataset if it exists
if "df" in st.session_state:
    # Product filter dropdown
    st.subheader("🔍 Filter by Product")
    product = st.selectbox("Choose a product", ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique()))
    st.subheader(f"📁 Reviews for {product}")

    if product != "All Products":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    else:
        filtered_df = st.session_state["df"]
    st.dataframe(filtered_df)

    st.subheader("Sentiment Score by Product - chart")
    grouped = st.session_state["df"].groupby(["PRODUCT"])[("SENTIMENT_SCORE")].mean()
    st.bar_chart(grouped)
    st.subheader("Sentiment Score by Product - Line")
    st.line_chart(grouped)
    st.subheader("Sentiment Score by Product - Area chart")
    st.area_chart(grouped)
    st.subheader("Sentiment Score by Product - Scatter chart")
    st.scatter_chart(grouped)