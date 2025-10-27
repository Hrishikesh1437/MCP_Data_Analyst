# streamlit_app.py
import streamlit as st
import requests
import pandas as pd
from io import StringIO

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="MCP Gemini Data Analyzer", layout="wide")
st.title("🚀 MCP Gemini Data Analysis Dashboard")

st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    # Display basic info
    st.write("### 📂 Uploaded File:", uploaded_file.name)
    
    # Display dataframe preview
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    # Send to FastAPI backend
    with st.spinner("Uploading and analyzing..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{BACKEND_URL}/upload_csv/", files=files)

    if response.ok:
        data = response.json()
        st.success("✅ File successfully analyzed!")
        st.write("### 🧾 Summary Statistics")
        st.json(data["summary"])
    else:
        st.error("Upload failed!")
        st.text(response.text)

    st.markdown("---")
    st.subheader("💬 Ask Gemini a Question About Your Data")

    question = st.text_area("Enter your question:")
    if st.button("Ask Gemini"):
        with st.spinner("Gemini thinking..."):
            files = {"file": uploaded_file.getvalue()}
            payload = {"question": question}
            res = requests.post(f"{BACKEND_URL}/analyze_csv_with_gemini/", files=files, data=payload)

        if res.ok:
            out = res.json()
            st.write("### 🧠 Gemini Insight")
            st.markdown(out["gemini_insight"])
        else:
            st.error("Gemini analysis failed!")
            st.text(res.text)
