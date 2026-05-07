# FILE NAME: main.py

import streamlit as st
from disease_detection import disease_detection_page

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AgriSens - Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🌿 AgriSens")

app_mode = st.sidebar.selectbox(
    "Select Page",
    [
        "HOME",
        "DISEASE RECOGNITION"
    ]
)

# =========================
# HOME PAGE
# =========================
if app_mode == "HOME":

    st.markdown(
        """
        <h1 style='text-align:center; color:green;'>
        SMART PLANT DISEASE DETECTION
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # =========================
    # FEATURE CARDS
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.info("🌱 Disease Detection")

    with col2:
        st.info("📄 PDF Report Generation")

    st.write("")

    # =========================
    # ABOUT SECTION
    # =========================
    st.write("""
    ### About

    AgriSens is an AI-powered smart agriculture assistant
    that helps farmers detect plant diseases using
    Deep Learning technology.

    ### Features

    - Plant Disease Detection
    - Confidence Score
    - Symptoms Analysis
    - Causes & Prevention
    - Treatment Suggestions
    - PDF Report Download
    """)

    st.success("System Ready ✅")

# =========================
# DISEASE RECOGNITION PAGE
# =========================
elif app_mode == "DISEASE RECOGNITION":

    disease_detection_page()