# =====================================================
# Parkinson's Disease Screener - Multi-Page Streamlit App
# Main Entry Point: app.py
# =====================================================

import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Parkinson's Disease Screener",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional theme
st.markdown("""
    <style>
    /* General background */
    .stApp {
        background: linear-gradient(135deg, #e6f2ff 0%, #cce6ff 100%);
        color: #000000;
        font-family: 'Open Sans', sans-serif;
    }
    /* Headers */
    h1, h2, h3, h4, h5 {
        color: #003366;
        font-family: 'Open Sans', sans-serif;
    }
    /* Buttons */
    .stButton>button {
        background-color: #007acc;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #005a9e;
    }
    /* Info and warning boxes */
    .stInfo, .stWarning, .stSuccess, .stError {
        border-left: 4px solid #007acc;
        padding: 10px;
        border-radius: 5px;
    }
    /* Deep blue sidebar */
    section[data-testid="stSidebar"] {
        background-color: #003366;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #ffffff !important;
        font-weight: bold;
    }
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "🔮 Predict & Recommend", "📊 Model Insights", "ℹ️ About"],
    key="nav_radio"
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **⚠️ Disclaimer:**  
    This app is for educational purposes only and should NOT be used for medical diagnosis. 
    Always consult with a qualified healthcare professional.
    """
)

# Main content routing
if page == "🏠 Home":
    import pages.home as home_page
    home_page.show()

elif page == "🔮 Predict & Recommend":
    import pages.predictions as pred_page
    pred_page.show()

elif page == "📊 Model Insights":
    import pages.model_insights as insights_page
    insights_page.show()

elif page == "ℹ️ About":
    import pages.about as about_page
    about_page.show()