# =============================================
# Parkinson's Disease Screener Web App
# =============================================

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import io

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="Parkinson's Disease Screener",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS for professional theme + deep blue sidebar ----------
st.markdown("""
    <style>
    /* General background */
    .stApp {
        background-color: #cce6ff;
        color: #000000;
        font-family: 'Open Sans', sans-serif;
    }
    /* Card styling */
    .stContainer {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
    }
    /* Info and warning boxes */
    .stInfo, .stWarning {
        border-left: 4px solid #007acc;
        background-color: #e6f2ff;
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

# --- Load the model, scaler, and feature names ---
model = joblib.load("parkinsons_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# --- Feature descriptions for tooltips ---
feature_descriptions = {
    "MDVP:Fo(Hz)": "Average vocal fundamental frequency (Hz)",
    "MDVP:Fhi(Hz)": "Maximum vocal fundamental frequency (Hz)",
    "MDVP:Flo(Hz)": "Minimum vocal fundamental frequency (Hz)",
    "MDVP:Jitter(%)": "Cycle-to-cycle pitch variation",
    "MDVP:Jitter(Abs)": "Absolute jitter in Hz",
    "MDVP:RAP": "Relative amplitude perturbation",
    "MDVP:PPQ": "Pitch period variation",
    "Jitter:DDP": "Difference of differences of pitch periods",
    "MDVP:Shimmer": "Amplitude variation",
    "MDVP:Shimmer(dB)": "Shimmer in decibels",
    "Shimmer:APQ3": "Amplitude perturbation quotient 3",
    "Shimmer:APQ5": "Amplitude perturbation quotient 5",
    "MDVP:APQ": "Average perturbation quotient",
    "Shimmer:DDA": "Difference of differences of amplitude",
    "NHR": "Noise-to-harmonics ratio",
    "HNR": "Harmonics-to-noise ratio",
    "RPDE": "Recurrence period density entropy",
    "DFA": "Detrended fluctuation analysis",
    "spread1": "Non-linear feature 1",
    "spread2": "Non-linear feature 2",
    "D2": "Correlation dimension",
    "PPE": "Pitch period entropy"
}

# ---------- Utility: animated circular gauge ----------
def show_animated_gauge(container, percent:int, color:str, label:str):
    percent = max(0, min(100, int(round(percent))))
    for val in range(0, percent + 1):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            number={'valueformat': "d", 'suffix': '%', 'font': {'size': 24}},
            title={'text': label, 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgrey"},
                'bar': {'color': color},
                'bgcolor': "#e6f7ff",
                'steps': [
                    {'range': [0, 50], 'color': "#cce6ff"},
                    {'range': [50, 100], 'color': "#99ccff"}
                ],
                'threshold': {
                    'line': {'color': color, 'width': 4},
                    'thickness': 0.75,
                    'value': val
                }
            }
        ))
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=10), height=320)
        container.plotly_chart(fig, use_container_width=True)
        time.sleep(0.01)

# --- Sidebar Navigation ---
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home / Predict", "📊 Explain Model", "ℹ️ About"])

# --- HOME / PREDICT PAGE ---
if page == "🏠 Home / Predict":
    st.title("🧠 Parkinson's Disease Screener")
    st.info("⚠️ For educational use only. Not a medical diagnostic tool.")

    # --- Input method selection ---
    method = st.radio(
        "Select Input Method:",
        ["📝 Manual Entry", "📁 Upload CSV File(s)"],
        horizontal=True
    )

    input_df = None

    if method == "📁 Upload CSV File(s)":
        st.subheader("Upload CSV File(s)")
        st.markdown("""
        💡 **Instructions:**  
        - You can upload one or more CSV files containing voice features.  
        - Column names can be in any order, but they **must match** the model’s feature names (case-insensitive).  
        - Extra columns will be ignored automatically.  
        - Missing required columns will trigger a warning.
        - You can download the **sample CSV file below** and use it as a guide.
        """)

        # --- Sample CSV file download ---
        sample_data = pd.DataFrame({feature: [0.0] for feature in feature_names})
        csv_buffer = io.StringIO()
        sample_data.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Sample CSV File",
            data=csv_buffer.getvalue(),
            file_name="sample_voice_features.csv",
            mime="text/csv"
        )

        uploaded_files = st.file_uploader("Upload one or more CSV files", type=["csv"], accept_multiple_files=True)
        all_dataframes = []

        for file in uploaded_files:
            try:
                df = pd.read_csv(file)
                df.columns = [c.strip() for c in df.columns]
                df.columns = [c.lower() for c in df.columns]
                fn_lower = [f.lower() for f in feature_names]
                match_dict = {f.lower(): f for f in feature_names if f.lower() in df.columns}

                valid_cols = [match_dict[f.lower()] for f in feature_names if f.lower() in match_dict]
                df_matched = df[[c for c in df.columns if c in fn_lower]].copy()
                df_matched.columns = valid_cols

                missing_cols = [f for f in feature_names if f.lower() not in df.columns]
                if missing_cols:
                    st.warning(f"File '{file.name}' is missing required columns: {missing_cols}")
                    continue
                all_dataframes.append(df_matched)
            except Exception as e:
                st.error(f"Could not process file '{file.name}': {e}")

        if len(all_dataframes) > 0:
            input_df = pd.concat(all_dataframes, ignore_index=True)
            st.write("### Uploaded data preview:")
            st.dataframe(input_df.head())

    else:
        st.subheader("Enter Voice Features Manually")
        col1, col2 = st.columns(2)
        user_input = {}
        for i, feature in enumerate(feature_names):
            column = col1 if i % 2 == 0 else col2
            user_input[feature] = column.number_input(
                f"{feature}",
                value=0.0,
                format="%.6f",
                help=feature_descriptions.get(feature, "No description available")
            )
        input_df = pd.DataFrame([user_input])

    # --- Prediction ---
    if st.button("Predict"):
        if input_df is None or input_df.empty:
            st.warning("No input data available. Please upload a CSV or use manual input.")
        else:
            try:
                input_scaled = scaler.transform(input_df[feature_names])
                predictions = model.predict(input_scaled)
                probabilities = model.predict_proba(input_scaled)[:, 1]

                if len(input_df) > 1:
                    results = []
                    for idx, (pred, prob) in enumerate(zip(predictions, probabilities), start=1):
                        label = "Parkinson’s likely" if pred == 1 else "No Parkinson’s likely"
                        conf = prob if pred == 1 else (1 - prob)
                        results.append({
                            "Row": idx,
                            "Prediction": label,
                            "Confidence": round(float(conf), 4)
                        })
                    results_df = pd.DataFrame(results)
                    st.write("### Batch prediction results")
                    st.dataframe(results_df)

                    for idx, (pred, prob) in enumerate(zip(predictions, probabilities), start=1):
                        st.write(f"#### Row {idx} detail")
                        conf = prob if pred == 1 else (1 - prob)
                        percent = int(round(conf * 100))
                        if pred == 1:
                            st.markdown(f"<h3 style='color:red'>⚠️ Parkinson’s likely (Confidence: {conf:.2f})</h3>", unsafe_allow_html=True)
                            gauge_color = "red"
                        else:
                            st.markdown(f"<h3 style='color:green'>✅ No Parkinson’s likely (Confidence: {conf:.2f})</h3>", unsafe_allow_html=True)
                            gauge_color = "green"
                        gauge_container = st.empty()
                        show_animated_gauge(gauge_container, percent, gauge_color, "Confidence")
                        st.write("---")
                else:
                    pred = int(predictions[0])
                    prob = float(probabilities[0])
                    conf = prob if pred == 1 else (1 - prob)
                    percent = int(round(conf * 100))
                    if pred == 1:
                        st.markdown(f"<h2 style='color:red'>⚠️ Parkinson’s likely (Confidence: {conf:.2f})</h2>", unsafe_allow_html=True)
                        gauge_color = "red"
                    else:
                        st.markdown(f"<h2 style='color:green'>✅ No Parkinson’s likely (Confidence: {conf:.2f})</h2>", unsafe_allow_html=True)
                        gauge_color = "green"
                    gauge_container = st.empty()
                    show_animated_gauge(gauge_container, percent, gauge_color, "Confidence")

                st.caption("This prediction is not a medical diagnosis.")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# --- EXPLAIN MODEL PAGE ---
elif page == "📊 Explain Model":
    st.title("📊 Model Insights")
    st.write("""
    The model used is a **Random Forest Classifier**, trained on voice features like frequency,
    jitter, shimmer, and noise ratios. These features measure how steady or shaky a person’s voice is.
    """)

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    st.write("### Top 10 Most Important Features")
    st.bar_chart(importance_df.set_index("Feature").head(10))

    with st.expander("See Full Feature Importance Table"):
        st.table(importance_df)

    st.info("""
    **Interpretation:**  
    - Features with higher bars contribute more to the model's decision.  
    - For example, high *PPE* and *spread1* values often correlate with vocal instability,
      which is a known symptom of Parkinson’s Disease.  
    - Frequency-based features like *MDVP:Fo(Hz)* show variations in vocal pitch stability.
    """)

# --- ABOUT PAGE ---
elif page == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.write("""
    - **Project:** Machine Learning-Based Detection of Parkinson’s Disease Using Voice Features  
    - **Developer:** Joseph Dabuo 
    - **Goal:** To design a simple, trustworthy ML-powered web app that helps screen
      for Parkinson’s Disease using voice features.  
    - **Frameworks Used:** Streamlit, scikit-learn, pandas, numpy  
    - **Disclaimer:** Educational tool only — not for clinical diagnosis.
    """)

    st.success("Thank you for exploring this project!")