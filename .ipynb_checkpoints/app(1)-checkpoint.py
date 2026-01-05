# =============================================
# Parkinson's Disease Screener Web App
# =============================================

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Load the model and features ---
model = joblib.load("parkinsons_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# --- Sidebar Navigation ---
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home / Predict", "📊 Explain Model", "ℹ️ About"])

# --- HOME / PREDICT PAGE ---
if page == "🏠 Home / Predict":
    st.title("🧠 Parkinson's Disease Screener")
    st.write("""
    This simple web app uses a trained Machine Learning model to screen for **Parkinson’s Disease**
    based on measurable **voice features**.

    ⚠️ *For educational use only. Not a medical device.*
    """)

    st.subheader("Enter Voice Features")

    user_input = {}
    for feature in feature_names:
        user_input[feature] = st.number_input(f"{feature}", value=0.0, format="%.6f")

    input_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        try:
            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            if prediction == 1:
                st.error(f"⚠️ The model predicts **Parkinson’s likely** (confidence: {prob:.2f})")
            else:
                st.success(f"✅ The model predicts **No Parkinson’s likely** (confidence: {1 - prob:.2f})")

            st.caption("This prediction is not a medical diagnosis.")
        except Exception as e:
            st.warning(f"Error: {e}")

# --- EXPLAIN MODEL PAGE ---
elif page == "📊 Explain Model":
    st.title("📊 Model Insights")

    st.write("""
    The model used is a **Random Forest Classifier**, trained on voice features like frequency,
    jitter, shimmer, and noise ratios. These features measure how steady or shaky a person’s voice is.
    """)

    # --- Get feature importances from the model ---
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.write("### Top 10 Most Important Features")
    st.bar_chart(importance_df.set_index("Feature").head(10))

    st.info("""
    **Interpretation:**  
    - Features with higher bars contribute more to the model's decision.  
    - For example, high *PPE* and *spread1* values often correlate with vocal instability,
      which is a known symptom of Parkinson’s Disease.
    """)

    # Load feature importance data (you can paste your importance table here)
    importance_data = {
        "Feature": ["PPE", "spread1", "MDVP:Fo(Hz)", "NHR", "Jitter:DDP"],
        "Importance": [0.152, 0.107, 0.064, 0.062, 0.056],
    }
    df_imp = pd.DataFrame(importance_data)
    st.table(df_imp)

    st.info("""
    **Interpretation:**  
    - Higher *PPE* and *spread1* values are often linked to irregularities in the voice,  
      which can indicate Parkinson’s symptoms.  
    - Frequency-based features like *MDVP:Fo(Hz)* show variations in vocal pitch stability.
    """)

# --- ABOUT PAGE ---
elif page == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.write("""
    - **Project:** Machine Learning Semester Project  
    - **Developer:** Joseph Dabuo 🇬🇭  
    - **Goal:** To design a simple, trustworthy ML-powered web app that helps screen
      for Parkinson’s Disease using voice features.  
    - **Frameworks Used:** Streamlit, scikit-learn, pandas, numpy  
    - **Disclaimer:** Educational tool only — not for clinical diagnosis.
    """)

    st.success("Thank you for exploring this project!")