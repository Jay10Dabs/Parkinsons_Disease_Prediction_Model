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
    /* Welcome section styling */
    .welcome-box {
        background: linear-gradient(135deg, #003366 0%, #005599 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Additional CSS for Streamlit Cloud compatibility ----------
st.markdown("""
    <style>
    /* Fix for Streamlit Cloud - Force text colors */
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #000000 !important;
    }
    
    /* Fix feature card text */
    .feature-card p, .feature-card li {
        color: #000000 !important;
    }
    
    /* Fix sidebar text visibility */
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #ffffff !important;
    }
    
    /* Fix radio button labels */
    .stRadio label {
        color: #000000 !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
    }
    
    /* Fix number input labels */
    .stNumberInput label {
        color: #000000 !important;
    }
    
    /* Fix text in info/warning/success boxes */
    .stAlert p, .stAlert li {
        color: #000000 !important;
    }
    
    /* Fix dataframe */
    .stDataFrame {
        color: #000000 !important;
    }
    
    /* Fix caption */
    .stCaptionContainer {
        color: #666666 !important;
    }
    
    /* Ensure expander content is visible */
    .streamlit-expanderContent p,
    .streamlit-expanderContent li {
        color: #000000 !important;
    }
    
    /* Fix table text */
    table {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load the model, scaler, and feature names ---
model = joblib.load("parkinsons_voice_model.pkl")
scaler = joblib.load("voice_feature_scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# --- Identify base features (exclude engineered ones) ---
engineered_features = ["mean_jitter", "mean_shimmer", "HNR_to_NHR"]
base_features = [f for f in feature_names if f not in engineered_features]

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

# ---------- Feature Engineering Function ----------
def engineer_features(df):
    """Apply the same feature engineering as in training"""
    df = df.copy()
    
    # Identify jitter columns
    jitter_cols = [col for col in df.columns if 'jitter' in col.lower()]
    if len(jitter_cols) > 0:
        df["mean_jitter"] = df[jitter_cols].mean(axis=1)
    
    # Identify shimmer columns
    shimmer_cols = [col for col in df.columns if 'shimmer' in col.lower()]
    if len(shimmer_cols) > 0:
        df["mean_shimmer"] = df[shimmer_cols].mean(axis=1)
    
    # Calculate HNR to NHR ratio
    if "HNR" in df.columns and "NHR" in df.columns:
        df["HNR_to_NHR"] = df["HNR"] / (df["NHR"] + 1e-6)
    
    return df

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

# --- Sidebar Navigation with session state ---
st.sidebar.title("🧭 Navigation")

# Initialize session state for page navigation if not exists
if 'page' not in st.session_state:
    st.session_state.page = "🏠 Home"

page = st.sidebar.radio(
    "Go to", 
    ["🏠 Home", "🔬 Predict", "📊 Explain Model", "ℹ️ About"],
    index=["🏠 Home", "🔬 Predict", "📊 Explain Model", "ℹ️ About"].index(st.session_state.page)
)

# Update session state when radio button changes
if page != st.session_state.page:
    st.session_state.page = page

# --- HOME PAGE ---
if page == "🏠 Home":
    # Welcome Section
    st.markdown("""
        <div class="welcome-box">
            <h1>🧠 Welcome to the Parkinson's Disease Voice Screener</h1>
            <p style="font-size: 1.2rem; margin-top: 1rem;">
                An AI-powered tool using voice analysis to assist in early Parkinson's Disease detection
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # About the App Section
    st.markdown("## 🎯 About This Application")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 How It Works</h3>
            <p>This application uses <strong>machine learning</strong> to analyze voice features and predict the likelihood of Parkinson's Disease. The model was trained on acoustic measurements from voice recordings, including:</p>
            <ul>
                <li>Frequency variations (jitter)</li>
                <li>Amplitude variations (shimmer)</li>
                <li>Harmonic and noise ratios</li>
                <li>Non-linear complexity measures</li>
            </ul>
            <p>The system achieves high accuracy by analyzing 22 distinct voice features simultaneously.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>✨ Key Features</h3>
            <ul>
                <li><strong>Manual Entry:</strong> Input voice features individually</li>
                <li><strong>Batch Processing:</strong> Upload CSV files for multiple predictions</li>
                <li><strong>Real-time Analysis:</strong> Instant prediction with confidence scores</li>
                <li><strong>Model Transparency:</strong> View feature importance and model insights</li>
                <li><strong>Educational Tool:</strong> Learn about Parkinson's Disease indicators</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # About Parkinson's Disease
    st.markdown("## 🧬 Understanding Parkinson's Disease")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📋 What is Parkinson's?</h4>
            <p>Parkinson's Disease (PD) is a progressive neurodegenerative disorder that affects movement control. It occurs when nerve cells in the brain don't produce enough dopamine, a chemical that coordinates movement.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>⚠️ Common Symptoms</h4>
            <ul style="font-size: 0.9rem;">
                <li>Tremor (shaking)</li>
                <li>Slowed movement</li>
                <li>Rigid muscles</li>
                <li>Impaired balance</li>
                <li><strong>Voice changes</strong></li>
                <li>Writing changes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🎤 Voice & Parkinson's</h4>
            <p>Voice changes are often early indicators of PD. Patients may experience:</p>
            <ul style="font-size: 0.9rem;">
                <li>Softer speech</li>
                <li>Monotone voice</li>
                <li>Hoarseness</li>
                <li>Breathiness</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Why Voice Analysis
    st.markdown("## 🔬 Why Voice Analysis?")
    st.markdown("""
    <div class="feature-card">
        <p>Voice analysis offers several advantages for Parkinson's Disease screening:</p>
        <ul>
            <li><strong>Non-invasive:</strong> No physical examination or blood tests required</li>
            <li><strong>Early Detection:</strong> Voice changes can appear before motor symptoms</li>
            <li><strong>Objective Measurement:</strong> Quantifiable data reduces subjective bias</li>
            <li><strong>Accessible:</strong> Can be performed remotely with proper equipment</li>
            <li><strong>Cost-effective:</strong> Lower cost compared to traditional diagnostic methods</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Video Section
    st.markdown("## 🎥 Learn More About Parkinson's Disease")
    st.markdown("""
    <div class="feature-card">
        <p>Watch this short video to learn more about Parkinson's Disease:</p>
        <p style="font-size: 0.9rem; color: #666; margin-top: 0.5rem;">Video credit: <strong>ParkinsonNet</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # YouTube video embed
    video_url = "https://www.youtube.com/watch?v=u_tozEV7f4k"  
    st.video(video_url)
    
    # Important Disclaimer
    st.markdown("## ⚠️ Important Disclaimer")
    st.error("""
    **This tool is for educational and research purposes only.**
    
    - This is NOT a medical diagnostic tool
    - Results should NOT replace professional medical consultation
    - Always consult qualified healthcare professionals for diagnosis
    - This tool is designed to support, not replace, clinical judgment
    - False positives and false negatives are possible
    """)
    
    # Call to Action
    st.markdown("---")
    
    # Create columns for centered button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h3>Ready to Get Started?</h3>
            <p>Click the button below to begin your analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Button to navigate to Predict page
        if st.button("🔬 Go to Prediction Page", use_container_width=True):
            st.session_state.page = "🔬 Predict"
            st.rerun()

# --- PREDICT PAGE ---
elif page == "🔬 Predict":
    st.title("🔬 Parkinson's Disease Prediction")
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
        - Upload one or more CSV files containing voice features.  
        - Column names can be in any order, but they **must match** the feature names (case-insensitive).  
        - Extra columns will be ignored automatically.  
        - Missing required columns will trigger a warning.
        - Download the **sample CSV file below** as a template.
        """)

        # --- Sample CSV file download ---
        sample_data = pd.DataFrame({feature: [0.0] for feature in base_features})
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
                bf_lower = [f.lower() for f in base_features]
                match_dict = {f.lower(): f for f in base_features if f.lower() in df.columns}

                valid_cols = [match_dict[f.lower()] for f in base_features if f.lower() in match_dict]
                df_matched = df[[c for c in df.columns if c in bf_lower]].copy()
                df_matched.columns = valid_cols

                missing_cols = [f for f in base_features if f.lower() not in df.columns]
                if missing_cols:
                    st.warning(f"File '{file.name}' is missing required columns: {missing_cols}")
                    continue
                
                # Add file identifier column
                df_matched.insert(0, 'File_Source', file.name)
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
        for i, feature in enumerate(base_features):
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
                # Store file source information before feature engineering
                file_sources = input_df['File_Source'].values if 'File_Source' in input_df.columns else None
                
                # Remove File_Source column for processing
                input_df_features = input_df.drop('File_Source', axis=1) if 'File_Source' in input_df.columns else input_df
                
                # Apply feature engineering
                input_engineered = engineer_features(input_df_features)
                
                # Ensure all required features are present in the correct order
                input_final = input_engineered[feature_names]
                
                # Scale and predict
                input_scaled = scaler.transform(input_final)
                predictions = model.predict(input_scaled)
                probabilities = model.predict_proba(input_scaled)[:, 1]

                if len(input_df) > 1:
                    results = []
                    for idx, (pred, prob) in enumerate(zip(predictions, probabilities), start=1):
                        label = "Parkinson's likely" if pred == 1 else "No Parkinson's likely"
                        conf = prob if pred == 1 else (1 - prob)
                        result_row = {
                            "Row": idx,
                            "Prediction": label,
                            "Confidence": round(float(conf), 4)
                        }
                        # Add file source if available
                        if file_sources is not None:
                            result_row["Source_File"] = file_sources[idx - 1]
                            # Move Source_File to second position
                            result_row = {"Row": result_row["Row"], "Source_File": result_row["Source_File"], 
                                        "Prediction": result_row["Prediction"], "Confidence": result_row["Confidence"]}
                        results.append(result_row)
                    results_df = pd.DataFrame(results)
                    st.write("### Batch prediction results")
                    st.dataframe(results_df)

                    for idx, (pred, prob) in enumerate(zip(predictions, probabilities), start=1):
                        # Show file source in header if available
                        if file_sources is not None:
                            st.write(f"#### Row {idx} - File: {file_sources[idx - 1]}")
                        else:
                            st.write(f"#### Row {idx} detail")
                        
                        conf = prob if pred == 1 else (1 - prob)
                        percent = int(round(conf * 100))
                        if pred == 1:
                            st.markdown(f"<h3 style='color:red'>⚠️ Parkinson's likely (Confidence: {conf:.2f})</h3>", unsafe_allow_html=True)
                            gauge_color = "red"
                        else:
                            st.markdown(f"<h3 style='color:green'>✅ No Parkinson's likely (Confidence: {conf:.2f})</h3>", unsafe_allow_html=True)
                            gauge_color = "green"
                        gauge_container = st.empty()
                        show_animated_gauge(gauge_container, percent, gauge_color, "Confidence")
                        
                        # Recommendations based on prediction
                        st.markdown("### 💡 Recommendations")
                        if pred == 1:
                            st.warning("""
                            **Based on this prediction, we recommend:**
                            
                            - 🏥 **Consult a neurologist** or movement disorder specialist as soon as possible
                            - 📋 **Request a comprehensive evaluation** including physical examination and medical history
                            - 🔍 **Additional tests** may include: DaTscan, MRI, or other neurological assessments
                            - 📝 **Document your symptoms** including when they started and how they've progressed
                            - 👨‍👩‍👧‍👦 **Bring a family member** to your appointment for additional observations
                            - ⏰ **Don't delay** - early detection can lead to better management outcomes
                            
                            **Remember:** This screening tool is not a diagnosis. Only a qualified healthcare professional can diagnose Parkinson's Disease.
                            """)
                        else:
                            st.success("""
                            **Based on this prediction:**
                            
                            - ✅ **Low likelihood** of Parkinson's Disease based on voice analysis
                            - 🔄 **Continue monitoring** - if you notice any symptoms, consult a doctor
                            - 🎤 **Voice health matters** - maintain good vocal hygiene and health
                            - 📅 **Regular check-ups** - consider periodic screenings, especially if you have risk factors
                            - 👀 **Watch for symptoms** such as tremors, stiffness, or changes in movement
                            - 💪 **Stay healthy** - regular exercise and a balanced diet support overall neurological health
                            
                            **Note:** A negative result does not guarantee the absence of Parkinson's Disease. Consult a healthcare professional if you have concerns.
                            """)
                        
                        st.write("---")
                else:
                    pred = int(predictions[0])
                    prob = float(probabilities[0])
                    conf = prob if pred == 1 else (1 - prob)
                    percent = int(round(conf * 100))
                    if pred == 1:
                        st.markdown(f"<h2 style='color:red'>⚠️ Parkinson's likely (Confidence: {conf:.2f})</h2>", unsafe_allow_html=True)
                        gauge_color = "red"
                    else:
                        st.markdown(f"<h2 style='color:green'>✅ No Parkinson's likely (Confidence: {conf:.2f})</h2>", unsafe_allow_html=True)
                        gauge_color = "green"
                    gauge_container = st.empty()
                    show_animated_gauge(gauge_container, percent, gauge_color, "Confidence")
                    
                    # Recommendations based on prediction
                    st.markdown("---")
                    st.markdown("### 💡 Recommendations")
                    if pred == 1:
                        st.warning("""
                        **Based on this prediction, we recommend:**
                        
                        - 🏥 **Consult a neurologist** or movement disorder specialist as soon as possible
                        - 📋 **Request a comprehensive evaluation** including physical examination and medical history
                        - 🔍 **Additional tests** may include: DaTscan, MRI, or other neurological assessments
                        - 📝 **Document your symptoms** including when they started and how they've progressed
                        - 👨‍👩‍👧‍👦 **Bring a family member** to your appointment for additional observations
                        - ⏰ **Don't delay** - early detection can lead to better management outcomes
                        
                        **Remember:** This screening tool is not a diagnosis. Only a qualified healthcare professional can diagnose Parkinson's Disease.
                        """)
                    else:
                        st.success("""
                        **Based on this prediction:**
                        
                        - ✅ **Low likelihood** of Parkinson's Disease based on voice analysis
                        - 🔄 **Continue monitoring** - if you notice any symptoms, consult a doctor
                        - 🎤 **Voice health matters** - maintain good vocal hygiene and health
                        - 📅 **Regular check-ups** - consider periodic screenings, especially if you have risk factors
                        - 👀 **Watch for symptoms** such as tremors, stiffness, or changes in movement
                        - 💪 **Stay healthy** - regular exercise and a balanced diet support overall neurological health
                        
                        **Note:** A negative result does not guarantee the absence of Parkinson's Disease. Consult a healthcare professional if you have concerns.
                        """)

                st.caption("This prediction is not a medical diagnosis. Please consult a healthcare professional for proper evaluation.")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.error("Please ensure all required features are provided correctly.")

# --- EXPLAIN MODEL PAGE ---
elif page == "📊 Explain Model":
    st.title("📊 Model Insights")
    st.write("""
    The model used is a **Random Forest Classifier**, trained on voice features like frequency,
    jitter, shimmer, and noise ratios. These features measure how steady or shaky a person's voice is.
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
      which is a known symptom of Parkinson's Disease.  
    - Frequency-based features like *MDVP:Fo(Hz)* show variations in vocal pitch stability.
    - **Engineered features** like *mean_jitter*, *mean_shimmer*, and *HNR_to_NHR* are automatically 
      calculated from the base features to enhance model performance.
    """)

# --- ABOUT PAGE ---
elif page == "ℹ️ About":
    st.title("ℹ️ About This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 👨‍💻 Project Information
        - **Project:** Machine Learning-Based Detection of Parkinson's Disease Using Voice Features  
        - **Developer:** Joseph Dabuo  
        - **Model:** Random Forest Classifier  
        - **Features:** 25 voice acoustic measurements (22 base + 3 engineered)
        """)
        
        st.markdown("""
        ### 🛠️ Technology Stack
        - **Frontend:** Streamlit
        - **ML Framework:** scikit-learn
        - **Data Processing:** pandas, numpy
        - **Visualization:** plotly
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Project Goals
        - Design a simple, trustworthy ML-powered web app
        - Help screen for Parkinson's Disease using voice features
        - Provide educational insights about the disease
        - Demonstrate the power of AI in healthcare support
        """)
        
        st.markdown("""
        ### 📚 Feature Engineering
        The model uses three engineered features:
        - **mean_jitter:** Average of all jitter measurements
        - **mean_shimmer:** Average of all shimmer measurements
        - **HNR_to_NHR:** Ratio of harmonics to noise
        
        These are automatically calculated from base features.
        """)
    
    st.markdown("---")
    st.warning("""
    ### ⚠️ Important Disclaimer
    This is an **educational tool only** — not for clinical diagnosis. 
    
    - Results should be interpreted by qualified healthcare professionals
    - This tool does not replace proper medical examination
    - Always consult a doctor if you have concerns about Parkinson's Disease
    """)
    
    st.success("Thank you for exploring this project!")
