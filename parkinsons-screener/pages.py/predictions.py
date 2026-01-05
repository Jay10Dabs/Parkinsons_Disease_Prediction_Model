# =====================================================
# Predictions & Recommendations: pages/predictions.py
# Input, Prediction, Risk Assessment, Clinical Recommendations
# =====================================================

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import time

def show_gauge(confidence, label, prediction):
    """Display animated confidence gauge"""
    percent = int(round(confidence * 100))
    color = "red" if prediction == 1 else "green"
    pred_label = "⚠️ Parkinson's Likely" if prediction == 1 else "✅ No Parkinson's Likely"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=percent,
        number={'valueformat': "d", 'suffix': '%', 'font': {'size': 28}},
        title={'text': pred_label, 'font': {'size': 16}},
        delta={'reference': 50, 'increasing': {'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgrey"},
            'bar': {'color': color},
            'bgcolor': "#e6f7ff",
            'steps': [
                {'range': [0, 33], 'color': "#90EE90"},
                {'range': [33, 67], 'color': "#FFE4B5"},
                {'range': [67, 100], 'color': "#FFB6C1"}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': percent
            }
        }
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=400, font=dict(family="Open Sans"))
    st.plotly_chart(fig, use_container_width=True)

def get_recommendations(confidence, prediction):
    """Generate clinical recommendations based on prediction"""
    
    if prediction == 1:  # Parkinson's Likely
        if confidence > 0.85:
            risk_level = "🔴 HIGH RISK"
            recommendations = [
                "🏥 **Schedule a neurological consultation immediately**",
                "📋 Request comprehensive neurological testing (MRI, PET scan)",
                "🧪 Discuss biomarkers and advanced diagnostic procedures",
                "📅 Consider regular monitoring appointments every 3-6 months",
                "💊 Discuss treatment options with neurologist (Levodopa, dopamine agonists)",
                "🏃 Begin or continue mild physical therapy and exercise",
                "👨‍⚕️ Inform your primary care physician of this screening result"
            ]
            color = "#FF6B6B"
        elif confidence > 0.70:
            risk_level = "🟠 MODERATE-HIGH RISK"
            recommendations = [
                "📞 **Schedule an appointment with a neurologist within 2-4 weeks**",
                "📝 Keep a symptom diary (tremor, rigidity, voice changes)",
                "🧪 Request voice analysis and motor testing",
                "🏃 Increase physical activity (yoga, walking, strength training)",
                "🎵 Consider speech therapy evaluation",
                "📅 Plan follow-up screening in 3-6 months",
                "💬 Discuss family history with healthcare provider"
            ]
            color = "#FFA500"
        else:
            risk_level = "🟡 MODERATE RISK"
            recommendations = [
                "👨‍⚕️ **Inform your healthcare provider about this screening result**",
                "📋 Request follow-up voice analysis in 6 months",
                "🏃 Maintain healthy lifestyle: exercise, good sleep, stress management",
                "🧠 Monitor for early symptoms (tremor, stiffness, voice changes)",
                "📞 Schedule a routine check-up with your doctor",
                "💪 Continue or start physical activity program",
                "📅 Periodic re-screening recommended"
            ]
            color = "#FFD700"
    
    else:  # No Parkinson's Likely
        if confidence > 0.85:
            risk_level = "🟢 LOW RISK"
            recommendations = [
                "✅ **Current screening suggests low Parkinson's risk**",
                "🏃 Maintain healthy habits: regular exercise, good nutrition",
                "🧠 Continue cognitive and physical activities",
                "😴 Ensure adequate sleep and stress management",
                "📅 Routine annual health check-ups recommended",
                "👨‍⚕️ No urgent neurological consultation needed",
                "📊 Consider periodic screening (annually) for long-term monitoring"
            ]
            color = "#90EE90"
        else:
            risk_level = "🟢 LOW-MODERATE RISK"
            recommendations = [
                "✅ **Current screening suggests relatively low Parkinson's risk**",
                "🏃 Maintain active lifestyle and healthy habits",
                "🧠 Stay cognitively engaged (reading, puzzles, learning)",
                "😴 Prioritize good sleep quality and stress reduction",
                "📞 Consult with doctor if any new symptoms develop",
                "📅 Follow-up screening recommended in 6-12 months",
                "👨‍⚕️ Routine health monitoring appropriate"
            ]
            color = "#98FB98"
    
    return risk_level, recommendations, color

def show():
    st.title("🔮 Predict & Recommendations")
    st.info("⚠️ This tool is for screening purposes only. Results should be verified by healthcare professionals.")
    
    # Load model and preprocessing objects
    try:
        model = joblib.load("parkinsons_model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_names = joblib.load("feature_names.pkl")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Feature descriptions
    feature_descriptions = {
        "MDVP:Fo(Hz)": "Average vocal fundamental frequency",
        "MDVP:Fhi(Hz)": "Maximum vocal fundamental frequency",
        "MDVP:Flo(Hz)": "Minimum vocal fundamental frequency",
        "MDVP:Jitter(%)": "Cycle-to-cycle pitch variation (higher = unstable)",
        "MDVP:Jitter(Abs)": "Absolute jitter in Hz",
        "MDVP:RAP": "Relative amplitude perturbation",
        "MDVP:PPQ": "Pitch period variation",
        "Jitter:DDP": "Difference of differences of pitch periods",
        "MDVP:Shimmer": "Amplitude variation (higher = less stable)",
        "MDVP:Shimmer(dB)": "Shimmer in decibels",
        "Shimmer:APQ3": "Amplitude perturbation quotient 3",
        "Shimmer:APQ5": "Amplitude perturbation quotient 5",
        "MDVP:APQ": "Average perturbation quotient",
        "Shimmer:DDA": "Difference of differences of amplitude",
        "NHR": "Noise-to-harmonics ratio (higher = noisier)",
        "HNR": "Harmonics-to-noise ratio (higher = cleaner)",
        "RPDE": "Recurrence period density entropy",
        "DFA": "Detrended fluctuation analysis",
        "spread1": "Non-linear complexity measure",
        "spread2": "Non-linear complexity measure",
        "D2": "Correlation dimension",
        "PPE": "Pitch period entropy (higher = irregular pitch)"
    }
    
    st.subheader("Input Method")
    method = st.radio("Choose input method:", ["📝 Manual Entry", "📁 Upload CSV"], horizontal=True)
    
    input_df = None
    
    if method == "📁 Upload CSV":
        st.markdown("""
        **Instructions:**
        - Upload CSV file(s) with voice features
        - Column names must match feature names (case-insensitive)
        - Download sample CSV below for reference
        """)
        
        # Sample CSV download
        sample_data = pd.DataFrame({feature: [0.0] for feature in feature_names})
        csv_buffer = io.StringIO()
        sample_data.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv_buffer.getvalue(),
            file_name="sample_voice_features.csv",
            mime="text/csv"
        )
        
        uploaded_files = st.file_uploader("Upload CSV file(s)", type=["csv"], accept_multiple_files=True)
        all_dfs = []
        
        for file in uploaded_files:
            try:
                df = pd.read_csv(file)
                df.columns = [c.strip().lower() for c in df.columns]
                fn_lower = {f.lower(): f for f in feature_names}
                
                # Match columns
                matched_cols = [fn_lower[c] for c in df.columns if c in fn_lower]
                df_matched = df[[c.lower() for c in matched_cols]].copy()
                df_matched.columns = matched_cols
                
                # Check for missing columns
                missing = [f for f in feature_names if f not in df_matched.columns]
                if missing:
                    st.warning(f"File '{file.name}' missing: {missing}")
                    continue
                
                all_dfs.append(df_matched)
            except Exception as e:
                st.error(f"Error processing '{file.name}': {e}")
        
        if all_dfs:
            input_df = pd.concat(all_dfs, ignore_index=True)
            st.success(f"✅ Loaded {len(input_df)} samples from {len(uploaded_files)} file(s)")
            st.dataframe(input_df.head(), use_container_width=True)
    
    else:  # Manual Entry
        st.markdown("Enter voice feature values below:")
        col1, col2 = st.columns(2)
        user_input = {}
        
        for i, feature in enumerate(feature_names):
            column = col1 if i % 2 == 0 else col2
            user_input[feature] = column.number_input(
                f"{feature}",
                value=0.0,
                format="%.6f",
                help=feature_descriptions.get(feature, "Voice feature")
            )
        
        input_df = pd.DataFrame([user_input])
    
    # Make predictions
    if st.button("🔮 Make Prediction", use_container_width=True):
        if input_df is None or input_df.empty:
            st.warning("No input data. Please upload CSV or use manual entry.")
        else:
            with st.spinner("Analyzing voice features..."):
                try:
                    # Scale and predict
                    input_scaled = scaler.transform(input_df[feature_names])
                    predictions = model.predict(input_scaled)
                    probabilities = model.predict_proba(input_scaled)[:, 1]
                    
                    # Handle batch vs single predictions
                    if len(input_df) > 1:
                        st.subheader("📊 Batch Results")
                        
                        results = []
                        for idx, (pred, prob) in enumerate(zip(predictions, probabilities), 1):
                            conf = prob if pred == 1 else (1 - prob)
                            label = "Parkinson's Risk" if pred == 1 else "No Parkinson's"
                            results.append({
                                "Sample #": idx,
                                "Prediction": label,
                                "Confidence": f"{conf:.2%}"
                            })
                        
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv_results = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv_results,
                            "predictions_results.csv",
                            "text/csv"
                        )
                        
                        # Show detailed results
                        st.subheader("Detailed Analysis")
                        for idx, (pred, prob) in enumerate(zip(predictions, probabilities), 1):
                            conf = prob if pred == 1 else (1 - prob)
                            risk_level, recommendations, color = get_recommendations(conf, int(pred))
                            
                            with st.expander(f"Sample {idx} - {risk_level}", expanded=(idx == 1)):
                                show_gauge(conf, "Confidence", int(pred))
                                st.markdown(f"**Risk Level:** {risk_level}")
                                st.markdown("**Recommendations:**")
                                for rec in recommendations:
                                    st.markdown(f"- {rec}")
                    
                    else:
                        # Single prediction
                        pred = int(predictions[0])
                        prob = float(probabilities[0])
                        conf = prob if pred == 1 else (1 - prob)
                        risk_level, recommendations, color = get_recommendations(conf, pred)
                        
                        st.subheader("Prediction Result")
                        show_gauge(conf, "Confidence", pred)
                        
                        # Risk level with color
                        st.markdown(f"""
                        <div style='
                            background-color: {color};
                            color: white;
                            padding: 20px;
                            border-radius: 10px;
                            text-align: center;
                            font-size: 1.2em;
                            font-weight: bold;
                        '>
                            {risk_level}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.subheader("Clinical Recommendations")
                        for rec in recommendations:
                            st.markdown(rec)
                        
                        st.markdown("---")
                        st.warning("""
                        **⚠️ Important Disclaimer:**
                        - This screening is NOT a medical diagnosis
                        - Results must be confirmed by qualified healthcare professionals
                        - Consult a neurologist for proper evaluation
                        - This tool supports clinical decision-making, not replaces it
                        """)
                
                except Exception as e:
                    st.error(f"Prediction error: {e}")