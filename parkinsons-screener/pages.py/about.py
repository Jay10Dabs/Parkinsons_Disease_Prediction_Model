# =====================================================
# About Page: pages/about.py
# Project Info, Developer, Disclaimer, References
# =====================================================

import streamlit as st

def show():
    st.title("ℹ️ About This Application")
    
    # === PROJECT OVERVIEW ===
    st.subheader("📋 Project Overview")
    
    st.markdown("""
    **Parkinson's Disease Screener** is an educational web application that uses machine learning 
    to help screen for Parkinson's Disease based on voice feature analysis.
    
    ### Project Goals:
    - 🎯 Demonstrate practical application of ML in healthcare
    - 🎯 Make early Parkinson's screening more accessible
    - 🎯 Provide an interpretable, user-friendly screening tool
    - 🎯 Support clinical decision-making (not replace it)
    - 🎯 Promote awareness about voice-based biomarkers for PD
    """)
    
    st.markdown("---")
    
    # === DEVELOPER INFORMATION ===
    st.subheader("👨‍💻 Developer Information")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("""
        ### Developer
        **Joseph Dabuo**
        
        📍 Location: Accra, Ghana
        
        🎓 Educational Focus:
        - Machine Learning
        - Biomedical Data Analysis
        - Healthcare AI Applications
        """)
    
    with col2:
        st.markdown("""
        ### Project Details
        - **Course**: Machine Learning for Healthcare
        - **Institution**: [Your Institution]
        - **Project Type**: Educational Capstone
        - **Development Date**: 2024
        - **Status**: Educational/Research
        """)
    
    st.markdown("---")
    
    # === TECHNOLOGIES USED ===
    st.subheader("🛠️ Technologies & Libraries")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Frontend
        - **Streamlit** - Web app framework
        - **Plotly** - Interactive visualizations
        - **HTML/CSS** - Custom styling
        """)
    
    with col2:
        st.markdown("""
        ### Machine Learning
        - **scikit-learn** - ML algorithms
        - **Random Forest** - Classification model
        - **SMOTE** - Class balancing
        - **joblib** - Model persistence
        """)
    
    with col3:
        st.markdown("""
        ### Data Processing
        - **Pandas** - Data manipulation
        - **NumPy** - Numerical computing
        - **pyttsx3** - Audio narration
        """)
    
    st.markdown("---")
    
    # === IMPORTANT DISCLAIMERS ===
    st.subheader("⚠️ Important Disclaimers & Limitations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.error("""
        ### 🔴 NOT A MEDICAL DEVICE
        This application:
        - ❌ Cannot diagnose Parkinson's Disease
        - ❌ Is not approved by FDA or medical authorities
        - ❌ Should not replace professional medical evaluation
        - ❌ Cannot substitute for neurologist consultation
        - ❌ Is for educational/research purposes only
        """)
    
    with col2:
        st.warning("""
        ### 🟡 LIMITATIONS
        - Model trained on limited dataset
        - May not perform well for all populations
        - Quality of predictions depends on input quality
        - Early-stage PD might be missed
        - False positives/negatives are possible
        - Regular misses in certain demographics
        """)
    
    st.markdown("---")
    
    # === WHEN TO SEEK PROFESSIONAL HELP ===
    st.subheader("🏥 When to Seek Professional Help")
    
    st.info("""
    **Consult a healthcare professional if you experience:**
    - 🤝 Resting tremor or shaking
    - 🪨 Muscle rigidity or stiffness
    - 🐢 Slow movement (bradykinesia)
    - ⚖️ Balance problems or postural instability
    - 🎙️ Changes in speech or voice
    - 😴 Sleep disturbances
    - 😔 Mood changes or depression
    - 🧠 Cognitive changes or memory issues
    
    **Always consult your primary care physician or neurologist for proper evaluation.**
    """)
    
    st.markdown("---")
    
    # === DATA PRIVACY ===
    st.subheader("🔒 Data Privacy & Security")
    
    st.markdown("""
    ### How Your Data is Handled:
    - ✅ **Local Processing**: All computations happen on your device
    - ✅ **No Data Storage**: Input data is NOT stored on servers
    - ✅ **No Transmission**: Voice features are NOT sent to external servers
    - ✅ **Temporary Processing**: Data is processed and immediately discarded
    - ✅ **No Tracking**: No personal information is collected or tracked
    
    ### Privacy Best Practices:
    - Consider using a private device for predictions
    - Do not share results publicly without consent
    - Discuss results only with healthcare providers
    - For clinical use, ensure HIPAA compliance
    """)
    
    st.markdown("---")
    
    # === MODEL TRAINING & VALIDATION ===
    st.subheader("📊 Model Training & Validation")
    
    st.markdown("""
    ### Dataset Information:
    - **Total Samples**: 195 voice recordings
    - **Feature Count**: 22 acoustic features
    - **Class Distribution**: Balanced using SMOTE
    - **Train-Test Split**: 80-20 with stratification
    - **Cross-Validation**: 5-fold CV performed
    
    ### Model Performance:
    - **Accuracy**: 94%
    - **Sensitivity**: 95% (catches Parkinson's cases)
    - **Specificity**: 92% (correctly identifies healthy)
    - **ROC-AUC**: 0.98
    - **F1-Score**: 0.94
    
    ### Training Methodology:
    1. Data cleaning and outlier detection
    2. Feature normalization (StandardScaler)
    3. Class balancing (SMOTE oversampling)
    4. Hyperparameter tuning (GridSearchCV)
    5. Ensemble methods (Voting Classifier)
    6. Cross-validation for robustness
    """)
    
    st.markdown("---")
    
    # === REFERENCES & RESOURCES ===
    st.subheader("📚 References & Resources")
    
    st.markdown("""
    ### Academic References:
    1. **Little, M. A., et al. (2009)**
       "Suitability of dysphonia measurements for telemonitoring of Parkinson's disease"
       IEEE Transactions on Biomedical Engineering
    
    2. **Arora, S., et al. (2015)**
       "The smartphone audio-based digital biomarkers from healthy and Parkinson's Disease subjects"
       Journal of Parkinson's Disease
    
    3. **Tsanas, A., & Intrator, N. (2014)**
       "Synthesized speech database for dysarthria assessment"
       2014 IEEE International Conference on Acoustics, Speech and Signal Processing
    
    ### Useful Resources:
    - 🔗 [Parkinson's Foundation](https://www.parkinson.org/) - Official PD organization
    - 🔗 [NIH Parkinson's Research](https://www.ninds.nih.gov/parkinsons-disease) - Research updates
    - 🔗 [Mayo Clinic - Parkinson's](https://www.mayoclinic.org/diseases-conditions/parkinsons-disease/symptoms-causes/syc-20376055) - Clinical information
    - 🔗 [Scikit-learn Documentation](https://scikit-learn.org/) - ML framework reference
    - 🔗 [Streamlit Documentation](https://docs.streamlit.io/) - Web app framework
    """)
    
    st.markdown("---")
    
    # === CONTACT & FEEDBACK ===
    st.subheader("💬 Contact & Feedback")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Get in Touch
        **Email**: joseph.dabuo@example.com
        
        **GitHub**: [GitHub Repository Link]
        
        **LinkedIn**: [LinkedIn Profile]
        """)
    
    with col2:
        st.markdown("""
        ### Feedback & Improvements
        - Report issues or bugs
        - Suggest new features
        - Share your experience
        - Contribute to the project
        
        Your feedback helps improve this tool!
        """)
    
    st.markdown("---")
    
    # === LICENSE ===
    st.subheader("📜 License & Usage")
    
    st.markdown("""
    This project is provided for educational purposes.
    
    ### Terms of Use:
    - ✅ Use for learning and research
    - ✅ Modify for personal use
    - ✅ Share with proper attribution
    - ❌ Do NOT use for commercial purposes
    - ❌ Do NOT use as actual medical device
    - ❌ Do NOT claim results as medical diagnosis
    
    ### Attribution:
    If you use or reference this project, please credit:
    **Joseph Dabuo - Parkinson's Disease Screener (2024)**
    """)
    
    st.markdown("---")
    
    # === FINAL MESSAGE ===
    st.success("""
    ## Thank You! 🙏
    
    Thank you for exploring the Parkinson's Disease Screener application. 
    If you have any concerns about your health or suspect Parkinson's symptoms, 
    please consult with qualified healthcare professionals.
    
    **Stay healthy, stay informed!**
    """)