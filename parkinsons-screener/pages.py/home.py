# =====================================================
# Home Page: pages/home.py
# Disease Overview, Video, Audio Narration, Carousel
# =====================================================

import streamlit as st
import pyttsx3
import io

def show():
    st.title("🧠 Understanding Parkinson's Disease")
    
    # === CAROUSEL / SLIDESHOW ===
    st.subheader("📸 Visual Overview")
    
    carousel_images = [
        {
            "title": "What is Parkinson's Disease?",
            "description": "Parkinson's Disease is a progressive neurodegenerative disorder affecting movement, characterized by tremors, rigidity, and bradykinesia.",
            "emoji": "🔬"
        },
        {
            "title": "Key Symptoms",
            "description": "Tremors (shaking), Rigidity (stiffness), Bradykinesia (slow movement), Postural instability, Speech & voice changes.",
            "emoji": "⚠️"
        },
        {
            "title": "Affected Areas",
            "description": "Primarily affects the substantia nigra in the brain, leading to dopamine deficiency and motor control issues.",
            "emoji": "🧠"
        },
        {
            "title": "Voice Changes",
            "description": "Parkinsonian speech often shows reduced volume, monotone pitch, irregular rhythm - key detection markers.",
            "emoji": "🎙️"
        },
        {
            "title": "Prevalence",
            "description": "Affects ~1 million people in the US, more common in older adults (average onset: 60+ years).",
            "emoji": "📊"
        },
        {
            "title": "Early Detection",
            "description": "Voice analysis is non-invasive, accessible, and can enable early screening for timely intervention.",
            "emoji": "✅"
        }
    ]
    
    # Initialize carousel index in session state
    if 'carousel_index' not in st.session_state:
        st.session_state.carousel_index = 0
    
    col1, col2, col3 = st.columns([1, 8, 1])
    
    with col1:
        if st.button("⬅️", key="prev_carousel"):
            st.session_state.carousel_index = (st.session_state.carousel_index - 1) % len(carousel_images)
            st.rerun()
    
    with col2:
        current_slide = carousel_images[st.session_state.carousel_index]
        st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #007acc 0%, #003366 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                min-height: 200px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            '>
                <h1 style='font-size: 3em; margin: 10px 0;'>{current_slide["emoji"]}</h1>
                <h2 style='margin: 10px 0;'>{current_slide["title"]}</h2>
                <p style='font-size: 1.1em; margin: 10px 0;'>{current_slide["description"]}</p>
                <p style='font-size: 0.9em; margin-top: 20px; opacity: 0.8;'>
                    Slide {st.session_state.carousel_index + 1} of {len(carousel_images)}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.button("➡️", key="next_carousel"):
            st.session_state.carousel_index = (st.session_state.carousel_index + 1) % len(carousel_images)
            st.rerun()
    
    st.markdown("---")
    
    # === DISEASE OVERVIEW ===
    st.subheader("📖 What is Parkinson's Disease?")
    
    st.markdown("""
    **Parkinson's Disease (PD)** is a progressive neurodegenerative disorder that primarily affects movement. 
    It occurs due to the loss of dopamine-producing cells in the brain, leading to a variety of motor and non-motor symptoms.
    
    ### Key Characteristics:
    - **Progressive Nature**: Symptoms gradually worsen over time
    - **Motor Symptoms**: Tremors, rigidity, slow movement, balance problems
    - **Non-Motor Symptoms**: Speech changes, sleep issues, cognitive changes, mood disorders
    - **Age of Onset**: Typically affects people over 60, but can occur earlier (Young-Onset PD)
    
    ### Why Voice Analysis?
    Voice changes are among the earliest detectable signs of Parkinson's Disease. Our model analyzes voice features 
    to help with early screening and monitoring.
    """)
    
    st.markdown("---")
    
    # === EMBEDDED VIDEO ===
    st.subheader("🎥 Educational Video")
    st.markdown("Watch this comprehensive overview of Parkinson's Disease:")
    
    st.video("https://www.youtube.com/watch?v=u_tozEV7f4k")
    
    st.markdown("---")
    
    # === AUDIO NARRATION ===
    st.subheader("🔊 Listen to Overview (Audio Narration)")
    
    audio_text = """
    Welcome to the Parkinson's Disease Screener application. 
    Parkinson's Disease is a progressive neurodegenerative disorder that affects movement and coordination. 
    It occurs when dopamine-producing cells in the brain gradually deteriorate.
    
    Common symptoms include tremors or shaking, muscle rigidity or stiffness, slow movement, balance and posture problems, 
    and changes in speech and voice.
    
    Voice changes in Parkinson's patients often include reduced volume, monotone pitch, and irregular rhythm. 
    These changes occur because the disease affects the muscles controlling speech.
    
    Our machine learning model analyzes voice features like pitch variation, amplitude perturbation, and noise ratios 
    to help screen for Parkinson's Disease. This is a non-invasive, accessible method for early detection.
    
    However, this tool is for educational and screening purposes only. 
    Any concerns about Parkinson's Disease should be discussed with a qualified healthcare professional.
    
    Now, let's proceed to the prediction section to analyze voice features and generate a risk assessment.
    """
    
    if st.button("🔊 Generate Audio Narration"):
        with st.spinner("Generating audio..."):
            try:
                # Initialize text-to-speech engine
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)  # Speech rate
                engine.setProperty('volume', 0.9)  # Volume
                
                # Save audio to bytes buffer
                audio_buffer = io.BytesIO()
                engine.save_to_file(audio_text, "temp_audio.wav")
                engine.runAndWait()
                
                # Display audio player
                with open("temp_audio.wav", "rb") as audio_file:
                    st.audio(audio_file.read(), format="audio/wav")
                
                st.success("✅ Audio narration generated successfully!")
                
            except Exception as e:
                st.error(f"Could not generate audio: {e}")
                st.info("You can still read the text above or visit the Prediction page to continue.")
    
    st.markdown("""
    > **Note:** If audio generation fails, try installing the required package:  
    > `pip install pyttsx3`
    """)
    
    st.markdown("---")
    
    # === KEY STATISTICS ===
    st.subheader("📊 Quick Facts About Parkinson's")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Global Cases", "~10 Million", "Millions")
    col2.metric("US Prevalence", "~1 Million", "People")
    col3.metric("Avg Onset Age", "60+ Years", "Age Range")
    col4.metric("Voice Involvement", "~90%", "Of Patients")
    
    st.markdown("---")
    
    # === CALL TO ACTION ===
    st.success("""
    ✅ **Ready to Screen?**  
    Head to the **🔮 Predict & Recommend** page to analyze voice features and get a risk assessment.  
    Remember: This is a screening tool only, not a medical diagnosis.
    """)