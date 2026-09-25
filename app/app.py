import streamlit as st
from predictor import predict_comment

# Set page config for a wider layout and custom title
st.set_page_config(
    page_title="ToxicGuard",
    page_icon="🛡️",
    layout="centered"
)

# Custom CSS for dark modern theme and professional styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Headers */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: -10px;
        background: -webkit-linear-gradient(45deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    
    .subtitle {
        font-size: 1.5rem;
        font-weight: 500;
        color: #a0aec0;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .lang-support {
        text-align: center;
        font-size: 0.9rem;
        color: #718096;
        letter-spacing: 1px;
        margin-bottom: 30px;
        background-color: #1a202c;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #2d3748;
    }
    
    .description {
        text-align: center;
        font-size: 1.1rem;
        color: #e2e8f0;
        margin-bottom: 40px;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        font-size: 0.8rem;
        color: #4a5568;
        padding: 15px 0;
        border-top: 1px solid #2d3748;
        background-color: #0e1117;
        left: 0;
    }
    
    /* Button full width */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 50px;
        font-weight: bold;
        background-color: #3182ce;
        color: white;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #2b6cb0;
        border-color: #2b6cb0;
    }
    
    /* Text area */
    .stTextArea textarea {
        background-color: #1a202c;
        color: #f7fafc;
        border: 1px solid #4a5568;
        border-radius: 8px;
    }
    
</style>
""", unsafe_allow_html=True)

def main():
    # Header Section
    st.markdown('<h1 class="main-title">ToxicGuard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Multilingual Toxic Comment Detection</p>', unsafe_allow_html=True)
    
    st.markdown(
        '<div class="lang-support">English • Sinhala • Tamil • Singlish • Tanglish</div>', 
        unsafe_allow_html=True
    )
    
    st.markdown(
        '<p class="description">Analyze multilingual comments for potentially toxic content '
        'using our transformer-based classification system.</p>',
        unsafe_allow_html=True
    )
    
    # Analysis Section
    st.markdown("### Analysis Panel")
    
    comment = st.text_area(
        label="Enter text to analyze", 
        placeholder="Type or paste a comment here...",
        label_visibility="collapsed",
        height=150
    )
    
    # Analyze Button
    if st.button("Analyze Comment", use_container_width=True):
        if not comment.strip():
            st.warning("Please enter a comment before predicting.")
        else:
            with st.spinner("Analyzing comment..."):
                result = predict_comment(comment)
                
                status = result.get("status")
                
                if status == "Model not connected yet":
                    st.info("Model integration pending — the interface is ready for the final trained model.")
                else:
                    # Keep space ready for the future result section
                    prediction = result.get("prediction")
                    confidence = result.get("confidence")
                    
                    if prediction == 1:
                        st.error(f"Prediction: **Toxic** (Confidence: {confidence:.2f})")
                    elif prediction == 0:
                        st.success(f"Prediction: **Non-toxic** (Confidence: {confidence:.2f})")
                    else:
                        st.write("Prediction unknown.")

    # Footer
    st.markdown(
        '<div class="footer">Multilingual Toxic Comment Detection • University Research Project</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
