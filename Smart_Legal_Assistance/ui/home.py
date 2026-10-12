import streamlit as st
import base64
import os

def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return ""

def render_home(set_page_callback=None):
    st.markdown('<div class="animate-fade-in">', unsafe_allow_html=True)
    
    # Hero Section
    logo_base64 = get_image_base64("assets/logo.png")
    hero_img_base64 = get_image_base64("assets/legal_assistant.png")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        if logo_base64:
            st.markdown(
                f'<img src="data:image/png;base64,{logo_base64}" style="width: 120px; margin-bottom: 20px; filter: drop-shadow(0 0 15px rgba(99,102,241,0.5));">', 
                unsafe_allow_html=True
            )
        st.markdown('<span class="badge-gradient">⚖️ Smart Legal Assistance</span>', unsafe_allow_html=True)
        st.title("Smart Legal Assistance")
        st.subheader("Your AI-Powered Legal Document Expert")
        st.markdown(
            """
            Analyze, query, and synthesize your legal documents instantly. Smart Legal Assistance 
            combines state-of-the-art hybrid semantic retrieval, cross-encoder reranking, and 
            local LLM execution with Qwen-2.5 7B to provide grounded answers to your criminal law questions.
            """
        )
        
        # Interactive buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("💬 Open Legal Chatbot"):
                if set_page_callback:
                    set_page_callback("Chatbot")
                    st.rerun()
        with btn_col2:
            if st.button("📤 Upload PDF Documents"):
                if set_page_callback:
                    set_page_callback("Upload PDF")
                    st.rerun()
                    
    with col2:
        if hero_img_base64:
            st.markdown(
                f'<div style="text-align: center; margin-top: 20px;">'
                f'<img src="data:image/png;base64,{hero_img_base64}" style="width: 100%; max-width: 380px; border-radius: 24px; box-shadow: 0 15px 40px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.1);">'
                f'</div>', 
                unsafe_allow_html=True
            )

    st.markdown('<hr style="border-color: rgba(255,255,255,0.1); margin: 40px 0;">', unsafe_allow_html=True)
    
    # Feature Cards Section
    st.subheader("🤖 Key Core Features")
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <h3>Advanced Hybrid RAG</h3>
                <p>Hybrid semantic retrieval using FAISS dense vectors combined with BM25 keyword search, powered by spaCy NLP preprocessing.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🛡️</div>
                <h3>Hallucination Guard</h3>
                <p>Ensures answers are strictly grounded in your context. If confidence is insufficient, it safeguards by falling back safely.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔊</div>
                <h3>Piper TTS</h3>
                <p>Fully local, high-fidelity speech synthesis supporting both Male and Female voice outputs for all legal responses.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <h3>Metrics Dashboard</h3>
                <p>Explore usage telemetry, uploaded documents statistics, and feedback analysis in an interactive Plotly-driven interface.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
