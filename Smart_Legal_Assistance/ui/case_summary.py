import streamlit as st
import os

def render_case_summary():
    st.title("Case Summary")
    st.write("Generate and view summaries of uploaded legal documents.")
    
    # In a real app, you'd fetch available documents from DB
    st.selectbox("Select a Document to Summarize", ["Mock_Document_1.pdf", "Mock_Document_2.pdf"])
    
    if st.button("Generate Summary"):
        st.info("Generating summary using t5-small... (This may take a moment)")
        
        # Placeholder for backend summary generation
        summary_text = "This document outlines the primary responsibilities and liabilities of both parties. Key sections indicate breach of contract conditions and jurisdictional definitions."
        key_topics = ["Breach of Contract", "Liability", "Jurisdiction"]
        
        st.subheader("Summary")
        st.write(summary_text)
        
        st.subheader("Key Legal Topics")
        for topic in key_topics:
            st.markdown(f"- {topic}")
            
        st.subheader("Simplified Explanation")
        st.write("Basically, if someone breaks the rules of this contract, they have to pay damages, and any lawsuit must happen in the specified court.")
        
        # Download button for ReportLab generated PDF
        st.download_button(
            label="Download PDF Report",
            data=b"Mock PDF Data",
            file_name="case_summary_report.pdf",
            mime="application/pdf"
        )
