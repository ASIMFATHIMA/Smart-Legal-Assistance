import streamlit as st
from database.connection import get_db
from database.models import ChatHistory

def render_settings():
    st.title("Settings")
    
    st.subheader("Appearance")
    st.write("Theme selection is handled automatically by Streamlit based on your system preferences. You can force Light/Dark mode via the top-right settings menu in Streamlit.")
    
    st.subheader("Data Management")
    if st.button("Clear Chat Memory"):
        st.session_state.messages = []
        st.success("Chat memory cleared from current session.")
        
    if st.button("Delete All Chat History (Database)"):
        user_id = st.session_state.get('user_id')
        if user_id:
            db_generator = get_db()
            db = next(db_generator)
            try:
                db.query(ChatHistory).filter(ChatHistory.user_id == user_id).delete()
                db.commit()
                st.session_state.messages = []
                st.success("All chat history permanently deleted.")
            except Exception as e:
                db.rollback()
                st.error(f"Failed to delete history: {e}")
            finally:
                db_generator.close()
        else:
            st.error("You must be logged in.")
            
    st.subheader("Export")
    if st.button("Export Conversations (JSON)"):
        user_id = st.session_state.get('user_id')
        if user_id:
            db_generator = get_db()
            db = next(db_generator)
            try:
                history = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).all()
                export_data = [{"Q": h.question, "A": h.answer, "Time": str(h.timestamp)} for h in history]
                st.json(export_data)
                # Could also provide a file download here
            finally:
                db_generator.close()
        else:
            st.error("You must be logged in.")
