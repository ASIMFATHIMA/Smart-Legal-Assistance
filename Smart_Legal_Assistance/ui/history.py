import streamlit as st
import pandas as pd
import json
from database.connection import get_db
from database.models import ChatHistory

def render_history():
    st.markdown('<div class="animate-fade-in">', unsafe_allow_html=True)
    st.title("📜 Conversation History")
    st.write("Browse, search, filter, and export your historical chatbot interactions.")

    user_id = st.session_state.get('user_id')
    if not user_id:
        st.error("You must be logged in to view history.")
        return

    db_generator = get_db()
    db = next(db_generator)

    try:
        # Fetch all histories for user
        history = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).order_by(ChatHistory.timestamp.desc()).all()

        if not history:
            st.info("No conversational history found yet. Start chatting with the assistant!")
            return

        # Prepare dataframe
        data = [{
            "ID": h.id,
            "Question": h.question,
            "Answer": h.answer,
            "Timestamp": h.timestamp
        } for h in history]

        df = pd.DataFrame(data)

        # Filters
        st.subheader("🔍 Filters & Search")
        col_search, col_start, col_end = st.columns([2, 1, 1])

        with col_search:
            search_query = st.text_input("Search questions or answers", "")
        with col_start:
            start_date = st.date_input("Start Date", value=df['Timestamp'].min().date())
        with col_end:
            end_date = st.date_input("End Date", value=df['Timestamp'].max().date())

        # Filter the DataFrame
        df['Date'] = pd.to_datetime(df['Timestamp']).dt.date
        filtered_df = df[
            (df['Date'] >= start_date) & 
            (df['Date'] <= end_date)
        ]

        if search_query:
            filtered_df = filtered_df[
                filtered_df['Question'].str.contains(search_query, case=False, na=False) |
                filtered_df['Answer'].str.contains(search_query, case=False, na=False)
            ]

        if filtered_df.empty:
            st.warning("No records matched the filter criteria.")
            return

        # Export Options
        st.subheader("📥 Export & Batch Operations")
        exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 1])

        # JSON Export
        export_list = filtered_df[["Question", "Answer", "Timestamp"]].to_dict(orient="records")
        # Custom serializer for datetimes
        for item in export_list:
            item["Timestamp"] = str(item["Timestamp"])
        json_data = json.dumps(export_list, indent=2)

        # CSV Export
        csv_data = filtered_df[["Question", "Answer", "Timestamp"]].to_csv(index=False)

        with exp_col1:
            st.download_button(
                label="⬇ Export to JSON",
                data=json_data,
                file_name="legal_chat_history.json",
                mime="application/json"
            )
        with exp_col2:
            st.download_button(
                label="⬇ Export to CSV",
                data=csv_data,
                file_name="legal_chat_history.csv",
                mime="text/csv"
            )
        with exp_col3:
            if st.button("🗑 Clear Filtered Logs", type="secondary"):
                try:
                    ids_to_del = filtered_df['ID'].tolist()
                    db.query(ChatHistory).filter(ChatHistory.id.in_(ids_to_del)).delete(synchronize_session=False)
                    db.commit()
                    st.success(f"Cleared {len(ids_to_del)} logs successfully!")
                    st.rerun()
                except Exception as e:
                    db.rollback()
                    st.error(f"Failed to clear logs: {e}")

        # Display history
        st.subheader("💬 Logged Conversations")
        for idx, row in filtered_df.iterrows():
            with st.container():
                st.markdown(
                    f"""
                    <div class="glass-card" style="margin-bottom: 20px;">
                        <div style="font-size: 12px; color: #94a3b8; margin-bottom: 8px;">🕒 {row['Timestamp']}</div>
                        <div style="font-weight: 600; color: #38bdf8; margin-bottom: 6px;">👤 Question:</div>
                        <div style="margin-bottom: 12px; font-size: 15px; color: #f1f5f9;">{row['Question']}</div>
                        <div style="font-weight: 600; color: #a855f7; margin-bottom: 6px;">🤖 AI Assistant:</div>
                        <div style="white-space: pre-wrap; font-size: 15px; color: #e2e8f0;">{row['Answer']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Delete single chat entry
                if st.button(f"Delete Log #{row['ID']}", key=f"del_{row['ID']}"):
                    try:
                        db.query(ChatHistory).filter(ChatHistory.id == row['ID']).delete()
                        db.commit()
                        st.success(f"Log #{row['ID']} deleted.")
                        st.rerun()
                    except Exception as e:
                        db.rollback()
                        st.error(f"Error: {e}")

    finally:
        db_generator.close()
        st.markdown('</div>', unsafe_allow_html=True)
