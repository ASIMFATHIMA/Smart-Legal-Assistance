"""Smart Legal Assistance – Chatbot UI

This UI integrates with `src/text_to_speech.py` (Piper).

Requirement highlights implemented here:
- Voice generation is optional.
- If Piper/voice model assets are missing, the UI continues normally.
- The "🔊 Listen" button is hidden when voice model is unavailable.
"""

from __future__ import annotations

import hashlib
import logging
import os

import streamlit as st

from database.connection import get_db
from database.models import ChatHistory

logger = logging.getLogger("rag_engine")


# ---------------------------------------------------------------------------
# TTS helpers
# ---------------------------------------------------------------------------
def _get_tts(voice: str):
    """Return a PiperTTS instance for the given voice, cached in session_state."""
    cache_key = f"_tts_instance_{voice}"
    if cache_key not in st.session_state:
        try:
            from src.text_to_speech import PiperTTS

            st.session_state[cache_key] = PiperTTS(voice=voice)
            logger.info("[TTS] PiperTTS instance created for voice: %s", voice)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[TTS] PiperTTS unavailable: %s", exc)
            st.session_state[cache_key] = None
    return st.session_state[cache_key]


def _get_tts_status(voice: str) -> dict:
    """Return TTS status dict cached in session_state."""
    status_key = f"_tts_status_{voice}"
    if status_key not in st.session_state:
        tts = _get_tts(voice)
        if tts is not None:
            try:
                st.session_state[status_key] = tts.check_status()
            except Exception:  # noqa: BLE001
                st.session_state[status_key] = {}
        else:
            st.session_state[status_key] = {}
    return st.session_state[status_key]


def _invalidate_tts_status_cache(voice: str) -> None:
    st.session_state.pop(f"_tts_status_{voice}", None)


def _voice_available_for_status(status: dict) -> bool:
    return bool(
        status.get("piper_installed")
        and status.get("voice_model_exists")
        and status.get("voice_config_exists")
    )


def _render_tts_sidebar() -> tuple[bool, str]:
    """Render TTS-related sidebar controls.

    Returns:
        (voice_enabled, selected_voice)
    """
    st.sidebar.divider()
    st.sidebar.subheader("🔊 Voice Settings")

    voice_enabled = st.sidebar.checkbox(
        "Enable Voice Response",
        value=st.session_state.get("voice_enabled", False),
        key="voice_enabled_cb",
        help="Automatically read every chatbot response aloud using Piper TTS.",
    )
    st.session_state["voice_enabled"] = voice_enabled

    prev_voice = st.session_state.get("_prev_voice_selection", None)
    selected_voice = st.sidebar.selectbox(
        "Voice",
        options=["English Female", "English Male"],
        index=0,
        key="voice_selection",
        help="Choose the Piper TTS voice model.",
    )

    if selected_voice != prev_voice:
        _invalidate_tts_status_cache(selected_voice)
        st.session_state["_prev_voice_selection"] = selected_voice

    # Developer debug mode (kept internal)
    st.sidebar.divider()
    st.sidebar.subheader("🛠️ Developer")
    debug_enabled = bool(os.environ.get("DEBUG_MODE", "0") == "1")
    st.session_state["debug_mode"] = debug_enabled

    # Single source of truth: voice availability
    status = _get_tts_status(selected_voice)
    voice_available = _voice_available_for_status(status)
    st.session_state["voice_available"] = voice_available

    if voice_available:
        st.sidebar.success("✅ Piper TTS ready")
    else:
        # Requirement: do not spam warnings; just indicate unavailable.
        st.sidebar.caption("🔇 Voice unavailable (Piper/voice model missing).")

    # If voice isn't available, auto-disable to avoid repeated errors.
    if not voice_available:
        voice_enabled = False
        st.session_state["voice_enabled"] = False

    return voice_enabled, selected_voice


def _render_audio_controls(
    message_id: str,
    answer_text: str,
    voice_enabled: bool,
    selected_voice: str,
    auto_play: bool = False,
) -> None:
    """Render audio controls.

    Requirement #8: hide "Listen" button when voice model is unavailable.
    """
    tts = _get_tts(selected_voice)
    if tts is None:
        return

    status = _get_tts_status(selected_voice)
    voice_available = _voice_available_for_status(status)

    col_listen, col_download = st.columns([1, 1])

    listen_key = f"listen_{message_id}"
    auto_play_key = f"auto_played_{message_id}"

    # Auto-play: generate audio once on first render of a new message
    if auto_play and voice_enabled and voice_available:
        if not st.session_state.get(auto_play_key, False):
            with st.spinner("Generating speech…"):
                audio_path = tts.generate_audio(answer_text)
            if audio_path and os.path.exists(audio_path):
                st.session_state[auto_play_key] = True
                st.audio(audio_path, format="audio/wav")
        return

    # Listen button (hidden when voice unavailable)
    with col_listen:
        if not voice_available:
            st.empty()
        else:
            if st.button("🔊 Listen", key=listen_key):
                with st.spinner("Generating speech…"):
                    audio_path = tts.generate_audio(answer_text)
                if audio_path and os.path.exists(audio_path):
                    st.audio(audio_path, format="audio/wav")
                else:
                    from src.text_to_speech import get_last_piper_diagnostics

                    diag = get_last_piper_diagnostics() if callable(get_last_piper_diagnostics) else {}
                    last_err = diag.get("last_error") if isinstance(diag, dict) else None
                    if last_err:
                        st.error(f"Audio generation failed.\n\nPiper details:\n{last_err}")
                    else:
                        st.error("Audio generation failed. Check logs/piper.log for details.")

    # Download button (only when cached WAV exists)
    with col_download:
        if not voice_available:
            st.empty()
            return

        try:
            cache_dir = getattr(tts, "cache_dir", None)
            if cache_dir is None:
                # Fallback: internal module constant
                from src import text_to_speech

                cache_dir = getattr(text_to_speech, "CACHE_DIR", None)

            if not cache_dir:
                return

            key_hash = hashlib.sha256(f"{selected_voice}:{answer_text}".encode()).hexdigest()[:20]
            cached_wav = os.path.join(cache_dir, f"{key_hash}.wav")

            if os.path.exists(cached_wav):
                with open(cached_wav, "rb") as f:
                    wav_bytes = f.read()
                st.download_button(
                    label="⬇ Download Audio",
                    data=wav_bytes,
                    file_name="legal_response.wav",
                    mime="audio/wav",
                    key=f"dl_{message_id}",
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[Chatbot] Download audio cache lookup failed: %s", exc)


def _render_sources(sources: list, retrieved_chunks: list) -> None:
    """Render only the document actually used for the answer."""
    if sources:
        with st.expander("📚 Source Documents", expanded=False):
            for i, s in enumerate(sources, 1):
                doc = s.get("document", s.get("source_document", "Unknown"))
                st.markdown(f"**{i}. {doc}**")


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------
def render_chatbot(chatbot=None):
    st.title("⚖️ Legal Assistant Chat")

    if not chatbot:
        st.error("❌ Chatbot pipeline is not initialized.")
        return

    voice_enabled, selected_voice = _render_tts_sidebar()

    # Session state init (idempotent)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "message_data" not in st.session_state:
        st.session_state.message_data = {}
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    if "_last_db_prompt" not in st.session_state:
        st.session_state["_last_db_prompt"] = None

    # Replay history
    for idx, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant":
                msg_id = f"hist_{idx}"
                data = st.session_state.message_data.get(msg_id, {})
                sources = data.get("sources", [])
                chunks = data.get("retrieved_chunks", [])
                _render_sources(sources, chunks)
                _render_audio_controls(
                    message_id=msg_id,
                    answer_text=content,
                    voice_enabled=voice_enabled,
                    selected_voice=selected_voice,
                    auto_play=False,
                )

    # New user input
    if prompt := st.chat_input(
        "Ask a legal question… e.g. 'What is the punishment for murder?' or 'IPC Section 302'"
    ):
        if st.session_state.pending_prompt == prompt:
            logger.debug("[Chatbot] Ignoring duplicate prompt submission: %s", prompt[:60])
            return

        st.session_state.pending_prompt = prompt
        logger.info("[Chatbot] New user prompt: %s", prompt[:120])
        st.chat_message("user").markdown(prompt)

        if not st.session_state.messages or st.session_state.messages[-1] != {"role": "user", "content": prompt}:
            st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching legal documents…"):
                try:
                    session_history = [
                        {"question": m["content"]}
                        for m in st.session_state.messages
                        if m["role"] == "user"
                    ]
                    if session_history:
                        session_history.pop()

                    def _safe_call_engine() -> dict:
                        try:
                            if hasattr(chatbot, "generate_response"):
                                return chatbot.generate_response(
                                    query=prompt,
                                    original_query=prompt,
                                    history=session_history,
                                )
                            if hasattr(chatbot, "answer"):
                                try:
                                    return chatbot.answer(user_query=prompt)
                                except TypeError:
                                    return chatbot.answer(prompt)
                            if hasattr(chatbot, "chat"):
                                return chatbot.chat(
                                    prompt,
                                    session_history=session_history,
                                    last_sources=st.session_state.last_sources,
                                )
                        except Exception:
                            logger.exception("[Chatbot] Engine call failed")
                        return {}

                    response_data = _safe_call_engine()
                    answer = response_data.get("answer", "")
                    sources = response_data.get("sources", [])
                    retrieved_chunks = response_data.get("retrieved_chunks", [])
                    confidence = response_data.get("confidence", 0.0)

                    if not answer or not answer.strip():
                        answer = "I couldn't find relevant information in the uploaded legal documents."

                    logger.info("[Chatbot] Answer generated (len=%d, confidence=%.3f)", len(answer), confidence or 0.0)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("[Chatbot] Error generating response: %s", exc)
                    st.error("Unable to generate legal response.")
                    answer = "Unable to generate legal response."
                    sources = []
                    retrieved_chunks = []

            st.markdown(answer)
            _render_sources(sources, retrieved_chunks)

            msg_id = f"new_{len(st.session_state.messages)}"
            _render_audio_controls(
                message_id=msg_id,
                answer_text=answer,
                voice_enabled=voice_enabled,
                selected_voice=selected_voice,
                auto_play=True,
            )

        if not st.session_state.messages or st.session_state.messages[-1] != {"role": "assistant", "content": answer}:
            st.session_state.messages.append({"role": "assistant", "content": answer})

        hist_idx = len(st.session_state.messages) - 1
        st.session_state.message_data[f"hist_{hist_idx}"] = {"sources": sources, "retrieved_chunks": retrieved_chunks}

        if sources:
            st.session_state.last_sources = [s.get("document", s.get("source_document", "")) for s in sources]

        if st.session_state.get("user_id") and st.session_state.get("_last_db_prompt") != prompt:
            db_generator = get_db()
            db = next(db_generator)
            try:
                chat_log = ChatHistory(
                    user_id=st.session_state["user_id"],
                    question=prompt,
                    answer=answer,
                )
                db.add(chat_log)
                db.commit()
                st.session_state["_last_db_prompt"] = prompt
                logger.info("[Chatbot] Chat entry saved to database.")
            except Exception as exc:  # noqa: BLE001
                db.rollback()
                logger.error("[Chatbot] DB write failed: %s", exc)
                st.error(f"Failed to log chat to database: {exc}")
            finally:
                db_generator.close()

        st.session_state.pending_prompt = None

