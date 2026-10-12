"""Smart Legal Assistance – Diagnostics UI

This Streamlit page validates the local Piper + voice model installation and
shows the last Piper return code / stderr summary.

The app imports this module from Smart_Legal_Assistance/streamlit_app.py.
So this file must exist to prevent import-time crashes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from src.text_to_speech import (
    PiperTTS,
    get_last_piper_diagnostics,
    validate_piper_installation,
)


def _render_status_table(rows: Dict[str, bool]) -> None:
    """Render a two-column status table for validation checks."""
    for label, ok in rows.items():
        st.write(f"{'✅' if ok else '❌'} {label}")


def render_diagnostics() -> None:
    st.title("🔧 Piper TTS Diagnostics")
    st.write(
        "Validates `piper.exe` + required DLLs and checks that the configured voice "
        "model/config look usable before attempting synthesis."
    )

    st.subheader("1) Installation checks")
    ok = False
    reason = ""
    try:
        ok, reason = validate_piper_installation()
    except Exception as exc:  # pragma: no cover
        st.error(f"Validation failed: {exc}")
        return

    # Show exact file-based reason
    st.write(f"{'✅' if ok else '❌'} Piper installation")
    if ok:
        st.success("Piper + voice model assets are present.")
    else:
        st.warning(reason or "Voice model is not installed. Chatbot will continue without voice.")


    st.subheader("2) Voice generation")

    tts = None
    try:
        # Voice selection affects hashing/cache only in this repo; model files are fixed.
        tts = PiperTTS(voice="English Female")
        status = tts.check_status()
        # PiperTTS.check_status() fields
        st.json({
            "piper_installed": status.get("piper_installed"),
            "piper_path": status.get("piper_path"),
            "voice": status.get("voice"),
            "voice_model_exists": status.get("voice_model_exists"),
            "voice_config_exists": status.get("voice_config_exists"),
            "model_path": status.get("model_path"),
            "voice_config_path": status.get("voice_config_path"),
            "cache_dir": status.get("cache_dir"),
        })

    except Exception as exc:  # pragma: no cover
        st.warning(f"Could not load PiperTTS status: {exc}")

    last = get_last_piper_diagnostics()
    st.subheader("3) Last Piper error")
    st.write(f"Last return code: `{last.get('last_return_code')}`")
    if last.get("last_error"):
        st.code(last["last_error"])
    else:
        st.info("No Piper run diagnostics available yet in this session.")

    st.subheader("4) Quick synthesis test")
    test_text = st.text_input(
        "Text to synthesize",
        value="This is a Piper TTS diagnostics test.",
    )

    run_clicked = st.button("Run synthesis", key="diag_run")
    if run_clicked:
        if not tts:
            st.error("PiperTTS could not be initialized.")
            return

        with st.spinner("Running Piper…"):
            wav_path = tts.generate_audio(test_text)

        if wav_path and os.path.exists(wav_path):
            st.success("✅ WAV generated")
            st.audio(wav_path, format="audio/wav")
        else:
            st.error("❌ WAV was not generated. Check logs/piper.log and the last error above.")


