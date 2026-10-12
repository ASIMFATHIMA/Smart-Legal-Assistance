"""Smart Legal Assistance – Diagnostics page for Piper + voice files.

Shows:
- ✅/❌ Piper exe + required DLLs
- ✅/❌ voice model + voice config
- espeak-ng-data/ existence
- Python version
- CWD
- Project root
- last return code + last Piper error

Never raises.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

import streamlit as st

from src.text_to_speech import (
    PIPER_EXE,
    _PIPER_DIR,
    _VOICE_MODEL,
    _VOICE_CONFIG,
    _logger,
    validate_piper_installation,
    get_last_piper_diagnostics,
    _PROJECT_ROOT,
)


def _row_status(label: str, ok: bool, path: Path | None = None) -> None:
    status = "✅ Found" if ok else "❌ Missing"
    if path is not None:
        st.write(f"{status} — {label}")
        st.caption(str(path))
    else:
        st.write(f"{status} — {label}")


def render_diagnostics() -> None:
    try:
        st.title("Diagnostics")

        st.subheader("Piper installation")
        install: Dict[str, bool] = validate_piper_installation()

        _row_status("piper.exe", install.get("piper.exe", False), PIPER_EXE)
        # DLLs
        for dll_label in [
            "onnxruntime.dll",
            "onnxruntime_providers_shared.dll",
            "piper_phonemize.dll",
            "espeak-ng.dll",
        ]:
            # The label exists in validate_piper_installation via fixed paths
            # but we rebuild paths here for captions.
            p = None
            if dll_label == "onnxruntime.dll":
                p = _PIPER_DIR / "onnxruntime.dll"
            elif dll_label == "onnxruntime_providers_shared.dll":
                p = _PIPER_DIR / "onnxruntime_providers_shared.dll"
            elif dll_label == "piper_phonemize.dll":
                p = _PIPER_DIR / "piper_phonemize.dll"
            elif dll_label == "espeak-ng.dll":
                p = _PIPER_DIR / "espeak-ng.dll"
            _row_status(dll_label, install.get(dll_label, False), p)

        # espeak-ng-data/
        data_dir = _PIPER_DIR / "espeak-ng-data"
        _row_status("espeak-ng-data/", data_dir.exists(), data_dir)

        st.subheader("Voice model")
        _row_status("Voice Model (.onnx)", install.get("Voice Model (.onnx)", False), _VOICE_MODEL)
        _row_status(
            "Voice Config (.onnx.json)", install.get("Voice Config (.onnx.json)", False), _VOICE_CONFIG
        )

        st.subheader("Environment")
        st.write(f"Python Version: {sys.version}")
        st.write(f"Current Working Directory: {os.getcwd()}")
        st.write(f"Project Root: {_PROJECT_ROOT}")

        st.subheader("Last Piper error")
        last = get_last_piper_diagnostics()
        st.write(f"Return Code: {last.get('last_return_code')}")
        err = last.get("last_error")
        st.write(f"Last Error: {err if err else 'N/A'}")

        st.caption(f"Log file: {Path('logs') / 'piper.log'}")

    except Exception as exc:  # never crash streamlit
        try:
            st.error(f"Diagnostics page failed: {exc}")
        except Exception:
            pass

