"""Smart Legal Assistance – Piper Text-to-Speech (crash-proof)

Production-ready defensive Piper TTS wrapper for Streamlit.

Key behaviors:
- Validates Piper + required voice assets before attempting synthesis.
- If anything is missing/unusable, voice generation is disabled automatically.
- Never crashes the chatbot.
- Detailed logging to logs/piper.log and (via logging handlers) console.
- Ensures Piper receives only plain text (no emojis/markdown/HTML/symbols) to
  avoid UnicodeEncodeError on Windows code pages.

Task additions (2026):
- clean_text_for_tts(text: str) -> str
- UnicodeEncodeError mitigation by cleaning before calling Piper.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple


# ---------------------------- Paths -------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPER_DIR = PROJECT_ROOT / "piper"
PIPER_EXE = PIPER_DIR / "piper.exe"

FEMALE_ONNX = "en_US-lessac-medium.onnx"
FEMALE_JSON = "en_US-lessac-medium.onnx.json"

VOICE_MODELS: Dict[str, Tuple[str, str]] = {
    "English Female": (FEMALE_ONNX, FEMALE_JSON),
    # Best-effort: fallback to the bundled female voice files.
    "English Male": (FEMALE_ONNX, FEMALE_JSON),
}


def _voice_paths(voice: str) -> Tuple[Path, Path]:
    onnx_name, json_name = VOICE_MODELS.get(voice, VOICE_MODELS["English Female"])
    return PIPER_DIR / onnx_name, PIPER_DIR / json_name


CACHE_DIR = PROJECT_ROOT / "utils" / "audio_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Backward-compatible alias (some older code may import this)
_CACHE_DIR = CACHE_DIR


# ---------------------------- Logging -------------------
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PIPER_LOG_PATH = LOG_DIR / "piper.log"

logger = logging.getLogger("piper")
_last_piper_return_code: Optional[int] = None
_last_piper_error: Optional[str] = None
_last_piper_stdout: Optional[str] = None
_last_piper_stderr: Optional[str] = None

if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(PIPER_LOG_PATH, encoding="utf-8")
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def get_last_piper_diagnostics() -> Dict[str, Optional[str]]:
    return {
        "last_return_code": str(_last_piper_return_code)
        if _last_piper_return_code is not None
        else None,
        "last_error": _last_piper_error,
        "last_stdout": _last_piper_stdout,
        "last_stderr": _last_piper_stderr,
    }


def _log_exception(context: str, exc: BaseException) -> None:
    global _last_piper_error
    tb = traceback.format_exc()
    _last_piper_error = f"{exc.__class__.__name__}: {exc}"
    logger.error("%s – %s\n%s", context, exc.__class__.__name__, tb)


# Optional Streamlit import
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None


def _ui_warning(message: str) -> None:
    logger.warning(message)
    if st is not None:
        try:
            st.warning(message)
        except Exception:
            pass


def _set_session_voice_disabled() -> None:
    if st is None:
        return
    st.session_state["voice_disabled"] = True


def _is_session_voice_disabled() -> bool:
    if st is None:
        return False
    return bool(st.session_state.get("voice_disabled", False))


# ---------------------------- Validation -------------------
_REQUIRED_DLLS = [
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
    "piper_phonemize.dll",
    "espeak-ng.dll",
]


def validate_piper_installation() -> Tuple[bool, str]:
    """Validate Piper installation and bundled voice model assets.

    Requirement #5: return (True, "") or (False, "Reason").
    """

    if not PIPER_EXE.exists():
        return (False, f"Missing piper executable: {PIPER_EXE}")

    voice_model_path, voice_config_path = _voice_paths("English Female")

    if not voice_model_path.exists():
        return (False, f"Missing voice model ONNX: {voice_model_path}")

    if not voice_config_path.exists():
        return (False, f"Missing voice model JSON config: {voice_config_path}")

    # Sanity check some required Piper assets are present.
    espeak_data = PIPER_DIR / "espeak-ng-data"
    if not espeak_data.exists():
        return (False, f"Missing espeak-ng-data folder: {espeak_data}")

    return (True, "")


# ---------------------------- TTS Text Cleaning -------------------
def clean_text_for_tts(text: str) -> str:
    """Clean model output before passing it to Piper.

    Piper (Windows) must receive plain text without emojis/symbols/markdown/HTML.

    Removes:
    - All emojis
    - Unicode symbols
    - Markdown
    - HTML tags
    - Extra spaces
    - Invisible characters

    Also converts common "label value" patterns into a readable multi-line form.
    """

    if text is None:
        return ""

    s = str(text)

    # Invisible characters / zero-width / control
    s = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]", " ", s)

    # Remove HTML tags
    s = re.sub(r"<[^>]+>", " ", s)

    # Remove markdown links/images (keep link text)
    s = re.sub(r"!\[[^\]]*\]\(([^)]+)\)", r"\1", s)
    s = re.sub(r"\[[^\]]*\]\(([^)]+)\)", r"\1", s)

    # Remove markdown emphasis/bullets/headers
    s = s.replace("#", " ")
    s = re.sub(r"[*_>`~\-]{1,3}", " ", s)

    # Remove supplementary-plane unicode (includes many emojis)
    s = re.sub(r"[\U00010000-\U0010FFFF]", " ", s)

    # Remove common emoji ranges in BMP
    s = re.sub(r"[\u2600-\u27BF\uFE0F]", " ", s)

    # Remove specific trademark/copyright etc.
    s = re.sub(r"[\u00A9\u00AE\u2122\u203C\u2049\u20E3]", " ", s)

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    raw_lines = [ln.strip() for ln in re.split(r"\r?\n", s) if ln.strip()]

    label_map = {
        "Crime": "Crime",
        "Description": "Description",
        "Ingredients": "Ingredients",
        "Punishment": "Punishment",
        "Important": "Important Case",
        "ImportantCase": "Important Case",
        "Case": "Important Case",
        "BNS": "BNS Equivalent",
        "Equivalent": "BNS Equivalent",
    }

    out_lines: list[str] = []

    for ln in raw_lines:
        parts = ln.split()
        if not parts:
            continue

        # Important Cases -> Important Case: <name>
        if ln.lower().startswith("important"):
            cleaned = re.sub(
                r"^important\s+cases?\s*:\s*",
                "",
                ln,
                flags=re.IGNORECASE,
            )
            cleaned = cleaned.strip()
            if cleaned:
                out_lines.append("Important Case:")
                out_lines.append(cleaned)
            else:
                out_lines.append("Important Case")
            continue


        # BNS Equivalent
        if ln.lower().startswith("bns") or "bns" in ln.lower():
            cleaned = re.sub(
                r"^bns\s+(equivalent)?\s*:\s*",
                "",
                ln,
                flags=re.IGNORECASE,
            )
            cleaned = cleaned.strip()
            if cleaned:
                out_lines.append("BNS Equivalent:")
                out_lines.append(cleaned)
            else:
                out_lines.append("BNS Equivalent")
            continue


        # Generic label split (Label Value...)
        if len(parts) >= 2 and parts[0] in label_map:
            label = label_map[parts[0]]
            rest = " ".join(parts[1:]).rstrip(".")
            out_lines.append(f"{label}:")
            out_lines.append(rest)
            continue

        # Special-case: IPC/BNS section line ("IPC Section 300")
        if re.match(r"^(IPC\s+Section|BNS\s+Equivalent\s*|IPC\s+Section)\b", ln, flags=re.IGNORECASE):
            out_lines.append(ln)
            continue

        # Default: keep line, ensure it ends with period/colon-like punctuation.
        out_lines.append(ln)

    # Post-format: collapse repeated whitespace and add periods for single-line items.
    final_lines: list[str] = []
    for ln in out_lines:
        ln = ln.strip()
        if not ln:
            continue
        if ln.endswith(":"):
            final_lines.append(ln)
            continue
        if not ln.endswith(".") and not ln.endswith(":"):
            ln = ln + "."
        final_lines.append(ln)

    final = "\n".join(final_lines)
    final = re.sub(r"\n{3,}", "\n\n", final).strip()
    final = re.sub(r"[ \t]+", " ", final)
    return final


# ---------------------------- Piper Command -------------------
def _build_command(voice_model: Path, voice_config: Path, wav_path: Path) -> Tuple[list[str], str]:
    cmd = [
        str(PIPER_EXE),
        "--model",
        str(voice_model),
        "--config",
        str(voice_config),
        "--output_file",
        str(wav_path),
    ]
    return cmd, " ".join(cmd)


def _expected_wav_path(cache_dir: Path, voice: str, text: str) -> Path:
    key = f"{voice}:{text.strip()}".encode("utf-8")
    file_hash = hashlib.sha256(key).hexdigest()[:20]
    return cache_dir / f"{file_hash}.wav"


# ---------------------------- Public API -------------------
class PiperTTS:
    def __init__(self, voice: str = "English Female"):
        self.voice = voice

    def check_status(self) -> dict:
        voice_model_path, voice_config_path = _voice_paths(self.voice)
        return {
            "piper_installed": PIPER_EXE.exists(),
            "piper_path": str(PIPER_EXE),
            "voice": self.voice,
            "voice_model_exists": voice_model_path.exists(),
            "voice_config_exists": voice_config_path.exists(),
            "model_path": str(voice_model_path),
            "voice_config_path": str(voice_config_path),
            "cache_dir": str(CACHE_DIR),
        }

    def generate_audio(self, text: object) -> Optional[str]:
        """Generate speech audio.

        Returns:
            Absolute path to WAV on success, else None.
        """

        global _last_piper_return_code, _last_piper_error, _last_piper_stdout, _last_piper_stderr

        try:
            if _is_session_voice_disabled():
                return None

            if not isinstance(text, str) or not text.strip():
                return None

            # Validate installation first
            ok, reason = validate_piper_installation()
            if not ok:
                _set_session_voice_disabled()
                _ui_warning("Voice model is not installed. Chatbot will continue without voice.")
                logger.error("Piper validation failed: %s", reason)
                return None

            if self.voice not in VOICE_MODELS:
                self.voice = "English Female"

            voice_model_path, voice_config_path = _voice_paths(self.voice)
            if not voice_model_path.exists() or not voice_config_path.exists():
                _set_session_voice_disabled()
                _ui_warning("Voice model is not installed. Chatbot will continue without voice.")
                return None

            # Clean text for Piper (UnicodeEncodeError mitigation)
            tts_text = clean_text_for_tts(text)
            if not tts_text.strip():
                return None

            CACHE_DIR.mkdir(parents=True, exist_ok=True)

            wav_path = _expected_wav_path(CACHE_DIR, self.voice, tts_text)

            if wav_path.exists() and wav_path.stat().st_size > 0:
                return str(wav_path.resolve())

            # Write cleaned UTF-8 text file for Piper (if Piper expects stdin text,
            # it also tends to work; but requirement asks to save UTF-8 text).
            text_file = CACHE_DIR / f"{wav_path.stem}.txt"
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(tts_text)

            cmd, cmd_str = _build_command(voice_model_path, voice_config_path, wav_path)
            cwd = str(PIPER_DIR)

            logger.info(
                "Piper executable path=%s\nVoice model path=%s\nJSON config path=%s\n"
                "Output WAV path=%s\nExact subprocess command=%s\nWorking directory=%s",
                PIPER_EXE,
                voice_model_path,
                voice_config_path,
                wav_path,
                cmd_str,
                cwd,
            )

            try:
                result = subprocess.run(
                    cmd,
                    input=tts_text,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    capture_output=True,
                    cwd=cwd,
                    timeout=60,
                    check=False,
                )
            except Exception as exc:
                _last_piper_return_code = None
                _last_piper_stdout = None
                _last_piper_stderr = None
                _last_piper_error = f"{exc.__class__.__name__}: {exc}"
                _log_exception("Piper subprocess exception", exc)
                _ui_warning("Voice generation unavailable.\nThe chatbot will continue normally.")
                return None

            _last_piper_return_code = result.returncode
            _last_piper_stdout = result.stdout or ""
            _last_piper_stderr = result.stderr or ""
            _last_piper_error = (
                (_last_piper_stderr.strip() or _last_piper_stdout.strip())[:2000]
                if result.returncode != 0
                else None
            )

            logger.info("Piper subprocess return code=%s", result.returncode)
            logger.info("Piper stdout=%s", _last_piper_stdout)
            logger.info("Piper stderr=%s", _last_piper_stderr)

            if result.returncode != 0:
                _ui_warning("Voice generation unavailable.\nThe chatbot will continue normally.")
                return None

            if not wav_path.exists() or wav_path.stat().st_size == 0:
                _ui_warning("Voice generation unavailable.\nThe chatbot will continue normally.")
                return None

            return str(wav_path.resolve())

        except Exception as exc:
            _log_exception("PiperTTS.generate_audio top-level", exc)
            _ui_warning("Voice generation unavailable.\nThe chatbot will continue normally.")
            return None

    def speak(self, text: object) -> Optional[str]:
        """Best-effort playback (never raises)."""
        wav_path = self.generate_audio(text)
        if not wav_path:
            return None

        try:
            if sys.platform == "win32":
                subprocess.Popen(
                    ["cmd", "/c", "start", "", wav_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception as exc:
            _log_exception("PiperTTS.speak playback", exc)
            return None

        return wav_path

