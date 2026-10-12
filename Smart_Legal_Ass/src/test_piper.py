"""Piper CLI test runner.

Runs required Piper commands and prints:
- Return Code
- STDOUT
- STDERR

This script is intentionally NOT used by Streamlit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPER_DIR = PROJECT_ROOT / "piper"
PIPER_EXE = PIPER_DIR / "piper.exe"
VOICE_MODEL = PIPER_DIR / "en_US-lessac-medium.onnx"
VOICE_CONFIG = PIPER_DIR / "en_US-lessac-medium.onnx.json"


def _run(cmd: list[str], input_text: str | None = None, timeout: int = 60) -> None:
    print("\n=== Executing ===")
    print("Command:", cmd)
    print("Python:", sys.executable)
    print("CWD:", str(Path.cwd()))

    try:
        cp = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            capture_output=True,
            cwd=str(PIPER_DIR),
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        print("Return Code: TIMEOUT")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        return

    print("Return Code:", cp.returncode)
    print("STDOUT:\n", cp.stdout)
    print("STDERR:\n", cp.stderr)


def main() -> None:
    print("Piper dir:", PIPER_DIR)
    print("Piper exe:", PIPER_EXE)
    print("Voice model:", VOICE_MODEL)
    print("Voice config:", VOICE_CONFIG)

    if not PIPER_EXE.exists():
        raise SystemExit(f"Missing Piper executable: {PIPER_EXE}")

    _run([str(PIPER_EXE), "--help"])

    # test.wav output created in piper dir
    test_out = PIPER_DIR / "test.wav"
    if test_out.exists():
        try:
            test_out.unlink()
        except Exception:
            pass

    _run(
        [
            str(PIPER_EXE),
            "--model",
            str(VOICE_MODEL),
            "--config",
            str(VOICE_CONFIG),
            "--output_file",
            "test.wav",
        ],
        input_text="Hello\n",
        timeout=120,
    )

    created = test_out.exists() and test_out.stat().st_size > 0
    if not created:
        print("\n❌ test.wav was not created or is empty.")
        print("Reason: check STDERR output above for the root cause.")
    else:
        print("\n✅ test.wav created:", test_out, "size=", test_out.stat().st_size)


if __name__ == "__main__":
    main()

