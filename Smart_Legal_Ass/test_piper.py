"""Manual Piper verification script.

Expected behavior:
- Print environment details.
- Run `piper.exe --help`.
- Synthesize "Hello" via stdin -> wav file.
- Exit non-zero only if WAV file is NOT created.

Run:
  python Smart_Legal_Ass/test_piper.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _print_bool(label: str, ok: bool) -> None:
    print(f"{label}: {ok}")


def main() -> int:
    project_root = Path(__file__).resolve().parent
    piper_dir = project_root / "piper"

    python_exe = sys.executable
    cwd = os.getcwd()

    piper_exe = piper_dir / "piper.exe"
    voice_model = piper_dir / "en_US-lessac-medium.onnx"
    voice_config = piper_dir / "en_US-lessac-medium.onnx.json"

    print("=== Piper Test ===")
    print(f"Python executable: {python_exe}")
    print(f"Project root: {project_root}")
    print(f"Piper executable: {piper_exe}")
    print(f"Working directory: {cwd}")

    print("\n--- Voice files ---")
    print(f"Voice model: {voice_model}")
    print(f"Voice config: {voice_config}")

    print("\n--- DLL files ---")
    dlls = [
        "onnxruntime.dll",
        "onnxruntime_providers_shared.dll",
        "piper_phonemize.dll",
        "espeak-ng.dll",
    ]
    for dll in dlls:
        p = piper_dir / dll
        _print_bool(dll, p.exists())

    print("\n--- espeak-ng-data/ ---")
    espeak_data = piper_dir / "espeak-ng-data"
    _print_bool("espeak-ng-data", espeak_data.exists() and espeak_data.is_dir())

    print("\n--- Voice validation ---")
    for p in [voice_model, voice_config]:
        if not p.exists():
            print(f"Missing: {p}")
        else:
            try:
                print(f"{p.name} size: {p.stat().st_size} bytes")
            except Exception:
                pass

    if not piper_exe.exists():
        print("❌ piper.exe missing; aborting")
        return 1

    # Run piper.exe --help
    print("\n--- Running: piper.exe --help ---")
    proc_help = subprocess.run(
        [str(piper_exe), "--help"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        cwd=str(piper_dir),
    )
    print(f"Return code: {proc_help.returncode}")
    print("STDOUT:")
    print(proc_help.stdout[:2000])
    print("STDERR:")
    print(proc_help.stderr[:2000])

    # Run synth: echo Hello | piper.exe ...
    out_wav = project_root / "utils" / "audio_cache" / "test.wav"
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if out_wav.exists():
        try:
            out_wav.unlink()
        except Exception:
            pass

    print("\n--- Running synthesis to test.wav ---")

    cmd = [
        str(piper_exe),
        "--model",
        str(voice_model),
        "--config",
        str(voice_config),
        "--output_file",
        str(out_wav),
    ]

    text = "Hello"
    proc = subprocess.run(
        cmd,
        input=text,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
        cwd=str(piper_dir),
    )

    print(f"Return code: {proc.returncode}")
    print("STDOUT:")
    print(proc.stdout[:2000])
    print("STDERR:")
    print(proc.stderr[:2000])

    # Success criteria
    if not out_wav.exists():
        print("❌ test.wav was not created")
        return 1

    try:
        size = out_wav.stat().st_size
        print(f"test.wav size: {size} bytes")
        if size <= 0:
            print("❌ test.wav is empty")
            return 1
    except Exception as exc:
        print(f"❌ Could not stat test.wav: {exc}")
        return 1

    print("✅ Piper synthesis succeeded; test.wav created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

