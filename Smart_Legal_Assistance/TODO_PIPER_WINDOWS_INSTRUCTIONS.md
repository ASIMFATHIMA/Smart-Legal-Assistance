# Piper TTS Installation (Windows)

This project uses **Piper** for Text-to-Speech.

## Required folder structure

Place Piper and the voice assets exactly under:

```text
Smart_Legal_Assistance/
│
├── piper/
│   ├── piper.exe
│   ├── en_US-lessac-medium.onnx
│   ├── en_US-lessac-medium.onnx.json
│
├── src/
├── ui/
├── streamlit_app.py
```

## Files you must have

1. **Piper executable**
   - `Smart_Legal_Assistance/piper/piper.exe`

2. **Voice model (ONNX)**
   - `Smart_Legal_Assistance/piper/en_US-lessac-medium.onnx`

3. **Voice config (JSON)**
   - `Smart_Legal_Assistance/piper/en_US-lessac-medium.onnx.json`

> If any of the above files are missing, the app will **disable voice generation automatically** and continue answering questions normally.

## Notes

- The app validates at runtime that `piper.exe`, the `.onnx` model, and the `.onnx.json` file exist.
- All voice assets are looked up relative to `Smart_Legal_Assistance/piper/`.
