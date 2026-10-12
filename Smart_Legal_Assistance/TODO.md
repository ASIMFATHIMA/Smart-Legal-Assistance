# TODO – Smart Legal Assistance (Piper TTS crash-proof)

- [ ] Step 1: Fix `validate_piper_installation()` to return `(True, "")` or `(False, "Reason")`.
- [ ] Step 2: Update `src/text_to_speech.py` to validate Piper + voice model + JSON config before generation.
- [ ] Step 3: Add/ensure session auto-disable for voice when validation fails.
- [ ] Step 4: Update Streamlit UI (`streamlit_app.py`) to treat voice as optional.
- [ ] Step 5: Update `ui/diagnostics.py` to reflect new validation signature.
- [ ] Step 6: Update `ui/chatbot.py` to hide “Listen” when voice model unavailable and use required warning text.
- [ ] Step 7: Add Windows Piper installation instructions (folder structure) in relevant docs/comments.
- [ ] Step 8: Quick run/import sanity checks.

