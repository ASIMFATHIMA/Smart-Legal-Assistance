"""Smart Legal Assistance - Phase 2 Module: Response Templates

Centralizes response strings for consistent UX.
"""

from __future__ import annotations


def greeting_template() -> str:
    # Friendly greeting for greeting-only intents (no document search)
    return (
        "👋 Hello! Welcome to Smart Legal Assistance.\n\n"
        "I can help you with:\n"
        "• IPC Sections\n"
        "• BNS Sections\n"
        "• Crime Information\n"
        "• Punishments\n"
        "• Legal Rights\n"
        "• Legal Procedures\n"
        "• Case Summaries\n\n"
        "Please ask your legal question."
    )


def low_confidence_template() -> str:
    return "I could not find sufficient information in the legal documents to answer confidently."


def unknown_template() -> str:
    return "I’m not sure how to help with that request. Please ask a legal question, or say 'summarize' or 'simplify'."

