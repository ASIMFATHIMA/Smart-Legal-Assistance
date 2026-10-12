import pytest

from src.intent_classifier import IntentClassifier


def test_intent_greeting():
    c = IntentClassifier()
    out = c.classify("hi")
    assert out.intent == "GREETING"


def test_intent_summarize():
    c = IntentClassifier()
    out = c.classify("summarize this")
    assert out.intent == "SUMMARIZE"


def test_intent_simplify():
    c = IntentClassifier()
    out = c.classify("simplify this")
    assert out.intent == "SIMPLIFY"


def test_intent_legal_query():
    c = IntentClassifier()
    out = c.classify("What is section 376DA?")
    assert out.intent in ("LEGAL_QUERY", "FOLLOWUP")


def test_intent_unknown():
    c = IntentClassifier()
    out = c.classify("blablabla")
    assert out.intent == "UNKNOWN"

