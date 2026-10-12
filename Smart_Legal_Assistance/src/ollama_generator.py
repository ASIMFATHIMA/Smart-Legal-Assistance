"""Smart Legal Assistance – Ollama Answer Generator
====================================================
Fixes applied:
  - Default model changed to qwen2.5:7b (was qwen3:0.6b).
  - Prompt updated to request structured legal response format.
  - Handle both dict and object-style ollama message chunks (newer library).
  - DEBUG_MODE prints the full prompt sent to Ollama.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("rag_engine")

# Ollama is optional – graceful degradation if not installed
try:
    import ollama  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    ollama = None

DEBUG_MODE: bool = os.environ.get("DEBUG_MODE", "0") == "1"

# Structured legal response template injected into the prompt
_STRUCTURE_INSTRUCTIONS = """
Respond in the following structured format when applicable:

**Answer:** (summarized legal answer explaining the response)
**Relevant IPC/BNS Section:** (specific section number, e.g. IPC Section 300 / BNS Section 101, or N/A)
**Punishment:** (details of imprisonment/fine under the section, or N/A if not available)
**Legal Procedure:** (investigation, cognizable/non-cognizable trial, bail terms, or N/A if not available)
**Legal Rights:** (rights of the accused / victim relevant to this, or N/A if not available)
**Case Summary:** (brief case precedents / case descriptions from context, or N/A if not available)
**Source Document:** (name of the document where this was retrieved, or N/A)
**Page Number:** (page number where this was retrieved, or N/A)

If any field is not available in the Context, write N/A for that field. Do NOT invent information. Answer ONLY based on the provided Context.
"""


class OllamaGenerator:
    def __init__(self, model_name: str = "c"):
        self.model_name = model_name

    def generate_answer(self, query: str, context: str, history: str = ""):
        """Stream answer tokens from Ollama given the retrieved context."""

        # Trim history to the last 4 lines for small models
        history_lines = history.strip().split("\n") if history.strip() else []
        if len(history_lines) > 4:
            history_lines = history_lines[-4:]
        trimmed_history = "\n".join(history_lines)

        prompt = f"""You are Smart Legal Assistant – an expert in Indian Criminal Law (IPC, BNS, CrPC).

Rules:
1. Answer ONLY using the provided Context. Do NOT use external knowledge.
2. If information is not in the Context, state: "I couldn't find sufficient information in the provided documents."
3. Use the structured format below when answering legal questions.
4. Mention IPC section and BNS section when available in the Context.
5. Write in clear, plain English understandable by a layperson.
6. Use conversation history ONLY to resolve pronouns (it, this, that) – not as a source of facts.

{_STRUCTURE_INSTRUCTIONS}

Previous Conversation (reference only – NOT a source of facts):
{trimmed_history}

Context (answer ONLY from this):
{context}

Question:
{query}

Answer:
"""

        if DEBUG_MODE:
            print("\n[DEBUG] === PROMPT SENT TO OLLAMA ===")
            print(prompt[:2000])
            print("[DEBUG] === END PROMPT ===\n")

        if ollama is None:
            yield (
                "I couldn't generate an answer because the Ollama service is not available. "
                "Please ensure Ollama is running and the model is pulled."
            )
            return

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in response:
                # Handle both dict and object-style responses (ollama library v0.1+ vs v0.2+)
                content = None
                if isinstance(chunk, dict):
                    msg = chunk.get("message")
                    if isinstance(msg, dict):
                        content = msg.get("content")
                    elif isinstance(msg, str):
                        content = msg
                else:
                    # Object-style: chunk.message.content
                    try:
                        content = chunk.message.content  # type: ignore[attr-defined]
                    except AttributeError:
                        pass

                if content:
                    yield content

        except Exception as exc:
            logger.warning(f"[OllamaGenerator] Error calling model {self.model_name}: {exc}")
            # Try to give a graceful degraded answer
            yield (
                "I encountered an error while generating the answer. "
                f"Model '{self.model_name}' may not be available. "
                "Please run: ollama pull qwen2.5:7b"
            )
