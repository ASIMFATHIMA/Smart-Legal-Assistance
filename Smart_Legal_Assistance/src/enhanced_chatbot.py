import re
from src.enhanced_rag_engine import EnhancedRAGEngine
from src.intent_classifier import IntentClassifier
from src.context_resolver import ContextResolver
from src.memory_manager import MemoryManager
from src.memory import ConversationMemory
import os
from src.legal_term_mapper import LegalTermMapper


class EnhancedChatbot:
    def __init__(self, rag_engine: EnhancedRAGEngine):
        self.rag_engine = rag_engine
        self.intent_classifier = IntentClassifier()
        self.context_resolver = ContextResolver()
        self.term_mapper = LegalTermMapper()

        # Initialize memory
        mem_path = os.path.join("logs", "conversation_history.json")
        base_memory = ConversationMemory(memory_path=mem_path)
        base_memory.load()
        self.memory_manager = MemoryManager(base_memory)

    def _is_legal_query(self, query: str) -> bool:
        legal_keywords = {
            "section", "ipc", "bns", "crpc", "law", "court", "punish", "imprisonment", "fine", "arrest",
            "jail", "bail", "rights", "accused", "victim", "complaint", "fir", "police", "offence",
            "offense", "illegal", "legal", "murder", "theft", "rape", "cheating", "assault", "robbery",
            "kidnapping", "homicide", "forgery", "dowry", "defamation", "bribe", "conspiracy", "trespass",
            "matrix", "document", "statute", "code", "act", "case", "precedent", "culpable", "constitution"
        }
        words = re.findall(r"\w+", query.lower())
        return any(w in legal_keywords for w in words)

    def chat(self, user_query: str, session_history: list[dict] = None, last_sources: list = None) -> dict:
        """Main entry point for chat interactions."""
        try:
            q_lower = user_query.lower().strip()
            state = self.memory_manager.get_state()

            # Strict intent: Where did you get this information?
            source_phrases = ["where did you get", "what is the source", "source", "where is it from"]
            if any(phrase in q_lower for phrase in source_phrases):
                current_doc = state.get("current_document")
                if current_doc:
                    return {
                        "answer": f"This information comes from {current_doc}.",
                        "sources": [current_doc],
                        "confidence": 1.0,
                    }
                elif last_sources:
                    sources_str = ", ".join(last_sources)
                    return {
                        "answer": f"This information comes from {sources_str}.",
                        "sources": last_sources,
                        "confidence": 1.0,
                    }
                else:
                    return {
                        "answer": "I don't have a specific source document referenced for the previous answer.",
                        "sources": [],
                        "confidence": 1.0,
                    }

            # Greeting intent
            greetings_map = {
                "hi": "Hello! How can I assist you with Indian Criminal Law today?",
                "hello": "Hello! I am your AI Legal Assistant. How can I help you today?",
                "hey": "Hey there! What legal query or document can I help you explore?",
                "good morning": "Good morning! How can I assist you today?",
                "good afternoon": "Good afternoon! How can I assist you today?",
                "good evening": "Good evening! How can I help you tonight?"
            }
            # Clean punctuation and space
            q_clean = re.sub(r'[^\w\s]', '', q_lower).strip()
            if q_clean in greetings_map:
                reply = greetings_map[q_clean]
                self.memory_manager.memory.add_turn(user_query, reply)
                return {
                    "answer": reply,
                    "sources": [],
                    "confidence": 1.0,
                }

            # Get formatted history
            history_str = self.memory_manager.memory.to_context_string(max_turns=6)

            # Check if it is a general conversation query (non-legal)
            if not self._is_legal_query(user_query):
                answer_parts = []
                try:
                    for token in self.rag_engine.answer_generator.generate_answer(
                        query=user_query, 
                        context="No legal document context is available. Answer general questions as a friendly AI assistant.", 
                        history=history_str
                    ):
                        answer_parts.append(token)
                    answer = "".join(answer_parts).strip()
                except Exception as exc:
                    answer = f"Hello! I am here to help you with legal documents. For general questions, I encountered an error: {exc}"
                
                self.memory_manager.memory.add_turn(user_query, answer)
                return {
                    "answer": answer,
                    "sources": [],
                    "confidence": 1.0,
                }

            # Rewrite query using state
            rewritten_query = self.context_resolver.resolve(user_query, state)

            # Map terms on the raw query to store in memory or log
            mapped_query = self.term_mapper.map_terms(rewritten_query)

            # Get answer (legal queries only)
            response = self.rag_engine.generate_response(
                mapped_query,
                original_query=user_query,
                history=history_str,
            )

            # Validation: if the engine couldn't extract a valid legal record,
            # enforce the required output message.
            if response.get("answer") == "No relevant legal information found.":
                return {"answer": "No relevant legal information found.", "sources": [], "confidence": 0.0}


            # Update memory history
            self.memory_manager.memory.add_turn(user_query, response.get("answer", ""))

            # Update state — use original user_query so topic keywords are detectable
            self.memory_manager.update_state(
                user_query=user_query,
                answer=response.get("answer", ""),
                sources=response.get("sources", []),
            )

            return response
        except Exception as e:
            import logging

            logger = logging.getLogger("rag_engine")
            logger.exception(e)
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "confidence": 0.0,
            }

