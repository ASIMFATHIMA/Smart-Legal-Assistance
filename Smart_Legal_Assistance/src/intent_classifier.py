import re

class IntentClassifier:
    def __init__(self):
        # Greeting-only detection. Normalize input then match against these variants.
        self.greetings = {
            "hi",
            "hii",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "good night",
            "greetings",
            "hola",
            "how are you",
            "what's up",
            "good day",
        }

        # Allow optional punctuation at ends/spaces around the phrase.
        self._strip_re = re.compile(r"^[\W_]+|[\W_]+$")

    def get_intent(self, query: str) -> str:
        """Returns 'greeting' if it's a greeting-only message, else 'query'."""
        q = query.lower().strip()

        # Remove extra punctuation/spaces and collapse whitespace.
        q = self._strip_re.sub("", q)
        q = re.sub(r"\s+", " ", q).strip()

        # Special cases: punctuation/encoding variants.
        q = q.replace("â€™", "'")
        q = q.replace("’", "'")

        # Strict greeting-only intent:
        # - only return greeting when the normalized entire message matches
        # - never classify a longer legal query as greeting.
        if q in self.greetings and not re.search(
            r"\b(section|ipc|bns|crime|punishment|rights|licen[sc]e|act|offence|offense|court|murder|cheating|sedition|forgery|dowry|kidnapping|ndps|wildlife|environment|trade|commerce|consumer)\b",
            q,
            flags=re.IGNORECASE,
        ):
            return "greeting"

        return "query"


