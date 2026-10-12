import difflib
import re

class LegalTermMapper:
    def __init__(self):
        self.mapping = {
            "culpable homicide": "299",
            "murder": "300",
            "attempt to murder": "307",
            "rape": "375",
            "punishment for rape": "376",
            "child rape": "376AB",
            "kidnapping": "361",
            "acid attack": "326A",
            "cheating": "420",
            "dowry death": "304B"
        }
        
    def get_section(self, term: str) -> str:
        term_clean = term.lower().strip()
        if term_clean in self.mapping:
            return self.mapping[term_clean]
            
        matches = difflib.get_close_matches(term_clean, self.mapping.keys(), n=1, cutoff=0.7)
        if matches:
            return self.mapping[matches[0]]
        return None

    def map_terms(self, query: str) -> str:
        mapped_query = query
        for key in sorted(self.mapping.keys(), key=len, reverse=True):
            mapped_query = re.sub(r'\b' + re.escape(key) + r'\b', f"IPC Section {self.mapping[key]}", mapped_query, flags=re.IGNORECASE)
        return mapped_query
