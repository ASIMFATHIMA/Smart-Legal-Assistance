import re

class EntityTracker:
    def extract_entities(self, text: str) -> dict:
        entities = {}
        
        # Extract section (e.g., 376DA)
        section_match = re.search(r'\b(376[A-Z]*)\b', text, re.IGNORECASE)
        if not section_match:
            section_match = re.search(r'(?i)\b(?:section|sec\.?)\s*(\d+[A-Z]*)\b', text)
            
        if section_match:
            entities['section_number'] = section_match.group(1).upper()
            
        # Extract document (e.g., something.pdf)
        doc_match = re.search(r'\b(\w+\.pdf)\b', text, re.IGNORECASE)
        if doc_match:
            entities['document_name'] = doc_match.group(1)
            
        return entities
