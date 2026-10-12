class ContextBuilder:
    def build(self, chunks: list[dict]) -> str:
        if not chunks:
            return ""
            
        seen = set()
        unique_chunks = []
        for c in chunks:
            text = c.get("text", "").strip()
            if not text:
                continue
            
            text_hash = hash(text)
            if text_hash not in seen:
                seen.add(text_hash)
                unique_chunks.append(c)
                
        top_chunks = unique_chunks[:5]
        
        context_parts = []
        for c in top_chunks:
            # Always use the text field — it contains the complete record
            # (section number, crime name, description, punishment, etc.)
            text = c.get("text", "").strip()
            src = c.get("source_document", "").strip()
            
            if text:
                part = text
                if src:
                    part += f"\n(Source: {src})"
                context_parts.append(part)
                
        return "\n\n---\n\n".join(context_parts)

