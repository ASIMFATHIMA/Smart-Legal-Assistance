"""
Document Chunking Module
========================
Break documents into manageable semantic chunks with metadata preservation.

Implements:
- Semantic Section-aware chunking for structured legal records.
- Metadata tracking (source, section, crime_name, description, punishment, bns_section)
- Fallback sliding window chunking for non-structured text.

Functions:
    - DocumentChunker: Main chunking engine
"""

from typing import List, Dict, Tuple, Optional
import re

class DocumentChunker:
    """
    Break documents into semantic chunks for processing.
    """
    
    def __init__(self, chunk_size: int = 700,
                 overlap_size: int = 100,
                 separator: str = "\n"):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.separator = separator
        
        print("Initialized Semantic Document Chunker")
        
    def chunk_document(self, text: str, 
                      doc_id: str = "unknown",
                      doc_source: str = "unknown") -> List[Dict]:
        """
        Chunk a document semantically by extracting explicit legal sections.
        If no strict sections are found, fall back to simple chunking.
        """
        chunks = []
        
        # Regex to find records:
        # Example target text block:
        # Section 299
        # Culpable Homicide
        # Description: Causing death with intention or knowledge but not amounting to murder.
        # Punishment: Life imprisonment or imprisonment up to 10 years and fine.
        # BNS Section: 101
        
        pattern = r"(?i)Section\s+(\d+[A-Z]*)\s*\n+(.*?)\s*\n+Description:\s*(.*?)\s*\n+Punishment:\s*(.*?)\s*\n+BNS Section:\s*(\d+[A-Z]*)"
        
        matches = list(re.finditer(pattern, text, re.DOTALL))
        
        if matches:
            for i, match in enumerate(matches):
                section_number = match.group(1).strip()
                crime_name = match.group(2).strip()
                description = match.group(3).strip()
                punishment = match.group(4).strip()
                bns_section = match.group(5).strip()
                
                complete_record = f"Section {section_number}\n{crime_name}\nDescription:\n{description}\nPunishment:\n{punishment}\nBNS Section:\n{bns_section}"
                
                chunks.append({
                    "chunk_id": f"{doc_id}_{i}",
                    "section_number": section_number,
                    "crime_name": crime_name,
                    "description": description,
                    "punishment": punishment,
                    "bns_section": bns_section,
                    "source_document": doc_source,
                    "doc_id": doc_id,
                    "text": complete_record,
                    "keywords": [],
                    "page_number": None,
                })

        else:
            # FALLBACK: sliding window
            start_pos = 0
            chunk_id = 0
            
            while start_pos < len(text):
                end_pos = min(start_pos + self.chunk_size, len(text))
                chunk_text = text[start_pos:end_pos]
                
                if end_pos < len(text):
                    last_sentence = max(
                        chunk_text.rfind('.'),
                        chunk_text.rfind('?'),
                        chunk_text.rfind('!')
                    )
                    if last_sentence > self.chunk_size * 0.8:
                        end_pos = start_pos + last_sentence + 1
                        chunk_text = text[start_pos:end_pos]
                
                if chunk_text.strip():
                    chunks.append({
                        "chunk_id": f"{doc_id}_{chunk_id}",
                        "section_number": "",
                        "crime_name": "",
                        "description": "",
                        "punishment": "",
                        "bns_section": "",
                        "source_document": doc_source,
                        "doc_id": doc_id,
                        "text": chunk_text.strip(),
                        "keywords": [],
                        "page_number": None,
                    })
                    chunk_id += 1
                
                if end_pos >= len(text):
                    break
                start_pos = end_pos - self.overlap_size
            
        # Ensure no chunk has empty text
        chunks = [c for c in chunks if c.get("text", "").strip()]

        # Print chunk verification diagnostics (Step 2)
        print(f"--- Chunking Debug for Document '{doc_id}' ---")
        print(f"Total Chunks: {len(chunks)}")
        for ch in chunks:
            print(f"  Chunk ID: {ch.get('chunk_id')}")
            print(f"  Length: {len(ch.get('text', ''))}")
            print(f"  Metadata: section={ch.get('section_number')}, crime={ch.get('crime_name')}, bns={ch.get('bns_section')}, keywords={ch.get('keywords')}")
        print("---------------------------------------------")

        return chunks

    def chunk_documents(self, documents: List[Tuple[str, str, str]]) -> List[Dict]:
        all_chunks = []
        for text, doc_id, doc_source in documents:
            chunks = self.chunk_document(text, doc_id, doc_source)
            all_chunks.extend(chunks)
        print(f"✓ Chunked {len(documents)} documents → {len(all_chunks)} semantic chunks")
        return all_chunks
