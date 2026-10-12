# pdfplumber is only required when extracting tables from PDFs.
# Make it optional so Streamlit can start without it.
try:
    import pdfplumber  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pdfplumber = None


class TableExtractor:
    def extract_from_pdf(self, file_path: str) -> list[dict]:
        """
        Extracts structured table rows from a PDF and returns them as a list of dictionaries.
        Assumes columns: IPC SEC, NAME OF CRIME, DESCRIPTION, ESSENTIAL INGREDIENTS, PUNISHMENT, IMPORTANT CASES, BNS SEC, KEYWORDS
        """
        records = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if not row or not any(row):
                            continue
                        
                        # Clean row strings
                        clean_row = [str(cell).strip().replace("\n", " ") if cell else "" for cell in row]
                        
                        # Check if header
                        if len(clean_row) > 0 and "IPC SEC" in clean_row[0].upper():
                            continue
                            
                        # Map to dictionary
                        if len(clean_row) >= 8:
                            keywords = [k.strip() for k in clean_row[7].split(",") if k.strip()]
                            record = {
                                "ipc_section": clean_row[0],
                                "crime": clean_row[1],
                                "description": clean_row[2],
                                "ingredients": clean_row[3],
                                "punishment": clean_row[4],
                                "important_cases": clean_row[5],
                                "bns_section": clean_row[6],
                                "keywords": keywords
                            }
                            records.append(record)
        return records
