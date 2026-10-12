import sys, os
os.environ['DEBUG_MODE'] = '1'

from src.vector_store import VectorStore
from src.embedding_generator import EmbeddingGenerator
from src.retriever import Retriever

vs = VectorStore(index_path='models/faiss_index.bin', metadata_path='models/metadata.pkl')
try:
    vs.load_index()
    print('FAISS index loaded. Total vectors:', vs.index.ntotal)
    print('Metadata count:', len(vs.metadata))
except Exception as e:
    print('ERROR loading index:', e)
    sys.exit(1)

embedder = EmbeddingGenerator()
retriever = Retriever(vs, embedder)
print('BM25 built over', len(retriever.chunks), 'chunks')
print()

test_queries = [
    'Adult Female',
    'Murder',
    'Air Pollution',
    'IPC Section 302',
    'Trade and Commerce without Licence',
    'Legal Rights',
    'Punishment',
    'Procedure'
]

all_ok = True
for q in test_queries:
    results = retriever.retrieve(q, top_k=5)
    status = 'OK' if results else 'FAIL - NO RESULTS'
    print(f'QUERY: {q!r}  -> {len(results)} result(s)  [{status}]')
    for r in results[:2]:
        print(f'    sim={float(r.get("similarity",0)):.4f}  '
              f'crime={str(r.get("crime_name",""))[:35]}  '
              f'src={r.get("source_document","")}')
    if not results:
        all_ok = False
    print()

print()
print('SMOKE TEST:', 'PASSED' if all_ok else 'SOME QUERIES FAILED')
sys.exit(0 if all_ok else 1)
