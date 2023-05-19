from pyserini.index.lucene import IndexReader

# Initialize from an index path:
index_reader = IndexReader('datasets/msmarco-index/')

index_reader.dump_documents_BM25('datasets/bm25_dump.jsonl')
