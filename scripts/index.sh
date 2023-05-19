if [ -z "$JAVA_HOME" ] || [ "$JAVA_HOME" != "/usr/lib/jvm/java-1.11.0-openjdk-amd64" ]; then
  export JAVA_HOME="/usr/lib/jvm/java-1.11.0-openjdk-amd64"
fi

.venv/bin/python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/collection_jsonl \
  --index data/msmarco-index/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 12 \
  --storePositions --storeDocvectors --storeRaw