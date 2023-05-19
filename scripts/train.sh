if [ -z "$JAVA_HOME" ] || [ "$JAVA_HOME" != "/usr/lib/jvm/java-1.11.0-openjdk-amd64" ]; then
  export JAVA_HOME="/usr/lib/jvm/java-1.11.0-openjdk-amd64"
fi

.venv/bin/python -m pyserini.search.lucene \
  --index data/msmarco-index/ \
  --topics data/msmarco-passage/queries.train.tsv \
  --output train.bm25.txt \
  --threads 12 \
  --bm25 --k1 0.82 --b 0.68