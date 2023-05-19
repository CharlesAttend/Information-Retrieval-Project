if [ -z "$JAVA_HOME" ] || [ "$JAVA_HOME" != "/usr/lib/jvm/java-1.11.0-openjdk-amd64" ]; then
  export JAVA_HOME="/usr/lib/jvm/java-1.11.0-openjdk-amd64"
fi

python pyserini/tools/scripts/msmarco/convert_collection_to_jsonl.py \
 --collection-path data/msmarco-passage/collection.tsv \
 --output-folder data/msmarco-passage/collection_jsonl