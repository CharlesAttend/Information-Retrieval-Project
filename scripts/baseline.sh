if [ -z "$JAVA_HOME" ] || [ "$JAVA_HOME" != "/usr/lib/jvm/java-1.11.0-openjdk-amd64" ]; then
  export JAVA_HOME="/usr/lib/jvm/java-1.11.0-openjdk-amd64"
fi

.venv/bin/python -m pyserini.search.lucene \
  --index data/msmarco-index \
  --topics msmarco-passage-dev-subset \
  --output runs/run.msmarco-passage.bm25tuned.txt \
  --output-format msmarco \
  --hits 1000 \
  --threads 12 \
  --bm25 --k1 0.82 --b 0.68

.venv/bin/python -m pyserini.eval.convert_msmarco_run_to_trec_run \
   --input runs/run.msmarco-passage.bm25tuned.txt \
   --output runs/run.msmarco-passage.bm25tuned.trec

.venv/bin/python pyserini/tools/scripts/msmarco/convert_msmarco_to_trec_qrels.py \
   --input pyserini/tools/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt \
   --output datasets/collectionandqueries/qrels.dev.small.trec

pyserini/tools/eval/trec_eval.9.0.4/trec_eval -c -mrecall.1000 -mmap \
   datasets/collectionandqueries/qrels.dev.small.trec runs/run.msmarco-passage.bm25tuned.trec