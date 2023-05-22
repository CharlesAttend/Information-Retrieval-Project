if [ -z "$JAVA_HOME" ] || [ "$JAVA_HOME" != "/usr/lib/jvm/java-1.11.0-openjdk-amd64" ]; then
  export JAVA_HOME="/usr/lib/jvm/java-1.11.0-openjdk-amd64"
fi

.venv/bin/python -m pyserini.eval.convert_msmarco_run_to_trec_run \
   --input runs/run.msmarco-passage.reranking.txt \
   --output runs/run.msmarco-passage.reranking.trec

.venv/bin/python pyserini/tools/scripts/msmarco/msmarco_passage_eval.py \
   pyserini/tools/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt runs/run.msmarco-passage.reranking.txt

pyserini/tools/eval/trec_eval.9.0.4/trec_eval -c -mrecall.1000 -mmap -mndcg \
   data/msmarco-passage/qrels.dev.small.trec runs/run.msmarco-passage.reranking.trec