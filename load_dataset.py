import os
import gzip
from tqdm import tqdm
from sentence_transformers import InputExample
from src.bm25 import (
    min_max_global,
    min_max_local,
    z_score_global,
    z_score_local,
    sum_normalisation,
)


def load_corpus(collection_filepath):
    """Read the corpus files, that contain all the passages.
    Store them in the corpus dict.
    """
    corpus = {}
    with open(collection_filepath, "r", encoding="utf8") as fIn:
        for line in tqdm(fIn, unit_scale=True):
            pid, passage = line.strip().split("\t")
            corpus[pid] = passage
    return corpus


def load_queries(queries_filepath):
    queries = {}
    with open(queries_filepath, "r", encoding="utf8") as fIn:
        for line in tqdm(fIn, unit_scale=True):
            qid, query = line.strip().split("\t")
            queries[qid] = query
    return queries


def load_eval(
    train_eval_filepath, corpus, queries, num_dev_queries=200, num_max_dev_negatives=200
):
    """We use 200 random queries from the train set for evaluation during training.
    Each query has at least one relevant and up to 200 irrelevant (negative) passages.
    """
    dev_samples = {}
    with gzip.open(train_eval_filepath, "rt") as fIn:
        for line in tqdm(fIn, unit_scale=True):
            qid, pos_id, neg_id = line.strip().split()

            if qid not in dev_samples and len(dev_samples) < num_dev_queries:
                dev_samples[qid] = {
                    "query": queries[qid],
                    "positive": set(),
                    "negative": set(),
                }

            if qid in dev_samples:
                dev_samples[qid]["positive"].add(corpus[pos_id])

                if len(dev_samples[qid]["negative"]) < num_max_dev_negatives:
                    dev_samples[qid]["negative"].add(corpus[neg_id])
    return dev_samples


def load_train(train_filepath, corpus, queries):
    train_samples = []
    with gzip.open(train_filepath, "rt") as fIn:
        for line in tqdm(fIn, unit_scale=True):
            qid, corpus_id, label, bm25_score_per_doc = line.strip().split("\t")

            query = queries[qid]
            passage = corpus[corpus_id]

            bm25_normalized = min_max_global(float(bm25_score_per_doc), 0, 50)
            bm25_score = int(bm25_normalized * 100)

            train_samples.append(
                InputExample(texts=[query, str(bm25_score), passage], label=label)
            )

    return train_samples


if __name__ == "__main__":
    data_folder = "data/msmarco-passage"
    corpus = load_corpus(os.path.join(data_folder, "collection.tsv"))
    queries = load_queries(os.path.join(data_folder, "queries.train.tsv"))
    train_samples = load_train(
        os.path.join(data_folder, "msmarco.bm25.train.tsv.gz"), corpus, queries
    )
