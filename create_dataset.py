import os
import gzip
import csv
from tqdm import tqdm
from pyserini.index.lucene import IndexReader
from load_dataset import load_corpus, load_queries, load_eval


def save_train(
    train_filepath, output_filepath, dev_samples, max_train_samples=2e7, pos_neg_ration=4
):
    index_reader = IndexReader("data/msmarco-index/")
    counter = 0

    with gzip.open(output_filepath, "wt", encoding="utf8") as fOut:
        with gzip.open(train_filepath, "rt") as fIn:
            for line in tqdm(fIn, unit_scale=True):
                qid, pos_id, neg_id = line.strip().split()

                if qid in dev_samples:
                    continue

                query = queries[qid]
                if (counter % (pos_neg_ration + 1)) == 0:
                    corpus_id = int(pos_id)
                    label = 1
                else:
                    corpus_id = int(neg_id)
                    label = 0

                # Compute BM25
                bm25_score_per_doc = index_reader.compute_query_document_score(
                    str(corpus_id), str(query)
                )

                # Save
                writer = csv.writer(fOut, delimiter="\t", lineterminator="\n")
                writer.writerow([qid, corpus_id, label, bm25_score_per_doc])

                counter += 1

                if counter >= max_train_samples:
                    break


def save_dev_retrieval(dev_filepath, output_filepath, corpus, queries):
    index_reader = IndexReader("data/msmarco-index/")

    with gzip.open(output_filepath, "wt", encoding="utf8") as fOut:
        with open(dev_filepath, "r") as fIn:
            for line in tqdm(fIn, unit_scale=True):
                qid, corpus_id, rank = line.strip().split("\t")
                
                query = queries[qid]
                
                score = index_reader.compute_query_document_score(corpus_id, query)

                # Save
                writer = csv.writer(fOut, delimiter="\t", lineterminator="\n")
                writer.writerow([qid, corpus_id, rank, score])


if __name__ == "__main__":
    data_folder = "data/msmarco-passage"
    corpus = load_corpus(os.path.join(data_folder, "collection.tsv"))
    queries = load_queries(os.path.join(data_folder, "queries.dev.small.tsv"))
    # dev_samples = load_eval(
    #     os.path.join(data_folder, "msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz"),
    #     corpus,
    #     queries,
    # )
    # save_train(
    #     os.path.join(data_folder, "msmarco-qidpidtriples.rnd-shuf.train.tsv.gz"),
    #     os.path.join(data_folder, "msmarco.bm25.train.tsv.gz"),
    # )
    # save_dev_retrieval(
    #     "runs/run.msmarco-passage.bm25tuned.txt",
    #     "data/msmarco-passage/msmarco.bm25.dev.small.tsv.gz",
    #     corpus,
    #     queries,
    # )
