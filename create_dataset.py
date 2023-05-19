from pyserini.index.lucene import IndexReader
import gzip
import os
from tqdm import tqdm
from src.bm25 import *
import pandas as pd

data_folder = "data/msmarco-passage"
train_filepath = os.path.join(
    data_folder, 'msmarco-qidpidtriples.rnd-shuf.train.tsv.gz')
train_samples = []
dev_samples = {}

queries = {}
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query
pos_neg_ration = 4

corpus = {}
collection_filepath = os.path.join(data_folder, 'collection.tsv')

with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage


cnt = 0
max_train_samples = 2e7


# Initialize from an index path:
index_reader = IndexReader('data/msmarco-index/')

output = "out.tsv"
import csv

with open(output, "w", encoding="utf8") as fOut:
    with gzip.open(train_filepath, 'rt') as fIn:
        for line in tqdm(fIn, unit_scale=True):
            qid, pos_id, neg_id = line.strip().split()
            if qid in dev_samples:
                continue

            query = queries[qid]
            if (cnt % (pos_neg_ration+1)) == 0:
                corpus_id = int(pos_id)
                passage = corpus[pos_id]
                label = 1
            else:
                corpus_id = int(neg_id)
                passage = corpus[neg_id]
                label = 0

            bm25_score_per_doc = index_reader.compute_query_document_score(str(corpus_id), str(query))
            # train_samples.append(texts=[f'{query} [SEP] {bm25_score_per_doc}', passage], label=label)
            
            # fOut.write(f"{qid}\t{corpus_id}\t{label}\t{bm25_score_per_doc}\n")

            writer = csv.writer(fOut, delimiter='\t', lineterminator='\n')
            writer.writerow([qid, corpus_id, label, bm25_score_per_doc])

            cnt += 1

            if cnt >= max_train_samples:
                break