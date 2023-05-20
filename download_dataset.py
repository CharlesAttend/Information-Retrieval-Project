import os
import logging
import tarfile
from sentence_transformers import util


data_folder = "./data/msmarco-passage/"
os.makedirs(data_folder, exist_ok=True)

collection_filepath = os.path.join(data_folder, "collection.tsv")
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, "collection.tar.gz")
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get(
            "https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz",
            tar_filepath,
        )

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

queries_filepath = os.path.join(data_folder, "queries.train.tsv")
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, "queries.tar.gz")
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get(
            "https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz",
            tar_filepath,
        )

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

train_eval_filepath = os.path.join(
    data_folder, "msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz"
)
if not os.path.exists(train_eval_filepath):
    logging.info("Download " + os.path.basename(train_eval_filepath))
    util.http_get(
        "https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz",
        train_eval_filepath,
    )

train_filepath = os.path.join(
    data_folder, "msmarco-qidpidtriples.rnd-shuf.train.tsv.gz"
)
if not os.path.exists(train_filepath):
    logging.info("Download " + os.path.basename(train_filepath))
    util.http_get(
        "https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train.tsv.gz",
        train_filepath,
    )
