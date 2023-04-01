import numpy as np
from numpy.typing import NDArray
from typing import List
from rank_bm25 import BM25Okapi


class BM25:
    def __init__(self) -> None:
        self.bm25 = None
        self.corpus = None

    def fit(self, corpus, **kwargs) -> None:
        """Fit un BM25Okapi sur le corpus en paramètre

        Parameters
        ----------
        corpus : List(str)
            Le corpus de document
        """
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus, **kwargs)
        self.corpus = corpus

    def predict(self, queries: List) -> NDArray:
        """Renvoie le score BM25 associés à chaque document du corpus pour chaque queries en paramètre.

        Parameters
        ----------
        queries : List
            Liste de query

        Returns
        -------
        NDArray (n_queries, n_corpus)
            score BM25 associés à chaque document du corpus pour chaque queries
        """
        # Support pour prendre une str nécéssaire ?
        if isinstance(queries, str):
            queries = list(queries)
            tokenized_queries = [query.split(" ") for query in queries]
            return np.array(
                [
                    self.bm25.get_scores(tokenized_query)
                    for tokenized_query in tokenized_queries
                ]
            ).squeeze()
        else:
            tokenized_queries = [query.split(" ") for query in queries]

            return np.array(
                [
                    self.bm25.get_scores(tokenized_query)
                    for tokenized_query in tokenized_queries
                ]
            )

    def predict_top_n(self, queries: List, n=5) -> NDArray:
        """Renvoie les `n` document du corpus ayant les plus grand scores pour chaque queries en paramètre.

        Parameters
        ----------
        queries : List
            _description_
        n : int, optional
            _description_, by default 5

        Returns
        -------
        NDArray (n_queries, n)
            Liste de document
        """
        tokenized_queries = [query.split(" ") for query in queries]

        return np.array([
            self.bm25.get_top_n(tokenized_query, self.corpus, n=n)
            for tokenized_query in tokenized_queries
        ])


def min_max_global(scores: NDArray, s_min: float, s_max: float) -> NDArray:
    return (scores - s_min) / (s_max - s_min)


def min_max_local(scores: NDArray):
    s_min = scores.min(axis=1, keepdims=True)
    s_max = scores.max(axis=1, keepdims=True)
    return (scores - s_min) / (s_max - s_min)


def z_score_global(scores: NDArray, mu: float, sig: float) -> NDArray:
    return (scores - mu) / sig


def z_score_local(scores: NDArray):
    mu = scores.mean(axis=1, keepdims=True)
    sig = scores.std(axis=1, keepdims=True)

    return (scores - mu) / sig

def sum_normalisation(scores: NDArray):
    return scores / scores.sum(axis=1, keepdims=True)