from collections import Counter
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from kwe.base import StatsTopWordsBase


class CountWords(StatsTopWordsBase):
    def __init__(
        self,
        token_freq_threshold=1,
        stopwords: List[str] = [],
        min_word_length: int = 3,
    ):
        super().__init__()
        self.token_freq_threshold = token_freq_threshold
        self.token2id = {}
        self.id2token = {}
        self.stopwords = stopwords
        self.min_word_length = min_word_length

    def build_vocab(self, tokens):
        all_tokens = [
            token
            for doc in tokens
            for token in doc
            if token not in self.stopwords and len(token) >= self.min_word_length
        ]
        counter = Counter(all_tokens)
        unique_tokens = [
            token
            for token, count in counter.items()
            if count >= self.token_freq_threshold
        ]

        for i, token in tqdm(
            enumerate(unique_tokens), desc="Building Vocab...", total=len(unique_tokens)
        ):
            self.token2id[token] = i
            self.id2token[i] = token

    def train(self, tokens):
        self.build_vocab(tokens)

    def inference(self, tokens):
        count_vectors = []
        for doc in tqdm(tokens, desc="Count Vectorizing..."):
            vec = np.zeros(len(self.token2id))
            for token in doc:
                if token in self.token2id:
                    vec[self.token2id[token]] += 1
            count_vectors.append(vec.tolist())
        count_vectors = np.array(count_vectors)
        return count_vectors


class BM25(CountWords):
    """
    아래의 BM25 산식을 기준으로 구현하였습니다
    https://en.wikipedia.org/wiki/Okapi_BM25
    """

    def __init__(
        self,
        k1=1.5,
        b=0.75,
        epsilon=0.25,
        token_freq_threshold=1,
        min_word_length=3,
        stopwords=[],
    ):
        super().__init__(
            token_freq_threshold=token_freq_threshold,
            min_word_length=min_word_length,
            stopwords=stopwords,
        )
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.idf = None
        self.tf = None
        self.scores = None
        self.avg_doc_len = None
        self.doc_lens = None

    def train(self, tokens: List[List[str]]):
        super().train(tokens)
        count_vectors = super().inference(tokens)
        # 전체 문서의 수
        self.n_docs = count_vectors.shape[0]

        # 각 토큰이 등장한 문서의 수
        doc_freqs = np.count_nonzero(count_vectors, axis=0)

        # 각 문서의 길이
        self.doc_lens = np.sum(count_vectors, axis=1)

        # 문서 길이평균
        self.avg_doc_len = np.mean(self.doc_lens)

        # IDF
        self.idf = np.log((self.n_docs - doc_freqs + 0.5) / (doc_freqs + 0.5) + 1)

    def inference(self, tokens):
        count_vectors = np.zeros((self.n_docs, len(self.token2id)))
        for i, doc in enumerate(tokens):
            vec = np.zeros(len(self.token2id))
            for token in doc:
                if token in self.token2id:
                    vec[self.token2id[token]] += 1
            count_vectors[i] = vec

        # TF
        tf = (count_vectors * (self.k1 + 1)) / (
            count_vectors
            + (
                self.k1 * (1 - self.b + self.b * self.doc_lens / self.avg_doc_len)
            ).reshape(-1, 1)
        )
        # bm25
        # self.scores = self.tf * self.idf
        scores = (tf * self.idf)[: len(tokens), :]
        return scores


if __name__ == "__main__":
    corpus = [
        ["hello", "world"],
        ["hello", "python", "hello"],
        ["hello", "python", "world", "hi"],
    ]
    model = BM25(corpus)
    model.train()
    # model = CountVectorizer(corpus)
    words, scores = model.get_top_keywords()
