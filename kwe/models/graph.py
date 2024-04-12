from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from tqdm import tqdm

from kwe.models.statistics import BM25, CountWords

# 아래 코드를 참고하여 구현하였습니다
# https://lovit.github.io/nlp/2019/04/30/textrank/


class TextRank(CountWords):
    def __init__(
        self,
        df: float = 0.85,
        max_iter: int = 30,
        window: int = 2,
        min_cooccurrence: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.df = df
        self.max_iter = max_iter
        self.window = window
        self.min_cooccurrence = min_cooccurrence

    def train(self, tokens):
        super().train(tokens)

    def inference(self, tokens):
        scores = []
        for token in tqdm(tokens, desc="Calculate textrank...", total=len(tokens)):
            matrix = self.calc_co_occurence([token])
            score = self.page_rank(matrix)
            scores.append(score.tolist())
        scores = np.array(scores)
        return scores

    def calc_co_occurence(
        self,
        tokens,
    ):
        counter = defaultdict(int)
        for s, tokens_i in enumerate(tokens):
            vocabs = [self.token2id[w] for w in tokens_i if w in self.token2id]
            n = len(vocabs)
            for i, v in enumerate(vocabs):
                if self.window <= 0:
                    b, e = 0, n
                else:
                    b = max(0, i - self.window)
                    e = min(i + self.window, n)
                for j in range(b, e):
                    if i == j:
                        continue
                    counter[(v, vocabs[j])] += 1
                    counter[(vocabs[j], v)] += 1
        counter = {k: v for k, v in counter.items() if v >= self.min_cooccurrence}
        n_vocabs = len(self.token2id)

        rows, cols, data = [], [], []

        for (i, j), v in counter.items():
            rows.append(i)
            cols.append(j)
            data.append(v)
        return csr_matrix((data, (rows, cols)), shape=(n_vocabs, n_vocabs))

    def page_rank(self, matrix):
        assert 0 < self.df < 1, "Damping factor must be between 0 and 1"

        # initialize
        normalized_matrix = normalize(matrix, axis=0, norm="l1")
        scores = np.ones(normalized_matrix.shape[0]).reshape(-1, 1)
        bias = (1 - self.df) * np.ones(normalized_matrix.shape[0]).reshape(-1, 1)

        # iteration
        for _ in range(self.max_iter):
            scores = self.df * (normalized_matrix * scores) + bias
        return scores.reshape(-1)


# # 예시 문장
# corpus = [
#     "TextRank is an algorithm for keyword and sentence extraction.",
#     "It is based on PageRank algorithm and used in natural language processing.",
#     "TextRank builds a graph representing the relationships between words or sentences.",
#     "It assigns a score to each word or sentence based on its importance in the graph.",
#     "The top-ranked words or sentences are then extracted as the summary or keywords.",
# ]
# tokens = [sentence.lower().split() for sentence in corpus]
# # vocab_to_idx = {w: i for i, w in enumerate(set(sum(tokens, [])))}
# # idx_to_vocab = {i: w for w, i in vocab_to_idx.items()}
# # result = cooccurrence(tokens, vocab_to_idx=vocab_to_idx)

# model = TextRank()
# model.train(tokens)


# TextRank 알고리즘을 사용하여 중요한 문장 2개를 추출
# result = textrank_keyword(
#     tokens,
#     window=2,
#     min_cooccurrence=2,
#     idx_to_vocab={i: w for w, i in vocab_to_idx.items()},
# )
# print(1)
