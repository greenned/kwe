from typing import List, Tuple

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity

from kwe.base import NNTopWordsBase


class KeyBert(NNTopWordsBase):
    def __init__(self, model_kwargs={}):
        super().__init__()
        self.model = KeyBERT(**model_kwargs)
        self.corpus = None

    def train(self, corpus: List[str]):
        # TODO: Implement train method
        pass

    def get_top_keywords(
        self, corpus, top_k: int = 5, n_docs: int = None, extract_kwargs: dict = {}
    ) -> Tuple[List[List[str]], List[float]]:
        keywords = self.model.extract_keywords(
            docs=corpus, top_n=top_k, **extract_kwargs
        )
        top_words = []
        top_scores = []
        for keyword in keywords:
            arr = np.array(keyword)
            if arr.shape[0] == 0:
                top_words.append([])
                top_scores.append([])
                continue
            top_words.append(arr[:, 0].tolist())
            top_scores.append([float(s) for s in arr[:, 1].tolist()])
        return top_words, top_scores


class EmbedRank(NNTopWordsBase):
    # 아래 논문과 코드를 참고하여 구현하였습니다
    # https://arxiv.org/pdf/1801.04470.pdf
    # https://github.com/yagays/embedrank/blob/master/src/embedrank.py

    def __init__(
        self,
        doc2vec_kwargs={},
        epochs: int = 1,
        stopwords: List[str] = [],
        min_word_length: int = 3,
    ):
        super().__init__()
        self.model = Doc2Vec(**doc2vec_kwargs)
        self.epochs = epochs
        self.min_word_length = min_word_length
        self.stopwords = stopwords

    def _preprocess(self, tokens: List[str]):
        return [
            token
            for token in tokens
            if len(token) >= self.min_word_length and token not in self.stopwords
        ]

    def train(self, tokens: List[List[str]]):

        tokens = [self._preprocess(doc) for doc in tokens]
        import logging

        logging.basicConfig(
            format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
        )
        tagged_data = [
            TaggedDocument(words=token, tags=[str(i)]) for i, token in enumerate(tokens)
        ]
        self.model.build_vocab(tagged_data)
        self.model.train(
            tagged_data, total_examples=self.model.corpus_count, epochs=self.epochs
        )

    def mmr(self, doc, n=4):
        doc = self._preprocess(doc)
        if len(doc) == 0:
            return [], []
        keyword_embeds = []
        document_embed = self.model.infer_vector(doc)

        for token in doc:
            token_embed = self.model.infer_vector([token])
            keyword_embeds.append(token_embed)
        keyword_embeds = np.array(keyword_embeds)

        doc_similarities = cosine_similarity(
            keyword_embeds, document_embed.reshape(1, -1)
        )
        keyword_similarities = cosine_similarity(keyword_embeds)

        unselected = list(range(len(doc)))
        select_idx = np.argmax(doc_similarities)

        selected = [select_idx]
        unselected.remove(select_idx)
        scores = []

        for _ in range(n):
            mmr_distance_to_doc = doc_similarities[unselected, :]
            mmr_distance_between_tokens = np.max(
                keyword_similarities[unselected][:, selected], axis=1
            )
            mmr = 0.5 * mmr_distance_to_doc - (
                1 - 0.5
            ) * mmr_distance_between_tokens.reshape(-1, 1)
            mmr_idx = unselected[np.argmax(mmr)]
            selected.append(mmr_idx)
            scores.append(mmr[np.argmax(mmr)])
            unselected.remove(mmr_idx)

            if len(unselected) == 0:
                break

        return selected, scores

    def get_top_keywords(self, tokens, top_k: int = 5):
        top_words = []
        top_scores = []
        for token_doc in tokens:
            seleced_idx, scores = self.mmr(token_doc, n=top_k)
            keywords = [self.model.wv.index_to_key[idx] for idx in seleced_idx]
            top_words.append(keywords[:top_k])
            top_scores.append(scores[:top_k])
        return top_words, top_scores
