from typing import List, Tuple

import numpy as np


class StatsTopWordsBase:

    def get_top_keywords(
        self, tokens: List[List[str]], top_k: int = 5
    ) -> Tuple[List[str], List[float]]:
        scores = self.inference(tokens)
        top_words = []
        top_scores = []

        top_token_ids = np.argsort(-1 * scores, axis=1)[:, :top_k]
        top_token_scores = np.flip(np.sort(scores, axis=1), axis=1)[:, :top_k].tolist()

        for token_ids, scores in zip(top_token_ids, top_token_scores):
            _words = []
            _scores = []
            for token_id, score in zip(token_ids, scores):
                if score > 0:
                    _words.append(self.id2token[token_id])
                    _scores.append(score)
            top_words.append(_words)
            top_scores.append(_scores)

        return top_words, top_scores


class NNTopWordsBase:
    pass
