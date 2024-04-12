from typing import Any, Dict

from kwe.models.graph import TextRank
from kwe.models.nn import EmbedRank, KeyBert
from kwe.models.statistics import BM25, CountWords

MODELS = {
    "bm25": BM25,
    "countwords": CountWords,
    "keybert": KeyBert,
    "textrank": TextRank,
    "embedrank": EmbedRank,
}


def load_model(model_name: str, model_kwargs: Dict[str, Any] = {}):
    return MODELS[model_name](**model_kwargs)
