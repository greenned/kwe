
# KWE

`kwe` is a simple tool to extract keywords from a text file. It provides a simple interface to extract keywords using different models. The models are based on statistical, graph-based, and embed-based methods. The tool is designed to be simple and easy to use.

## Table of Contents

* [Installation](#installation)
* [Example](#minimal-example)
* [Getting started](#getting-started)
* [Implemented models](#implemented-models)

## Installation

```bash
pip install kwe
```

## Example

```python
from kwe.models import load_model
from nltk.corpus import stopwords

corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy dog and the quick brown fox jumps over the lazy dog."
]
tokens = [sentence.split() for sentence in corpus]

stop_words_list = stopwords.words("english")

model_configs = [
    {
        "model_name": "countwords",
        "model_kwargs": {
            "token_freq_threshold": 100,
            "stopwords": stop_words_list,
        },
        "use_tokens": True,
        "do_train": True,
    },
    {
        "model_name": "bm25",
        "model_kwargs": {"token_freq_threshold": 100},
        "use_tokens": True,
        "do_train": True,
    },
    {
        "model_name": "textrank",
        "model_kwargs": {
            "token_freq_threshold": 100,
            "stopwords": stop_words_list,
        },
        "use_tokens": True,
        "do_train": True,
    },
    {
        "model_name": "embedrank",
        "model_kwargs": {
            "epochs": 15,
            "stopwords": stop_words_list,
        },
        "use_tokens": True,
        "do_train": True,
    },
    {
        "model_name": "keybert",
        "model_kwrags": {},
        "use_tokens": False,
        "do_train": False,
    },
]

sample_size = None
results = {}
for model_config in model_configs:
    model_name = model_config["model_name"]
    model_kwargs = model_config.get("model_kwargs", {})
    model = load_model(model_name, model_kwargs)
    train_data = tokens if model_config["use_tokens"] else corpus

    if model_config["do_train"]:
        model.train(train_data)

    top_words, top_scores = model.get_top_keywords(train_data[:sample_size])

    results[model_name] = top_words

print(results)
```

## Implmented models

* Statistical Model
  * CountWords
  * BM25
* Graph-based Model
  * [TextRank](https://aclanthology.org/W04-3252/)
* Embed-based Model
  * [EmbedRank](https://arxiv.org/abs/1801.04470)
  * [KeyBERT](https://github.com/MaartenGr/KeyBERT)
