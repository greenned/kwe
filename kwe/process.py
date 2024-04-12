import re
import string
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset


def clean_text(
    example,
    target_column: str,
    regex: List[Tuple[str, str]] = None,
    stop_chars: str = string.punctuation,
):

    text = example[target_column]

    text = replace_regex(
        remove_chars(
            remove_linebreaks(
                do_lower(text),
            ),
            stop_chars,
        ),
        regex,
    )

    example[f"{target_column}_cleaned"] = text
    # example[f"{target_column}_tokens"] = data.split()

    return example


def do_lower(text: str) -> str:
    return text.lower()


def replace_regex(text: str, regex: Optional[List[Tuple[str, str]]]) -> str:
    if regex is None:
        return text
    for from_, to in regex:
        text = re.sub(from_, to, text)
    return text


def remove_chars(text: str, stop_chars: str = string.punctuation) -> str:
    return "".join([char for char in text if char not in stop_chars])


def remove_linebreaks(text: str) -> str:
    return text.replace("\n", " ").strip()


def split_tokenizer(example, target_column: str, split: str = " ") -> List[str]:

    example[f"{target_column}_tokens"] = example[f"{target_column}_cleaned"].split(
        split
    )
    return example


if __name__ == "__main__":
    text = "The \tquick \nbrown fox jumps over the  lazy dog  "
