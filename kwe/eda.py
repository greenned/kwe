from collections import defaultdict
from typing import List

import nltk
import pandas as pd
from nltk.corpus import stopwords

from .constant import Color


def show_length_stats(df: pd.DataFrame, columns: List[str]):
    pd.set_option("display.float_format", "{:.0f}".format)
    print(
        f"\n{Color.header}EDA for the length of the columns\n-----------------------------{Color.endc}"
    )
    stats = []
    for column in columns:
        stats.append(df[column].str.len().describe())
    stats = pd.concat(stats, axis=1)
    stats.columns = columns
    print(stats)
    # stats.hist(by="column", bins=100)


def check_missing_value(df: pd.DataFrame, target_columns: List[str]):
    print(
        f"\n{Color.header}Start Missing Value Check.......\n-----------------------------{Color.endc}"
    )
    for col, val in df[target_columns].isna().sum().items():
        if val > 0:
            print(f"- There are `{val}` missing values in column `{col}`")
        else:
            print(f"- There are no missing values in column `{col}`")


def check_duplicates(df: pd.DataFrame, target_columns: List[str]):
    print(
        f"\n\n{Color.header}Start Duplication Check.......\n-----------------------------{Color.endc}"
    )
    if df[target_columns].duplicated().any():
        print("- There are duplicated rows in the dataset")
    else:
        print("- There are no duplicated rows in the dataset")


def fill_missing_value(df: pd.DataFrame, fill_value: str, inplace: bool = True):
    print(
        f"\n{Color.header}Fill the missing values with empty string\n-----------------------------{Color.endc}"
    )
    print(f"- Total {df.isna().sum().sum()} missing values filled with empty string")
    if inplace:
        df.fillna(fill_value, inplace=True)
        return df
    else:
        return df.fillna(fill_value)


def check_length(
    df: pd.DataFrame, columns: List[str], fill_value: str = "", inplace: bool = True
):
    check_missing_value(df, columns)
    check_duplicates(df, columns)
    show_length_stats(df, columns)
    if not inplace:
        return df


def check_stopwords(
    df: pd.DataFrame, target_columns: List[str], fields: str = "english"
):
    nltk.download("stopwords")
    stop = set(stopwords.words(fields))

    corpus = []
    dic = defaultdict(int)
    print(
        f"\n{Color.header}Start Stopwords Analysis for `{target_columns}`.....\n-----------------------------------{Color.endc}"
    )
    for column in target_columns:
        new = df[column].str.split()
        new = new.values.tolist()
        corpus += [word for i in new for word in i]
        len_corpus = len(corpus)

        cnt = {}
        for word in corpus:
            if word in stop:
                cnt[word] = cnt.get(word, 0) + 1
        dic[column] = cnt
        len_stopwords = sum(dic[column].values())
        print(
            f"\nNumber of stopwords in `{column}` is {Color.point}{len_stopwords}{Color.endc}"
        )
        print(f"Number of words in `{column}` is {Color.point}{len_corpus}{Color.endc}")
        print(
            f"Percentage of stopwords in `{column}` is {Color.point}{len_stopwords/len_corpus:.5f}{Color.endc}"
        )
        print(
            f"Top 10 stopwords in {column} are {sorted(dic[column].items(), key=lambda x: x[1], reverse=True)[:10]}"
        )
        print(f"\n")
