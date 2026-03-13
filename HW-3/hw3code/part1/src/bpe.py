import json
import sys
from collections import Counter
from typing import TypeAlias

import regex as re
from tqdm.auto import trange

Bigram: TypeAlias = tuple[int, int]


def string_to_ascii(s: str) -> list[int]:
    """Convert string to ascii.

    Args:
        s: An ASCII string

    Returns:
        An array of integers (0 <= i < 128) under ASCII encoding
    """
    return list(s.encode("ascii"))


def string_to_unicode_bytes(s: str) -> bytes:
    """Convert string to unicode bytes.

    Args:
        s: A unicode string

    Returns:
        An array of integers (0 <= i < 256) under UTF-8 encoding
    """
    return list(s.encode("utf-8"))


def compute_bigram_statistics(
    token_ids: list[int], counter: Counter | None = None
) -> Counter[Bigram, int]:
    """Compute adjacent token (i.e., bigram) statistics.

    Args:
        token_ids (list[int]): sequence of token ids

    Returns:
        A counter (dict-like object) with frequencies of bigrams

    For examples, [1, 2, 3, 1, 2, 3] -> {(1, 2): 2, (2, 3): 2, (3, 1): 1}
    """

    bigram_counter = Counter() if counter is None else counter
    for left, right in zip(token_ids, token_ids[1:]):
        bigram_counter[(left, right)] += 1
    return bigram_counter


def replace_bigram(token_ids: list[int], bigram: Bigram, bigram_id: int) -> list[int]:
    """Replaces all copies of a bigram with `bigram_id`.

    Args:
        token_ids: List of token ids
        bigram: The bigram to replace. A bigram is a tuple of exactly two token ids.
        bigram_id: New id for the bigram

    Returns:
        list[int]: sequence with bigram replaced
    """

    idx = 0
    new_token_ids = []
    while idx < len(token_ids):
        if token_ids[idx : idx + 2] == list(bigram):
            new_token_ids.append(bigram_id)
            idx += 2
        else:
            new_token_ids.append(token_ids[idx])
            idx += 1
    return new_token_ids


class ASCIIBPETokenizer:
    def __init__(self):
        self.vocab: list[str] = [chr(i) for i in range(128)]
        self.merge_rules: dict[Bigram, int] = {}

    def merge(self, token_ids: list[int]) -> list[int]:
        """Perform one merge in the BPE algorithm.

        Specifically, implement the following:
          1. Find the most frequent bigram B, and create a new token T for it.
            If two bigram have the same frequency, break tie by taking
            the lexicographically smaller bigram.
          2. Replace all occurrences of B in token_ids with T

        Args:
            token_ids (list[int]): Current list of token ids

        Returns:
            list[int]: New list of token ids, after one merge step
        """
        bigram_counts = compute_bigram_statistics(token_ids)
        if not bigram_counts:
            return token_ids

        best_bigram = min(
            bigram_counts.keys(),
            key=lambda b: (-bigram_counts[b], b),
        )
        left_id, right_id = best_bigram
        new_id = len(self.vocab)
        new_token_str = self.vocab[left_id] + self.vocab[right_id]
        self.vocab.append(new_token_str)
        self.merge_rules[best_bigram] = new_id
        return replace_bigram(token_ids, best_bigram, new_id)

    def encode(self, text: str) -> list[int]:
        """Convert text to tokens.

        Args:
            text: An arbitrary ASCII string

        Returns:
            Tokens produced under BPE
        """

        assert all(ord(c) < 128 for c in text), "input text is not ASCII"

        token_ids = string_to_ascii(text)

        for bigram, merged_id in self.merge_rules.items():
            token_ids = replace_bigram(token_ids, bigram, merged_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Convert tokens back to text.

        Args:
            token_ids: A list of token ids.

        Returns:
            str: An ASCII string.
        """
        return "".join(self.vocab[t] for t in token_ids)

    @classmethod
    def from_config(cls, config_file: str):
        """Load tokenizer from a JSON config file.

        Args:
            config_file: Path to the config file

        Returns:
            Initialized tokenizer object
        """
        with open(config_file, "r") as f:
            config = json.load(f)

        tokenizer = cls()
        tokenizer.vocab = config["vocab"]
        tokenizer.merge_rules = {
            tuple(bigram): bigram_id for bigram, bigram_id in config["merge-rules"]
        }
        return tokenizer

    def save(self, path: str) -> None:
        """Dumps tokenizer configuration

        Args:
            path: Path to write config
        """
        config = {
            "vocab": self.vocab,
            "merge-rules": [
                [bigram, bigram_id] for bigram, bigram_id in self.merge_rules.items()
            ],
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_data(cls, train_data: str, n_merges: int):
        """Train an ASCIIBPETokenizer from data.

        Args:
            train_data (str): A text corpus in ASCII format.
            n_merges (int): The number of merges to perform.

        Returns:
            Trained BPE tokenizer.
        """
        tokenizer = cls()
        assert all(ord(c) < 128 for c in train_data), "train_data is not ASCII"

        token_ids = string_to_ascii(train_data)
        for _ in trange(n_merges, desc="merging..", file=sys.stdout):
            token_ids = tokenizer.merge(token_ids)

        return tokenizer

