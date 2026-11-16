"""Shared Qwen token counting for source chunks and user queries."""

from functools import lru_cache

import litellm
from tokenizers import Tokenizer

MAX_SOURCE_TOKENS = 1000
WIKI_WINDOW_TOKENS = 2000
WIKI_OVERLAP_TOKENS = 500
MAX_QUERY_TOKENS = 32_768


@lru_cache(maxsize=1)
def tokenizer() -> Tokenizer:
    """Use the embedding model's tokenizer, without adding special tokens."""
    return litellm.create_pretrained_tokenizer(
        "Qwen/Qwen3-Embedding-0.6B"
    )["tokenizer"]


def count_tokens(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False).ids)


def validate_query(query: str) -> None:
    size = count_tokens(query)
    if size > MAX_QUERY_TOKENS:
        raise ValueError(
            f"query exceeds the {MAX_QUERY_TOKENS:,}-token limit ({size:,} tokens)"
        )
