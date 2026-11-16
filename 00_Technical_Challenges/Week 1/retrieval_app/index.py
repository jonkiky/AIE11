"""Stage 3 and 4: embed the chunks, store them, and search them.

YOUR CODE GOES HERE. `Hit` and the LiteLLM client are written for you, because
plumbing is not the lesson. `embed_texts`, `cosine_similarity`, `build_index`
and `search` are the lesson.
"""

import os
from dataclasses import dataclass
from types import SimpleNamespace

import litellm
import numpy as np

from .tokens import MAX_QUERY_TOKENS, MAX_SOURCE_TOKENS, count_tokens, validate_query

EMBEDDING_MODEL = "ollama/qwen3-embedding:0.6b"
EMBEDDING_DIMENSIONS = 1024


class _Embeddings:
    def __init__(self, api_base: str):
        self._api_base = api_base

    def create(self, input: list[str], model: str, **kwargs):
        response = litellm.embedding(model=model, input=input, api_base=self._api_base, **kwargs)
        # Ollama returns plain dicts; expose `.embedding` like the OpenAI SDK does.
        response.data = [
            SimpleNamespace(**d) if isinstance(d, dict) else d for d in response.data
        ]
        return response


class _Completions:
    def __init__(self, api_base: str):
        self._api_base = api_base

    def create(self, model: str, messages: list[dict], **kwargs):
        return litellm.completion(model=model, messages=messages, api_base=self._api_base, **kwargs)


class LiteLLMClient:
    """A small OpenAI-shaped wrapper around LiteLLM.

    Supports `client().embeddings.create(...)` and
    `client().chat.completions.create(...)`, so the hints below still apply.
    """

    def __init__(self, api_base: str):
        self.embeddings = _Embeddings(api_base)
        self.chat = SimpleNamespace(completions=_Completions(api_base))


_cached: LiteLLMClient | None = None


def client() -> LiteLLMClient:
    """The LiteLLM client, built on first use and reused after that.

    It is built lazily, not at import time, so that .env has already been
    loaded by the time it looks for LLM_API_BASE (defaults to local Ollama).
    """
    global _cached
    if _cached is None:
        _cached = LiteLLMClient(os.getenv("LLM_API_BASE", "http://localhost:11434"))
    return _cached


@dataclass
class Hit:
    """One retrieved chunk, and everything you need to cite it.

    `chunk_id` is what makes an answer traceable: when an answer is wrong in
    Week 4, this is how you find out which chunk caused it. Keep it.
    """

    chunk_id: int
    text: str
    score: float
    label: str | None = None
    topic_ids: tuple[str, ...] = ()
    line_ranges: tuple[tuple[int, int], ...] = ()

    def context_label(self) -> str:
        return self.label or f"chunk {self.chunk_id}"

    def citation(self) -> str:
        """A short, human-readable label for this chunk. Printed under answers."""
        preview = " ".join(self.text.split())[:60]
        return f"[{self.context_label()}] {preview}... (score {self.score:.3f})"


def embed_texts(texts: list[str]) -> np.ndarray:
    """Turn a list of strings into a matrix of embedding vectors.

    Args:
        texts: the strings to embed. One API call can take many at once —
            do NOT loop and call once per chunk.

    Returns:
        An array of shape (len(texts), 1024), one row per input string,
        in the same order as `texts`.

    Hint: `client().embeddings.create(input=texts, model=EMBEDDING_MODEL)`.
    The response has a `.data` list; each item has an `.embedding`. Wrap the
    result in np.array(). Print `response.usage` once — knowing what a corpus
    costs to embed is the number that decides architecture at scale.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSIONS), dtype=np.float32)

    response = client().embeddings.create(input=texts, model=EMBEDDING_MODEL)
    if len(response.data) != len(texts):
        raise ValueError(
            f"Expected {len(texts)} embeddings, received {len(response.data)}"
        )
    data = response.data
    if all(hasattr(item, "index") for item in data):
        data = sorted(data, key=lambda item: item.index)
    return np.asarray([item.embedding for item in data], dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """How similar are two vectors? 1.0 is identical, 0.0 is unrelated.

    A zero vector has no direction, so its similarity is defined here as 0.0.
    """
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    if norm_product == 0:
        return 0.0
    return float(np.dot(a, b) / norm_product)


def build_index(chunks: list[str]) -> np.ndarray:
    """Embed every chunk once, and return the matrix to search against.

    Args:
        chunks: from corpus.split_corpus().

    Returns:
        The embedding matrix. Row i corresponds to chunks[i] — that
        correspondence is the only thing holding your citations together,
        so do not sort or filter one without the other.

    Hint: this is mostly a call to embed_texts(). Notice that it happens once
    per process, and that a fresh process has no memory of the last one.
    """
    if any(count_tokens(chunk) > MAX_SOURCE_TOKENS for chunk in chunks):
        raise ValueError(f"Corpus chunks must not exceed {MAX_SOURCE_TOKENS} tokens")
    return embed_texts(chunks)


def search(query: str, chunks: list[str], matrix: np.ndarray, k: int = 3) -> list[Hit]:
    """Find the k chunks most similar to the query.

    Args:
        query: the user's question, limited to 32,768 Qwen tokens.
        chunks: the chunk texts.
        matrix: the embedding matrix from build_index().
        k: how many hits to return.

    Returns:
        Up to k Hits, most similar first.

    Raises:
        ValueError: if the query exceeds 32,768 tokens or the matrix shape
            does not match the chunks or query embedding.

    Hints:
      - Embed the query with the SAME model you built the index with. A query
        embedded by a different model is not comparable, and the failure is
        silent — you get plausible-looking nonsense, not an error.
      - Score the query against every row, then sort.
      - Read the signature again: it says "up to k". Nothing here forces you to
        return exactly k. That freedom is the whole of Step 4's threshold option.
    """
    validate_query(query)

    if k <= 0:
        return []

    vectors = np.asarray(matrix)
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("matrix must have one embedding row per chunk")
    if not chunks:
        return []

    query_vector = embed_texts([query])[0]
    if vectors.shape[1] != query_vector.shape[0]:
        raise ValueError("query and chunk embeddings must have the same dimensions")

    hits = [
        Hit(chunk_id=chunk_id, text=text, score=cosine_similarity(query_vector, vector))
        for chunk_id, (text, vector) in enumerate(zip(chunks, vectors))
    ]
    return sorted(hits, key=lambda hit: hit.score, reverse=True)[:k]
