"""Stage 1 and 2: load the source text, and split it into chunks.

The loader returns lines, and the splitter turns those lines into chunks.
Session 2's `aimakerspace/text_utils.py` is a reference for these stages.

Neither function knows or cares what your corpus is about. That is the point:
swap the .txt in data/ and nothing in this file should need to change.




"""

import re
from pathlib import Path

from .tokens import MAX_SOURCE_TOKENS, count_tokens, tokenizer


def load_corpus(path: Path) -> list[str]:
    """Read a UTF-8 .txt file and return its lines.

    Args:
        path: the .txt file to read, whichever corpus the app was pointed at.

    Returns:
        Nonblank lines without newline characters. An empty file returns an
        empty list.
    """
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def split_text(
    text: str, max_tokens: int = MAX_SOURCE_TOKENS, prefix: str = ""
) -> list[str]:
    """Keep sentences whole when possible, with a hard Qwen-token limit.

    An oversized sentence splits at whitespace, or at a Unicode character
    boundary if necessary. A prefix can reserve room for wiki line labels.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")
    remaining = text.strip()
    chunks = []
    while remaining:
        encoded = tokenizer().encode(prefix + remaining, add_special_tokens=False)
        if len(encoded.ids) <= max_tokens:
            chunks.append(remaining)
            break
        end = encoded.offsets[max_tokens - 1][1] - len(prefix)
        while end > 0 and count_tokens(prefix + remaining[:end].rstrip()) > max_tokens:
            end -= 1
        if end <= 0:
            raise ValueError("Token limit is too small for the line label and one character")
        boundaries = list(
            re.finditer(r"""[.!?。！？]+["')\]”’]*(?=\s|$)""", remaining)
        )
        sentence_ends = [match.end() for match in boundaries if match.end() <= end]
        if sentence_ends:
            end = sentence_ends[-1]
        else:
            whitespace = list(re.finditer(r"\s+", remaining[:end]))
            if whitespace and remaining[:whitespace[-1].start()].strip():
                end = whitespace[-1].start()
        while end > 0 and count_tokens(prefix + remaining[:end].strip()) > max_tokens:
            end -= 1
        chunk = remaining[:end].strip()
        if not chunk:
            raise ValueError("Could not split text within the token limit")
        chunks.append(chunk)
        remaining = remaining[end:].strip()
    return chunks


def split_corpus(
    lines: list[str], chunk_size: int = MAX_SOURCE_TOKENS
) -> list[str]:
    """Split lines independently into at most chunk_size Qwen tokens.

    Short lines remain separate and chunks do not overlap. Unlike the old
    character target, chunk_size is now a hard token limit.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    return [chunk for line in lines for chunk in split_text(line, chunk_size)]
