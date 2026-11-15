"""Stage 1 and 2: load the source text, and split it into chunks.

YOUR CODE GOES HERE. `load_corpus` still raises NotImplementedError.
You saw working versions of both in Session 2's `aimakerspace/text_utils.py`:
read it, understand it, then write your own. Copying it verbatim is allowed;
not being able to explain it is not.

Neither function knows or cares what your corpus is about. That is the point:
swap the .txt in data/ and nothing in this file should need to change.
"""

from pathlib import Path


def load_corpus(path: Path) -> str:
    """Read a .txt file and return its entire contents as one string.

    Args:
        path: the .txt file to read, whichever corpus the app was pointed at.

    Returns:
        The whole file as a single string.

    Hint: this is three lines. `open()`, or `Path.read_text()`. Mind the
    encoding: read it as utf-8, and your own corpus may need you to mean it.
    """
    raise NotImplementedError("Write load_corpus() in retrieval_app/corpus.py")


def split_corpus(text: str, chunk_size: int = 1000) -> list[str]:
    """Split each nonblank line independently, without overlap.

    Args:
        text: the whole corpus, from load_corpus().
        chunk_size: target number of characters per chunk, not a hard limit.

    Returns:
        Chunks with surrounding whitespace removed. Short lines stay separate.
        Long lines split after the last period within chunk_size, or after the
        next period if none fits. A remainder without periods stays intact,
        even if it exceeds chunk_size. Only literal periods are boundaries.

    Raises:
        ValueError: if chunk_size is not positive.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    chunks = []
    for line in text.splitlines():
        line = line.strip()
        start = 0
        while start < len(line):
            end = len(line)
            if end - start > chunk_size:
                period = line.rfind(".", start, start + chunk_size)
                if period == -1:
                    period = line.find(".", start + chunk_size)
                if period != -1:
                    end = period + 1
            chunks.append(line[start:end].strip())
            start = end
            while start < len(line) and line[start].isspace():
                start += 1
    return chunks
