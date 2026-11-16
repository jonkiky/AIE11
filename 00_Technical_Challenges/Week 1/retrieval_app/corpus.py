"""Stage 1 and 2: load the source text, and split it into chunks.

The loader returns lines, and the splitter turns those lines into chunks.
Session 2's `aimakerspace/text_utils.py` is a reference for these stages.

Neither function knows or cares what your corpus is about. That is the point:
swap the .txt in data/ and nothing in this file should need to change.




"""

from pathlib import Path


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


def split_corpus(lines: list[str], chunk_size: int = 1000) -> list[str]:
    """Split each nonblank line independently, without overlap.

    Args:
        lines: the corpus lines, from load_corpus().
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
    for line in lines:
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
