"""Human-editable wiki topics and a source-aware embedding cache."""

import hashlib
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .index import EMBEDDING_MODEL, embed_texts
from .tokens import (
    MAX_SOURCE_TOKENS, WIKI_OVERLAP_TOKENS, WIKI_WINDOW_TOKENS, count_tokens,
)

FORMAT_VERSION = 1
WIKI_ROOT = Path(__file__).resolve().parent.parent / "wiki"


@dataclass
class WikiTopic:
    topic_id: str
    title: str
    summary: str
    keywords: list[str]
    ranges: list[tuple[int, int]]

    def embedding_text(self) -> str:
        return f"{self.title}. {self.summary} Keywords: {', '.join(self.keywords)}"


@dataclass
class WikiIndex:
    topics: list[WikiTopic]
    matrix: np.ndarray
    lines: list[str]
    corpus_path: Path


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def validate_topic(topic: WikiTopic, line_count: int) -> None:
    for value in [topic.title, topic.summary, *topic.keywords]:
        if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
            raise ValueError("Wiki topic fields must be nonempty single-line strings")
    if not 3 <= len(topic.title.split()) <= 10:
        raise ValueError("Wiki topic titles must contain 3 to 10 words")
    if not 3 <= len(topic.keywords) <= 8 or any("," in word for word in topic.keywords):
        raise ValueError("Wiki topics need 3 to 8 comma-free keyword phrases")
    if not topic.ranges or any(
        not 1 <= start <= end <= line_count for start, end in topic.ranges
    ):
        raise ValueError("Wiki source ranges must be within the corpus line numbers")
    if count_tokens(topic.embedding_text()) > MAX_SOURCE_TOKENS:
        raise ValueError("Wiki topic text exceeds the 1,000-token embedding limit")


def render_wiki(
    corpus_name: str, metadata: dict, topics: list[WikiTopic], source_path: str
) -> str:
    sections = [
        f"# Wiki: {corpus_name}",
        f"<!-- wiki-meta: {json.dumps(metadata, ensure_ascii=False)} -->",
    ]
    for topic in topics:
        sources = ", ".join(
            f"[L{start}-L{end}]({source_path}#L{start}-L{end})"
            for start, end in topic.ranges
        )
        sections.append(
            f"## {topic.topic_id} \u00b7 {topic.title}\n"
            f"- **Summary:** {topic.summary}\n"
            f"- **Keywords:** {', '.join(topic.keywords)}\n"
            f"- **Sources:** {sources}"
        )
    return "\n\n".join(sections) + "\n"


def parse_wiki(text: str) -> tuple[dict, list[WikiTopic]]:
    header = re.match(r"\A# Wiki: [^\n]+\n\n<!-- wiki-meta: ([^\n]+) -->\n", text)
    if header is None:
        raise ValueError("Wiki header or metadata is invalid; use --rebuild-wiki")
    metadata = json.loads(header[1])
    if not isinstance(metadata, dict) or type(metadata.get("lines")) is not int:
        raise ValueError("Wiki metadata must include an integer line count")
    entry = re.compile(
        r"## (T\d{2,}) \u00b7 ([^\n]+)\n"
        r"- \*\*Summary:\*\* ([^\n]+)\n"
        r"- \*\*Keywords:\*\* ([^\n]+)\n"
        r"- \*\*Sources:\*\* ([^\n]+)"
    )
    topics = []
    remaining = text[header.end():].strip()
    seen = set()
    while remaining:
        match = entry.match(remaining)
        if match is None:
            raise ValueError("Invalid wiki entry format; preserve the documented Markdown fields")
        links = match[5].split(", ")
        ranges = []
        for link in links:
            source = re.fullmatch(r"\[L(\d+)-L(\d+)\]\([^)\n]+#L(\d+)-L(\d+)\)", link)
            if source is None or source.groups()[:2] != source.groups()[2:]:
                raise ValueError("Wiki source labels must match their line anchors")
            ranges.append((int(source[1]), int(source[2])))
        topic = WikiTopic(match[1], match[2], match[3], match[4].split(", "), ranges)
        validate_topic(topic, metadata["lines"])
        if topic.topic_id in seen:
            raise ValueError(f"Duplicate wiki topic ID: {topic.topic_id}")
        seen.add(topic.topic_id)
        topics.append(topic)
        remaining = remaining[match.end():].strip()
    if not topics:
        raise ValueError("Wiki contains no topics")
    covered = {line for topic in topics for start, end in topic.ranges for line in range(start, end + 1)}
    if covered != set(range(1, metadata["lines"] + 1)):
        raise ValueError("Wiki source ranges do not cover every corpus line")
    return metadata, topics


def _validate_matrix(matrix: np.ndarray, topic_count: int) -> None:
    if (
        matrix.ndim != 2 or matrix.shape[0] != topic_count
        or matrix.shape[1] == 0 or not np.isfinite(matrix).all()
    ):
        raise ValueError("Wiki embedding cache is invalid; use --rebuild-wiki")


def load_or_build_wiki(
    corpus_path: Path, rebuild: bool = False, cache_dir: Path | None = None
) -> WikiIndex:
    from .answer import CHAT_MODEL
    from .wiki_builder import build_topics

    corpus_path = corpus_path.resolve()
    raw = corpus_path.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("Wiki corpus must be nonempty and contain no blank lines")
    digest = hashlib.sha256(raw).hexdigest()
    directory = cache_dir if cache_dir is not None else WIKI_ROOT / corpus_path.stem
    wiki_path, matrix_path = directory / "wiki.md", directory / "topics.npy"
    metadata = {}
    topics = []
    if wiki_path.exists() and not rebuild:
        metadata, topics = parse_wiki(wiki_path.read_text(encoding="utf-8"))
    stale = (
        rebuild or not topics or metadata.get("corpus_sha256") != digest
        or metadata.get("format_version") != FORMAT_VERSION
        or metadata.get("corpus") != str(corpus_path)
        or metadata.get("builder_model") != CHAT_MODEL
        or metadata.get("source_token_limit") != MAX_SOURCE_TOKENS
        or metadata.get("wiki_window_tokens") != WIKI_WINDOW_TOKENS
        or metadata.get("wiki_overlap_tokens") != WIKI_OVERLAP_TOKENS
    )
    if stale:
        print(f"Building wiki for {corpus_path.name}...", file=sys.stderr, flush=True)
        topics = build_topics(lines)
        metadata = {
            "corpus": str(corpus_path), "corpus_sha256": digest, "lines": len(lines),
            "builder_model": CHAT_MODEL, "format_version": FORMAT_VERSION,
            "source_token_limit": MAX_SOURCE_TOKENS,
            "wiki_window_tokens": WIKI_WINDOW_TOKENS,
            "wiki_overlap_tokens": WIKI_OVERLAP_TOKENS,
        }
    texts = [topic.embedding_text() for topic in topics]
    text_digest = hashlib.sha256(json.dumps(texts).encode("utf-8")).hexdigest()
    cache_bytes = matrix_path.read_bytes() if matrix_path.exists() else b""
    refresh_vectors = (
        stale or not cache_bytes or metadata.get("embedding_text_sha256") != text_digest
        or metadata.get("embedding_model") != EMBEDDING_MODEL
        or metadata.get("matrix_sha256") != hashlib.sha256(cache_bytes).hexdigest()
    )
    if refresh_vectors:
        print("Embedding wiki topics and refreshing cache...", file=sys.stderr, flush=True)
        matrix = embed_texts(texts)
        _validate_matrix(matrix, len(topics))
        buffer = io.BytesIO()
        np.save(buffer, matrix, allow_pickle=False)
        cache_bytes = buffer.getvalue()
        metadata.update(
            embedding_text_sha256=text_digest, embedding_model=EMBEDDING_MODEL,
            matrix_sha256=hashlib.sha256(cache_bytes).hexdigest(),
        )
        source_path = quote(Path(os.path.relpath(corpus_path, directory)).as_posix(), safe="/")
        markdown = render_wiki(corpus_path.name, metadata, topics, source_path)
        parse_wiki(markdown)
        directory.mkdir(parents=True, exist_ok=True)
        matrix_path.write_bytes(cache_bytes)
        wiki_path.write_text(markdown, encoding="utf-8")
    else:
        matrix = np.load(io.BytesIO(cache_bytes), allow_pickle=False)
    _validate_matrix(matrix, len(topics))
    return WikiIndex(topics, matrix, lines, corpus_path)
