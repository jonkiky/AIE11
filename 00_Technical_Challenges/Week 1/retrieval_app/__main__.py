"""Run retrieval with source chunks or an optional cached topic wiki.

    uv run python -m retrieval_app "your question here"
    uv run python -m retrieval_app "..." --corpus data/my_own_notes.txt
    uv run python -m retrieval_app "..." --retriever wiki
"""

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from . import answer as answer_stage
from . import corpus as corpus_stage
from . import index as index_stage

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent / "data"


def find_corpus() -> tuple[Path | None, list[Path]]:
    """Find the .txt file to search, whatever it happens to be called.

    Swapping corpora is meant to be a file drop, not a code edit: put your own
    .txt in data/ and this finds it. Two files in there and it will not guess
    which you meant, so it asks for --corpus instead.

    Returns:
        (the corpus, every .txt found). The first is None when there is not
        exactly one candidate, and then the caller explains what to do.
    """
    found = sorted(DATA_DIR.glob("*_kb.txt"))
    return (found[0] if len(found) == 1 else None), found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retrieval_app", description=__doc__)
    parser.add_argument("question", help="the question to answer")
    parser.add_argument("-k", type=int, default=3, help="how many chunks to retrieve (default: 3)")
    parser.add_argument("--corpus", type=Path, default=None, help="the .txt file to search (default: the one in data/)")
    parser.add_argument("--show-chunks", action="store_true", help="print the retrieved text, not just the citations")
    parser.add_argument("--retriever", choices=["chunks", "wiki"], default="chunks")
    parser.add_argument("--rebuild-wiki", action="store_true", help="regenerate the wiki and its embeddings")
    parser.add_argument("--wiki-threshold", type=float, default=0.55)
    parser.add_argument("--wiki-delta", type=float, default=None, help="deprecated; has no effect on top-three matching")
    parser.add_argument("--wiki-strict-top1", action="store_true", help="match only the top topic per extracted phrase")
    args = parser.parse_args(argv)
    if args.rebuild_wiki and args.retriever != "wiki":
        parser.error("--rebuild-wiki requires --retriever wiki")
    if not 0 <= args.wiki_threshold <= 1 or (
        args.wiki_delta is not None and not 0 <= args.wiki_delta <= 1
    ):
        parser.error("--wiki-threshold and --wiki-delta must be between 0 and 1")

    # Load optional settings (e.g. LLM_API_BASE) from the .env file next to this package.
    load_dotenv(APP_DIR.parent / ".env")

    corpus_path = args.corpus
    if corpus_path is None:
        corpus_path, found = find_corpus()
        if corpus_path is None:
            if not found:
                print(f"No .txt file in {DATA_DIR}/", file=sys.stderr)
                print("  Put one there, or point at it with --corpus", file=sys.stderr)
            else:
                print(f"More than one .txt in {DATA_DIR}/ - say which one:", file=sys.stderr)
                for candidate in found:
                    print(f"  --corpus data/{candidate.name}", file=sys.stderr)
            return 2
    if not corpus_path.is_file():
        print(f"No such corpus file: {corpus_path}", file=sys.stderr)
        return 2

    started = time.perf_counter()

    lines = corpus_stage.load_corpus(corpus_path)
    wiki_index = None
    if args.retriever == "wiki":
        from .wiki import load_or_build_wiki
        wiki_index = load_or_build_wiki(corpus_path, rebuild=args.rebuild_wiki)
        print(f"wiki {corpus_path.name}: {len(wiki_index.topics)} topics", file=sys.stderr)
    else:
        chunks = corpus_stage.split_corpus(lines)
        print(f"corpus {corpus_path.name}: {len(lines):,} lines -> {len(chunks)} chunks", file=sys.stderr, flush=True)
        matrix = index_stage.build_index(chunks)
    indexed = time.perf_counter()
    print(f"indexed in {indexed - started:.2f}s", file=sys.stderr, flush=True)

    if wiki_index is not None:
        from .wiki_search import no_match_reply, search_wiki
        hits = search_wiki(
            args.question, wiki_index, threshold=args.wiki_threshold,
            delta=args.wiki_delta, strict_top1=args.wiki_strict_top1,
        )
        reply = answer_stage.answer(args.question, hits) if hits else no_match_reply(wiki_index)
    else:
        hits = index_stage.search(args.question, chunks, matrix, k=args.k)
        reply = answer_stage.answer(args.question, hits)

    print(f"\nQ: {args.question}\n")
    print(reply)

    # Every answer names its sources. This is not decoration: an answer a user
    # cannot trace is a rumour, and in Week 4 you debug wrong answers by chunk.
    print("\nSources:")
    if not hits:
        print("  (none retrieved)")
    for hit in hits:
        print(f"  {hit.citation()}")
        if args.show_chunks:
            print(f"      {' '.join(hit.text.split())[:400]}...")

    if wiki_index is not None:
        topics_by_id = {topic.topic_id: topic for topic in wiki_index.topics}
        picked = dict.fromkeys(topic_id for hit in hits for topic_id in hit.topic_ids)
        print("\nWiki topics picked:")
        if not picked:
            print("  (none matched)")
        for topic_id in picked:
            topic = topics_by_id[topic_id]
            ranges = ", ".join(f"L{start}-L{end}" for start, end in topic.ranges)
            print(f"  {topic_id} · {topic.title} ({ranges})")

    sys.stdout.flush()
    print(f"\ntotal {time.perf_counter() - started:.2f}s", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NotImplementedError as exc:
        print(f"\n  Not written yet: {exc}\n", file=sys.stderr)
        print("  Start with corpus.py, then index.py, then answer.py.", file=sys.stderr)
        raise SystemExit(1)
