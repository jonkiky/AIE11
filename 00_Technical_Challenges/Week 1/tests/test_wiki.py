import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from retrieval_app import wiki, wiki_builder
from retrieval_app.tokens import count_tokens


class WikiTests(unittest.TestCase):
    def test_invalid_extraction_is_retried_with_validation_feedback(self):
        topic = {"title": "A useful source topic", "summary": "Source.", "keywords": ["one", "two", "three"], "ranges": [[1, 1]]}
        with patch.object(
            wiki_builder, "chat_json",
            side_effect=[{"topics": [{**topic, "ranges": [[1]]}]}, {"topics": [topic]}],
        ) as chat:
            topics = wiki_builder._extract_window("L1: Source.", ["Source."])
        self.assertEqual(topics[0].ranges, [(1, 1)])
        self.assertIn("received [1]", chat.call_args.args[0])

    def test_extra_model_keywords_are_trimmed_to_eight(self):
        topic = {"title": "A useful source topic", "summary": "Source.", "keywords": [f"k{n}" for n in range(10)], "ranges": [[1, 1]]}
        with patch.object(wiki_builder, "chat_json", return_value={"topics": [topic]}) as chat:
            topics = wiki_builder._extract_window("L1: Source.", ["Source."])
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(topics[0].keywords, [f"k{n}" for n in range(8)])

    def test_round_trip_and_rejects_bad_ranges(self):
        topics = [wiki.WikiTopic("T01", "Working with senior leaders", "Be concise.", ["time", "context", "goals"], [(1, 2)])]
        text = wiki.render_wiki("notes.txt", {"lines": 2}, topics, "../../data/notes.txt")
        metadata, parsed = wiki.parse_wiki(text)
        self.assertEqual(metadata, {"lines": 2})
        self.assertEqual(parsed, topics)
        with self.assertRaises(ValueError):
            wiki.parse_wiki(text.replace("#L1-L2", "#L1-L9").replace("[L1-L2]", "[L1-L9]"))
        with self.assertRaises(ValueError):
            wiki.parse_wiki(text + "\nUnexpected content\n")

    def test_numbered_windows_obey_cap_and_keep_source_lines(self):
        lines = ["A short sentence.", "word " * 4600, "The final sentence."]
        windows = wiki_builder.source_windows(lines)
        self.assertTrue(all(count_tokens(window) <= 2000 for window in windows))
        combined = "\n".join(windows)
        self.assertIn("L1: A short sentence.", combined)
        self.assertIn("L3: The final sentence.", combined)
        self.assertGreater(combined.count("L2:"), 1)
        self.assertEqual(
            wiki_builder.source_windows(["First.", "Third."], [1, 3]),
            ["L1: First.", "L3: Third."],
        )

    def test_windows_overlap_by_at_most_500_tokens(self):
        lines = [f"Sentence number {n} explains one idea." for n in range(1, 400)]
        windows = wiki_builder.source_windows(lines)
        self.assertGreater(len(windows), 1)
        for first, second in zip(windows, windows[1:]):
            self.assertLessEqual(count_tokens(first), 2000)
            shared = [line for line in second.split("\n") if line in first.split("\n")]
            self.assertTrue(shared)
            self.assertGreater(count_tokens("\n".join(shared)), 400)
            self.assertLessEqual(count_tokens("\n".join(shared)), 500)

    def test_build_repairs_uncovered_lines_and_assigns_ids(self):
        lines = ["First line.", "Second line.", "Third line."]
        extracted = {"topics": [{"title": "First source topic", "summary": "First.", "keywords": ["first", "start", "topic"], "ranges": [[1, 1]]}]}
        with (
            patch.object(wiki_builder, "chat_json", side_effect=[extracted, {"topics": []}]),
            patch.object(wiki_builder, "embed_texts", return_value=np.array([[1.0, 0.0]])),
        ):
            topics = wiki_builder.build_topics(lines)
        self.assertEqual(topics[0].topic_id, "T01")
        self.assertEqual(topics[0].ranges, [(1, 3)])

    def test_build_retries_initial_extraction_but_bounds_coverage_repair(self):
        topic = {"title": "A useful source topic", "summary": "First.", "keywords": ["first", "start", "topic"], "ranges": [[1, 1]]}
        invalid = {"topics": [{**topic, "ranges": [[1]]}]}
        with (
            patch.object(wiki_builder, "chat_json", side_effect=[invalid, {"topics": [topic]}, invalid]) as chat,
            patch.object(wiki_builder, "embed_texts", return_value=np.array([[1.0, 0.0]])),
        ):
            topics = wiki_builder.build_topics(["First.", "Second."])
        self.assertEqual(chat.call_count, 3)
        self.assertEqual(topics[0].ranges, [(1, 2)])

    def test_cache_reuses_then_reembeds_manual_summary_edit(self):
        topic = wiki.WikiTopic("T01", "A useful source topic", "A short summary.", ["one", "two", "three"], [(1, 1)])
        with TemporaryDirectory() as directory:
            corpus = Path(directory) / "notes.txt"
            corpus.write_text("Source text.\n", encoding="utf-8")
            cache = Path(directory) / "wiki"
            with (
                patch.object(wiki_builder, "build_topics", return_value=[topic]) as build,
                patch.object(wiki, "embed_texts", return_value=np.array([[1.0, 0.0]], dtype=np.float32)) as embed,
            ):
                first = wiki.load_or_build_wiki(corpus, cache_dir=cache)
                second = wiki.load_or_build_wiki(corpus, cache_dir=cache)
                self.assertEqual(build.call_count, 1)
                self.assertEqual(embed.call_count, 1)
                np.testing.assert_array_equal(first.matrix, second.matrix)
                path = cache / "wiki.md"
                path.write_text(path.read_text().replace("A short summary.", "An edited summary."), encoding="utf-8")
                edited = wiki.load_or_build_wiki(corpus, cache_dir=cache)
                self.assertEqual(edited.topics[0].summary, "An edited summary.")
                self.assertEqual(embed.call_count, 2)
                with patch.object(wiki, "WIKI_WINDOW_TOKENS", 1000):
                    wiki.load_or_build_wiki(corpus, cache_dir=cache)
                self.assertEqual(build.call_count, 2)
                corpus.write_text("Changed source text.\n", encoding="utf-8")
                wiki.load_or_build_wiki(corpus, cache_dir=cache)
                self.assertEqual(build.call_count, 3)
            corpus.write_text("Source.\n\nMore.\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "blank"):
                wiki.load_or_build_wiki(corpus, cache_dir=cache)


if __name__ == "__main__":
    unittest.main()
