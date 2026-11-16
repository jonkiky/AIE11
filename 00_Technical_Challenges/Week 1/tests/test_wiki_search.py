import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from retrieval_app import answer as answer_stage
from retrieval_app import wiki_search
from retrieval_app.wiki import WikiIndex, WikiTopic


class WikiSearchTests(unittest.TestCase):
    def setUp(self):
        self.index = WikiIndex(
            [
                WikiTopic("T01", "First useful source topic", "First.", ["one", "two", "three"], [(1, 2)]),
                WikiTopic("T02", "Second useful source topic", "Second.", ["one", "two", "three"], [(2, 3)]),
            ],
            np.array([[1.0, 0.0], [0.99, 0.1]]),
            ["First source.", "Shared source.", "Third source."],
            Path("notes.txt"),
        )

    def test_overlapping_topics_merge_source_ranges_and_labels(self):
        with (
            patch.object(wiki_search, "chat_json", return_value={"topics": ["first"]}),
            patch.object(wiki_search, "embed_texts", return_value=np.array([[1.0, 0.0], [1.0, 0.0]])),
        ):
            hits = wiki_search.search_wiki("question", self.index)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].text, "First source.\nShared source.\nThird source.")
        self.assertEqual(hits[0].topic_ids, ("T01", "T02"))
        self.assertEqual(hits[0].line_ranges, ((1, 3),))
        messages = answer_stage.build_messages("question", hits)
        self.assertIn("[T01,T02 L1-L3]", messages[1]["content"])
        self.assertIn("[T01,T02 L1-L3]", hits[0].citation())

    def test_selects_at_most_three_topics_and_applies_threshold(self):
        scores = [1.0, 0.9, 0.8, 0.795, 0.2]
        index = WikiIndex(
            [
                WikiTopic(f"T{i + 1:02d}", f"Topic {i + 1}", "Summary.", [], [(2 * i + 1, 2 * i + 1)])
                for i in range(len(scores))
            ],
            np.array([[score, np.sqrt(1 - score ** 2)] for score in scores]),
            [f"Source {i}." for i in range(1, 10)],
            Path("notes.txt"),
        )
        with (
            patch.object(wiki_search, "chat_json", return_value={"topics": ["question"]}),
            patch.object(wiki_search, "embed_texts", return_value=np.array([[1.0, 0.0]])),
        ):
            hits = wiki_search.search_wiki("question", index)
            high_threshold_hits = wiki_search.search_wiki("question", index, threshold=0.85)

        self.assertEqual([hit.topic_ids for hit in hits], [("T01",), ("T02",), ("T03",)])
        self.assertEqual([hit.topic_ids for hit in high_threshold_hits], [("T01",), ("T02",)])

    def test_selects_three_topics_for_each_query_vector(self):
        vectors = np.array([
            [1.0, 0.0], [0.9, np.sqrt(0.19)], [0.8, 0.6],
            [0.6, 0.8], [np.sqrt(0.19), 0.9], [0.0, 1.0],
        ])
        index = WikiIndex(
            [
                WikiTopic(f"T{i + 1:02d}", f"Topic {i + 1}", "Summary.", [], [(2 * i + 1, 2 * i + 1)])
                for i in range(len(vectors))
            ],
            vectors, [f"Source {i}." for i in range(1, 12)], Path("notes.txt"),
        )
        with (
            patch.object(wiki_search, "chat_json", return_value={"topics": ["focused phrase"]}),
            patch.object(wiki_search, "embed_texts", return_value=np.array([[1.0, 0.0], [0.0, 1.0]])) as embed,
        ):
            hits = wiki_search.search_wiki("whole question", index)

        embed.assert_called_once_with(["whole question", "focused phrase"])
        self.assertEqual([hit.topic_ids for hit in hits], [(f"T{i:02d}",) for i in range(1, 7)])

    def test_no_match_does_not_invent_context(self):
        with (
            patch.object(wiki_search, "chat_json", return_value={"topics": ["unrelated"]}),
            patch.object(wiki_search, "embed_texts", return_value=np.array([[0.0, -1.0]])),
        ):
            self.assertEqual(wiki_search.search_wiki("unrelated", self.index), [])
        self.assertIn("First useful source topic", wiki_search.no_match_reply(self.index))

    def test_bad_query_topics_warn_and_use_raw_question(self):
        with (
            patch.object(wiki_search, "chat_json", return_value={"invalid": []}),
            patch.object(wiki_search, "embed_texts", return_value=np.array([[1.0, 0.0]])) as embed,
            self.assertLogs("retrieval_app.wiki_search", level="WARNING"),
        ):
            wiki_search.search_wiki("question", self.index)
        embed.assert_called_once_with(["question"])

    def test_legacy_delta_warns_and_does_not_limit_top_three(self):
        with (
            patch.object(wiki_search, "chat_json", return_value={"topics": ["question"]}),
            patch.object(wiki_search, "embed_texts", return_value=np.array([[1.0, 0.0]])),
            self.assertWarnsRegex(FutureWarning, "deprecated"),
        ):
            hits = wiki_search.search_wiki("question", self.index, delta=0)
        self.assertEqual(hits[0].topic_ids, ("T01", "T02"))

    def test_context_budget_drops_lowest_topic_and_strict_mode_selects_top_one(self):
        self.index.topics[1].ranges = [(3, 3)]
        with (
            patch.object(wiki_search, "chat_json", return_value={"topics": ["first"]}),
            patch.object(wiki_search, "embed_texts", return_value=np.array([[1.0, 0.0], [1.0, 0.0]])),
        ):
            hits = wiki_search.search_wiki("question", self.index, context_budget=30)
        self.assertEqual(hits[0].topic_ids, ("T01",))
        with (
            patch.object(wiki_search, "chat_json", return_value={"topics": ["first"]}),
            patch.object(wiki_search, "embed_texts", return_value=np.array([[1.0, 0.0]])) as embed,
        ):
            hits = wiki_search.search_wiki("question", self.index, strict_top1=True)
        embed.assert_called_once_with(["first"])
        self.assertEqual(hits[0].topic_ids, ("T01",))


if __name__ == "__main__":
    unittest.main()
