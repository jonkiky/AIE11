import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from retrieval_app.corpus import load_corpus, split_corpus


class LoadCorpusTests(unittest.TestCase):
    def test_reads_utf8_lines_and_skips_blank_lines(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "book.txt"
            path.write_bytes(
                "  Café.  \r\n\r\n \t\r\nNext line.\nLast line.".encode("utf-8")
            )
            lines = load_corpus(path)

        self.assertEqual(lines, ["  Café.  ", "Next line.", "Last line."])
        self.assertEqual(split_corpus(lines), ["Café.", "Next line.", "Last line."])

    def test_empty_file_returns_empty_list(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empty.txt"
            path.write_text("", encoding="utf-8")
            self.assertEqual(load_corpus(path), [])

    def test_missing_file_raises(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_corpus(Path(directory) / "missing.txt")


class SplitCorpusTests(unittest.TestCase):
    def test_short_lines_stay_separate_and_blank_lines_are_skipped(self):
        self.assertEqual(
            split_corpus(["  First line  ", "", " \t", "Second line"], chunk_size=20),
            ["First line", "Second line"],
        )
        self.assertEqual(split_corpus([]), [])
        self.assertEqual(split_corpus([" ", "\t"]), [])

    def test_long_line_uses_last_period_before_limit_without_overlap(self):
        self.assertEqual(
            split_corpus(["One. Two. Three. Four."], chunk_size=10),
            ["One. Two.", "Three.", "Four."],
        )

    def test_next_period_is_used_if_none_fits(self):
        self.assertEqual(
            split_corpus(["A longer sentence. End."], chunk_size=8),
            ["A longer sentence.", "End."],
        )

    def test_text_without_periods_is_not_cut(self):
        self.assertEqual(
            split_corpus(["A long line without periods", "Next"], chunk_size=5),
            ["A long line without periods", "Next"],
        )
        self.assertEqual(
            split_corpus(["Start. A long remainder without periods"], chunk_size=8),
            ["Start.", "A long remainder without periods"],
        )

    def test_period_at_limit_and_exact_size_line(self):
        self.assertEqual(split_corpus(["Four. End."], chunk_size=5), ["Four.", "End."])
        self.assertEqual(split_corpus(["abcde", "fghij"], chunk_size=5), ["abcde", "fghij"])

    def test_invalid_chunk_size(self):
        for size in (0, -1):
            with self.subTest(chunk_size=size):
                with self.assertRaises(ValueError):
                    split_corpus(["Example."], chunk_size=size)


if __name__ == "__main__":
    unittest.main()
