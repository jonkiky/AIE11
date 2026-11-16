import unittest

from retrieval_app.corpus import split_corpus
from retrieval_app.tokens import count_tokens


class TokenChunkTests(unittest.TestCase):
    def test_short_lines_stay_separate(self):
        self.assertEqual(split_corpus([" First. ", "", "Second."]), ["First.", "Second."])

    def test_sentence_boundary_is_preferred_over_filling_the_chunk(self):
        first = "The first sentence explains the main point."
        second = "The second sentence introduces a different idea."
        limit = max(count_tokens(first), count_tokens(second))
        self.assertEqual(split_corpus([f"{first} {second}"], limit), [first, second])

    def test_long_sentence_and_unicode_are_split_without_losing_text(self):
        for text in [" ".join(["unbroken"] * 2300), "世界你好" * 1500]:
            with self.subTest(text=text[:15]):
                chunks = split_corpus([text])
                self.assertGreater(len(chunks), 1)
                self.assertTrue(all(count_tokens(chunk) <= 1000 for chunk in chunks))
                self.assertEqual("".join("".join(chunks).split()), "".join(text.split()))

    def test_exact_token_limit_and_invalid_limit(self):
        text = "word " * 999 + "word"
        limit = count_tokens(text)
        self.assertEqual(split_corpus([text], limit), [text])
        with self.assertRaises(ValueError):
            split_corpus(["text"], 0)


if __name__ == "__main__":
    unittest.main()
