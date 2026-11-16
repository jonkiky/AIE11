import unittest
from unittest.mock import patch

import numpy as np

from retrieval_app import index


class TokenLimitTests(unittest.TestCase):
    def test_query_limit_remains_32768_tokens_before_retrieval(self):
        query = " ".join(["word"] * 32_768)
        with patch.object(index, "embed_texts") as embed:
            self.assertEqual(index.search(query, [], np.empty((0, 2))), [])
            with self.assertRaisesRegex(ValueError, "32,768-token limit"):
                index.search(query + " word", [], np.empty((0, 2)))
        embed.assert_not_called()

    def test_index_rejects_oversized_source_before_embedding(self):
        with patch.object(index, "embed_texts") as embed:
            with self.assertRaisesRegex(ValueError, "1000 tokens"):
                index.build_index([" ".join(["word"] * 1001)])
        embed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
