import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from retrieval_app import index
from retrieval_app.index import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    Hit,
    build_index,
    cosine_similarity,
    embed_texts,
    search,
)


class EmbeddingTests(unittest.TestCase):
    def test_embed_texts_batches_inputs_and_preserves_response_order(self):
        vectors = [np.ones(EMBEDDING_DIMENSIONS), np.zeros(EMBEDDING_DIMENSIONS)]
        embeddings = Mock()
        embeddings.create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=vectors[1]),
                SimpleNamespace(index=0, embedding=vectors[0]),
            ],
            usage=SimpleNamespace(total_tokens=5),
        )
        fake_client = SimpleNamespace(embeddings=embeddings)

        with patch.object(index, "client", return_value=fake_client):
            result = embed_texts(["first", "second"])

        embeddings.create.assert_called_once_with(
            input=["first", "second"], model=EMBEDDING_MODEL
        )
        self.assertEqual(result.shape, (2, EMBEDDING_DIMENSIONS))
        np.testing.assert_array_equal(result[0], vectors[0])
        np.testing.assert_array_equal(result[1], vectors[1])

    def test_build_index_embeds_all_chunks_in_one_call(self):
        chunks = ["first", "second"]
        expected = np.ones((2, EMBEDDING_DIMENSIONS), dtype=np.float32)
        with patch.object(index, "embed_texts", return_value=expected) as embed:
            result = build_index(chunks)

        embed.assert_called_once_with(chunks)
        np.testing.assert_array_equal(result, expected)

    def test_empty_embed_and_build_index_have_two_dimensional_shape(self):
        with patch.object(index, "client") as get_client:
            result = embed_texts([])
            matrix = build_index([])

        get_client.assert_not_called()
        self.assertEqual(result.shape, (0, EMBEDDING_DIMENSIONS))
        self.assertEqual(matrix.shape, (0, EMBEDDING_DIMENSIONS))

    def test_embed_texts_rejects_mismatched_response_count(self):
        fake_client = SimpleNamespace(
            embeddings=SimpleNamespace(
                create=Mock(return_value=SimpleNamespace(data=[]))
            )
        )
        with patch.object(index, "client", return_value=fake_client):
            with self.assertRaises(ValueError):
                embed_texts(["missing embedding"])


class SimilarityTests(unittest.TestCase):
    def test_cosine_similarity(self):
        self.assertAlmostEqual(
            cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 1.0])),
            1 / np.sqrt(2),
        )
        self.assertEqual(
            cosine_similarity(np.zeros(2), np.array([1.0, 0.0])), 0.0
        )


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.encode = Mock(return_value=SimpleNamespace(ids=[1]))
        tokenizer_patch = patch.object(
            index,
            "_query_tokenizer",
            return_value={"tokenizer": SimpleNamespace(encode=self.encode)},
        )
        tokenizer_patch.start()
        self.addCleanup(tokenizer_patch.stop)

    def test_query_at_token_limit_is_allowed(self):
        self.encode.return_value = SimpleNamespace(ids=[1] * 32_768)
        with patch.object(index, "embed_texts", return_value=np.array([[1.0, 0.0]])):
            hits = search("at the limit", ["chunk"], np.array([[1.0, 0.0]]))

        self.assertEqual(hits, [Hit(chunk_id=0, text="chunk", score=1.0)])
        self.encode.assert_called_once_with("at the limit", add_special_tokens=False)

    def test_query_above_limit_raises_before_embedding_or_early_returns(self):
        self.encode.return_value = SimpleNamespace(ids=[1] * 32_769)
        with patch.object(index, "embed_texts") as embed:
            for chunks, matrix, k in [
                (["chunk"], np.ones((1, 2)), 3),
                ([], np.empty((0, EMBEDDING_DIMENSIONS)), 3),
                (["chunk"], np.ones((1, 2)), 0),
            ]:
                with self.subTest(chunks=chunks, k=k):
                    with self.assertRaisesRegex(ValueError, "32,768.*32,769"):
                        search("too long", chunks, matrix, k=k)
        embed.assert_not_called()

    def test_returns_hits_by_descending_similarity(self):
        vectors = np.zeros((3, EMBEDDING_DIMENSIONS), dtype=np.float32)
        vectors[0, 0] = 1
        vectors[1, 1] = 1
        vectors[2, 0] = 2
        query_vector = np.zeros(EMBEDDING_DIMENSIONS)
        query_vector[0] = 1
        chunks = ["first", "unrelated", "same direction"]
        with patch.object(index, "embed_texts", return_value=query_vector[None, :]) as embed:
            hits = search("question", chunks, vectors, k=2)

        embed.assert_called_once_with(["question"])
        self.assertEqual(
            hits,
            [
                Hit(chunk_id=0, text="first", score=1.0),
                Hit(chunk_id=2, text="same direction", score=1.0),
            ],
        )

    def test_empty_corpus_and_nonpositive_k_return_no_hits(self):
        with patch.object(index, "embed_texts") as embed:
            self.assertEqual(search("question", [], np.empty((0, EMBEDDING_DIMENSIONS))), [])
            self.assertEqual(
                search("question", ["chunk"], np.ones((1, 2)), k=0), []
            )
        embed.assert_not_called()

    def test_matrix_rows_must_match_chunks(self):
        with self.assertRaises(ValueError):
            search("question", ["chunk"], np.empty((0, EMBEDDING_DIMENSIONS)))


if __name__ == "__main__":
    unittest.main()
