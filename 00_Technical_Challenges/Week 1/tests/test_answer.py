import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from retrieval_app import answer as answer_module
from retrieval_app.answer import (
    CHAT_MODEL,
    NO_CONTEXT_ANSWER,
    answer,
    build_messages,
)
from retrieval_app.index import Hit


class BuildMessagesTests(unittest.TestCase):
    def test_includes_chunk_labels_and_question(self):
        hits = [
            Hit(chunk_id=4, text="First source.", score=0.9),
            Hit(chunk_id=9, text="Second source.", score=0.7),
        ]

        messages = build_messages("What is supported?", hits)

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("[chunk 4]\nFirst source.", messages[1]["content"])
        self.assertIn("[chunk 9]\nSecond source.", messages[1]["content"])
        self.assertIn("What is supported?", messages[1]["content"])
        self.assertIn(NO_CONTEXT_ANSWER, messages[0]["content"])
        self.assertIn("only the provided context", messages[0]["content"])

    def test_empty_hits_are_explicit(self):
        messages = build_messages("An unanswered question", [])

        self.assertIn("No context chunks were retrieved.", messages[1]["content"])
        self.assertIn(NO_CONTEXT_ANSWER, messages[0]["content"])


class AnswerTests(unittest.TestCase):
    def test_calls_chat_model_and_returns_trimmed_content(self):
        create = Mock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="  Grounded answer.  ")
                    )
                ]
            )
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        hits = [Hit(chunk_id=2, text="Source text.", score=0.8)]

        with patch.object(answer_module, "client", return_value=fake_client):
            result = answer("Question?", hits)

        self.assertEqual(result, "Grounded answer.")
        create.assert_called_once_with(
            model=CHAT_MODEL,
            messages=build_messages("Question?", hits),
            temperature=0,
        )

    def test_empty_choices_or_answer_raise_clear_errors(self):
        for choices, expected_error in [
            ([], "no choices"),
            ([SimpleNamespace(message=SimpleNamespace(content=None))], "empty answer"),
            ([SimpleNamespace(message=SimpleNamespace(content="  "))], "empty answer"),
        ]:
            with self.subTest(expected_error=expected_error):
                create = Mock(return_value=SimpleNamespace(choices=choices))
                fake_client = SimpleNamespace(
                    chat=SimpleNamespace(completions=SimpleNamespace(create=create))
                )
                with patch.object(answer_module, "client", return_value=fake_client):
                    with self.assertRaisesRegex(ValueError, expected_error):
                        answer("Question?", [])


if __name__ == "__main__":
    unittest.main()
