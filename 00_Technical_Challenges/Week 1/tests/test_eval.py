import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from eval import run_eval
from retrieval_app.index import Hit


class LoadCasesTests(unittest.TestCase):
    def test_reads_cases_and_rejects_malformed_input(self):
        valid = {"id": 1, "prompt": "Question?", "expected_output": "Reference."}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "evals.json"
            path.write_text(json.dumps({"questions": [valid]}), encoding="utf-8")
            self.assertEqual(
                run_eval.load_cases(path),
                [run_eval.EvaluationCase(1, "Question?", "Reference.")],
            )
            for questions in [[], [valid, valid], [{**valid, "expected_output": ""}]]:
                with self.subTest(questions=questions):
                    path.write_text(json.dumps({"questions": questions}), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        run_eval.load_cases(path)


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.case = run_eval.EvaluationCase(1, "Question?", "Reference.")
        self.create = Mock()
        client_patch = patch.object(
            run_eval,
            "client",
            return_value=SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
            ),
        )
        client_patch.start()
        self.addCleanup(client_patch.stop)

    def set_response(self, content):
        self.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def test_judges_actual_answer_against_reference(self):
        self.set_response(json.dumps({
            "key_points": [
                {"point": "Be concise.", "verdict": "covered", "evidence": "Keep it SHORT,"},
                {"point": "Give background.", "verdict": "partial", "evidence": "some context"},
                {"point": "Tie to goals.", "verdict": "missing", "evidence": ""},
            ],
            "contradictions": [],
        }))
        score, reason = run_eval.judge_answer(
            self.case, "Keep it short, and add some context [chunk 1].", "judge-model"
        )
        self.assertEqual(score, 50.0)
        self.assertIn("missing: Tie to goals.", reason)
        kwargs = self.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "judge-model")
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        payload = json.loads(kwargs["messages"][1]["content"])
        self.assertEqual(payload["prompt"], "Question?")
        self.assertEqual(payload["expected_output"], "Reference.")
        self.assertEqual(payload["actual_output"], "Keep it short, and add some context [chunk 1].")

    def test_claims_without_verbatim_evidence_earn_no_credit(self):
        judgment = {
            "key_points": [
                {"point": "List costs.", "verdict": "covered", "evidence": "list the costs"},
                {"point": "Involve others.", "verdict": "covered",
                 "evidence": "involve others to strengthen the analysis"},
            ],
            "contradictions": [
                {"issue": "Says to skip costs.", "evidence": "skip the costs"},
            ],
        }
        score, reason = run_eval.score_judgment(judgment, "First, list the costs.")
        self.assertEqual(score, 50.0)
        self.assertIn("quote is not in the answer", reason)
        self.assertNotIn("contradiction", reason)

    def test_verified_contradictions_reduce_score(self):
        judgment = {
            "key_points": [{"point": "Act fast.", "verdict": "covered", "evidence": "act fast"}],
            "contradictions": [{"issue": "Says to wait.", "evidence": "wait forever"}],
        }
        self.assertEqual(
            run_eval.score_judgment(judgment, "Act fast, or wait forever.")[0], 75.0
        )

    def test_invalid_judgments_are_not_converted_to_scores(self):
        for content in [
            "not JSON", "[]", '{"key_points": []}',
            '{"key_points": [{"point": "A", "verdict": "maybe", "evidence": ""}]}',
            '{"key_points": [{"point": "", "verdict": "missing", "evidence": ""}]}',
            '{"key_points": ["A"]}',
            '{"key_points": [{"point": "A", "verdict": "missing"}], "contradictions": {}}',
            None,
        ]:
            with self.subTest(content=content):
                self.set_response(content)
                with self.assertRaises(ValueError):
                    run_eval.judge_answer(self.case, "Actual.", "judge")


class EvaluationTests(unittest.TestCase):
    def test_indexes_once_and_passes_answers_to_judge(self):
        cases = [
            run_eval.EvaluationCase(1, "First?", "First reference."),
            run_eval.EvaluationCase(2, "Second?", "Second reference."),
        ]
        hit = Hit(4, "Source.", 0.9)
        matrix = np.ones((1, 2))
        with (
            patch.object(run_eval, "load_corpus", return_value=["Source."]),
            patch.object(run_eval, "build_index", return_value=matrix) as build,
            patch.object(run_eval, "search", return_value=[hit]) as search,
            patch.object(run_eval, "answer", side_effect=["First actual.", "Second actual."]),
            patch.object(run_eval, "judge_answer", side_effect=[(100.0, "Complete."), (50.0, "Partial.")]) as judge,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            results = run_eval.evaluate(cases, Path("book.txt"), "judge", k=2)

        build.assert_called_once_with(["Source."])
        self.assertEqual(search.call_count, 2)
        self.assertEqual(search.call_args.kwargs["k"], 2)
        judge.assert_any_call(cases[0], "First actual.", "judge")
        judge.assert_any_call(cases[1], "Second actual.", "judge")
        self.assertEqual([result.score for result in results], [100.0, 50.0])
        self.assertEqual(results[0].retrieved_chunk_ids, [4])

    def test_cli_writes_answers_scores_and_mean(self):
        cases = [
            {"id": 1, "prompt": "First?", "expected_output": "First reference."},
            {"id": 2, "prompt": "Second?", "expected_output": "Second reference."},
        ]
        results = [
            run_eval.EvaluationResult(1, "First?", "First reference.", "Actual.", 100.0, "Complete.", [1]),
            run_eval.EvaluationResult(2, "Second?", "Second reference.", "Actual.", 50.0, "Partial.", [2]),
        ]
        with TemporaryDirectory() as directory:
            evals_path = Path(directory) / "evals.json"
            output_path = Path(directory) / "report.json"
            evals_path.write_text(json.dumps({"questions": cases}), encoding="utf-8")
            with (
                patch.object(run_eval, "evaluate", return_value=results),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                code = run_eval.main([
                    "--evals", str(evals_path), "--corpus", "book.txt",
                    "--output", str(output_path),
                ])
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(report["mean_score"], 75.0)
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["results"][0]["actual_output"], "Actual.")
        self.assertEqual(report["results"][1]["expected_output"], "Second reference.")


if __name__ == "__main__":
    unittest.main()
