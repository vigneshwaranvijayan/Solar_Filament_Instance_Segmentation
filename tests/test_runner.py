import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_pipeline


class RunnerTests(unittest.TestCase):
    def test_dry_run_creates_no_output_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            arguments = ['run_pipeline.py', '--input', str(base / 'official inputs'), '--data-root', str(base / 'prepared'), '--work-root', str(base / 'work'), '--pretrained-root', str(base / 'weights'), '--out', str(base / 'outputs/submission.csv'), '--dry-run']
            with patch.object(sys, 'argv', arguments), contextlib.redirect_stdout(io.StringIO()) as output:
                run_pipeline.main()
            self.assertIn('test_predict:', output.getvalue())
            self.assertIn('prepare_fold_0:', output.getvalue())
            self.assertEqual(list(base.iterdir()), [])

    def test_selected_stages_use_active_python_and_no_test_annotations(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            arguments = ['run_pipeline.py', '--stages', 'evaluate', 'predict', '--data-root', str(base / 'prepared data'), '--work-root', str(base / 'work'), '--out', str(base / 'output/submission.csv')]
            with patch.object(sys, 'argv', arguments), patch.object(run_pipeline.subprocess, 'run', fake_run), contextlib.redirect_stdout(io.StringIO()):
                run_pipeline.main()
            stage_calls = calls[1:]  # The first subprocess records pip freeze.
            self.assertEqual([Path(command[1]).name for command in stage_calls], ['evaluate_experiment.py', 'predict_test.py', 'validate_submission.py'])
            self.assertTrue(all(command[0] == sys.executable for command in stage_calls))
            self.assertIn('--annotations', stage_calls[0])
            self.assertNotIn('--annotations', stage_calls[1])
            journal = next((base / 'work').glob('stage_history_*.jsonl'))
            self.assertTrue(all(json.loads(line)['status'] == 'completed' for line in journal.read_text().splitlines()))

    def test_failed_stage_is_recorded_and_stops_pipeline(self):
        def fake_run(command, **kwargs):
            if Path(command[1]).name == 'evaluate_experiment.py':
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            arguments = ['run_pipeline.py', '--stages', 'evaluate', 'predict', '--work-root', str(base / 'work'), '--out', str(base / 'output/submission.csv')]
            with patch.object(sys, 'argv', arguments), patch.object(run_pipeline.subprocess, 'run', fake_run), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError):
                    run_pipeline.main()
            journal = next((base / 'work').glob('stage_history_*.jsonl'))
            rows = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['status'], 'interrupted_or_failed')
            self.assertFalse((base / 'output').exists())


if __name__ == '__main__':
    unittest.main()
