from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import signal
from med_prism.rgpa import early_eval as e


class EarlyEvalTests(TestCase):
    def test_only_completed_stage_cells(self):
        self.assertEqual(e.CELLS, [(1, 1), (2, 1), (2, 2)])

    def test_only_coordinator_signalled_and_resumed_on_error(self):
        cmd = b'python\0-m\0med_prism.rgpa.separated\0--root\0/run\0--output\0/run/evaluation_lite_x\0--mode\0all\0'
        with patch.object(Path, 'read_bytes', return_value=cmd), \
             patch.object(Path, 'read_text', return_value='State:\tT (stopped)\n'), \
             patch.object(Path, 'exists', return_value=False), \
             patch.object(e.os, 'kill') as kill:
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                with e.hold_scheduler(123, Path('/run'), Path('/run/evaluation_lite_x')):
                    raise RuntimeError('test failure')
            self.assertEqual([c.args for c in kill.call_args_list],
                             [(123, signal.SIGSTOP), (123, signal.SIGCONT)])

    def test_unrelated_process_never_signalled(self):
        with patch.object(Path, 'read_bytes', return_value=b'other\0'), patch.object(e.os, 'kill') as kill:
            with self.assertRaises(ValueError):
                with e.hold_scheduler(123, Path('/run'), Path('/run/evaluation_lite_x')):
                    self.fail('must not enter')
            kill.assert_not_called()

    def test_existing_main_evaluation_refused_and_resumed(self):
        cmd = b'med_prism.rgpa.separated\0/run\0/run/evaluation_lite_x\0all\0'
        with patch.object(Path, 'read_bytes', return_value=cmd), \
             patch.object(Path, 'read_text', return_value='State:\tT (stopped)\n'), \
             patch.object(Path, 'exists', return_value=True), patch.object(e.os, 'kill') as kill:
            with self.assertRaisesRegex(RuntimeError, 'already be active'):
                with e.hold_scheduler(123, Path('/run'), Path('/run/evaluation_lite_x')):
                    self.fail('must not enter')
            self.assertEqual(kill.call_args.args, (123, signal.SIGCONT))
