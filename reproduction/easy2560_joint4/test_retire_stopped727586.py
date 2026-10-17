import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.easy2560_joint4 import retire_stopped727586 as op


class ExactRetirementTests(unittest.TestCase):
    def test_scope_is_exactly39_tensor_names_without_heads_or_shared_assets(self):
        self.assertEqual(len(op.NAMES), 13)
        self.assertEqual(len(set(op.NAMES)), 13)
        self.assertEqual(op.STEPS, (5, 50, 100))
        self.assertTrue(all(x.endswith(('.pt', '.safetensors')) for x in op.NAMES))
        self.assertFalse(any('commit_probe' in x or 'boundary_probe' in x for x in op.NAMES))

    def test_durable_claim_never_overwrites(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'claim.json'; op.write(path, dict(first=True))
            with self.assertRaises(FileExistsError): op.write(path, dict(first=False))
            self.assertEqual(json.loads(path.read_text()), dict(first=True))

    def test_source_identity_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'real'; p.write_bytes(b'checkpoint')
            q = Path(td) / 'link'; q.symlink_to(p)
            with self.assertRaisesRegex(ValueError, 'SYMLINK'): op.identity(q)

    def test_insufficient_proofs_cannot_reach_unlink_or_terminal(self):
        with patch.object(op.os, 'getuid', return_value=14090), patch.object(
                op.socket, 'gethostname', return_value='ACD-Login-1'), patch.object(op, 'terminal') as terminal:
            with self.assertRaises(ValueError):
                op.execute(dict(files=[]), dict(status='not-verified'), {}, {}, Path('/unused'))
            terminal.assert_not_called()


if __name__ == '__main__': unittest.main()
