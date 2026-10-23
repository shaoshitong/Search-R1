"""CPU-only guard regressions; no Slurm invocation or task restart."""
import json
from pathlib import Path
import tempfile
import unittest

from reproduction.dynrpo7b.retry_before_actor import ensure_no_actor_writes


class PreActorRetryTests(unittest.TestCase):
    def test_absent_or_zero_write_journals_allow_pre_actor_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            ensure_no_actor_writes(base)
            (base/'logs').mkdir()
            for rank in range(4):
                rows=[dict(kind='POLICY_MONITOR'),
                      dict(kind='OPPORTUNITY',optimizer_write_succeeded=False,
                           optimizer_writes_before=0,optimizer_writes_after=0),
                      dict(kind='OUTER_BUDGET',optimizer_writes=0,cumulative_writes=0)]
                (base/'logs'/('optimizer-rank-%d.jsonl'%rank)).write_text(
                    '\n'.join(json.dumps(row) for row in rows)+'\n')
            ensure_no_actor_writes(base)

    def test_any_actual_rank_write_blocks_fresh_retry_before_checkpoint(self):
        evidence=[dict(kind='OPTIMIZER_WRITE'),
                  dict(kind='OPPORTUNITY',optimizer_write_succeeded=True),
                  dict(kind='OUTER_BUDGET',optimizer_writes=1),
                  dict(kind='OUTER_BUDGET',optimizer_writes=0,cumulative_writes=1)]
        for rank in range(4):
            for row in evidence:
                with self.subTest(rank=rank,kind=row['kind']),tempfile.TemporaryDirectory() as directory:
                    base=Path(directory)
                    (base/'logs').mkdir()
                    (base/'logs'/('optimizer-rank-%d.jsonl'%rank)).write_text(json.dumps(row)+'\n')
                    self.assertFalse((base/'checkpoints').exists())
                    with self.assertRaisesRegex(RuntimeError,'actual actor Adam write'):
                        ensure_no_actor_writes(base)

    def test_damaged_or_uninterpretable_journal_blocks_fresh_retry(self):
        for content in ('{"kind":"OPPORTUNITY"','[]\n',
                        '{"kind":"OPPORTUNITY","optimizer_write_succeeded":null}\n'):
            with self.subTest(content=content),tempfile.TemporaryDirectory() as directory:
                base=Path(directory)
                (base/'logs').mkdir()
                (base/'logs/optimizer-rank-2.jsonl').write_text(content)
                with self.assertRaisesRegex(RuntimeError,'damaged journal'):
                    ensure_no_actor_writes(base)


if __name__=='__main__':
    unittest.main()
