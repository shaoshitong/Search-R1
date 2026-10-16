import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import json

from reproduction.prefix20_joint4 import recover727050 as recovery


class RecoveryTests(unittest.TestCase):
    def test_only_both_actual_stopped_pretraining_roles_are_accepted(self):
        value=dict(run_id=recovery.RUN,jobs={name:dict(state='STOPPED',exit_code=code,
            leader=dict(uid=14090)) for name,code in [('gpu-preflight',0),('model-initial',1)]})
        recovery.validate_ledger(value)
        for case in ('live','wrong_exit','foreign','training'):
            bad=copy.deepcopy(value)
            if case=='live':bad['jobs']['model-initial']['state']='RUNNING'
            elif case=='wrong_exit':bad['jobs']['model-initial']['exit_code']=0
            elif case=='foreign':bad['jobs']['gpu-preflight']['leader']['uid']=1
            else:bad['jobs']['actor']=copy.deepcopy(bad['jobs']['gpu-preflight'])
            with self.assertRaises(ValueError):recovery.validate_ledger(bad)

    def test_extra_2gib_is_cumulative_and_scoped_walk_never_follows_links(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)
            (source/'source-publication.json').write_text(json.dumps(dict(files={'a':dict(bytes=10)})))
            paths=['/data/user/jxu957/experiments/'+x for x in (
                'qprobe-teacher-warmup200-prefix20-joint4-20261016',
                'qprobe-teacher-warmup200-prefix20-joint4-recovery726927-20261016',
                'qprobe-teacher-warmup200-prefix20-joint4-recovery727000-20261016')]
            def observed(size):return SimpleNamespace(returncode=0,stderr='',stdout=''.join(str(size)+'\t'+p+'\n' for p in paths))
            with patch.object(recovery.subprocess,'run',return_value=observed(400*2**20)) as run:
                proof=recovery.prelaunch_footprint(source)
                self.assertEqual(run.call_args.args[0],['du','-s','-B1',*paths])
                self.assertLess(proof['projected_cumulative_prelaunch_bytes'],2*2**30)
                self.assertFalse(proof['full_root_measurement_repeated'])
            with patch.object(recovery.subprocess,'run',return_value=observed(700*2**20)):
                with self.assertRaisesRegex(ValueError,'CUMULATIVE_EXTRA_RESERVE'):
                    recovery.prelaunch_footprint(source)


if __name__=='__main__':unittest.main()
