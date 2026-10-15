import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe import repair_actual_storage_denials as r


class ActualDenialScope(unittest.TestCase):
    def test_only_reported_periodic_temporary_root_is_accepted(self):
        path=r.ROOT.parent/'qprobe-teacher-warmup200-periodic25-thread5-20261016-r2/tmp/torchelastic_wfz3wjm8'
        self.assertEqual(r.reported_roots("du: cannot read directory '"+str(path)+"': Permission denied\n"),[path])

    def test_model_cache_outside_scope_refuses(self):
        for path in (r.ROOT.parent/'topd-20261008/model',r.ROOT.parent/'qprobe-teacher-warmup200-periodic25-thread5-20261016-r2/checkpoints',Path('/etc/private')):
            with self.assertRaises(ValueError):r.reported_roots("du: cannot read directory '"+str(path)+"': Permission denied\n")

    def test_no_permission_error_and_mixed_unknown_error_refuse(self):
        with self.assertRaises(ValueError):r.reported_roots("du: cannot access '"+str(r.ROOT.parent)+"/x': No such file or directory\n")
        path=r.ROOT.parent/'qprobe-teacher-warmup200-periodic25-thread5-20261016-r2/tmp/torchelastic_x'
        with self.assertRaises(ValueError):r.reported_roots("du: cannot read directory '"+str(path)+"': Permission denied\nI/O error\n")

    def test_live_measurement_refuses_before_directory_inventory(self):
        error="du: cannot read directory '"+str(r.ROOT.parent)+"/qprobe-teacher-warmup200-periodic25-thread5-20261016-r2/tmp/torchelastic_x': Permission denied\n"
        own=dict(pid=1,sid=1,uid=14090,start_ticks=2);child=dict(own,pid=3)
        value=dict(status='ACTUAL_DU_FAILED_NO_USAGE_ACCEPTED',actual_du_exit_code=1,
            actual_du_child_still_live=False,uid=14090,measure=str(r.ROOT.parent),stderr=error,
            stderr_truncated=False,stderr_bytes=len(error.encode()),stderr_sha256=hashlib.sha256(error.encode()).hexdigest(),
            process=own,du_process=child)
        path=r.ROOT/'observations/storage-sigma3-after-r2-permission-repaired-20261016/result.json'
        with patch.object(r,'document',return_value=(value,'s')),patch.object(r,'birth',return_value=own):
            with self.assertRaisesRegex(ValueError,'actually be terminal'):r.failure(path,'s')


if __name__=='__main__':unittest.main()
