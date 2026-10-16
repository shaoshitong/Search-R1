import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import hashlib

from reproduction.prefix20_joint4 import retire_predecessor as retire


class PredecessorRetirementTests(unittest.TestCase):
    def test_published_plan_cannot_delete_another_task_or_shared_file(self):
        plan=json.loads(Path(__file__).with_name('retire_old10_100_plan.json').read_text())
        retire.validate_plan(plan)
        for key,value in [('path','/data/user/jxu957/experiments/topd-20261008/model.safetensors'),
                          ('nlink',2),('uid',21930),('symlink',True)]:
            changed=copy.deepcopy(plan);changed['files'][0][key]=value
            with self.assertRaises(ValueError):retire.validate_plan(changed)
        changed=copy.deepcopy(plan);changed['files'][-1]=changed['files'][0]
        with self.assertRaises(ValueError):retire.validate_plan(changed)

    def test_retirement_preserves_metadata_and_rejects_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();old=root/'old';base=root/'new'
            (base/'preparation').mkdir(parents=True);old.mkdir();(old/'gpu-job.lock').touch()
            rows=[];metadata=[]
            for step in (10,100):
                tree=old/'checkpoints/search-g1-common8/actor'/('global_step_'+str(step))
                for name in retire.NAMES:
                    path=tree/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'unused-old-state')
                    s=path.stat();rows.append(dict(path=str(path),bytes=s.st_size,allocated=s.st_blocks*512,
                        inode=s.st_ino,nlink=1,mtime_ns=s.st_mtime_ns,uid=s.st_uid,gid=s.st_gid,symlink=False))
                path=tree/'complete.json';path.write_text('{"old_evaluation":"retained"}');metadata.append(path)
            plan=dict(kind='authorized_previous_qprobe_checkpoint_retirement',source_base=str(old),
                run_id=retire.RUN,files=rows,logical_bytes=sum(r['bytes'] for r in rows),
                allocated_file_bytes=sum(r['allocated'] for r in rows),authorization='fixture authorization',
                metadata_logs_evaluations_retained=True)
            path=base/'preparation/plan.json';path.write_text(json.dumps(plan));digest=hashlib.sha256(path.read_bytes()).hexdigest()
            original_validate=retire.validate_plan
            def validate(value):
                with patch.object(retire,'RUN',retire.RUN):
                    # Ownership is separately authenticated on the real host.
                    changed=copy.deepcopy(value)
                    for row in changed['files']:row['uid']=14090
                    original_validate(changed)
            with patch.object(retire,'OLD',old),patch.object(retire,'own_base',lambda value:Path(value)),\
                 patch.object(retire.os,'getuid',return_value=14090),\
                 patch.object(retire,'inactive',return_value={'fixture_terminal':True}),\
                 patch.object(retire,'validate_plan',side_effect=validate):
                result=retire.retire(base,path,digest)
                self.assertTrue(result['all_deleted_paths_absent'])
                self.assertTrue(all(p.read_text()=='{"old_evaluation":"retained"}' for p in metadata))
                self.assertEqual(len((base/'preparation/retire-old10-100-deleted.jsonl').read_text().splitlines()),20)
                with self.assertRaisesRegex(ValueError,'ALREADY_ATTEMPTED'):retire.retire(base,path,digest)


if __name__=='__main__':unittest.main()
