import copy
import json
import os
from pathlib import Path
import resource
import tempfile
import unittest
from unittest.mock import patch

from reproduction.prefix20_joint4 import native_ray_resources as core


def manifest():
    return dict(profile='common8',execution_site='hpc3',execution_backend='slurm',training_world_size=4,
        data_selection_mode='prefix20_probe',run_id='r'*32,source_commit='a'*40)


class Limits:
    def __init__(self,soft=2048,hard=100000):self.value=(soft,hard);self.writes=[]
    def get(self,kind):
        if kind!=resource.RLIMIT_NPROC:raise AssertionError('wrong limit')
        return self.value
    def set(self,kind,value):
        if kind!=resource.RLIMIT_NPROC:raise AssertionError('wrong limit')
        self.writes.append(tuple(value));self.value=tuple(value)


class NativeRayResourceTests(unittest.TestCase):
    def test_self_soft_raised_and_kernel_read_back_without_hard_change(self):
        limit=Limits();value=core.raise_self_soft(getlimit=limit.get,setlimit=limit.set)
        self.assertEqual(limit.writes,[(8192,100000)])
        self.assertEqual(value['before'],[2048,100000]);self.assertEqual(value['after'],[8192,100000])
        self.assertTrue(value['soft_raised']);self.assertTrue(value['hard_unchanged'])

    def test_higher_and_infinite_soft_never_lowered(self):
        for soft,hard in ((8192,100000),(16384,100000),(resource.RLIM_INFINITY,resource.RLIM_INFINITY)):
            with self.subTest(soft=soft):
                limit=Limits(soft,hard);value=core.raise_self_soft(getlimit=limit.get,setlimit=limit.set)
                self.assertFalse(value['soft_raised']);self.assertEqual(limit.writes,[])
                self.assertEqual(value['after'],[soft,hard])

    def test_insufficient_hard_fails_before_any_set(self):
        limit=Limits(2048,4096)
        with self.assertRaisesRegex(ValueError,'HARD_LIMIT_BELOW8192'):
            core.raise_self_soft(getlimit=limit.get,setlimit=limit.set)
        self.assertEqual(limit.writes,[])

    def test_unexpected_actual_kernel_readback_fails(self):
        limit=Limits()
        with self.assertRaisesRegex(RuntimeError,'READBACK_DIFFERS'):
            core.raise_self_soft(getlimit=limit.get,setlimit=lambda kind,value:None)

    def test_only_native4_prefix20_gets_public_ray_options(self):
        run=manifest();before=copy.deepcopy(run)
        self.assertEqual(core.options(run),dict(num_cpus=16,object_store_memory=8*1024**3))
        self.assertEqual(run,before)
        for key,value in [('training_world_size',8),('data_selection_mode','question_probe'),('execution_site','h1008'),
                          ('execution_backend','ssh_local'),('profile','other')]:
            changed=dict(run);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):core.options(changed)

    def fixture(self,temporary,run=None):
        base=Path(temporary).resolve();(base/'logs').mkdir();(base/'preparation').mkdir()
        run=manifest() if run is None else run
        (base/'logs/run-manifest.json').write_text(json.dumps(run))
        (base/'logs/prefix20-configuration-prepared.json').write_text(json.dumps(dict(
            run_id=run['run_id'],source_commit=run['source_commit'],actual_slurm_job_binding={'JobId':'123'})))
        (base/'preparation/restore-submission.json').write_text(json.dumps(dict(job='123',outcome='SUBMITTED',returncode=0)))
        env=dict(SEARCH_G1_BASE=str(base),SEARCH_G1_TRAINING_WORLD_SIZE='4',SLURM_JOB_ID='123')
        return base,run,env

    def test_existing_actual_job_binding_uid_and_role_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            base,run,env=self.fixture(temporary)
            with patch('reproduction.prefix20_joint4.control.own_base',side_effect=Path):
                self.assertEqual(core.scope(base,run,'train',environ=env,uid=14090)[0],base)
                for key,value in [('SLURM_JOB_ID','999'),('SLURM_JOB_ID',''),('SEARCH_G1_TRAINING_WORLD_SIZE','8'),
                                  ('SEARCH_G1_BASE','/other')]:
                    changed=dict(env);changed[key]=value
                    with self.subTest(key=key),self.assertRaises(ValueError):core.scope(base,run,'train',environ=changed,uid=14090)
                with self.assertRaises(ValueError):core.scope(base,run,'train',environ=env,uid=21930)
                with self.assertRaises(ValueError):core.scope(base,run,'restore',environ=env,uid=14090)
                restore=dict(run,restore_only_role=core.RESTORE_ROLE)
                self.assertEqual(core.scope(base,restore,'restore',environ=env,uid=14090)[0],base)
                with self.assertRaises(ValueError):core.scope(base,restore,'train',environ=env,uid=14090)

    def test_entry_then_ray_logs_actual_inherited_limits_and_exact_public_kwargs(self):
        with tempfile.TemporaryDirectory() as temporary:
            base,run,env=self.fixture(temporary);limit=Limits()
            with patch('reproduction.prefix20_joint4.control.own_base',side_effect=Path),patch.dict(os.environ,env),patch.object(os,'getuid',return_value=14090):
                entry=core.configure(base,run,role='train',phase='entry',getlimit=limit.get,setlimit=limit.set)
                ray=core.configure(base,run,role='train',phase='ray',getlimit=limit.get,setlimit=limit.set)
            files=sorted((base/'logs').glob('native-ray-resources-*.json'));self.assertEqual(len(files),2)
            rows=[json.loads(path.read_text()) for path in files];by_phase={r['phase']:r for r in rows}
            self.assertEqual(by_phase['entry']['actual_before'],[2048,100000])
            self.assertEqual(by_phase['entry']['actual_after'],[8192,100000])
            self.assertEqual(by_phase['ray']['actual_before'],[8192,100000]);self.assertFalse(by_phase['ray']['soft_raised'])
            self.assertEqual(limit.writes,[(8192,100000)]);self.assertEqual(entry,ray)
            self.assertEqual(ray,dict(num_cpus=16,object_store_memory=8*1024**3))
            self.assertTrue(all(r['accepted'] and not r['scientific_configuration_changed'] for r in rows))

    def test_hard_failure_records_real_before_after_and_no_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            base,run,env=self.fixture(temporary);limit=Limits(2048,4096)
            with patch('reproduction.prefix20_joint4.control.own_base',side_effect=Path),patch.dict(os.environ,env),patch.object(os,'getuid',return_value=14090):
                with self.assertRaisesRegex(ValueError,'HARD_LIMIT_BELOW8192'):
                    core.configure(base,run,role='train',phase='entry',getlimit=limit.get,setlimit=limit.set)
            row=json.loads(next((base/'logs').glob('native-ray-resources-*.json')).read_text())
            self.assertFalse(row['accepted']);self.assertEqual(row['actual_before'],[2048,4096]);self.assertEqual(row['actual_after'],[2048,4096])
            self.assertEqual(limit.writes,[])

    def test_cli_raises_then_execs_same_original_pipeline_without_wrapper_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            base,run,env=self.fixture(temporary);events=[]
            def configured(*args,**kwargs):events.append(('limits',kwargs['role'],kwargs['phase']));return {}
            with patch.object(core,'configure',side_effect=configured),patch.object(core.os,'execv',side_effect=lambda *args:events.append(('exec',args))),\
                    patch('sys.argv',['native_ray_resources','--base',str(base),'--role','train']):
                core.main()
            self.assertEqual(events[0],('limits','train','entry'))
            self.assertEqual(events[1][1][1],core.command(base,'train',core.sys.executable))
            self.assertEqual(core.command(base,'restore','python'),['python','-m','reproduction.prefix20_joint4.restore_only','execute','--base',str(base)])


if __name__=='__main__':unittest.main()
