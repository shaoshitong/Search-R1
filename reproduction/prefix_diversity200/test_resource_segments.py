"""Local scalar fixtures only: all remote paths, commands and clocks are mocked."""
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import fresh_observe_local as observer
from . import resource_control as control
from . import resource_sample as sampler

RUN = '4a1cc2d500fe40979ef8ca4792959438'
BASE = '/mnt/data/shaoshitong/search-g1-prefix-diversity200-20261017-4a1cc2d500fe-calibration1'
JOB = '3433'
UNIT = 'g1-resource5s-' + RUN[:12] + '-' + JOB


def summary(start, end, cpu, used, free, utilization):
    return dict(run_id=RUN, base=BASE, job=JOB, started_unix=start, last_observed_unix=end,
        samples=3, cpu_accounting_complete=False, latest=dict(owned_cpu_core_seconds_lower_bound=cpu),
        whole_card_peaks={'GPU-0':dict(max_used_mib=used,min_free_mib=free,max_utilization_percent=utilization)},
        gpu_scope='whole physical card including other users; sampled, not continuous',
        segment_source={'path':'summary-'+str(start),'sha256':'a'*64})


class MemoryFS:
    """Lexical /mnt fixtures never touch the real filesystem."""
    def __init__(self): self.files={}; self.dirs=set()
    def put(self,path,text):
        path=PurePosixPath(str(path)); self.files[str(path)]=text
        self.dirs.update(map(str,path.parents))
    def document(self,path,value): self.put(path,json.dumps(value))
    def path(self,value): return MemoryPath(self,value)


class MemoryPath:
    def __init__(self,fs,value): self.fs=fs; self.p=PurePosixPath(str(value))
    def __str__(self): return str(self.p)
    def __eq__(self,other): return str(self)==str(other)
    def __truediv__(self,other): return MemoryPath(self.fs,self.p/str(other))
    @property
    def parent(self): return MemoryPath(self.fs,self.p.parent)
    @property
    def name(self): return self.p.name
    def resolve(self): return self
    def exists(self): return str(self)in self.fs.files or str(self)in self.fs.dirs
    def mkdir(self,**kwargs): self.fs.dirs.add(str(self))
    def read_text(self,**kwargs): return self.fs.files[str(self)]
    def read_bytes(self): return self.read_text().encode()
    def write_text(self,value): self.fs.put(self,value)
    def chmod(self,mode): pass
    def stat(self): return SimpleNamespace(st_size=len(self.read_bytes()))
    @contextlib.contextmanager
    def open(self,mode):
        if mode=='x' and self.exists(): raise FileExistsError(str(self))
        stream=io.StringIO(); stream.fileno=lambda:123
        try: yield stream
        finally: self.fs.put(self,stream.getvalue()); stream.close()


def execute_control(*, segment='resource5s-v2', previous_patch=None,
                    previous_state='failed', previous_pid='0', fs=None, payload_patch=None):
    fs=fs or MemoryFS(); source='# scalar sampler fixture\n'
    payload=dict(phase='deploy',segment=segment,base=BASE,run_id=RUN,job=JOB,
        commit='a'*40,source=source,sha256=hashlib.sha256(source.encode()).hexdigest())
    if payload_patch: payload.update(payload_patch)
    fs.document(BASE+'/preparation/launch-config.json',dict(run_id=RUN))
    fs.document(BASE+'/preparation/submit-receipt.json',dict(job_id=JOB))
    if segment=='resource5s-v2' and BASE+'/preparation/resource5s-v1/intent.json'not in fs.files:
        old_source='# original v1 reader remains preserved\n'
        previous=dict(run_id=RUN,base=BASE,job=JOB,unit=UNIT,
            reader_sha256=hashlib.sha256(old_source.encode()).hexdigest())
        if previous_patch: previous.update(previous_patch)
        fs.document(BASE+'/preparation/resource5s-v1/intent.json',previous)
        fs.put(BASE+'/preparation/resource5s-v1/reader.py',old_source)
    commands=[]
    def run(args,**kwargs):
        commands.append(args)
        if args[0]=='scontrol': text='JobId='+JOB+' JobState=RUNNING WorkDir='+BASE
        elif args[0]=='systemd-run': text=''
        elif args[0]=='systemctl' and args[-1]=='--property=ActiveState,MainPID':
            text='ActiveState='+previous_state+'\nMainPID='+previous_pid+'\n'
        else: text='LoadState=not-found\nActiveState=inactive\nMainPID=0\n'
        return SimpleNamespace(returncode=0,stdout=text,stderr='')
    tree=ast.parse(control.REMOTE.replace('PAYLOAD',repr(payload)))
    # Inject all side effects; separately compile the complete unmodified script below.
    tree.body=[node for node in tree.body if not isinstance(node,ast.Import)]
    namespace=dict(pathlib=SimpleNamespace(Path=fs.path),json=json,hashlib=hashlib,
        subprocess=SimpleNamespace(run=run,TimeoutExpired=TimeoutError),
        os=SimpleNamespace(fsync=lambda fd:None),time=SimpleNamespace(time=lambda:1000.))
    with contextlib.redirect_stdout(io.StringIO()):exec(compile(tree,'<mocked-resource-control>','exec'),namespace)
    return fs,commands


class ResourceSegmentsTests(unittest.TestCase):
    def test_merge_cpu_max_peaks_gap_and_original_inputs_preserved(self):
        old=summary(100.,200.,800.,12000,5000,80)
        new=summary(300.,400.,650.,10000,4000,90)
        original=copy.deepcopy([old,new]);result=sampler.combine_resource_summaries([new,old])
        self.assertEqual(result['latest']['owned_cpu_core_seconds_lower_bound'],800.)
        self.assertEqual(result['samples'],6);self.assertFalse(result['cpu_accounting_complete'])
        self.assertEqual(result['whole_card_peaks']['GPU-0'],dict(max_used_mib=12000,min_free_mib=4000,max_utilization_percent=90))
        self.assertEqual(result['resource_coverage_gaps'],[dict(after_unix=200.,before_unix=300.,peak_and_cpu_samples_available=False)])
        self.assertIn('gaps',result['coverage']);self.assertIn('not continuous',result['gpu_scope'])
        self.assertEqual([old,new],original)
        self.assertIsNone(sampler.combine_resource_summaries([]))

    def test_legacy_v1_summary_alone_retains_incomplete_scope(self):
        value=summary(100.,200.,800.,12000,5000,80)
        result=sampler.combine_resource_summaries([value])
        self.assertEqual(result['latest'],value['latest']);self.assertEqual(result['resource_coverage_gaps'],[])
        self.assertEqual(len(result['resource_segments']),1);self.assertFalse(result['cpu_accounting_complete'])

    def test_merge_rejects_identity_overlap_and_false_complete_cpu(self):
        for key in('run_id','base','job','overlap','cpu_accounting_complete','negative_cpu','nan_cpu'):
            with self.subTest(key=key):
                first=summary(100.,200.,800.,12000,5000,80);second=summary(300.,400.,650.,10000,4000,90)
                if key in('run_id','base','job'):second[key]='foreign'
                elif key=='overlap':second['started_unix']=150.
                elif key=='cpu_accounting_complete':second[key]=True
                else:second['latest']['owned_cpu_core_seconds_lower_bound']=-1. if key=='negative_cpu'else float('nan')
                with self.assertRaises(ValueError):sampler.combine_resource_summaries([first,second])

    def test_v2_memory_cpu_nice_namespace_and_no_resubmission(self):
        fs,commands=execute_control()
        submit=next(row for row in commands if row[0]=='systemd-run')
        for item in('--unit='+UNIT+'-v2','--property=MemoryMax=1G','--property=CPUQuota=10%',
                     '--property=Nice=19','--setenv=CUDA_VISIBLE_DEVICES='):
            self.assertIn(item,submit)
        self.assertEqual(submit[-1],BASE+'/preparation/resource5s-v2')
        before=copy.deepcopy(fs.files)
        _,commands=execute_control(fs=fs)
        self.assertFalse(any(row[0]=='systemd-run'for row in commands));self.assertEqual(fs.files,before)

    def test_v1_deployment_compatibility(self):
        _,commands=execute_control(segment='resource5s-v1')
        submit=next(row for row in commands if row[0]=='systemd-run')
        self.assertIn('--unit='+UNIT,submit);self.assertIn('--property=MemoryMax=128M',submit)
        self.assertEqual(submit[-1],BASE+'/preparation/resource5s-v1')

    def test_v2_rejects_live_or_unreaped_prior_sampler(self):
        for state,pid in(('active','123'),('active','0'),('failed','123')):
            with self.subTest(state=state,pid=pid),self.assertRaises(AssertionError):
                execute_control(previous_state=state,previous_pid=pid)

    def test_v2_rejects_predecessor_identity_mismatch(self):
        for key in('run_id','job','base','unit','reader_sha256'):
            with self.subTest(key=key),self.assertRaises(AssertionError):
                execute_control(previous_patch={key:'foreign'})

    def test_control_rejects_unowned_segment_base_or_run(self):
        for values in(dict(segment='../resource5s-v2'),dict(base='/tmp/foreign'),dict(run_id='foreign')):
            with self.subTest(values=values),self.assertRaises(AssertionError):
                execute_control(payload_patch=values)

    def test_sampler_v2_scans_cpu_60s_and_gpu_5s_v1_remains_5s(self):
        for segment,expected_cpu_reads in(('resource5s-v2',2),('resource5s-v1',14)):
            with self.subTest(segment=segment):
                fs=MemoryFS();clock=SimpleNamespace(now=1000.)
                def sleep(seconds):clock.now+=seconds
                gpu=[];ledger=[];summaries=[]
                def read(path):
                    name=str(path)
                    if name.endswith('launch-config.json'):return dict(run_id=RUN,physical_gpu_uuids=['GPU-'+str(i)for i in range(8)])
                    if name.endswith('submit-receipt.json'):return dict(job_id=JOB)
                    if name.endswith('process-ledger.json'):
                        ledger.append(clock.now)
                        return dict(run_id=RUN,jobs={'owned':dict(state='RUNNING',ledger={'1':dict(pid=1,start_ticks=1,sid=1,uid=1)})})
                    return dict(outer=121)
                def run(args,**kwargs):
                    if args[0]=='scontrol':text='JobId='+JOB+' JobState=RUNNING WorkDir='+BASE
                    else:
                        gpu.append(clock.now)
                        text='\n'.join(f'{i}, GPU-{i}, 100, 1000, 900, 10'for i in range(8))
                    return SimpleNamespace(returncode=0,stdout=text,stderr='')
                fake_time=SimpleNamespace(time=lambda:clock.now,monotonic=lambda:clock.now,sleep=sleep)
                fake_os=SimpleNamespace(sysconf=lambda key:100,fsync=lambda fd:None)
                with patch.object(sampler,'Path',fs.path),patch.object(sampler,'read',side_effect=read),\
                     patch.object(sampler,'time',fake_time),patch.object(sampler,'os',fake_os),\
                     patch.object(sampler,'subprocess',SimpleNamespace(run=run)),\
                     patch.object(sampler,'own_cpu',return_value=100),\
                     patch.object(sampler,'atomic',side_effect=lambda path,value:summaries.append(copy.deepcopy(value))):
                    sampler.sample(BASE,JOB,BASE+'/preparation/'+segment,max_seconds=66.)
                self.assertEqual(len(ledger),expected_cpu_reads);self.assertEqual(len(gpu),14)
                self.assertTrue(all(b-a==5 for a,b in zip(gpu,gpu[1:])))
                self.assertEqual(summaries[-1]['cpu_scan_interval_seconds'],60. if segment.endswith('v2') else 5.)
                self.assertFalse(summaries[-1]['cpu_accounting_complete']);self.assertFalse(summaries[-1]['new_model_forward'])

    def test_sampler_rejects_noncanonical_output_without_commands(self):
        fs=MemoryFS()
        with patch.object(sampler,'Path',fs.path),patch.object(sampler.subprocess,'run')as run:
            for output in(BASE+'/preparation/resource5s-v3',BASE+'/other/resource5s-v2'):
                with self.assertRaises(ValueError):sampler.sample(BASE,JOB,output)
            run.assert_not_called()

    def test_remote_and_observer_embedded_function_parse_without_globals(self):
        compile(control.REMOTE.replace('PAYLOAD',repr(dict(base=BASE))),'remote','exec')
        class Captured(Exception):pass
        captured=[]
        def capture(code,*args,**kwargs):captured.append(code);raise Captured
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);out=root/'out';out.mkdir()
            (out/'CURRENT_CONFIG.json').write_text(json.dumps(dict(base=BASE,node='h100-10')))
            connection=root/'connection.json';connection.write_text('{}')
            with patch.object(observer,'remote',side_effect=capture),self.assertRaises(Captured):
                observer.observe(root,out,connection)
        tree=ast.parse(captured[0]);compile(tree,'observer-remote','exec')
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name=='combine_resource_summaries')
        namespace={};exec(compile(ast.Module(body=[function],type_ignores=[]),'embedded','exec'),namespace)
        value=summary(100.,200.,800.,12000,5000,80)
        self.assertEqual(namespace['combine_resource_summaries']([value])['latest'],value['latest'])


if __name__=='__main__':unittest.main()
