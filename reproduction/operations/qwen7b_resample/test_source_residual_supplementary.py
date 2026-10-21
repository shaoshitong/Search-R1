"""CPU-only residual wrapper bootstrap, lifecycle lease and CLI regression."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import source_residual_supplementary_control as control

DIRECTORY=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cpu_source_residual',DIRECTORY/'source_residual_supplementary_worker.py')
worker=importlib.util.module_from_spec(spec)
with patch.object(sys,'path',[str(DIRECTORY)]+sys.path):spec.loader.exec_module(worker)
core=worker.original


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class ResidualSupplementaryTests(unittest.TestCase):
    def test_stable_full_bootstrap_new_mapping_caps_and_original_bundles_unchanged(self):
        from . import node10_supplementary_control
        legacy=control.original
        original_specs=legacy.SPECIFICATIONS
        frozen={name:(DIRECTORY/name).read_bytes()for name in ('source_supplementary_control.py','source_supplementary_worker.py',
            'shared_vram_worker.py','relay_consumer_worker.py','queued_generation_control.py')}
        old_source=legacy.bundle(slot='gpu1-tp1')
        old_node10=node10_supplementary_control.bundle()
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            with patch.object(legacy.original,'BASE',str(base)):
                identity,files,manifest=control.bundle()
                self.assertEqual((identity,files,manifest),control.bundle())
            self.assertEqual(manifest['supplementary_slots'],worker.SPECIFICATIONS)
            self.assertEqual(manifest['generation'],'guard-rollover-supp-residual-v1')
            self.assertEqual(manifest['anchor_unit'],control.ANCHOR_UNIT)
            self.assertEqual(manifest['consumer_max_seconds'],28500)
            self.assertEqual(manifest['slots'][control.SLOT]['indices'],[4])
            self.assertNotEqual(manifest['unit'],control.ANCHOR_UNIT)
            stage=base/'preparation'/manifest['directory']/identity;stage.mkdir(parents=True)
            for name,raw in files.items():
                (stage/name).write_bytes(raw)
                self.assertEqual(manifest['files'][name],dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
            put(stage/'manifest.json',manifest)
            ready=base/'owner/owner-ready.json';put(ready,dict(slots={control.SLOT:dict(indices=[4],tp=1)},
                consumer_deadline_epoch=50000,owner=dict(unit='owner.service')))
            put(stage.parent/'submit-context.json',dict(owner_ready=str(ready),owner_ready_sha256=sha(ready)))
            unit=manifest['unit'];group='/system.slice/'+unit
            detail='Id=%s\nMainPID=123\nActiveState=active\nControlGroup=%s\nMemoryMax=103079215104\nActiveEnterTimestampMonotonic=1000000000\n'%(unit,group)
            original_text=Path.read_text
            def text(path,*a,**k):return '0::'+group+'\n'if str(path)=='/proc/self/cgroup'else original_text(path,*a,**k)
            namespace={'__file__':str(stage/'generation_bootstrap.py')}
            try:
                with patch('os.geteuid',return_value=0),patch('os.getpid',return_value=123),patch('socket.gethostname',return_value='h100-5'), \
                     patch.object(Path,'read_text',text),patch('time.time',return_value=2000),patch('time.monotonic',return_value=1500), \
                     patch('subprocess.check_output',return_value=detail),patch('os.chdir'),patch('os.dup2'),patch('builtins.print'), \
                     patch('os.execvpe')as execute:
                    exec(compile(files['generation_bootstrap.py'],namespace['__file__'],'exec'),namespace)
                _,argv,env=execute.call_args.args;flag=lambda key:argv[argv.index(key)+1]
                self.assertTrue(argv[2].endswith('/source_residual_supplementary_worker.py'))
                self.assertEqual([flag(k)for k in ('--model-gpus','--tp','--model-max-seqs','--model-batch-tokens',
                    '--model-kv-block-cap','--model-reserve-gib','--run-seconds','--batch-reserve-seconds')],
                    ['1,3,4,6','4','32','8192','4096','0.5','28210','1200'])
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'1,3,4,6');self.assertEqual(env['Q7B_SLOT_NAME'],'gpu4-tp1')
                self.assertEqual(env['SEARCH_G1_PHYSICAL_WORLD_SIZE'],'4')
                self.assertEqual(env['Q7B_SUPPLEMENTARY_MANIFEST_SHA256'],sha(stage/'manifest.json'))
            finally:
                if 'log'in namespace:namespace['log'].close()
        self.assertIs(legacy.SPECIFICATIONS,original_specs)
        self.assertEqual(legacy.bundle(slot='gpu1-tp1'),old_source)
        self.assertEqual(node10_supplementary_control.bundle(),old_node10)
        for name,raw in frozen.items():self.assertEqual((DIRECTORY/name).read_bytes(),raw)

    def test_real_inherited_admission_keeps_gpu4_anchor_visible_and_releases_only_own_leases(self):
        with tempfile.TemporaryDirectory()as folder:
            base=Path(folder);owner=dict(pid=900,birth='111',unit='owner.service',cgroup='0::/system.slice/owner.service\n',
                netns_inode=11,mntns_inode=12);retriever=dict(owner,pid=901,birth='112')
            model=control.SPECIFICATIONS[control.SLOT];run_id=control.original.original.RUN_ID
            specs=core.rollover_source_slots.specifications(core.owner_module.SLOTS)
            slots={key:dict(row,cuda_uuids=['GPU-%d'%i for i in row['indices']])for key,row in specs.items()}
            allocation=base/'owner/allocation.json';put(allocation,dict(allocation_role='RETRIEVAL_ONLY_PRIMARY_LEASE_OWNER',
                run_id=run_id,owner=owner,slots=slots,gpu_lock_proofs={'GPU-%d'%i:dict(fd=90+i,device=1,inode=100+i)for i in range(8)}))
            ready=base/'owner/owner-ready.json';put(ready,dict(status='RETRIEVAL_ONLY_OWNER_READY',run_id=run_id,
                owner=owner,retriever=retriever,allocation_path=str(allocation),allocation_sha256=sha(allocation),slots=slots,
                consumer_deadline_epoch=50000));ready_bytes,allocation_bytes=ready.read_bytes(),allocation.read_bytes()
            upstream=ready.parent/'retriever-source-ready.json';put(upstream,dict(role='upstream',source_pid=901,listen_url='http://127.0.0.1:28831'))
            original=base/'original/submit-intent.json';put(original,dict(unit=control.ANCHOR_UNIT,
                manifest=dict(role='slot',slot=control.SLOT,instance='r1'),context=dict(owner_ready_sha256=sha(ready))))
            context=base/'supp/submit-context.json';put(context,dict(anchor_terminal=dict(unit=control.ANCHOR_UNIT,
                intent_path=str(original),intent_sha256=sha(original),observed=dict(MainPID='0',ActiveState='failed'))))
            identity,_,manifest_value=control.bundle();manifest_value['files']={}
            manifest=base/'stage/manifest.json';put(manifest,manifest_value)
            unit=manifest_value['unit'];group='/system.slice/'+unit
            actual=dict(pid=123,birth='113',cgroup='0::'+group+'\n',netns_inode=11,mntns_inode=12)
            evidence=base/'evidence.json';put(evidence,dict(node='h100-5',main_pid=123,unit=unit,cgroup=group,
                unit_show='Id=%s\nMainPID=123\nActiveState=active\nControlGroup=%s\n'%(unit,group),
                source_bundle_sha256=identity,unit_hard_end_epoch=40000,resource_limits=dict(runtime_max_seconds=28800)))
            args=argparse.Namespace(base=str(base),reuse_local_relay_ready=str(ready),reuse_local_relay_sha256=sha(ready),
                reuse_local_relay_unit=owner['unit'],model_gpus='1,3,4,6',tp=4,run_seconds=28500,model_max_seqs=32,
                model_batch_tokens=8192,model_kv_block_cap=4096,model_reserve_gib=.5,allocation_evidence=str(evidence),upstream_ready=str(upstream))
            held=set();leases=[]
            def lock(path):
                path=Path(path)
                if str(path)in held:return None
                path.parent.mkdir(parents=True,exist_ok=True);fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600)
                held.add(str(path));leases.append((path,fd));return fd
            pool=SimpleNamespace(read=read,sha=sha,RUN_ID=run_id,try_lock=lock,
                nvml=lambda:{'GPU-%d'%i:dict(index=i,uuid='GPU-%d'%i)for i in range(8)})
            states={900:owner,901:retriever,123:actual};uuids=['GPU-%d'%i for i in model['indices']]
            env=dict(Q7B_SLOT_NAME=control.SLOT,Q7B_SLOT_OWNER_READY=str(ready),Q7B_SLOT_OWNER_READY_SHA256=sha(ready),
                Q7B_SUPPLEMENTARY_MANIFEST=str(manifest),Q7B_SUPPLEMENTARY_MANIFEST_SHA256=sha(manifest),
                Q7B_SUPPLEMENTARY_CONTEXT=str(context),Q7B_SUPPLEMENTARY_CONTEXT_SHA256=sha(context))
            old_specs=core.SPECIFICATIONS
            with patch.dict(os.environ,env),patch('os.getpid',return_value=123),patch('socket.gethostname',return_value='h100-5'), \
                 patch('time.time',return_value=1000),patch.object(core.owner_module,'process',side_effect=lambda pid:states[pid]), \
                 patch.object(core.slot_worker,'held_lock',side_effect=lambda p,o,v:v)as primary, \
                 patch.object(core.subprocess,'check_output',return_value=json.dumps(dict(actual_cuda_runtime_uuids=uuids)))as probe:
                binding,locks=worker.admit(pool,args)
                try:
                    self.assertEqual(binding['model_indices'],[1,3,4,6]);self.assertEqual(primary.call_count,4)
                    self.assertEqual(binding['model_cuda_visible_devices'],'1,3,4,6');self.assertEqual(len(locks),4)
                    self.assertEqual(leases[0][0].name,'GPU-4.lock');self.assertTrue(all(p.parent.name=='gpu4-tp1'for p,_ in leases))
                    self.assertEqual(core.owner_module.busy_slots(pool,base,dict(owner=owner,slots=slots)),[dict(slot='gpu4-tp1',uuid='GPU-4')])
                    self.assertEqual(probe.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'1,3,4,6')
                    with self.assertRaisesRegex(ValueError,'occupied'):worker.admit(pool,args)
                    self.assertEqual(probe.call_count,1)
                finally:
                    for fd in locks:os.close(fd)
                held.clear();probe.return_value=json.dumps(dict(actual_cuda_runtime_uuids=['GPU-0']))
                with self.assertRaisesRegex(ValueError,'CUDA UUID'):worker.admit(pool,args)
                for _,fd in leases[-4:]:
                    with self.assertRaises(OSError):os.fstat(fd)
            self.assertIs(core.SPECIFICATIONS,old_specs)
            self.assertEqual(ready.read_bytes(),ready_bytes);self.assertEqual(allocation.read_bytes(),allocation_bytes)

    def test_actual_main_stage_submit_once_and_active_original_anchor_rejected(self):
        with tempfile.TemporaryDirectory()as folder:
            base=Path(folder);original=control.original.original;run_id=original.RUN_ID
            owner=dict(pid=900,birth='111',unit='fixture-owner.service');allocation=base/'owner/allocation.json';put(allocation,dict(owner=owner))
            ready=base/'owner/owner-ready.json';put(ready,dict(status='RETRIEVAL_ONLY_OWNER_READY',run_id=run_id,owner=owner,
                allocation_path=str(allocation),allocation_sha256=sha(allocation),slots={control.SLOT:{}},consumer_deadline_epoch=50000))
            anchor=base/'old/submit-intent.json';put(anchor,dict(unit=control.ANCHOR_UNIT,
                manifest=dict(unit=control.ANCHOR_UNIT,role='slot',slot=control.SLOT,instance='r1'),
                context=dict(owner_ready=str(ready),owner_ready_sha256=sha(ready))))
            local=base/'local';(local/'work').mkdir(parents=True);starts=[];loaded=False;anchor_active=True
            def run(argv,**kwargs):
                nonlocal loaded
                if argv[0]=='unshare':
                    stream=io.StringIO()
                    with contextlib.redirect_stdout(stream):exec(compile(kwargs['input'],'actual-main-inner','exec'),{})
                    return SimpleNamespace(returncode=0,stdout=stream.getvalue(),stderr='')
                if argv[:2]==['systemctl','show']:
                    unit=argv[2]
                    if unit==owner['unit']:p=dict(Id=unit,LoadState='loaded',MainPID='900',ActiveState='active')
                    elif unit==control.ANCHOR_UNIT:p=dict(Id=unit,LoadState='loaded',MainPID='901'if anchor_active else'0',ActiveState='active'if anchor_active else'failed')
                    else:p=dict(Id=unit,LoadState='loaded'if loaded else'not-found',MainPID='123'if loaded else'0',ActiveState='active'if loaded else'inactive')
                    return SimpleNamespace(returncode=0,stdout='\n'.join(k+'='+v for k,v in p.items()),stderr='')
                self.assertEqual(argv[0],'systemd-run');starts.append(argv);loaded=True
                return SimpleNamespace(returncode=0,stdout='started',stderr='')
            def transport(script,node):
                stream=io.StringIO()
                with patch('socket.gethostname',return_value=node),contextlib.redirect_stdout(stream):
                    try:exec(compile(script,'actual-main-transport','exec'),{})
                    except SystemExit as error:self.assertEqual(error.code,0)
                return SimpleNamespace(returncode=0,stdout=stream.getvalue(),stderr='')
            original_text=Path.read_text
            def text(path,*a,**k):
                if str(path)=='/proc/900/stat':return '900 (fixture) '+' '.join(['S']+['0']*18+['111'])
                return original_text(path,*a,**k)
            common=['--intent-id','residual-fixture','--published-commit','a'*40,'--published-ref','refs/heads/fixture',
                '--owner-ready',str(ready),'--owner-ready-sha256',sha(ready),'--anchor-intent',str(anchor),'--anchor-intent-sha256',sha(anchor)]
            with patch.object(original,'BASE',str(base)),patch.object(original,'PHYSICAL',str(base)),patch.object(original,'ROOT',local), \
                 patch('os.geteuid',return_value=0),patch.object(Path,'read_text',text),patch('time.time',return_value=1000), \
                 patch('subprocess.run',side_effect=run),patch.object(original.old,'transport',side_effect=transport), \
                 patch.object(original.old,'verify_publication',return_value=dict(status='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED')), \
                 contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys,'argv',['residual','stage',*common]):control.main()
                identity,files,manifest=control.bundle();root=base/'preparation'/manifest['directory']
                with patch.object(sys,'argv',['residual','submit',*common]):
                    with self.assertRaisesRegex(AssertionError,'not terminal'):control.main()
                self.assertFalse((root/'submit-intent.json').exists());self.assertFalse((root/'submit-context.json').exists())
                anchor_active=False
                for _ in range(2):
                    with patch.object(sys,'argv',['residual','submit',*common]):control.main()
            self.assertEqual(len(starts),1)
            intent=read(root/'submit-intent.json')
            self.assertEqual(intent['manifest']['generation'],'guard-rollover-supp-residual-v1')
            self.assertEqual(intent['context']['anchor_terminal']['unit'],control.ANCHOR_UNIT)
            self.assertEqual(intent['context']['anchor_terminal']['intent_sha256'],sha(anchor))
            self.assertIn('--property=RuntimeMaxSec=28800',starts[0])
            self.assertIn('--setenv=Q7B_NATIVE_UNIT_INSTANCE='+manifest['unit'],starts[0])


if __name__=='__main__':unittest.main()
