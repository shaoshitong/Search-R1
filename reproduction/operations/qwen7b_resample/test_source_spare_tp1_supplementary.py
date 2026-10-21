"""CPU-only spare TP1 topology, independent anchor lifecycle and bootstrap checks."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import source_spare_tp1_supplementary_control as control

DIRECTORY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cpu_spare_tp1', DIRECTORY / 'source_spare_tp1_supplementary_worker.py')
worker = importlib.util.module_from_spec(spec)
with patch.object(sys, 'path', [str(DIRECTORY)] + sys.path):spec.loader.exec_module(worker)


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def put(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True);path.write_text(json.dumps(value))


class SpareTP1SupplementaryTests(unittest.TestCase):
    def test_active_original_r1_anchor_rejected_before_any_submit_intent(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);identity,files,manifest=control.bundle(slot='gpu6-tp1')
            root=base/'preparation'/manifest['directory'];stage=root/identity;stage.mkdir(parents=True)
            for name,raw in files.items():(stage/name).write_bytes(raw)
            put(stage/'manifest.json',manifest);put(stage/'stage-receipt.json',dict(status='EXACT_OWNER_GENERATION_STAGE_SHA_PASS'))
            owner=dict(pid=900,birth='111',unit='fixture-owner.service')
            allocation=base/'owner/allocation.json';put(allocation,dict(owner=owner))
            ready=base/'owner/owner-ready.json';put(ready,dict(status='RETRIEVAL_ONLY_OWNER_READY',run_id=control.original.original.RUN_ID,
                owner=owner,allocation_path=str(allocation),allocation_sha256=sha(allocation),slots={'gpu6-tp1':{}},consumer_deadline_epoch=50000))
            oldunit='sst-q7b-g5-guard-rollover-owner-v2-gpu6-tp1-r1-'+control.original.original.RUN_ID[:12]+'.service'
            anchor=base/'old/submit-intent.json';put(anchor,dict(unit=oldunit,manifest=dict(unit=oldunit,role='slot',slot='gpu6-tp1',instance='r1'),
                context=dict(owner_ready=str(ready),owner_ready_sha256=sha(ready))))
            def run(argv,**kwargs):
                self.assertEqual(argv[:2],['systemctl','show'],'No systemd-run may occur with an active anchor')
                unit=argv[2]
                if unit==manifest['unit']:values=dict(Id=unit,LoadState='not-found',MainPID='0',ActiveState='inactive')
                elif unit==owner['unit']:values=dict(Id=unit,LoadState='loaded',MainPID='900',ActiveState='active')
                elif unit==oldunit:values=dict(Id=unit,LoadState='loaded',MainPID='901',ActiveState='active')
                else:raise AssertionError(unit)
                return SimpleNamespace(returncode=0,stdout='\n'.join(k+'='+v for k,v in values.items()),stderr='')
            original_text=Path.read_text
            def text(path,*a,**k):
                if str(path)=='/proc/900/stat':return '900 (fixture) '+' '.join(['S']+['0']*18+['111'])
                return original_text(path,*a,**k)
            values=dict(BASE=str(base),PHYSICAL=str(base),ACTION='submit',MANIFEST=manifest,IDENTITY=identity,
                PUBLICATION=dict(status='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'),OWNER_READY=str(ready),OWNER_READY_SHA=sha(ready),
                ANCHOR_INTENT=str(anchor),ANCHOR_INTENT_SHA=sha(anchor),RUN_ID=control.original.original.RUN_ID,INTENT='fixture')
            with patch('os.geteuid',return_value=0),patch('socket.gethostname',return_value='h100-5'),patch('subprocess.run',side_effect=run), \
                 patch.object(Path,'read_text',text),patch('time.time',return_value=1000):
                with self.assertRaisesRegex(AssertionError,'Original exact r1 anchor is not terminal'):
                    exec(compile(control.remote_source(),'supplementary-submit','exec'),values)
            self.assertFalse((root/'submit-intent.json').exists());self.assertFalse((root/'submit-context.json').exists())

    def test_both_tp1s_keep_separate_anchor_visible_and_existing_slot_unchanged(self):
        for name, model in control.SPECIFICATIONS.items():
            with self.subTest(anchor=name), tempfile.TemporaryDirectory() as folder:
                base = Path(folder);owner = dict(pid=900, birth='111', unit='owner.service',
                    cgroup='0::/system.slice/owner.service\n', netns_inode=11, mntns_inode=12)
                retriever = dict(owner, pid=901, birth='112');own_unit='supplementary.service';group='/system.slice/'+own_unit
                actual = dict(pid=123, birth='113', cgroup='0::'+group+'\n', netns_inode=11, mntns_inode=12)
                specifications = worker.rollover_source_slots.specifications(worker.owner_module.SLOTS)
                slots = {key:dict(value,cuda_uuids=['GPU-%d'%i for i in value['indices']]) for key,value in specifications.items()}
                allocation=base/'owner/allocation.json'
                put(allocation,dict(allocation_role='RETRIEVAL_ONLY_PRIMARY_LEASE_OWNER',run_id=control.original.original.RUN_ID,
                    owner=owner,slots=slots,gpu_lock_proofs={'GPU-%d'%i:dict(fd=90+i,device=1,inode=100+i)for i in range(8)}))
                ready=base/'owner/owner-ready.json'
                put(ready,dict(status='RETRIEVAL_ONLY_OWNER_READY',run_id=control.original.original.RUN_ID,owner=owner,retriever=retriever,
                    allocation_path=str(allocation),allocation_sha256=sha(allocation),slots=slots,consumer_deadline_epoch=50000))
                original_ready,original_allocation=ready.read_bytes(),allocation.read_bytes()
                upstream=ready.parent/'retriever-source-ready.json';put(upstream,dict(role='upstream',source_pid=901,listen_url='http://127.0.0.1:28831'))
                original=base/'original/submit-intent.json'
                put(original,dict(unit=control.ANCHOR_UNITS[name],manifest=dict(role='slot',slot=name,instance='r1'),
                    context=dict(owner_ready_sha256=sha(ready))))
                context=base/'supp/submit-context.json'
                put(context,dict(anchor_terminal=dict(unit=control.ANCHOR_UNITS[name],intent_path=str(original),intent_sha256=sha(original),
                    observed=dict(MainPID='0',ActiveState='failed'))))
                manifest=base/'stage/manifest.json'
                put(manifest,dict(role='slot',slot=name,anchor_unit=control.ANCHOR_UNITS[name],
                    supplementary_slots={name:model},files={},source_bundle_sha256='a'*64))
                evidence=base/'evidence.json';put(evidence,dict(node='h100-5',main_pid=123,unit=own_unit,cgroup=group,
                    unit_show='Id=%s\nMainPID=123\nActiveState=active\nControlGroup=%s\n'%(own_unit,group),
                    source_bundle_sha256='a'*64,unit_hard_end_epoch=40000,resource_limits=dict(runtime_max_seconds=28800)))
                args=argparse.Namespace(base=str(base),reuse_local_relay_ready=str(ready),reuse_local_relay_sha256=sha(ready),
                    reuse_local_relay_unit=owner['unit'],model_gpus=','.join(map(str,model['indices'])),tp=1,run_seconds=28500,
                    model_max_seqs=model['seqs'],model_batch_tokens=model['tokens'],model_kv_block_cap=model['blocks'],model_reserve_gib=model['reserve'],
                    allocation_evidence=str(evidence),upstream_ready=str(upstream))
                existing=worker.owner_module.slot_directory(base,control.original.original.RUN_ID,owner,'gpu2-tp1')/'GPU-2.lock'
                existing.parent.mkdir(parents=True);existing.write_bytes(b'original-live-slot-lease')
                held={str(existing)};leases=[];opened=[]
                def try_lock(path):
                    path=Path(path)
                    if str(path) in held:return None
                    path.parent.mkdir(parents=True,exist_ok=True)
                    fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600);held.add(str(path));leases.append(str(path));opened.append(fd);return fd
                pool=SimpleNamespace(read=read,sha=sha,RUN_ID=control.original.original.RUN_ID,try_lock=try_lock,
                    nvml=lambda:{'GPU-%d'%i:dict(index=i,uuid='GPU-%d'%i)for i in range(8)})
                states={900:owner,901:retriever,123:actual};uuids=['GPU-%d'%i for i in model['indices']]
                env=dict(Q7B_SLOT_NAME=name,Q7B_SLOT_OWNER_READY=str(ready),Q7B_SLOT_OWNER_READY_SHA256=sha(ready),
                    Q7B_SUPPLEMENTARY_MANIFEST=str(manifest),Q7B_SUPPLEMENTARY_MANIFEST_SHA256=sha(manifest),
                    Q7B_SUPPLEMENTARY_CONTEXT=str(context),Q7B_SUPPLEMENTARY_CONTEXT_SHA256=sha(context))
                with patch.dict(os.environ,env),patch('os.getpid',return_value=123),patch('socket.gethostname',return_value='h100-5'), \
                     patch.object(worker.owner_module,'process',side_effect=lambda pid:states[pid]), \
                     patch.object(worker.slot_worker,'held_lock',side_effect=lambda path,owner,proof:dict(proof)) as primary, \
                     patch.object(worker.subprocess,'check_output',return_value=json.dumps(dict(actual_cuda_runtime_uuids=uuids)))as probe, \
                     patch.object(worker.time,'time',return_value=1000):
                    binding,locks=worker.admit(pool,args)
                    try:
                        self.assertEqual(binding['model_indices'],model['indices']);self.assertEqual(binding['tensor_parallel_size'],1)
                        self.assertEqual(binding['actual_cuda_runtime_uuids'],uuids)
                        self.assertEqual(binding['model_cuda_visible_devices'],str(model['indices'][0]))
                        self.assertEqual(primary.call_count,2);self.assertEqual(len(locks),2)
                        self.assertEqual(set(binding['original_primary_lock_proofs']),set(slots[name]['cuda_uuids']+uuids))
                        self.assertTrue(all(Path(p).parent.name==name for p in leases))
                        self.assertNotIn(str(existing),leases)
                        self.assertEqual(Path(leases[0]).name,slots[name]['cuda_uuids'][0]+'.lock')
                        busy=worker.owner_module.busy_slots(pool,base,dict(owner=owner,slots=slots))
                        self.assertCountEqual(busy,[dict(slot='gpu2-tp1',uuid='GPU-2'),dict(slot=name,uuid=slots[name]['cuda_uuids'][0])])
                        self.assertEqual(binding['source_model_deadline_epoch'],39955)
                        with self.assertRaisesRegex(ValueError,'already occupied'):worker.admit(pool,args)
                    finally:
                        for fd in locks:os.close(fd)
                    # A busy supplementary lease releases only the newly acquired anchor.
                    held.clear();held.update((str(existing),binding['slot_lease_paths'][uuids[0]]))
                    with self.assertRaisesRegex(ValueError,'already occupied'):worker.admit(pool,args)
                    with self.assertRaises(OSError):os.fstat(opened[-1])
                    # A failed actual CUDA UUID probe releases both fresh leases.
                    held.clear();held.add(str(existing));probe.return_value=json.dumps(dict(actual_cuda_runtime_uuids=['wrong-GPU']))
                    with self.assertRaisesRegex(ValueError,'CUDA UUID'):worker.admit(pool,args)
                    for fd in opened[-2:]:
                        with self.assertRaises(OSError):os.fstat(fd)
                self.assertEqual(ready.read_bytes(),original_ready);self.assertEqual(allocation.read_bytes(),original_allocation)
                self.assertEqual(existing.read_bytes(),b'original-live-slot-lease')

    def test_bundle_bootstrap_uses_new_spec_and_binds_every_artifact_sha(self):
        import ast
        ast.parse(control.remote_source())
        self.assertIn("p.get('MainPID')=='0'",control.remote_source())
        self.assertIn("prior['instance']=='r1'",control.remote_source())
        for name, model in control.SPECIFICATIONS.items():
            with self.subTest(anchor=name),tempfile.TemporaryDirectory() as folder:
                base=Path(folder)
                with patch.object(control.original.original,'BASE',str(base)):
                    identity,files,manifest=control.bundle(slot=name)
                    self.assertEqual((identity,files,manifest),control.bundle(slot=name))
                self.assertEqual(manifest['supplementary_slots'],{name:model})
                self.assertEqual(manifest['anchor_unit'],control.ANCHOR_UNITS[name])
                self.assertEqual(manifest['generation'],'guard-rollover-'+control.GENERATION)
                self.assertNotEqual(manifest['slots'][name],model)
                for filename,raw in files.items():
                    self.assertEqual(manifest['files'][filename],dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
                    if filename in manifest['source_files']:self.assertEqual(manifest['source_files'][filename],manifest['files'][filename])
                stage=base/'preparation'/manifest['directory']/identity;stage.mkdir(parents=True)
                for filename,raw in files.items():(stage/filename).write_bytes(raw)
                put(stage/'manifest.json',manifest)
                ready=base/'owner/owner-ready.json';put(ready,dict(slots={name:dict(indices=[1],tp=1)},consumer_deadline_epoch=50000,owner=dict(unit='owner.service')))
                put(stage.parent/'submit-context.json',dict(owner_ready=str(ready),owner_ready_sha256=sha(ready)))
                unit=manifest['unit'];group='/system.slice/'+unit
                detail='Id=%s\nMainPID=123\nActiveState=active\nControlGroup=%s\nMemoryMax=103079215104\nActiveEnterTimestampMonotonic=1000000000\n'%(unit,group)
                original_text=Path.read_text
                def text(path,*a,**k):return '0::'+group+'\n' if str(path)=='/proc/self/cgroup' else original_text(path,*a,**k)
                namespace={'__file__':str(stage/'generation_bootstrap.py')}
                try:
                    with patch('os.geteuid',return_value=0),patch('os.getpid',return_value=123),patch('socket.gethostname',return_value='h100-5'), \
                         patch.object(Path,'read_text',text),patch('time.time',return_value=2000),patch('time.monotonic',return_value=1500), \
                         patch('subprocess.check_output',return_value=detail),patch('os.chdir'),patch('os.dup2'),patch('builtins.print'), \
                         patch('os.execvpe') as execute:
                        exec(compile(files['generation_bootstrap.py'],namespace['__file__'],'exec'),namespace)
                    _,argv,env=execute.call_args.args
                    flag=lambda key:argv[argv.index(key)+1]
                    self.assertTrue(argv[2].endswith('/source_spare_tp1_supplementary_worker.py'))
                    self.assertEqual(flag('--tp'),'1');self.assertEqual(flag('--model-gpus'),','.join(map(str,model['indices'])))
                    self.assertEqual(flag('--model-max-seqs'),str(model['seqs']));self.assertEqual(flag('--model-kv-block-cap'),str(model['blocks']))
                    self.assertEqual(flag('--model-batch-tokens'),str(model['tokens']));self.assertEqual(flag('--model-reserve-gib'),'1.5')
                    self.assertEqual(env['CUDA_VISIBLE_DEVICES'],str(model['indices'][0]));self.assertEqual(env['SEARCH_G1_PHYSICAL_WORLD_SIZE'],'4')
                    profile={'__package__':'reproduction.prefix_selection200'}
                    with patch.dict(os.environ,env):
                        exec(compile((DIRECTORY.parents[1]/'prefix_selection200/contract.py').read_bytes(),'contract.py','exec'),profile)
                    self.assertEqual(profile['WORLD'],4)
                    # Exercise the canonical writer: legacy profile4 keeps the real batch world1.
                    from . import pool_worker
                    inp=base/'inputs/shard-0000/batch-000000.json'
                    _,configuration=pool_worker.batch_config(SimpleNamespace(atomic=put),base,inp,dict(rows=[dict(question='fixture')]),
                        dict(base=str(base),model_dir='official-qwen7b'))
                    self.assertEqual(read(configuration)['world_size'],1)
                    self.assertEqual(env['Q7B_SLOT_NAME'],name)
                    self.assertEqual(env['Q7B_SUPPLEMENTARY_MANIFEST_SHA256'],sha(stage/'manifest.json'))
                    self.assertEqual(env['Q7B_SUPPLEMENTARY_CONTEXT_SHA256'],sha(stage.parent/'submit-context.json'))
                finally:
                    if 'log' in namespace:namespace['log'].close()

    def test_original_source_bundles_and_in_process_registry_unchanged(self):
        from . import source_residual_supplementary_control
        original=control.original;specifications=original.SPECIFICATIONS
        before=(original.bundle(slot='gpu1-tp1'),source_residual_supplementary_control.bundle())
        for name in control.SPECIFICATIONS:control.bundle(slot=name)
        self.assertIs(original.SPECIFICATIONS,specifications)
        self.assertEqual(before,(original.bundle(slot='gpu1-tp1'),source_residual_supplementary_control.bundle()))
        self.assertEqual(worker.original.SPECIFICATIONS,{'gpu1-tp1':dict(indices=[0,1,2,5],tp=4,seqs=32,tokens=8192,blocks=4096,reserve=.5),
            'gpu3-tp1':dict(indices=[3,4,6,7],tp=4,seqs=32,tokens=8192,blocks=4096,reserve=.5)})


if __name__=='__main__':unittest.main()
