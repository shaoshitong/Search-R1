"""CPU-only finite-slot bundle/bootstrap, code-only stage and no-replay checks."""
import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import relay_tp1_control as control


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class BundleTests(unittest.TestCase):
    def test_four_distinct_tp1_bundles_preserve_config_and_scientific_sources(self):
        identities,units,directories=set(),set(),set()
        self.assertEqual(control.SLOTS,dict(gpu1=1,gpu2=2,gpu3=3,gpu5=5))
        for slot,index in control.SLOTS.items():
            identity,files,manifest=control.bundle(slot)
            identities.add(identity);units.add(manifest['unit']);directories.add(manifest['directory'])
            self.assertEqual(manifest['model_gpus'],[index]);self.assertEqual(manifest['tensor_parallel_size'],1)
            self.assertEqual(manifest['model_limits'],dict(seqs=64,tokens=8192,blocks=4096,reserve=1.0))
            self.assertEqual(manifest['max_model_len'],4096)
            self.assertEqual(manifest['maximum_run_seconds'],14400);self.assertEqual(manifest['batch_reserve_seconds'],1200)
            self.assertEqual(manifest['owner_deadline_cutoff_seconds'],600)
            for name,data in files.items():
                self.assertEqual(manifest['files'][name],dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
            # The scientific sampler, claiming and exact retriever remain the existing bytes.
            for name in ('relay_consumer_worker.py','shared_vram_worker.py','shared_vram_vllm_adapter.py','pool_worker.py','retrieval_relay.py'):
                self.assertEqual(files[name],Path(control.__file__).with_name(name).read_bytes())
            ast.parse(files['relay_tp1_bootstrap.py'].decode())
        self.assertEqual((len(identities),len(units),len(directories)),(4,4,4))
        for slot in ('gpu0','gpu4','gpu6','gpu7','all'):
            with self.assertRaises(ValueError):control.bundle(slot)


class BootstrapTests(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self,slot='gpu1',now=1000,deadline=5000,owner_deadline=5600):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);unit,directory=control.names(slot);stage=base/'preparation'/directory/'code';stage.mkdir(parents=True)
            manifest=dict(run_id=control.RUN_ID,source_bundle_sha256='a'*64,files={})
            write(stage/'manifest.json',manifest)
            context=dict(unit=unit,run_id=control.RUN_ID,model_indices=[control.SLOTS[slot]],model_limits=control.LIMITS,
                model_deadline_epoch=deadline,relay_owner=dict(unit='original-v5.service',deadline_epoch=owner_deadline),
                lease_owner=dict(deadline_epoch=owner_deadline),source=dict(source_sampling_deadline_epoch=owner_deadline),
                upstream_ready=str(base/'original/upstream-ready.json'),reuse_local_relay_ready=str(base/'original/relay-ready.json'),
                reuse_local_relay_sha256='b'*64)
            path=stage.parent/'slot-context.json';write(path,context)
            write(stage.parent/'native-submit-intent.json',dict(unit=unit,source_bundle_sha256='a'*64,
                dependency_context_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            with patch.object(control,'BASE',str(base)):source=control.bootstrap(slot)
            group='/system.slice/'+unit
            detail='Id=%s\nMainPID=314\nActiveState=active\nControlGroup=%s\n'%(unit,group)
            real_stat,real_read=os.stat,Path.read_text
            def stat(path,*args,**kwargs):
                if str(path) in ('/proc/self/ns/mnt','/proc/1/ns/mnt','/proc/self/ns/net','/proc/1/ns/net'):
                    return SimpleNamespace(st_ino=2 if str(path)=='/proc/self/ns/mnt' else 1)
                return real_stat(path,*args,**kwargs)
            def read(path,*args,**kwargs):
                if str(path)=='/proc/self/cgroup':return '0::'+group+'\n'
                return real_read(path,*args,**kwargs)
            namespace=dict(__file__=str(stage/'relay_tp1_bootstrap.py'))
            try:
                with patch('socket.gethostname',return_value='h100-10'),patch('os.geteuid',return_value=0), \
                     patch('os.getpid',return_value=314),patch('time.time',return_value=now),patch('os.stat',side_effect=stat), \
                     patch.object(Path,'read_text',read),patch('subprocess.check_output',return_value=detail) as query, \
                     patch('os.dup2'),patch('os.chdir'),patch('builtins.print'),patch('os.execvpe') as execute:
                    yield SimpleNamespace(source=source,namespace=namespace,execute=execute,query=query,base=base,slot=slot)
            finally:
                if namespace.get('log'):namespace['log'].close()

    def test_each_slot_has_independent_tp1_cvd_and_borrows_only_existing_relay(self):
        for slot,index in control.SLOTS.items():
            with self.subTest(slot=slot),self.fixture(slot) as fixture:
                exec(compile(fixture.source,fixture.namespace['__file__'],'exec'),fixture.namespace)
                executable,argv,env=fixture.execute.call_args.args
                option=lambda flag:argv[argv.index(flag)+1]
                self.assertTrue(executable.endswith('/env/bin/python'))
                self.assertEqual(option('--model-gpus'),str(index));self.assertEqual(env['CUDA_VISIBLE_DEVICES'],str(index))
                self.assertEqual(option('--tp'),'1');self.assertEqual(option('--model-max-seqs'),'64')
                self.assertEqual(option('--model-batch-tokens'),'8192');self.assertEqual(option('--model-kv-block-cap'),'4096')
                self.assertEqual(option('--model-reserve-gib'),'1.0');self.assertEqual(option('--batch-reserve-seconds'),'1200')
                self.assertEqual(option('--run-seconds'),'4000')
                self.assertEqual(option('--reuse-local-relay-unit'),'original-v5.service')
                self.assertEqual(env['SEARCH_G1_B_PREFIX_TOKENS'],'20');self.assertEqual(env['SEARCH_G1_PHYSICAL_WORLD_SIZE'],'4')
                self.assertTrue(Path(env['HF_HOME']).is_relative_to(fixture.base))
                self.assertNotIn('--token-file',argv);self.assertNotIn('--quantization',argv)
                self.assertEqual(Path(env['Q7B_NODE10_TP1_CONTEXT']).parent.name,control.names(slot)[1])

    def test_absolute_deadline_survives_bootstrap_delay_and_never_exceeds_four_hours(self):
        for now,deadline,owner_deadline,expected in ((1000,25000,25600,14400),(2400,5000,5600,2600)):
            with self.subTest(now=now),self.fixture(now=now,deadline=deadline,owner_deadline=owner_deadline) as fixture:
                exec(compile(fixture.source,fixture.namespace['__file__'],'exec'),fixture.namespace)
                argv=fixture.execute.call_args.args[1]
                self.assertEqual(int(argv[argv.index('--run-seconds')+1]),expected)
        for deadline,owner_deadline in ((5001,5600),(2200,5600)):
            with self.subTest(deadline=deadline),self.fixture(deadline=deadline,owner_deadline=owner_deadline) as fixture:
                with self.assertRaises(AssertionError):exec(compile(fixture.source,fixture.namespace['__file__'],'exec'),fixture.namespace)
                fixture.execute.assert_not_called();fixture.query.assert_not_called()


class RemoteBranchTests(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self,action):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);identity,files,manifest=control.bundle('gpu1')
            root=base/'preparation'/manifest['directory'];root.mkdir(parents=True)
            namespace=dict(PHYSICAL=str(base),BASE=str(base),IDENTITY=identity,ACTION=action,INTENT='cpu-once',
                PUBLICATION=dict(status='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'),MANIFEST=manifest,
                FILES={name:__import__('base64').b64encode(data).decode() for name,data in files.items()})
            with patch('socket.gethostname',return_value='h100-10'),patch('os.geteuid',return_value=0), \
                 patch('subprocess.run') as run,patch('builtins.print'):
                yield SimpleNamespace(base=base,root=root,namespace=namespace,run=run,files=files,manifest=manifest)

    def test_stage_is_code_only_without_gpu_or_source_process_queries(self):
        with self.fixture('stage') as fixture:
            exec(compile(control.REMOTE,'<remote-stage>','exec'),fixture.namespace)
            fixture.run.assert_not_called()
            self.assertEqual(fixture.namespace['value']['status'],'NODE10_TP1_CODE_ONLY_STAGED')
            self.assertFalse((fixture.base/'pool').exists())
            self.assertFalse((fixture.root/'slot-context.json').exists())
            self.assertFalse((fixture.root/'native-submit-intent.json').exists())
            stage=fixture.root/fixture.namespace['IDENTITY']
            self.assertEqual({p.name for p in stage.iterdir()},set(fixture.files)|{'manifest.json','stage-receipt.json'})

    def test_existing_intent_never_replays_start_even_with_a_new_local_bundle(self):
        with self.fixture('submit') as fixture:
            path=fixture.root/'native-submit-intent.json';write(path,dict(intent_id='original',unit=fixture.manifest['unit']))
            original=path.read_bytes()
            exec(compile(control.REMOTE,'<remote-submit>','exec'),fixture.namespace)
            self.assertEqual(fixture.namespace['value']['status'],'EXISTING_INTENT_NO_START_REPLAY')
            fixture.run.assert_not_called();self.assertEqual(path.read_bytes(),original)
            self.assertFalse((fixture.root/'slot-context.json').exists())


if __name__=='__main__':unittest.main()
