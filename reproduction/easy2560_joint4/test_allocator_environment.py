"""Launch and actual Ray-init argument plumbing, with all remote/model work mocked."""
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from reproduction.easy2560_joint4 import native_ray_resources as resources


class AllocatorEnvironmentTests(unittest.TestCase):
    def test_fa3_recovery_shell_sets_allocator_before_the_first_python(self):
        entry = Path(resources.__file__).with_name('train.sh')
        for profile, receipt, expected in (('baseline','source.json',''),
                ('fa3-exact-cache-v1','',''), ('fa3-exact-cache-v1','source.json','expandable_segments:True')):
            with self.subTest(profile=profile,receipt=receipt), tempfile.TemporaryDirectory() as td:
                root = Path(td); base = root/'target'; (base/'env/bin').mkdir(parents=True); (base/'repo').mkdir()
                python = base/'env/bin/python'
                python.write_text('#!/bin/sh\nprintf "%s\\n" "${PYTORCH_CUDA_ALLOC_CONF-}" >> "$CAPTURE"\n'
                                  'printf "%s\\n" "$@" >> "$ARGUMENTS"\n')
                python.chmod(0o755)
                env = dict(os.environ, CAPTURE=str(root/'environment'), ARGUMENTS=str(root/'arguments'))
                env.pop('LD_PRELOAD',None); env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
                subprocess.run(['bash',str(entry),str(base),'a'*40,'8',receipt,profile],
                               env=env,check=True,capture_output=True)
                self.assertEqual((root/'environment').read_text().splitlines(),[expected,expected])
                args = (root/'arguments').read_text().splitlines()
                self.assertIn('reproduction.easy2560_joint4.prepare_training',args)
                self.assertIn('reproduction.easy2560_joint4.native_ray_resources',args)
                i=args.index('--actor-micro'); self.assertEqual(args[i+1],'8')

    def test_actual_main_body_passes_allocator_to_ray_driver_and_descendants(self):
        # Execute only the real main body, avoiding Hydra/model imports and any GPU work.
        path = Path(resources.__file__).resolve().parents[2]/'verl/trainer/main_ppo_eco.py'
        tree = ast.parse(path.read_text()); function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        function.decorator_list=[]
        code = compile(ast.Module(body=[function],type_ignores=[]), str(path), 'exec')
        for config_value in ('','expandable_segments:True'):
            with self.subTest(config=config_value):
                ray = SimpleNamespace(is_initialized=lambda:False,init=Mock(),get=Mock(),shutdown=Mock())
                task = SimpleNamespace(remote=Mock(return_value='driver-result'))
                namespace={'ray':ray,'main_task':task}; exec(code,namespace)
                env=dict(SEARCH_G1_BASE='/unused/isolated-target',RAY_TMPDIR='/unused/private-ray',
                         PYTORCH_CUDA_ALLOC_CONF=config_value)
                with patch.dict(os.environ,env), \
                     patch('reproduction.common8.topology._bound_run',return_value={'easy2560_selection':True}), \
                     patch.object(resources,'configure',return_value={'num_cpus':16}):
                    namespace['main'](SimpleNamespace(trainer=SimpleNamespace(n_gpus_per_node=4)))
                kwargs=ray.init.call_args.kwargs
                expected={'TOKENIZERS_PARALLELISM':'true','NCCL_DEBUG':'WARN'}
                if config_value: expected['PYTORCH_CUDA_ALLOC_CONF']=config_value
                self.assertEqual(kwargs['runtime_env'],{'env_vars':expected})
                self.assertEqual(kwargs['num_gpus'],4); self.assertEqual(kwargs['num_cpus'],16)
                ray.get.assert_called_once_with('driver-result'); ray.shutdown.assert_called_once_with()

    def test_existing_resource_receipt_records_actual_allocator_without_changing_ray_resources(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)
            run=dict(profile='common8',execution_site='hpc3',training_world_size=4,
                data_selection_mode='mixed_first',easy2560_selection=True,question_probe=None,
                all_correct_virtual_zero=False,run_id='r'*32,source_commit='a'*40)
            env=dict(SEARCH_G1_BASE=str(base),SLURM_JOB_ID='123',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            with patch.dict(os.environ,env), patch.object(resources,'own_base',side_effect=Path), \
                 patch.object(os,'getuid',return_value=14090), patch.object(resources,'raise_self_soft',return_value={}), \
                 redirect_stdout(io.StringIO()):
                kwargs=resources.configure(base,run,phase='ray')
            self.assertEqual(kwargs,dict(num_cpus=16,object_store_memory=8*1024**3))
            receipt,=(base/'logs').glob('native-ray-resources-train-ray-*.json')
            document=json.loads(receipt.read_text())
            self.assertEqual(document['torch_allocator_env'],{'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'})
            self.assertEqual(document['public_ray_init_kwargs'],kwargs)


if __name__ == '__main__': unittest.main()
