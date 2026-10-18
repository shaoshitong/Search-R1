"""CPU-only reader tests; no stored weights, native process or GPU is used."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.prefix_selection200 import ab_checkpoint_once as reader
from reproduction.prefix_selection200.b_progressive import MODE
from .normal_checkpoint_control import remote_inner


def config():
    return dict(run_id='test-b',selection_probe_experiment='balanced_online_ema',
        physical_gpu_indices=[4,5,6,7],probe_learning_rate=5e-6,
        probe_group_loss_weight=5.,probe_ema_rate=.99,probe_source_sha256='warmup',
        b_progressive=dict(mode=MODE,resume_outer=50))


class ProgressiveCheckpointTests(unittest.TestCase):
    def test_completed_stage_boundaries_and_A_stays_five(self):
        c=config()
        for outer,prefix in [(50,5),(51,20),(100,20),(101,40),(150,40),(151,80),(200,80)]:
            value,shapes=reader.probe_layout(c,outer)
            self.assertEqual(value,prefix)
            self.assertEqual(shapes['output.weight'],(1,(273+prefix)*2048))
            self.assertEqual({k:v for k,v in shapes.items()if k!='output.weight'},
                {k:v for k,v in reader.PARAMETER_SHAPES.items()if k!='output.weight'})
        self.assertEqual(reader.probe_layout({},150)[0],5)
        with self.assertRaises(ValueError):reader.probe_layout(dict(c,physical_gpu_indices=[0,1,2,3]),100)

    def fake_saved(self,outer):
        c=config();prefix,shapes=reader.probe_layout(c,outer);steps=outer*10
        binding=dict(selection_probe_experiment='balanced_online_ema',learning_rate=5e-6,
            group_weight=5.,ema_rate=.99,prefix_tokens=prefix,head_max_tokens=273+prefix,
            question_max_tokens=273,source_probe_sha256='warmup',source_epoch=5,
            source_model_kind='ema',warmup_optimizer_imported=False,world_size=4)
        model={k:SimpleNamespace(shape=v)for k,v in shapes.items()}
        buffers={k:SimpleNamespace(shape=v)for k,v in reader.NATIVE_BUFFER_LAYOUTS[0].items()}
        count=512 if outer==200 else 2560
        saved=dict(binding=binding,binding_sha256=reader.digest(binding),run_id='test-b',
            completed_outer=outer,actual_adam_steps_per_rank=[steps]*4,model=model,
            model_nonpersistent_buffers=buffers,ema_model=copy.deepcopy(model),
            ema_nonpersistent_buffers=copy.deepcopy(buffers),ema_updates=steps,
            sampling=dict(status='BALANCED_RESAMPLED_ONE_PASS',sampled_indices=list(range(count)),
                sampled_counts=[count//2]*2,group_multiplicities={str(i):1 for i in range(count)},
                rng_after=random.Random(1).getstate()),optimizer={'unchanged':'moments'},
            rank_rng=[dict(rank=i,actual_adam_steps=steps)for i in range(4)])
        receipt={k:saved[k]for k in ['binding_sha256','completed_outer','actual_adam_steps_per_rank','ema_updates']}
        return c,saved,receipt

    def test_raw_ema_and_adam_shapes_are_validated_without_reset(self):
        def tensor(value,**kw):
            if tuple(value.shape)!=tuple(kw['shape']):raise ValueError('Tensor shape mismatch')
        for outer in (100,150,200):
            c,saved,receipt=self.fake_saved(outer)
            with patch.object(reader,'tensor_cpu',side_effect=tensor),patch.object(reader,'validate_rng'),patch.object(reader,'validate_adam',return_value=14)as adam:
                result=reader.validate_probe_state(saved,receipt,c,outer)
                self.assertEqual(result['prefix_tokens'],{100:20,150:40,200:80}[outer])
                self.assertEqual(result['actual_adam_steps_per_rank'],[outer*10]*4)
                self.assertIs(adam.call_args.args[0],saved['optimizer'])
                self.assertEqual(adam.call_args.args[1], [v.shape for v in saved['model'].values()])
                saved['ema_model']['output.weight'].shape=(1,278*2048)
                with self.assertRaises(ValueError):reader.validate_probe_state(saved,receipt,c,outer)
                saved['binding']['prefix_tokens']=5
                saved['binding_sha256']=receipt['binding_sha256']=reader.digest(saved['binding'])
                with self.assertRaises(ValueError):reader.validate_probe_state(saved,receipt,c,outer)

    def test_control_stages_versioned_worker_and_sets_completed_prefix(self):
        class CapturedLaunch(Exception):pass
        for outer,prefix in [(100,20),(150,40),(200,80)]:
            with tempfile.TemporaryDirectory()as tmp:
                base=Path(tmp);prep=base/'preparation';prep.mkdir();(base/'repo').mkdir()
                (base/'repo/frozen-marker').write_text('unchanged')
                joint=base/'outers'/('outer-%06d'%outer);joint.mkdir(parents=True);(joint/'complete.json').write_text('{}')
                c=config();raw=json.dumps(c).encode();(prep/'launch-config.json').write_bytes(raw)
                arms={'B':dict(base=str(base),run_id='test-b',commit='frozen',config_sha256=hashlib.sha256(raw).hexdigest())}
                captured={}
                def launch(command,**kw):
                    captured.update(command=command,**kw);raise CapturedLaunch()
                script=remote_inner(arms,outer,'start');compile(script,'remote','exec')
                with patch('subprocess.Popen',side_effect=launch),self.assertRaises(CapturedLaunch):exec(script,{})
                self.assertEqual(captured['env']['CUDA_VISIBLE_DEVICES'],'')
                self.assertEqual(captured['env']['SEARCH_G1_B_PREFIX_TOKENS'],str(prefix))
                self.assertEqual(captured['command'][1],'-c')
                self.assertIn('._normal_checkpoint_control',captured['command'][2])
                workers=list((prep/'normal-savepoint-tools').glob('*/ab_checkpoint_once.py'))
                self.assertEqual(len(workers),1)
                self.assertEqual(hashlib.sha256(workers[0].read_bytes()).hexdigest(),workers[0].parent.name)
                self.assertEqual(list((base/'repo').iterdir()),[base/'repo/frozen-marker'])
                intent=json.loads((prep/('ab-cp%d-cpu-control-intent.json'%outer)).read_text())
                self.assertFalse(intent['frozen_repo_changed'])
                self.assertEqual(intent['checkpoint_prefix_tokens'],prefix)


if __name__=='__main__':unittest.main()
