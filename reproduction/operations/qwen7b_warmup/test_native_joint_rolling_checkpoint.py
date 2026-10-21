"""Small CPU full-state transaction tests; no real model/checkpoint is loaded."""
import copy,gc,json,random,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import probe_ablation_train as u
from .balanced_hard_g10_sampler import BalancedHardG10Sampler
from . import native_joint_rolling_checkpoint as c


def setup():
    random.seed(71);np.random.seed(71);torch.manual_seed(71)
    model=torch.nn.Sequential(torch.nn.Dropout(.2),torch.nn.Linear(2,1,dtype=torch.float64))
    model.register_buffer('native_clock',torch.zeros(1,dtype=torch.float64),persistent=False)
    ema=copy.deepcopy(model).eval().requires_grad_(False);opt=torch.optim.Adam(model.parameters(),lr=5e-6)
    opt.zero_grad();model(torch.ones(2,2,dtype=torch.float64)).sum().backward();opt.step()
    for state in opt.state.values():state['step'].fill_(14418)
    groups=[dict(qid='q%03d'%i,question_hash='h%03d'%i,trajectory_ids=['q%03d:t%d'%(i,j)for j in range(10)],labels=[int(j<k)for j in range(10)])for i,k in enumerate(list(range(11))*4)]
    sampler=BalancedHardG10Sampler(groups,batch_questions=8,steps_per_epoch=4)
    rng=u.rng_state(torch.device('cpu'));ranks=[dict(rank=i,rng=copy.deepcopy(rng))for i in range(4)]
    return model,ema,opt,sampler,ranks


def update(values):
    model,ema,opt,sampler,_=values;batch=sampler.next_batch();opt.zero_grad()
    model(torch.randn(3,2,dtype=torch.float64)).square().mean().backward();opt.step()
    with torch.no_grad():
        for e,p in zip(ema.parameters(),model.parameters()):e.mul_(.99).add_(p,alpha=.01)
        model.native_clock.add_(1);ema.native_clock.copy_(model.native_clock)
    sampler.observe(batch,[.1+float(torch.rand(()))for _ in batch['grouping']['groups']])


def blob(values):
    model,ema,opt,sampler,ranks=values;cursor=sampler.accepted_updates;state=sampler.state_dict()
    value=u.checkpoint_blob(model,ema,opt,torch.device('cpu'),cursor,dict(config_sha256='a'*64,matched_binding_sha256='b'*64),ranks)
    value.update(role=c.ROLE,phase='accepted_update_complete',source_steps=14418,sampler_state=state,sampler_state_sha256=u.digest(state),
        data_cursor=dict(cursor=cursor,sampler_binding=sampler.binding,sampler_state_sha256=u.digest(state)))
    return copy.deepcopy(value)


class RollingNativeCheckpointTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_full_restore_next_update_sampler_and_six_components(self):
        values=setup();update(values);saved=blob(values)
        with tempfile.TemporaryDirectory()as d:
            base=Path(d);receipt=c.save_native(base,saved);self.assertTrue(receipt['full_CPU_six_state_reload_passed'])
            self.assertEqual(Path(receipt['path']).stat().st_ino,(base/'checkpoints/rolling-latest.pt').stat().st_ino)
            loaded=torch.load(u.checked(receipt),map_location='cpu',mmap=True,weights_only=False)
            update(values);expected=u.state_hashes(*values[:3],torch.device('cpu'));expected_sampler=values[3].state_dict()
            u.restore_blob(loaded,*values[:3],torch.device('cpu'));values[3].load_state_dict(loaded['sampler_state']);update(values)
            self.assertEqual(u.state_hashes(*values[:3],torch.device('cpu')),expected);self.assertEqual(values[3].state_dict(),expected_sampler)
            self.assertEqual(u.steps(values[2]),14420)
            del loaded;gc.collect()

    def test_rolling_replacement_preserves_pins_and_only_one_rolling_body(self):
        values=setup();update(values)
        with tempfile.TemporaryDirectory()as d:
            base=Path(d);first=c.save_native(base,blob(values),pin=True);pinsha=first['sha256'];pinpath=Path(first['path'])
            update(values);second=c.save_native(base,blob(values));old=Path(second['path']);self.assertTrue(old.exists())
            update(values);third=c.save_native(base,blob(values));self.assertFalse(old.exists());self.assertEqual(u.sha(pinpath),pinsha)
            self.assertEqual(len(list((base/'checkpoints').glob('.rolling-accepted-cursor-*.pt'))),1)
            self.assertEqual(u.read(base/'latest.json'),third)
            self.assertEqual(c.save_native(base,blob(values)),third)
            update(values);last=c.save_native(base,blob(values),pin=True);self.assertTrue(pinpath.exists());self.assertFalse(Path(third['path']).exists());self.assertTrue(last['pinned'])

    def test_sampler_cursor_and_full_component_corruption_do_not_publish(self):
        values=setup();update(values)
        with tempfile.TemporaryDirectory()as d:
            base=Path(d);previous=c.save_native(base,blob(values));old=(base/'latest.json').read_bytes();update(values);good=blob(values)
            bad=copy.deepcopy(good);bad['sampler_state']['accepted_updates']-=1
            with self.assertRaisesRegex(ValueError,'sampler'):c.save_native(base,bad)
            bad=copy.deepcopy(good);bad['data_cursor']['sampler_binding']='wrong'
            with self.assertRaisesRegex(ValueError,'cursor'):c.save_native(base,bad)
            bad=copy.deepcopy(good);next(iter(bad['raw'].values())).add_(1)
            with self.assertRaisesRegex(ValueError,'hash'):c.save_native(base,bad)
            self.assertEqual((base/'latest.json').read_bytes(),old);u.checked(previous)

    def test_metadata_publish_failure_retains_old_valid_body_and_new_verified_state(self):
        values=setup();update(values)
        with tempfile.TemporaryDirectory()as d:
            base=Path(d);previous=c.save_native(base,blob(values));old=(base/'latest.json').read_bytes();update(values);state=blob(values);original=u.atomic
            def fail_latest(path,value):
                if Path(path)==base/'latest.json':raise OSError('synthetic final metadata interruption')
                return original(path,value)
            with patch.object(u,'atomic',side_effect=fail_latest):
                with self.assertRaises(OSError):c.save_native(base,state)
            self.assertEqual((base/'latest.json').read_bytes(),old);u.checked(previous)
            pending=u.read(base/'native-checkpoint-transaction.json');self.assertEqual(pending['phase'],'verified_new_body_before_publish');u.checked(pending['new_checkpoint'])
            self.assertEqual(pending['new_checkpoint']['cursor'],2);self.assertTrue(Path(previous['path']).exists())
            with self.assertRaisesRegex(ValueError,'Unresolved'):c.save_native(base,state)

if __name__=='__main__':unittest.main()
