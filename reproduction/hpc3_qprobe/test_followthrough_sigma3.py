"""Fresh-run routing and terminal/once guards; CPU fixtures never prove CUDA."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from reproduction.hpc3_qprobe import followthrough_sigma3 as worker
from reproduction.hpc3_qprobe.sigma3_experiment import LOSS

class Sigma3Followthrough(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.top=Path(self.temp.name).resolve()
        self.base=self.top/'base';self.root=self.top/'root';self.directory=self.root/'followthrough/721485'
        (self.base/'preparation').mkdir(parents=True);self.directory.mkdir(parents=True)
        self.settings=dict(target_successful_outers=200,fixed_validation=list(worker.control.STEPS),no_update201=True)
        (self.base/'preparation/sigma3-experiment.json').write_text(json.dumps(self.settings))
        self.run=dict(run_id=worker.RUN,source_commit=worker.PRODUCER,question_probe=dict(LOSS,head_hidden_size=64))
        self.binding=dict(producer_job=worker.JOB,producer_commit=worker.PRODUCER,original_failed_job='721449',
            recovery_submission_sha256='a',recovery_claim_sha256='b',preserved_pre_model_failure_sha256='c')
        self.patches=[]
        for target,key,value in ((worker,'BASE',self.base),(worker,'ROOT',self.root),(worker,'DIRECTORY',self.directory),(worker.control,'OWNER',os.getuid())):
            p=patch.object(target,key,value);p.start();self.patches.append(p)
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()
    def binding_call(self,run=None,binding=None):
        with patch.object(worker.control,'producer',return_value=(run or self.run,binding or self.binding,'fixture')),\
             patch('reproduction.hpc3_qprobe.sigma3_experiment.validate_settings',return_value=self.settings):
            return worker.fixed_binding()
    def test_fresh_loss_routes_exact_new_producer_and_preserved_failure(self):
        value=self.binding_call()
        self.assertEqual(value['producer_job'],'721485');self.assertIn('sigma3_settings_sha256',value)
        cases=[]
        for key,value in (('run_id','old-run'),('source_commit','old-source'),('full_state_restore',{'completed_outer':5})):
            r=copy.deepcopy(self.run);r[key]=value;cases.append((r,self.binding))
        for key,value in (('loss_mode','count_moment'),('mean_loss_weight',1),('auxiliary_count_loss_weight',1),('head_hidden_size',128),('initial_head_checkpoint','old-head.pt')):
            r=copy.deepcopy(self.run);r['question_probe'][key]=value;cases.append((r,self.binding))
        for key,value in (('producer_job','719225'),('original_failed_job','719225'),('recovery_claim_sha256',None)):
            b=dict(self.binding);b[key]=value;cases.append((self.run,b))
        for r,b in cases:
            with self.subTest(run=r,binding=b),self.assertRaises(ValueError):self.binding_call(r,b)
    def test_live_prefix_does_not_submit_final_native_or_claim_archive(self):
        (self.base/'logs').mkdir();(self.base/'logs/run-manifest.json').write_text('{}');(self.base/'logs/native-allocation-admission.json').write_text('{}')
        with patch.object(worker,'accounting',return_value=dict(state='RUNNING',exit_code='0:0')),\
             patch.object(worker.control,'producer',return_value=(self.run,self.binding,'fixture')),\
             patch.object(worker.control,'launch_archive') as archive,\
             patch.object(worker.control,'start_audit') as native:
            value=worker.one_tick('published-helper',{})
        self.assertEqual(value['phase'],'ACTUAL_PRODUCER_OR_ARCHIVE_WAIT');archive.assert_not_called();native.assert_not_called()
    def test_real_failure_preserved_without_restarting_or_native_submit(self):
        with patch.object(worker,'accounting',return_value=dict(state='FAILED',exit_code='1:0')),\
             patch.object(worker.control,'launch_archive') as archive,\
             patch.object(worker.control,'start_audit') as native:
            with self.assertRaisesRegex(ValueError,'ACTUAL_PRODUCER_FAILED'):worker.one_tick('published-helper',{})
        archive.assert_not_called();native.assert_not_called();self.assertTrue((self.directory/'producer-terminal-failure.json').is_file())

if __name__=='__main__':unittest.main()
