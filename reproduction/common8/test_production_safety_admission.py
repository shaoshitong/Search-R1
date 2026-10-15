"""CPU integrity/rule tests only; no native calibration or healthy artifact."""
import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.common8 import production_safety_admission as admission
from reproduction.common8.policy_calibration_plan import freeze_rule
from reproduction.common8 import test_calibration_evaluation as measurement_fixture


def health_fixtures():
    rows,ids,sources=measurement_fixture.canonical_rows()
    result=measurement_fixture.aggregate(rows,ids,sources)
    discovery=dict(evaluation_before=copy.deepcopy(result),evaluation_after=copy.deepcopy(result))
    confirmation=copy.deepcopy(discovery)
    rule=freeze_rule(dict(role='policy_safety_health_rule',provenance={'scope':'CPU rule fixture only'},
        criteria=dict(method='paired_fixed512_absolute_allowed_drop',
                      allowed_drop={'weighted_em':0.,'weighted_f1':0.,'valid_action_rate':0.})))
    return rule,discovery,confirmation


class ProductionSafetyAdmissionTests(unittest.TestCase):
    def test_explicit_rule_reads_values_and_never_selects_thresholds(self):
        rule,discovery,confirmation=health_fixtures()
        original=copy.deepcopy((rule,discovery,confirmation))
        decision=admission.evaluate_health(rule,discovery,confirmation)
        self.assertTrue(decision['accepted']);self.assertEqual(decision['rule_sha256'],rule['sha256'])
        self.assertEqual((rule,discovery,confirmation),original)
        self.assertIn('recognized actions',decision['action_definition'])

    def test_both_child_segments_must_pass_the_same_frozen_rule(self):
        for label in ('discovery','confirmation'):
            rule,discovery,confirmation=health_fixtures()
            segment=discovery if label=='discovery' else confirmation
            segment['evaluation_after']['weighted_em']-=.01
            with self.subTest(label=label),self.assertRaises(admission.AdmissionError):
                admission.evaluate_health(rule,discovery,confirmation)

    def test_missing_or_opaque_rule_cannot_become_healthy(self):
        _,discovery,confirmation=health_fixtures()
        for criteria in ({'definition':'opaque'},
                         {'method':'paired_fixed512_absolute_allowed_drop','allowed_drop':{'weighted_em':0.,'weighted_f1':0.}},
                         {'method':'choose_from_confirmation','allowed_drop':{}},
                         {'method':'paired_fixed512_absolute_allowed_drop','allowed_drop':{'weighted_em':0.,'weighted_f1':0.,'valid_action_rate':0.,'unknown':0.}}):
            rule=freeze_rule(dict(role='policy_safety_health_rule',provenance={'scope':'fixture'},criteria=criteria))
            with self.subTest(criteria=criteria),self.assertRaises((ValueError,KeyError)):
                admission.evaluate_health(rule,discovery,confirmation)

    def test_bool_negative_nan_inf_thresholds_and_modified_rule_hash_reject(self):
        for value in (True,-.01,float('nan'),float('inf')):
            rule,discovery,confirmation=health_fixtures()
            rule['document']['criteria']['allowed_drop']['weighted_em']=value
            try: rule=freeze_rule(rule['document'])
            except ValueError: pass
            with self.subTest(value=value),self.assertRaises(ValueError):
                admission.evaluate_health(rule,discovery,confirmation)
        rule,discovery,confirmation=health_fixtures();rule['sha256']='0'*64
        with self.assertRaises(ValueError): admission.evaluate_health(rule,discovery,confirmation)

    def test_explicit_minimums_have_no_inferred_default(self):
        rule,discovery,confirmation=health_fixtures()
        document=copy.deepcopy(rule['document']);document['criteria']['minimum_after']={'weighted_em':1.1}
        with self.assertRaises(ValueError): admission.evaluate_health(freeze_rule(document),discovery,confirmation)
        document['criteria']['minimum_after']={'weighted_em':0.}
        self.assertTrue(admission.evaluate_health(freeze_rule(document),discovery,confirmation)['accepted'])

    def test_pair_order_or_missing_question_is_rejected(self):
        for change in ('order','count'):
            rule,discovery,confirmation=health_fixtures()
            if change=='order': confirmation['evaluation_after']['question_ids'].reverse()
            else: confirmation['evaluation_after']['question_count']=511
            with self.subTest(change=change),self.assertRaises(ValueError):
                admission.evaluate_health(rule,discovery,confirmation)

    def test_receipt_original_bytes_sha_and_path_are_bound(self):
        with tempfile.TemporaryDirectory() as root:
            base=Path(root);path=base/'proof.json';content=b'{"scope":"CPU fixture"}';path.write_bytes(content)
            sha=hashlib.sha256(content).hexdigest()
            self.assertEqual(admission._receipt(base,'proof.json',sha)['scope'],'CPU fixture')
            path.write_bytes(content+b' ')
            with self.assertRaises(ValueError): admission._receipt(base,'proof.json',sha)
            with self.assertRaises(ValueError): admission._receipt(base,'../outside.json',sha)

    def test_produce_does_not_accept_free_cli_paths_or_child_parent_alias(self):
        with tempfile.TemporaryDirectory() as root:
            base=Path(root);(base/'logs').mkdir()
            (base/'logs/run-manifest.json').write_text(json.dumps(dict(policy_safety_required=True,
                policy_calibration_discovery_base=str(base/'bound-discovery'),policy_calibration_confirmation_base=str(base/'bound-confirmation'))))
            with self.assertRaises(ValueError): admission.produce(base,base/'other',base/'bound-confirmation',{})
            with self.assertRaises(ValueError): admission.produce(base,base,base/'bound-confirmation',{})
            self.assertFalse((base/'logs/policy-safety-admission-claim.json').exists())

    def test_cleanup_requires_terminal_stage_all_stopped_and_canonical_absent_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);(base/'logs').mkdir();run=dict(run_id='CPU-fixture-cleanup',policy_calibration_slurm_job_id='123')
            documents={'pipeline-status.json':dict(run_id=run['run_id'],stage='FINISHED_POLICY_CALIBRATION_MEASUREMENTS_AND_OWNED_CLEANUP'),
                       'cleanup-status.json':dict(run_id=run['run_id'],cleaned=True),
                       'process-ledger.json':dict(run_id=run['run_id'],jobs={name:dict(state='STOPPED',ledger={}) for name in ('trainer','retriever','probe-server')})}
            root=Path('/dev/shm/search-g1-common8-CPU-fixture-cleanup-123')
            for name,suffix in (('retrieval-staging-claim.json',''),('corpus-cache-claim.json','datasets-cache')):
                documents[name]=dict(run_id=run['run_id'],phase='CLEANED',slurm_job_id='123',uid=os.getuid(),root=str(root/suffix))
            for name,row in documents.items(): (base/'logs'/name).write_text(json.dumps(row))
            admission._cleanup(base,run,{},'fixture/')
            for name,field,value in (('pipeline-status.json','stage','FAILED'),('cleanup-status.json','cleaned',1),
                                    ('retrieval-staging-claim.json','root','/tmp/not-owned')):
                original=documents[name];changed=dict(original,**{field:value})
                (base/'logs'/name).write_text(json.dumps(changed))
                with self.subTest(name=name),self.assertRaises(ValueError): admission._cleanup(base,run,{},'fixture/')
                (base/'logs'/name).write_text(json.dumps(original))
            changed=copy.deepcopy(documents['process-ledger.json']);changed['jobs']['trainer']['state']='STARTED'
            (base/'logs/process-ledger.json').write_text(json.dumps(changed))
            with self.assertRaises(ValueError): admission._cleanup(base,run,{},'fixture/')

    def test_parent_probability_receipts_use_actual_writer_schema_and_precision(self):
        from reproduction.common8.test_model_smoke import ModelSmokeTests
        rows,ids,uuids=ModelSmokeTests().proofs()
        run=dict(run_id='cpu-only',profile='common8');receipts={}
        probability=dict(slurm_job_id='123',question_ids=ids,actual_cuda_runtime_uuids=uuids,resolved_config_sha256='same')
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);parent=base/'logs/policy-safety-admission-receipts/parent';parent.mkdir(parents=True)
            for phase in ('initial','coexistence'):
                proofs=[];hashes={}
                for rank,row in enumerate(rows):
                    original=dict(row,phase=phase,slurm_job_id='123',initial_parameters_finite=True)
                    path=parent/f'model-smoke-{phase}-rank-{rank}.json';path.write_text(json.dumps(original))
                    relative=path.relative_to(base).as_posix();receipts[relative]=admission.digest(path)
                    proofs.append(original);hashes[str(rank)]=receipts[relative]
                path=parent/f'model-smoke-{phase}-complete.json'
                path.write_text(json.dumps(dict(accepted=True,run_id='cpu-only',phase=phase,ranks=proofs,
                                               actual_grpo_updates=0,rank_file_sha256=hashes)))
                probability[phase+'_proof']=path.relative_to(base).as_posix()
                probability[phase+'_proof_sha256']=admission.digest(path)
            admission._verify_parent_probability(base,run,probability,receipts)
            path=parent/'model-smoke-initial-rank-7.json';original=json.loads(path.read_text())
            for field,value in (('actual_optimizer_writes',1),('initial_parameters_finite',False),
                                ('resolved_config_sha256','changed'),('rank',True)):
                changed=dict(original,**{field:value});path.write_text(json.dumps(changed))
                # Original raw SHA must reject even a plausible JSON schema.
                with self.subTest(field=field),self.assertRaises((ValueError,AssertionError)):
                    admission._verify_parent_probability(base,run,probability,receipts)
                path.write_text(json.dumps(original))

    def test_actual_group_receipts_preserve_duplicate_parent_ids_and_five_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);(base/'logs').mkdir();sha='1'*64
            run=dict(run_id='cpu-only',input_sha256={'train.parquet':'2'*64})
            parents=[dict(data_source='nq' if i<256 else 'hotpotqa',upstream_group_sha256=f'{i//2:064x}',
                          tensor_sha256={name:f'{i:064x}' for name in ('input_ids','attention_mask','position_ids')}) for i in range(512)]
            repeated=[row['upstream_group_sha256'] for row in parents for _ in range(5)]
            inputs=dict(run_id=run['run_id'],plan_sha256=sha,actual_trajectory_count=2560,production_admission=False,
                        actual_parent_count=512,n_agent=5,parents=parents,train_sha256='2'*64,repeated_group_sha256=repeated)
            groups=dict(run_id=run['run_id'],plan_sha256=sha,actual_trajectory_count=2560,production_admission=False,
                        upstream_group_multiset_unchanged=True,balanced_group_sha256=list(reversed(repeated)))
            paths={name:base/'logs'/('policy-calibration-training-'+suffix+'.json') for name,suffix in (('parents','inputs'),('groups','groups'))}
            paths['parents'].write_text(json.dumps(inputs));paths['groups'].write_text(json.dumps(groups))
            segment=dict(actual_training_inputs=inputs,actual_training_groups=groups,
                         training_input_proof_sha256={name:admission.digest(path) for name,path in paths.items()})
            admission._training_inputs(base,run,sha,segment,{},'')
            groups['balanced_group_sha256'][0]='f'*64;paths['groups'].write_text(json.dumps(groups))
            segment['training_input_proof_sha256']['groups']=admission.digest(paths['groups'])
            with self.assertRaises(ValueError): admission._training_inputs(base,run,sha,segment,{},'')

    def test_actual_adoption_writer_lineage_and_damaged_origin_reject(self):
        from reproduction.common8.test_adopt_initial_probes import AdoptionTests
        fixture=AdoptionTests();fixture.setUp()
        try:
            receipt=fixture.adopt()  # Actual owned byte-copy writer; only native boundary is a CPU fixture.
            run=copy.deepcopy(fixture.run);run['initial_probe_version']=receipt['version']
            live=json.loads((fixture.base/'artifacts/live/manifest.json').read_text())
            source_segment=json.loads((fixture.source/'logs/policy-calibration-segment.json').read_text())
            names=('logs/run-manifest.json','logs/policy-safety-calibration-plan.json','logs/policy-calibration-segment.json',
                   'logs/initial-probes-accepted.json','artifacts/live/manifest.json')
            hashes={name:admission.digest(fixture.source/name) for name in names}
            archive=fixture.round/'version-manifest.json';hashes[archive.relative_to(fixture.source).as_posix()]=admission.digest(archive)
            for name,key in (('commit_probe.pkl','commit'),('boundary_probe.pkl','boundary')):
                hashes[Path(fixture.live[key]).relative_to(fixture.source).as_posix()]=fixture.live['head_sha256'][name]
            def check(value,current_live=live):
                admission._adoption_lineage(run,'101',value,current_live,fixture.source_run,fixture.plan,source_segment,
                    fixture.live,hashes,admission.digest(archive),admission.digest(fixture.base/'artifacts/live/manifest.json'))
            check(receipt)
            for field,value in (('version','discovery:0:'+'a'*32),('live_manifest_sha256','0'*64),
                                ('slurm_job_id','100'),('production_admission',True),('adopted',1)):
                altered=copy.deepcopy(receipt);altered[field]=value
                with self.subTest(field=field),self.assertRaises(ValueError):check(altered)
            for key in ('plan_sha256','segment_sha256','archive_sha256','accepted_manifest_sha256','version','run_id'):
                altered=copy.deepcopy(receipt);altered['origin'][key]='changed'
                with self.subTest(origin=key),self.assertRaises(ValueError):check(altered)
            altered=copy.deepcopy(receipt);altered['source_file_sha256']['logs/policy-calibration-segment.json']='0'*64
            with self.assertRaises(ValueError):check(altered)
            altered=copy.deepcopy(live);altered['head_sha256']['commit_probe.pkl']='0'*64
            with self.assertRaises(ValueError):check(receipt,altered)
        finally:fixture.doCleanups()

    def test_small_real_checkpoint_finite_canonical_and_actual_delta(self):
        try:
            import torch
            from safetensors.torch import save_file
            from verl.utils.checkpoint_integrity import validate_saved_checkpoint
        except ModuleNotFoundError as failure:
            self.skipTest('Actual checkpoint dependencies unavailable locally: '+str(failure))
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);initial=base/'official-fixture';checkpoint=base/'checkpoints/actor/global_step_1'
            initial.mkdir();checkpoint.mkdir(parents=True)
            weight=torch.ones(2,3);key='model.embed_tokens.weight';index=dict(weight_map={key:'weights.safetensors'})
            for path,value in ((initial,weight),(checkpoint,weight+1e-6)):
                save_file({key:value},str(path/'weights.safetensors'))
                (path/'model.safetensors.index.json').write_text(json.dumps(index))
                (path/'config.json').write_text(json.dumps({'model_type':'qwen2','tie_word_embeddings':True,'architectures':['Qwen2ForCausalLM']}))
            count=validate_saved_checkpoint(checkpoint,initial,0.)
            complete=dict(run_id='cpu-only',step=1,finite_all_saved_tensors=True,validated_tensor_count=count,
                files_sha256={name:admission.digest(checkpoint/name) for name in ('weights.safetensors','model.safetensors.index.json','config.json')})
            (checkpoint/'complete.json').write_text(json.dumps(complete))
            run=dict(run_id='cpu-only',checkpoint_dir=str(base/'checkpoints'),initial_model=str(initial),started_unix=0.)
            segment=dict(checkpoint=complete,changed_parameters=[key])
            admission._checkpoint(base,run,segment,{},'fixture/')
            save_file({key:torch.full_like(weight,float('nan'))},str(checkpoint/'weights.safetensors'))
            with self.assertRaises(AssertionError): admission._checkpoint(base,run,segment,{},'fixture/')


if __name__=='__main__': unittest.main()
