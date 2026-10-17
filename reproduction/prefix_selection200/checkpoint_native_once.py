"""Normal CPU/full-file checkpoint readback for actor-hidden periodic probes.

Not a training gate, no GPU forward, no repeated existing proof, no backup claim.
Supports the actual latest complete outer as well as normal milestones.
"""
import argparse
import gc
import json
import os
from pathlib import Path

from .periodic import probe_boundary, is_probe_update
from .checkpoint_cpu_once import (
    read, need, digest, owned, hash_file, load_torch_cpu, validate_actor_state,
    head_shapes, tensor_cpu, validate_adam, validate_rng,
)


def actual_selection_binding(base, outer, joint, driver):
    """Use immutable scoring receipts; report stale legacy driver auxiliaries."""
    base = Path(base)
    root = base / 'outers' / ('outer-%06d' % outer)
    before = probe_boundary(outer - 1)
    if before['last_update_outer']:
        previous = base / 'outers' / ('outer-%06d' % before['last_update_outer'])
        expected = read(previous / 'probe-update/metrics.json')['checkpoint']
    else:
        expected = read(base / 'probe/initialization-complete.json')['receipt']
    selection = read(root / 'probe-score/selection.json')
    manifest = read(root / 'probe-score/manifest.json')
    complete = read(root / 'probe-score/complete.json')
    selection_sha = hash_file(root / 'probe-score/selection.json')['sha256']
    manifest_sha = hash_file(root / 'probe-score/manifest.json')['sha256']
    need(expected['completed_outer'] == before['last_update_outer']
         and expected['actual_adam_steps_per_rank'] == [before['actual_adam_per_rank']] * 8,
         'Original selected probe clock differs')
    need(selection_sha == complete['selection_sha256'] == manifest['selection_sha256']
         == joint['selection_sha256'] == joint['rollout']['selection_sha256']
         and manifest_sha == complete['manifest_sha256']
         and selection['run_id'] == manifest['run_id'] == complete['run_id'] == joint['run_id']
         and selection['outer'] == manifest['outer'] == complete['outer'] == outer
         and selection['actor_version'] == manifest['actor_version'] == complete['actor_version']
         == joint['actor']['sampling_actor_version']
         and selection['probe_before_checkpoint_sha256'] == manifest['probe_before_checkpoint_sha256']
         == expected['sha256'] and selection['outcome_labels_used'] is False,
         'Actual pre-update selection/probe/actor binding differs')
    auxiliary = driver['prefix_probe_receipt']['value']
    return dict(actual_probe=expected, selection_sha256=selection_sha, manifest_sha256=manifest_sha,
                driver_auxiliary_probe_matches_selection=auxiliary == expected,
                driver_auxiliary_probe_record=driver['prefix_probe_receipt'],
                actual_binding_source='original_immutable_selection_manifest_complete_and_probe_receipt',
                original_driver_modified=False)


def inspect(base,outer):
    import torch
    base=Path(base);config=read(base/'preparation/launch-config.json')
    proof=base/'preparation'/('joint-native-%04d-independent-full-sha-cpu.json'%outer)
    if proof.exists():return dict(status='EXISTING_CPU_PROOF_NO_REPLAY',proof=str(proof),sha256=hash_file(proof)['sha256'])
    root=base/'outers'/('outer-%06d'%outer)
    if not (root/'complete.json').exists():return dict(status='PENDING_NO_COMPLETE_CHECKPOINT',outer=outer)
    joint=read(root/'complete.json');actor=joint['actor'];probe=joint['probe'];boundary=probe_boundary(outer)
    need(config['actor_hidden_native'] and joint['run_id']==config['run_id'] and joint['outer']==outer,
         'Inspect the current native-actor run only')
    need(actor['completed_outer']==outer and actor['actual_adam_steps_per_rank']==[20*outer]*8
         and actor['scheduler_completed_outer']==outer,'Actual actor clock differs')
    need(probe['completed_outer']==boundary['last_update_outer']
         and probe['actual_adam_steps_per_rank']==[boundary['actual_adam_per_rank']]*8,
         'Probe may advance only100 writes at20-outer boundaries')
    files=[];ranks=[]
    for name,item in actor['hf_files'].items():
        files.append(hash_file(owned(base,Path(actor['hf_model_dir'])/name),item))
    for rank,item in enumerate(actor['rank_states']):
        state,meta=load_torch_cpu(owned(base,item['path']),item)
        ranks.append(validate_actor_state(state,actor,config,outer,rank));files.append(meta)
        del state;gc.collect()
    driver,meta=load_torch_cpu(owned(base,actor['driver_state']['path']),actor['driver_state']);files.append(meta)
    selection_binding=actual_selection_binding(base,outer,joint,driver)
    del driver;gc.collect()
    head,meta=load_torch_cpu(owned(base,joint['probe_snapshot']),probe);files.append(meta)
    need(head['run_id']==config['run_id'] and head['completed_outer']==boundary['last_update_outer']
         and head['binding_sha256']==digest(head['binding'])==probe['binding_sha256'],
         'Probe checkpoint version/binding differs')
    rate=config['probe_learning_rate']
    b=head['binding'];need(b['representation_source']=='current_sampling_actor_native_layer27_postblock'
         and b['independent_encoder_forward'] is False and b['probe_update_period']==20
         and b['updates_per_period']==100 and b['learning_rate']==rate,'Wrong periodic native probe recipe')
    model=head['model'];shapes=head_shapes();need(set(model)==set(shapes),'Missing original probe tensors')
    for name,value in model.items():tensor_cpu(value,dtype=torch.float32,shape=shapes[name],finite=True)
    steps=boundary['actual_adam_per_rank']
    if steps:
        adam=validate_adam(head['optimizer'],[v.shape for v in model.values()],steps,rate,weight_decay=0.)
        need(len(head['rank_rng'])==8,'Missing periodic probe rank RNG')
        for rank,rng in enumerate(head['rank_rng']):
            need(rng['rank']==rank and rng['actual_adam_steps']==steps,'Wrong periodic RNG clock');validate_rng(rng)
    else:
        need(head['optimizer'] is None and head['rank_rng'] is None,'No probe optimizer/RNG stream before first update20')
        adam=0
    del head;gc.collect()
    pool=read(owned(base,joint['pool_snapshot']));binding=read(base/'question-pool/binding.json')
    size=binding['eligible_questions']
    need(sorted(pool['order'])==list(range(size)) and pool['next_outer']==outer+1
         and pool['consumed_candidates']==5120*outer and pool['cycle']*size+pool['cursor']==5120*outer,
         'Saved question pool/permutation/cycle/cursor differs from completed outer')
    files.append(hash_file(owned(base,joint['pool_snapshot'])))
    grounding_lifecycle = None
    if config.get('grounding_refresh_enabled'):
        from .grounding import validate_joint, recovery_state, expected_round
        used = validate_joint(config, joint)
        effective = recovery_state(base, config, joint)
        assets = {}
        metadata = {joint['grounding_snapshot']}
        for state in (used, effective):
            path = Path(state['manifest_path']); manifest = read(path)
            metadata.add(str(path))
            if state.get('adoption_receipt'):
                metadata.add(state['adoption_receipt'])
            fitted = path.parent / 'fit-complete.json'
            if fitted.is_file(): metadata.add(str(fitted))
            for name, expected_sha in state['encoder_sha256'].items():
                file = Path(state['encoder']) / name
                assets[str(file)] = dict(path=str(file), bytes=file.stat().st_size, sha256=expected_sha,
                    sha256_source='actual_saved_grounding_manifest_producer_not_independent_cpu_read')
            for key, name in (('boundary', 'boundary_probe.pkl'), ('commit', 'commit_probe.pkl')):
                file = Path(state[key]); expected_sha = state['head_sha256'][name]
                item = dict(path=str(file), bytes=file.stat().st_size, sha256=expected_sha,
                    sha256_source='actual_saved_grounding_manifest_producer_not_independent_cpu_read')
                assets[str(file)] = item
        refresh = root / 'grounding-refresh.json'
        if refresh.is_file(): metadata.add(str(refresh))
        calibration = Path(config['grounding_calibration_questions'])
        assets[str(calibration)] = dict(path=str(calibration), bytes=calibration.stat().st_size,
            sha256=config['grounding_calibration_sha256'],
            sha256_source='bound_calibration_source_not_independent_cpu_read')
        existing = {item['path'] for item in files}
        # Read the actual small state/manifest/head bytes; do not repeatedly
        # hash the large immutable evaluator encoder already saved by actor or
        # retained as the separately backed-up official/fixed recovery asset.
        for path in sorted(metadata | {state[key] for state in (used, effective) for key in ('boundary', 'commit')}):
            if path not in existing:
                files.append(hash_file(Path(path), assets.get(path)))
        grounding_lifecycle = dict(used=used, effective_for_next_actor=effective,
            refit_pending_before_next_actor=(effective['round'] != expected_round(outer)
                                             or 'adoption_receipt' not in effective),
            assets=list(assets.values()), encoder_semantic_reload_performed=False)
    result=dict(status='NATIVE_ACTOR_PERIODIC_PROBE_JOINT_FULL_SHA_CPU_PASS',run_id=config['run_id'],outer=outer,
        actor_actual_adam_steps_per_rank=[20*outer]*8,probe_actual_adam_steps_per_rank=[steps]*8,
        probe_last_update_outer=boundary['last_update_outer'],actor_rank_cpu_states=ranks,
        probe_fp32_tensors=14,probe_fp32_adam_states=adam,files=files,
        pool_next_outer=pool['next_outer'],pool_consumed_candidates=pool['consumed_candidates'],
        actual_selection_binding=selection_binding,
        gpu_forward_performed=False,cuda_rng_applied=False,independent_backup_performed=False,
        grounding_lifecycle=grounding_lifecycle)
    with proof.open('x') as f:json.dump(result,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    return dict(status=result['status'],proof=str(proof),sha256=hash_file(proof)['sha256'],outer=outer,
                actor_actual_adam_steps_per_rank=[20*outer]*8,probe_actual_adam_steps_per_rank=[steps]*8)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--outer',type=int,required=True)
    a=p.parse_args();print(json.dumps(inspect(a.base,a.outer)))


if __name__=='__main__':main()
