"""Isolated CPU reconstruction of historical probe metrics; never training control.

Recomputes only the tiny saved probe in private process memory, from the actual
warmup model/Adam/RNG and original per-round features/labels. Every logged model
version/loss must match. At 5/50 the complete original state must be tensor-exact;
otherwise results are explicitly rejected, never calibrated to saved weights.
No checkpoints, actor model, rollout, or experiment files are written.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from search_g1.qprobe_model import OnlineQProbe
from reproduction.hpc3_qprobe.analyze_sigma3_fit_readonly import BASE, RUN, PRODUCER, LOSS, metrics, read_json, read_json_lines


class NotExact(ValueError): pass


def need(ok, message):
    if not ok: raise NotExact(message)


def exact_tree(actual, expected, path='state'):
    if isinstance(expected, torch.Tensor):
        need(isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype and
            actual.shape == expected.shape and torch.equal(actual.cpu(), expected.cpu()), path+' tensor differs')
    elif isinstance(expected, dict):
        need(isinstance(actual, dict) and actual.keys() == expected.keys(), path+' keys differ')
        for key in expected: exact_tree(actual[key], expected[key], path+'.'+str(key))
    elif isinstance(expected, (list, tuple)):
        need(type(actual) == type(expected) and len(actual) == len(expected), path+' sequence differs')
        for i, value in enumerate(expected): exact_tree(actual[i], value, path+'.'+str(i))
    else: need(type(actual) == type(expected) and actual == expected, path+' value differs')


def read_tensor_state(path, weights_only):
    path = Path(path)
    need(path.resolve() == path and not path.is_symlink() and path.stat().st_uid == 14090,
        'unsafe actual probe artifact path')
    raw = path.read_bytes()
    need(len(raw) <= 100*2**20, 'bounded probe artifact is unexpectedly large')
    state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=weights_only)
    proof = dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    need(hashlib.sha256(path.read_bytes()).hexdigest() == proof['sha256'], 'artifact changed during read')
    return state, proof


def summarize(probe, outer):
    counts = np.array([x['correct_count'] for x in probe.replay_labels])
    prediction = probe.predict(probe.replay_features)['probabilities'].numpy()
    weights = np.array([2**(-(outer-x['source_outer'])/probe.config.replay_half_life_outer)
        for x in probe.replay_labels])
    current = np.array([x['source_outer'] == outer for x in probe.replay_labels])
    need(int(current.sum()) == (5120 if outer == 0 else 512), 'current fit cohort differs')
    return dict(outer=outer, phase='POST_FIT_OFFLINE_RECONSTRUCTION', head_version=probe.version,
        replay_unweighted=metrics(prediction, counts),
        replay_age_weighted=metrics(prediction, counts, weights),
        current_cohort=metrics(prediction[current], counts[current]))


def reconstruct(cutoff, threads, start_outer=0):
    torch.set_num_threads(threads)
    import search_g1.qprobe_model as original_math
    original_path = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014/source')/PRODUCER/'search_g1/qprobe_model.py'
    need(Path(original_math.__file__).read_bytes() == original_path.read_bytes(),
        'diagnostic probe mathematics differs from actual training source')
    manifest, manifest_proof = read_json(BASE/'logs/run-manifest.json')
    need(manifest['run_id'] == RUN and manifest['source_commit'] == PRODUCER and
        manifest['training_world_size'] == 4, 'actual producer differs')
    ready, ready_proof = read_json(BASE/'artifacts/question-probe/teacher-warmup-ready.json')
    anchors=[]
    if start_outer:
        need(start_outer==50 and cutoff>=start_outer,'only actual saved head50 is an allowed tail boundary')
        path=BASE/'checkpoints/search-g1-common8/actor/global_step_50/training_state/question-probe-state.pt.head.pt'
        driver,driver_proof=read_tensor_state(path.with_name('question-probe-state.pt'),False)
        need(driver['head_file']==path.name and driver['completed_outer']==50 and
            driver['actor_snapshot_id']==RUN+':50','actual tail checkpoint binding differs')
        probe=OnlineQProbe.load(path,expected_actor_snapshot_id=RUN+':50',
            expected_checkpoint_sha=driver['head_sha256'])
        saved,proof=read_tensor_state(path,True);exact_tree(probe.state_dict(),saved)
        anchors.append(dict(outer=50,status='LOADED_ORIGINAL_FULL_STATE_TENSOR_EXACT',head_file=proof,binding_file=driver_proof))
    else:
        probe = OnlineQProbe.load(ready['head_checkpoint']['path'],
            expected_actor_snapshot_id=RUN+':0', expected_checkpoint_sha=ready['head_checkpoint']['sha256'])
    need(all(getattr(probe.config,k) == v for k,v in LOSS.items()), 'actual sigma3 loss differs')
    updates, updates_proof = read_json_lines(BASE/'logs/actual-updates.jsonl')
    need(len(updates) >= cutoff and [u['step'] for u in updates] == list(range(1,len(updates)+1)),
        'actual completed sequence differs')
    rows = [summarize(probe,start_outer)]
    current_step = start_outer
    try:
        for update in updates[start_outer:cutoff]:
            current_step = step = update['step']; aid = update['attempt_id']
            directory = BASE/'artifacts/question-probe/attempts'/aid
            head, head_proof = read_json(BASE/'artifacts/question-probe/completed'/(aid+'.json'))
            selection, selection_proof = read_json(directory/'selection.json')
            rollout, rollout_proof = read_json(directory/'rollout-audit.json')
            need(head['outer'] == selection['step'] == rollout['step'] == step and
                head['attempt_id'] == selection['attempt_id'] == rollout['attempt_id'] == aid and
                rollout['run_id'] == RUN and rollout['source_commit'] == PRODUCER,
                'attempt/run/source binding differs')
            need(probe.version == head['prediction_version'] == selection['head_version'],
                'original pre-fit version differs')
            selected = {q['ordinal']:q for q in selection['questions'] if q['selected']}
            groups = rollout['groups']
            need(len(groups) == len(selected) == 512 and len({g['ordinal'] for g in groups}) == 512,
                'actual 512 cohort differs')
            before, feature_proof = read_tensor_state(directory/'before-generation.pt',False)
            need(before['head_version'] == probe.version and before['identity']['attempt_id'] == aid and
                before['identity']['policy_snapshot_id'] == head['label_policy_snapshot_id'],
                'original feature/actor boundary differs')
            features = before['selected_features']; del before
            need(features.shape == (512,probe.feature_dim), 'actual feature shape differs')
            expected_prob = np.array([selected[g['ordinal']]['probabilities'] for g in groups])
            actual_prob = probe.predict(features)['probabilities'].numpy()
            need(np.allclose(actual_prob,expected_prob,atol=3e-6,rtol=3e-6),
                'original selected feature order/prediction differs')
            labels = []
            for g in groups:
                q = selected[g['ordinal']]
                need(g['question_id'] == q['qid'] and g['content_sha256'] == q['content_sha256'] and
                    g['source'] == q['selection_source'], 'original selected identity differs')
                labels.append(dict(complete=True,qid=q['qid'],question_hash=q['question_hash'],
                    strict_em=g['strict_correctness'],selection_source=g['source']))
            fit = probe.update_completed_outer(step,labels,features,
                policy_snapshot_id=head['label_policy_snapshot_id'],
                sampling_contract_sha=rollout['sampling_contract_sha'],
                strict_scoring_contract_sha=rollout['strict_scoring_contract_sha'],
                retriever_version=rollout['retriever_version'], run_id=RUN,attempt_id=aid)
            need(probe.version == head['head_version'], 'recomputed post-fit model hash differs')
            for key in ('replay_groups','head_optimizer_updates','update_count','completed_outer','version'):
                need(fit[key] == head['head_update'][key], 'recomputed fit counter/version differs: '+key)
            for key in ('replay_weighted_loss','count_ce','mean_mse','std_mse','sigma_ce'):
                need(abs(fit[key]-head['head_update'][key]) < 2e-6, 'recomputed fit loss differs: '+key)
            if step in (5,50,100,150,200):
                path = BASE/('checkpoints/search-g1-common8/actor/global_step_%d/training_state/' % step)/'question-probe-state.pt.head.pt'
                saved, proof = read_tensor_state(path,True)
                # save() binds only this metadata after taking a state snapshot.
                probe.actor_snapshot_id = RUN+':'+str(step)
                exact_tree(probe.state_dict(),saved)
                anchors.append(dict(outer=step,status='FULL_MODEL_ADAM_RNG_REPLAY_PROTOCOL_TENSOR_EXACT',head_file=proof))
            row = summarize(probe,step)
            row.update(original_head_file=head_proof, original_features=feature_proof,
                original_selection=selection_proof, original_rollout=rollout_proof,
                logged_fit=head['head_update'], every_round_model_version_equal=True)
            rows.append(row)
        last_anchor=max((a['outer'] for a in anchors),default=0)
        need(last_anchor >= min(cutoff,50), 'reconstruction lacks a complete saved-state anchor')
        return dict(status='VERIFIED_OFFLINE_RECONSTRUCTION' if last_anchor==cutoff else
            'VERSION_LOSS_VERIFIED_RECONSTRUCTION_TAIL_AFTER_FULL_STATE_ANCHOR',
            last_full_state_anchor=last_anchor, tail_verified_by='every original model version and all logged fit components',
            cutoff=cutoff,start_outer=start_outer,cpu_threads=threads,
            anchors=anchors,per_outer_post_fit=rows,manifest_file=manifest_proof,
            warmup_file=ready_proof,updates_file=updates_proof)
    except NotExact as exc:
        return dict(status='RECONSTRUCTION_NOT_EXACT_DO_NOT_USE_METRICS',failed_outer=current_step,
            reason=str(exc),cutoff=cutoff,cpu_threads=threads,anchors=anchors,
            verified_prefix_before_failure=len(rows)-1,per_outer_post_fit=[])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cutoff',type=int,required=True)
    parser.add_argument('--start-outer',type=int,choices=(0,50),default=0)
    parser.add_argument('--threads',type=int,choices=(1,8),default=1);args=parser.parse_args()
    need(5 <= args.cutoff <= 200 and os.getuid() == 14090 and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU owner/device/cutoff guard differs')
    value=reconstruct(args.cutoff,args.threads,args.start_outer)
    value.update(epoch=time.time(),run_id=RUN,training_actions=0,actor_loads=0,rollout_generations=0,
        scope='Historical tiny probe reconstructed in isolated CPU process memory; actual training and all experiment files unchanged')
    print(json.dumps(value,allow_nan=False),flush=True)


if __name__ == '__main__':main()
