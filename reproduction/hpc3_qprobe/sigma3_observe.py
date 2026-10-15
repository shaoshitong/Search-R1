"""Read-only observations of the new loss run; no training submission or restart."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from reproduction.hpc3_qprobe.sigma3_experiment import BASE, LOSS

PRODUCER='721449'
SOURCE='aca0e779e94ddc3fbd50e49c3d59e2e889fbcf12'


def document(path):
    path=Path(path)
    if not path.is_file():return None
    raw=path.read_bytes()
    return dict(value=json.loads(raw),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),path=str(path))


def rows(path):
    path=Path(path)
    if not path.is_file():return [],None
    raw=path.read_bytes();complete=raw if raw.endswith(b'\n') else raw[:raw.rfind(b'\n')+1]
    return [json.loads(line) for line in complete.splitlines()],dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
        incomplete_tail_ignored=len(raw)-len(complete),path=str(path))


def observe():
    if os.getuid()!=14090 or BASE.resolve()!=BASE:raise ValueError('SIGMA3_OBSERVER_OWNER_OR_ROOT_DIFFERS')
    out=dict(epoch=time.time(),base=str(BASE),job=PRODUCER,training_mutations=0,goal_complete=False)
    out['accounting']=subprocess.check_output(['sacct','-nP','-X','-j',PRODUCER,'--format=JobID,State,ExitCode,UID,AllocTRES,NodeList'],text=True,timeout=20)
    out['queue']=subprocess.check_output(['squeue','-h','-j',PRODUCER,'-o','%i|%T|%R'],text=True,timeout=20)
    settings=document(BASE/'preparation/sigma3-experiment.json')
    if not settings or settings['value']['head_overrides']!=LOSS:raise ValueError('SIGMA3_OBSERVED_LOSS_SETTINGS_DIFFERS')
    out['loss_settings']=settings
    for name in ('pipeline-status.json','storage-budget-admission.json','native-allocation-admission.json','cleanup-status.json'):
        d=document(BASE/'logs'/name)
        if d:out[name]=d
    m=document(BASE/'logs/run-manifest.json')
    if m:
        run=m['value']
        if run['source_commit']!=SOURCE or run.get('full_state_restore') or run['training_world_size']!=4:
            raise ValueError('SIGMA3_OBSERVED_ACTOR_OR_TOPOLOGY_DIFFERS')
        if any(run['question_probe'].get(k)!=v for k,v in LOSS.items()):raise ValueError('SIGMA3_ACTUAL_MANIFEST_LOSS_DIFFERS')
        out['run']=dict(run_id=run['run_id'],source_commit=SOURCE,training_world_size=4,
            actual_loss={k:run['question_probe'][k] for k in LOSS},manifest_sha256=m['sha256'])
    updates,proof=rows(BASE/'logs/actual-updates.jsonl')
    out['updates']=dict(count=len(updates),latest=updates[-1] if updates else None,proof=proof)
    out['rank_optimizer']={}
    for rank in range(4):
        rs,rp=rows(BASE/'logs'/('optimizer-rank-%d.jsonl'%rank))
        out['rank_optimizer'][str(rank)]=dict(rows=len(rs),latest=rs[-1] if rs else None,proof=rp)
    events=[]
    for p in (BASE/'artifacts/question-probe/completed').glob('*.json'):
        d=document(p);events.append(d)
    events.sort(key=lambda d:d['value']['outer'])
    out['head_events']=dict(count=len(events),latest=events[-1] if events else None)
    warm=document(BASE/'artifacts/question-probe/teacher-warmup-ready.json')
    if warm:
        w=warm['value'];metrics=w['head_metrics']
        expected=metrics['sigma_ce']+metrics['std_mse']+.1*metrics['count_ce']
        if abs(expected-metrics['replay_weighted_loss'])>2e-6:raise ValueError('SIGMA3_ACTUAL_WARMUP_LOSS_FORMULA_DIFFERS')
        if w['reused_questions']!=5120 or w['fitted_questions']!=5120 or len(w['actor_rng_restored'])!=4 or not all(x['rng_equal'] for x in w['actor_rng_restored']):
            raise ValueError('SIGMA3_TEACHER_REUSE_OR_RNG_BOUNDARY_DIFFERS')
        from search_g1.qprobe_warmup import verify_frozen
        frozen=verify_frozen(w['before_workers'],w['after_workers'],4)
        import torch
        head=Path(w['head_checkpoint']['path'])
        if hashlib.sha256(head.read_bytes()).hexdigest()!=w['head_checkpoint']['sha256']:raise ValueError('SIGMA3_BOOTSTRAP_HEAD_SHA_DIFFERS')
        state=torch.load(head,map_location='cpu',weights_only=True)
        if any(state['config'].get(k)!=v for k,v in LOSS.items()):raise ValueError('SIGMA3_SAVED_HEAD_LOSS_DIFFERS')
        count=sum(v.numel() for k,v in state['model'].items() if k.startswith('net.'))
        if count!=135622:raise ValueError('SIGMA3_HEAD_CAPACITY_DIFFERS')
        out['warmup']=dict(receipt_sha256=warm['sha256'],reused_questions=5120,fitted_questions=5120,
            head_metrics=metrics,head_config=state['config'],head_parameters=count,
            actual_four_rank_frozen=frozen['actual_frozen_rank_readback'],actual_four_rank_rng_equal=True)
    errors=[];logs={}
    for p in (BASE/'logs').glob('*.log'):
        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-65536));tail=f.read().decode(errors='replace')
        matches=[line[:1200] for line in tail.splitlines() if re.search(r'Traceback \(most recent call last\)|CUDA out of memory|SEARCH_G1_FATAL|AssertionError|ValueError:|RuntimeError:',line)]
        if matches:errors.append(dict(path=str(p),matches=matches[-12:]))
        logs[p.name]=dict(bytes=p.stat().st_size,tail=tail[-1600:])
    out['recent_error_patterns']=errors;out['logs']=logs
    if updates:
        last=updates[-1]['step']
        from reproduction.common8.behavior_runtime import read_gate_receipt
        from reproduction.common8.runtime import verify_scheduler_receipts
        gates={str(step):read_gate_receipt(BASE,run,step,update_record=updates[step-1]) for step in range(1,last+1)}
        verified=verify_scheduler_receipts(BASE,run,1,last,gates)
        if [x['value']['outer'] for x in events]!=list(range(1,last+1)):
            out['head_publication_in_flight']=True
        else:
            out['strict_formal_actor_head_prefix']=dict(first=1,last=last,verified_ranks=len(verified),
                actor_optimizer_writes=updates[-1]['optimizer_writes_after'],all_head_commits=True)
    return out


if __name__=='__main__':print(json.dumps(observe()),flush=True)
