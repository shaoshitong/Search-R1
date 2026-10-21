"""Terminal-only stdlib acceptance: inherit allocation reload, stream file SHA.

Never import torch or load checkpoint tensors on a login node. The producer's
allocation CPU nine-state reload is inherited and identified as such; only
file bytes, frozen own-EM/direct-q metrics and terminal/lock state are fresh.
"""
import argparse,fcntl,gzip,json,socket,subprocess,sys,time
from pathlib import Path
from . import peer_append_train as a
from .audit_direct_hidden_count import accept_rows,compare

KEYS=set(('legacy_raw','legacy_ema','legacy_adam','legacy_raw_buffers',
          'legacy_ema_buffers','rng','head_raw','head_ema','head_adam'))


def inherit_state(latest,complete,receipt,actual):
    cp=receipt['checkpoint']
    for saved in (latest,complete['checkpoint'],cp):
        if (any(saved[k]!=actual[k] for k in ('path','bytes','sha256')) or saved['cursor']!=938
                or saved['legacy_Adam_steps']!=11402 or saved['new_head_Adam_steps']!=938
                or saved.get('full_CPU_reload_passed') is not True
                or saved.get('full_file_SHA256_verified') is not True
                or saved.get('source_all_four_RNG_preserved') is not True
                or set(saved['accepted_hashes'])!=KEYS
                or saved['accepted_hashes']!=latest['accepted_hashes']):
            raise ValueError('Producer full-nine-state receipt/file binding differs')
    if (receipt['before_nine_component_hashes']!=latest['accepted_hashes']
            or receipt['after_nine_component_hashes']!=latest['accepted_hashes']
            or receipt['training_nine_components_unchanged'] is not True):
        raise ValueError('Evaluation altered producer accepted nine-state snapshot')
    return dict(mode='inherited_producer_allocation_full_nine_state_CPU_reload',
                fresh_checkpoint_tensor_reload=False,checkpoint=actual,
                producer_accepted_hashes=latest['accepted_hashes'],legacy_Adam=11402,new_head_Adam=938,
                original_four_rank_archive_preserved=True)


def frozen_groups(groups,plan):
    plan=plan.get('fixed_G5',plan)
    if plan['summary'].get('labels_consulted') is not False or plan['summary'].get('role')!='validation':
        raise ValueError('Frozen label-blind validation plan required')
    by={g['qid']:g for g in groups};selected=plan['groups'];seen=set();out=[]
    if len(by)!=len(groups) or len(selected)!=len(by) or {g['qid'] for g in selected}!=set(by):
        raise ValueError('Frozen question coverage changed')
    for v in selected:
        g=by[v['qid']];tids=v['trajectory_ids'];labels=dict(zip(g['trajectory_ids'],g['labels']))
        if (len(tids)!=5 or len(set(tids))!=5 or set(tids)&seen or not set(tids)<=set(labels)
                or len(labels)!=len(g['trajectory_ids']) or len(labels) not in (5,10)
                or any(type(y) is not int or y not in (0,1) for y in g['labels'])):
            raise ValueError('Five distinct real own IDs/EM required')
        seen.update(tids);out.append(dict(qid=g['qid'],trajectory_ids=tids,labels=[labels[t] for t in tids]))
    return out


def audit(base,job):
    if 'torch' in sys.modules:raise RuntimeError('Terminal inherited acceptance must not load torch/tensors')
    base=Path(base);attempt=base/'continuations/attempt-0001-v2';directory=base/'evaluations/cursor-000938'
    if base!=a.A/'direct-hidden-count30016-train-v2' or str(job)!='743872':
        raise ValueError('Only the actual completed continuation is accepted')
    if (attempt/'job.id').read_text().strip()!=str(job):raise ValueError('Actual continuation job differs')
    terminal=subprocess.check_output(['sacct','-j',str(job),'-X','-n','-P','--format=JobIDRaw,State,ExitCode,Elapsed']).decode()
    rows=[line.split('|') for line in terminal.splitlines() if line.split('|',1)[0]==str(job)]
    if len(rows)!=1 or rows[0][1:3]!=['COMPLETED','0:0']:raise ValueError('Exact main job is not COMPLETED0:0')
    lock=(base/'joint-training.lock').open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for p in (base/'failure.json',attempt/'failure.json'):
            if p.exists():raise ValueError('Actual failure receipt exists')
        config=a.proof(base/'config.json');c=a.read(base/'config.json')
        latest=a.read(base/'latest.json');done=a.read(base/'complete.json');receipt=a.read(directory/'complete.json')
        if (done['status']!='JOINT_HIDDEN_DIRECT_COUNT938_AND_FIXED_HF_VALIDATION_COMPLETE'
                or done['cursor']!=938 or done['legacy_steps']!=11402 or done['new_head_steps']!=938
                or done['config_sha256']!=config['sha256'] or done['final_pool_read'] is not False
                or receipt['config_sha256']!=config['sha256'] or receipt['cursor']!=938
                or receipt['legacy_steps']!=11402 or receipt['new_head_steps']!=938
                or receipt['final_pool_read'] is not False):raise ValueError('Actual final938 receipt/config differs')
        file=Path(receipt['checkpoint']['path'])
        if file.resolve()!=file or file.parent!=base/'checkpoints':raise ValueError('Unexpected checkpoint path')
        actual=a.proof(file)  # One bounded streaming read, never tensor deserialization.
        inherited=inherit_state(latest,done,receipt,actual)
        metrics={};predictions={}
        for e in c['evaluations']:
            if e['role']!='validation' or e['backend']!='huggingface_frozen_causal_prefill':raise ValueError('Only recipe-selection validation may be read')
            bank=a.read(a.checked(e['bank']))[e['split']]
            frozen=frozen_groups(bank,a.read(a.checked(e['fixed_g5'])))
            ref=receipt['predictions'][e['name']]
            with gzip.open(a.checked(ref),'rt') as f:pred=[json.loads(line) for line in f]
            metrics[e['name']]=accept_rows(pred,frozen);compare(metrics[e['name']],receipt['metrics'][e['name']]);predictions[e['name']]=ref
        if 'torch' in sys.modules:raise RuntimeError('Forbidden tensor framework imported')
        result=dict(status='TERMINAL938_INHERITED_PRODUCER_NINE_STATE_FRESH_FILE_SHA_OWN_EM_DIRECT_Q_ACCEPTED',
                    job_id=str(job),original_job_id='743619',cursor=938,host=socket.gethostname(),epoch=time.time(),
                    audit_mode='stdlib_only_no_tensor_deserialization',checkpoint=inherited,
                    config=config,producer_complete=a.proof(base/'complete.json'),producer_latest=a.proof(base/'latest.json'),
                    evaluation_receipt=a.proof(directory/'complete.json'),predictions=predictions,metrics=metrics,
                    slurm_main=terminal,joint_training_lock_free=True,actor_updates=0,final_pool_read=False,
                    threshold=.5,threshold_fitted=False,raw_predictions_stay_remote=True)
        a.immutable(directory/'terminal-inherited-independent-audit.json',result)
        return result
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--job-id',required=True);v=p.parse_args()
    result=audit(v.base,v.job_id)
    print(json.dumps(dict(status=result['status'],audit=a.proof(Path(v.base)/'evaluations/cursor-000938/terminal-inherited-independent-audit.json'),metrics=result['metrics'],checkpoint=result['checkpoint'],host=result['host'],mode=result['audit_mode'])))
