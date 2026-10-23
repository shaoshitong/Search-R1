"""Read existing scalar logs across the actual restore chain; no tensors or evaluation."""
import hashlib,json,re,time
from pathlib import Path
BASE=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-batch-searchstop-r1-20261023')
ANSI=re.compile(r'\x1b\[[0-9;]*m')
def read(p): return json.loads(Path(p).read_text())
def jlines(p):
    if not p.exists(): return []
    out=[]
    for line in p.read_text().splitlines():
        try: out.append(json.loads(line))
        except json.JSONDecodeError: pass
    return out
segments=[]; rows=[]; errors=[]; seen=set(); base=BASE; upper=None
while base not in seen:
    seen.add(base); runpath=base/'logs/run-manifest.json';run=read(runpath);binding=run.get('full_state_restore') or {}; lower=binding.get('completed_outer',0)+1
    updates=jlines(base/'logs/actual-updates.jsonl'); accepted={r.get('step',r.get('logical_update')):r for r in updates if r.get('update_status')=='complete'}
    if upper is None: upper=max(accepted,default=lower-1)
    valid={s for s in accepted if isinstance(s,int) and lower<=s<=upper}; metrics={}; logfiles=[]
    for name in ['trainer.log','search-g1-common8-training.log']:
        p=base/'logs'/name
        if not p.exists(): continue
        raw=p.read_bytes(); text=ANSI.sub('',raw.decode(errors='replace')); logfiles.append(dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        for m in re.finditer(r'\bstep:(\d+)\s*-\s*([^\n]+)',text):
            step=int(m.group(1)); vals={}
            for entry in m.group(2).split(' - '):
                if ':' in entry:
                    k,v=entry.split(':',1)
                    try: vals[k.strip()]=float(v)
                    except ValueError: pass
            if step in valid and 'train/perf/em' in vals and 'critic/rewards/mean' in vals:
                metrics[step]=dict(step=step,training_em=vals['train/perf/em'],reward=vals['critic/rewards/mean'],score_mean=vals.get('critic/score/mean'),selected_mixed_trajectories=vals.get('selection/mixed/trajectories'),actor_seconds=vals.get('timing_s/update_actor'),log_path=str(p),base=str(base),run_id=run.get('run_id'),runtime_commit=run.get('source_commit'))
    missing=sorted(valid-set(metrics));errors.extend([dict(base=str(base),missing_scalar_steps=missing)] if missing else [])
    for step in sorted(metrics): rows.append(metrics[step])
    evaluations=[]
    p=base/'logs/validation-identities.jsonl'
    for entry in jlines(p):
        step=entry.get('step',entry.get('logical_update',entry.get('completed_outer')))
        if step is None or (lower<=step<=upper): evaluations.append(entry)
    segment=dict(base=str(base),run_id=run.get('run_id'),commit=run.get('source_commit'),lower=lower,upper=upper,source_restore_outer=binding.get('completed_outer'),source_receipt=binding.get('source_receipt'),accepted_steps=sorted(valid),metric_steps=sorted(metrics),logs=logfiles,validation_records=evaluations)
    segments.append(segment)
    receipt_path=binding.get('source_receipt')
    if not receipt_path: break
    receipt=read(receipt_path); parentmanifest=receipt.get('source_run_manifest',{}).get('path')
    if parentmanifest: parent=Path(parentmanifest).parent.parent
    elif receipt.get('source_base'): parent=Path(receipt['source_base'])
    else:
        errors.append(dict(base=str(base),error='No parent source manifest/base in receipt',receipt=str(receipt_path)));break
    upper=binding['completed_outer'];base=parent
rows.sort(key=lambda r:r['step'])
print(json.dumps(dict(observed_unix=time.time(),current_base=str(BASE),latest_completed=max((r['step'] for r in rows),default=None),segments=segments,training_rows=rows,read_errors=errors,training_modified=False,torch_imported=False,tensor_files_read=False,evaluation_run=False,scalar_log_precision='Printed logger values rounded to three decimal places.'),ensure_ascii=False),flush=True)
