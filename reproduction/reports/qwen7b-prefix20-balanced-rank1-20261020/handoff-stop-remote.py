import pathlib,json,subprocess,time,hashlib,shutil
parent=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');old=parent/'mixed-only-ft-cp2-prefix20-20261020';new=parent/'balanced-rank1-from-epoch1-prefix20-20261020'
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);assert r.returncode==0,(args,r.stderr);return r.stdout.strip()
def atomic(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
assert (old/'job.id').read_text().strip()=='742046'
receipt=json.loads((old/'probe/epoch-0001.json').read_text());assert receipt['steps']==10464 and receipt['completed_epoch'] and receipt['cursor']==0
assert receipt['sha256']=='02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724'
proof=json.loads((old/'mixed-checkpoint-audit'/ 'epoch-0001-step-00010464-cursor-0000-02884a3c76e8.json').read_text()) if (old/'mixed-checkpoint-audit'/'epoch-0001-step-00010464-cursor-0000-02884a3c76e8.json').exists() else None
assert pathlib.Path(receipt['path']).stat().st_size==receipt['bytes']
state=command(['squeue','-h','-j','742046','-o','%i|%u|%j|%T|%P|%N'])
assert state.split('|')[:3]==['742046','mou176','q7b-mixed-p20'],state
assert state.split('|')[3]=='RUNNING',state
assert not new.exists(),'Inspect existing handoff before retry'
(new/'preparation').mkdir(parents=True)
record=dict(status='USER_AUTHORIZED_EPOCH1_RESUME_HANDOFF_INTENT',time=time.time(),old_job='742046',old_base=str(old),new_base=str(new),source_checkpoint=receipt,old_queue=state,user_instruction='Increase ranking weight and resume from current completed epoch1',pair_weight_before=.1,pair_weight_after=1.,sampling='genuine Mixed:nonMixed 1:1 by complete G5 question; no actor update',rollback_explicitly_authorized=True,preserve_all_old_outputs=True)
for name in ['progress.json','latest.json','metrics-latest.json','phase.json']:
 p=old/'probe'/name
 if p.exists():shutil.copyfile(str(p),str(new/'preparation'/('old-'+name)))
atomic(new/'preparation/handoff.json',record)
command(['scancel','742046'])
record.update(status='OLD_JOB_CANCEL_REQUESTED_FOR_AUTHORIZED_EPOCH1_RESTART',time=time.time());atomic(new/'preparation/handoff.json',record)
print(json.dumps(record))
print(command(['squeue','-h','-j','742046','-o','%i|%u|%j|%T|%N']))
