import pathlib,json,time,hashlib,fcntl
R=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hard-enrichment-20261021')
def read(p):return json.loads(p.read_text())
def fp(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(4*1024**2),b''):h.update(x)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
out=dict(status='ACTUAL_CPU_HARD_ENRICHMENT_STATUS',time=time.time())
for name in ('prepare-launch.json','prepare-controller.json','prepare-progress.json','prepare-failure.json'):
 p=R/name
 if p.exists():out[name]=read(p)
launch=out.get('prepare-launch.json',{})
if launch:
 p=pathlib.Path('/proc/'+str(launch['pid'])+'/stat')
 s=p.read_text().split() if p.exists() else []
 out['actual_process_identity_live']=bool(s and s[21]==launch['start_ticks'] and s[2]!='Z')
with (R/'prepare.lock').open('a') as f:
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);out['prepare_lock']='FREE'
 except BlockingIOError:out['prepare_lock']='HELD'
for name in ('identity-exclusions.json','prepare-config.json','candidates.json','prepare-complete.json'):
 p=R/name
 if p.exists():out[name]=dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
if (R/'prepare-complete.json').exists():
 v=read(R/'prepare-complete.json');assert fp(v['candidate_plan']['path'])==v['candidate_plan']
 assert fp(v['exclusions']['path'])==v['exclusions'];assert fp(v['source_config']['path'])==v['source_config']
 actual=read(pathlib.Path(v['candidate_plan']['path']));g=actual['groups']
 assert len(g)==len({r['qid'] for r in g})==len({r['question_hash'] for r in g})==32768
 assert len({t for r in g for t in r['trajectory_ids']})==163840
 assert all(set(r)=={'qid','question_hash','trajectory_ids','labels','split'} and r['split']=='train' and len(r['labels'])==5 for r in g)
 bank=read(pathlib.Path(v['source_bank']['path']));lookup={r['qid']:r for r in bank['train']}
 assert fp(v['source_bank']['path'])==v['source_bank'] and fp(v['features']['path'])==v['features']
 for row in g:assert all(row[k]==lookup[row['qid']][k] for k in ('qid','question_hash','trajectory_ids','labels'))
 kh=[sum(sum(row['labels'])==k for row in g) for k in range(6)]
 out['actual_candidate_K_histogram']=kh;out['common_reachable_fourstratum_quota']=min(kh[0],kh[5],sum(kh[1:5])//2)
 assert not (R/'prepare-failure.json').exists()
 out['complete']=dict(v,complete_ref=fp(R/'prepare-complete.json'),independent_readback_unique_questions=32768,independent_readback_own_trajectories=163840)
else:
 p=R/'cpu-prepare.log'
 if p.exists():out['log_tail']=p.read_text()[-1500:]
print(json.dumps(out))
