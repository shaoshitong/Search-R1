"""Recover omitted canonical metadata, not features or labels, from old raw31232."""
import importlib.util,json,pathlib,shlex,subprocess
R=pathlib.Path(__file__).resolve().parents[4];C=R.parents[1];HERE=pathlib.Path(__file__).parent
worker='''import pathlib,json,hashlib,time,os,resource
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):
 p=pathlib.Path(r['path']);assert p.stat().st_size==r['bytes'] if 'bytes'in r else True;return p
def sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
t=read(A/'peer-newseed8192-g10-train-v1/training-input.json');raw=read(checked(t['raw_complete']));cfg=read(checked(raw['config']));groups=read(checked(raw['outputs']['groups']))['train'];old=read(checked(t['identity_plan']));planpath=A/'peer-expanded-identity-v1/identity-plan.json';plan=read(planpath);controlled=read(checked(old['controlled6528']));cids=set(controlled['cohort_qids']);orig=read(checked(cfg['original_bank']))['train'];ch={g['question_hash']for g in orig if g['qid']in cids};del orig
by={g['qid']:g for g in groups};chosen={g['qid']for g in plan['groups']};missing=[g for g in groups if g['qid']not in chosen];assert len(groups)==31232 and len(chosen)==30016 and len(missing)==1216
scopes={name:dict(qids={g['qid']for g in x['groups']},hashes={g['question_hash']for g in x['groups']})for name,x in cfg['exclusions'].items()};overlaps={name:dict(qids=sum(g['qid']in x['qids']for g in missing),hashes=sum(g['question_hash']in x['hashes']for g in missing))for name,x in scopes.items()}
controlq=sum(g['qid']in cids for g in missing);controlh=sum(g['question_hash']in ch for g in missing);tail={g['qid']for g in plan['omitted_tail_identities']};dup=set(plan['duplicate_text_qids_excluded']);eligible=[g for g in missing if not any(g['qid']in x['qids']or g['question_hash']in x['hashes']for x in scopes.values())]
rows=[{k:g[k]for k in('qid','question_hash','trajectory_ids')}for g in eligible];assert all(len(g['trajectory_ids'])==10 for g in rows)
out=A/'peer-recovered1216-identity-v1';out.mkdir(exist_ok=True);identity=dict(role='train',groups=rows,questions=len(rows),trajectories=10*len(rows),source_raw_complete=t['raw_complete'],source_raw_groups=raw['outputs']['groups'],old_feature_identity_plan=proof(planpath),selection_rule='Exact raw31232 minus already-featured30016; reinclude controlled canonical training identities and full-batch tail, never evaluation identities',labels_consulted=False,scores_consulted=False,prior_controlled_exclusion_retired_for_full124194_training=True,external_final_qid_hash_exclusions_retained=True,original_warmup_unseen_claim=False)
p=out/'identity-plan.json';blob=json.dumps(identity,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()+bytes([10])
if p.exists():assert p.read_bytes()==blob
else:p.write_bytes(blob)
result=dict(status='OLD_RAW31232_MINUS_FEATURE30016_RECOVERED_METADATA_ONLY',epoch=time.time(),raw_questions=len(groups),old_feature_questions=len(chosen),omitted_questions=len(missing),eligible_recovered_questions=len(rows),eligible_recovered_trajectories=10*len(rows),old_plan_summary=plan['summary'],omission_breakdown=dict(controlled_qid=controlq,controlled_hash=controlh,full_batch_tail=sum(g['qid']in tail for g in missing),train_text_dedup=sum(g['qid']in dup for g in missing)),fixed_evaluation_identity_overlap=overlaps,recovered_identity_plan=proof(p),groups_source=raw['outputs']['groups'],data_source='canonical training metadata; no payload/model/tensor/feature read',GPU_submitted=False)
print(json.dumps(result))'''
s=importlib.util.spec_from_file_location('ctl',C/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl)
cmd='python3 -c '+shlex.quote(worker);assert len(cmd.encode())<16000
r=ctl.run(cmd,account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=55)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-1500:])
x=json.loads(r.stdout.decode().splitlines()[-1]);(HERE/'actual-recovered1216.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
