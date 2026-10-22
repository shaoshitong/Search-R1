"""Read-only CPU metadata witness of old/new full-training sampling seeds."""
import ast,hashlib,json,os,random,resource,time
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
OLD=A/'online-packed-full169028-M3to10-P20-20261022-v1';NEW=A/'online-packed-full169028-Z-seed19137-plan2026102302-P20-20261023-v1'
resource.setrlimit(resource.RLIMIT_CPU,(50,50));resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));started=time.time()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
raw=(OLD/'fixture/inputs/train-groups.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=='4528712a7dd5f34e676e2bf2c2e8746b07ebda8112ed59ba100551c4891ae6f2';groups=json.loads(raw);assert len(groups)==169028
pub=read(NEW/'source-publication.json');rel='reproduction/operations/qwen7b_warmup/packed_variance_probe.py';source=Path(pub['code_root'])/rel;payload=source.read_bytes();assert hashlib.sha256(payload).hexdigest()==pub['files'][rel]['sha256']
tree=ast.parse(payload);tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in('digest','view','plan')];ns=dict(hashlib=hashlib,json=json,random=random);exec(compile(tree,str(source),'exec'),ns)
old=ns['plan'](groups,1,2026102227);new=ns['plan'](groups,1,2026102302)
assert digest(old)==read(OLD/'mae/coverage-epoch-01.json')['plan_sha256']
old_by={v['qid']:v for v in old};new_by={v['qid']:v for v in new};assert set(old_by)==set(new_by)==set(groups)
changed=sum(old_by[q]['trajectory_ids']!=new_by[q]['trajectory_ids']for q in groups);M_changed=sum(old_by[q]['M']!=new_by[q]['M']for q in groups)
assert changed>0 and M_changed>0 and [v['qid']for v in old]!=[v['qid']for v in new]
ready=read(NEW/'ready.json');assert ready['training']['init_seed']==19137 and ready['training']['train_plan_seed']==2026102302
print(json.dumps(dict(status='ACTUAL_FULL169028_TRAINING_PLAN_SECOND_SEED_CHANGES_VERIFIED',time=time.time(),source_publication_commit=pub['commit'],groups_sha256=hashlib.sha256(raw).hexdigest(),questions=169028,unique_questions_each=169028,old_train_plan_seed=2026102227,new_train_plan_seed=2026102302,old_epoch1_plan_sha256=digest(old),new_epoch1_plan_sha256=digest(new),old_epoch1_ordered_qid_sha256=digest([v['qid']for v in old]),new_epoch1_ordered_qid_sha256=digest([v['qid']for v in new]),changed_own_trajectory_views=changed,changed_M=M_changed,fixed_evaluation_seed=2026102228,official_actor_or_feature_or_CP_read=False,new_rollouts_generated=False,training_or_Slurm_started=False,final4096_read=False,elapsed_seconds=time.time()-started),sort_keys=True))
