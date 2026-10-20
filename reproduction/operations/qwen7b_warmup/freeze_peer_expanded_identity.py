"""Freeze an enlarged training-only G10 identity cohort; remote CPU, no GPU."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path
P='reproduction/operations/qwen7b_warmup/'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-expanded-identity-v1'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--control-root',type=Path,required=True);args=ap.parse_args()
    repo=Path(__file__).resolve().parents[3];commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0];assert remote==commit,'Publish before freeze'
    data=(repo/(P+'peer_expanded_identity.py')).read_bytes();assert data==subprocess.check_output(['git','show',commit+':'+P+'peer_expanded_identity.py'],cwd=repo)
    encoded=base64.b64encode(data).decode();expected=hashlib.sha256(data).hexdigest()
    program='''import pathlib,base64,hashlib,json,os,time,fcntl,resource
b=pathlib.Path(BASE);b.mkdir(exist_ok=True,parents=True);guard=(b/'freeze.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[:2])
started=time.time();source=base64.b64decode(ENCODED);assert hashlib.sha256(source).hexdigest()==EXPECTED
p=b/'identity-source.py'
if p.exists():assert p.read_bytes()==source
else:p.write_bytes(source);p.chmod(0o444)
ns={};exec(compile(source,str(p),'exec'),ns)
def read(p):return json.loads(pathlib.Path(p).read_text())
def fp(p):
 p=pathlib.Path(p);v=p.read_bytes();return dict(path=str(p),bytes=len(v),sha256=hashlib.sha256(v).hexdigest())
def checked(r):
 p=pathlib.Path(r['path']);v=fp(p);assert v['sha256']==r['sha256']
 if 'bytes' in r:assert v['bytes']==r['bytes']
 return p
def atomic(p,v):
 p=pathlib.Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,sort_keys=True,separators=(',',':'))+'\\n');t.replace(p)
A=b.parent;t=read(A/'peer-newseed8192-g10-train-v1/training-input.json');old=read(checked(t['identity_plan']));raw=read(checked(t['raw_complete']))
assert raw['status']=='RECEIVED_NEWSEED5_PLUS_CANONICAL_OLD5_RAW_G10_STRICT_EM_CAUSAL_SEEDS_ACCEPTED' and raw['strict_EM_mismatches']==0 and raw['final_labels_or_rollouts_read'] is False
if (b/'complete.json').exists():
 done=read(b/'complete.json');assert done['source_sha256']==EXPECTED;checked(done['identity_plan']);print(json.dumps(done))
else:
 assert not (b/'identity-plan.json').exists(),'Unknown partial identity freeze preserved; reconcile instead of overwrite'
 config=read(checked(raw['config']));groups=read(checked(raw['outputs']['groups']))['train'];plan=read(checked(old['controlled6528']));assert len(plan['cohort_qids'])==6528
 original=read(checked(config['original_bank']))['train'];qhash={g['qid']:g['question_hash'] for g in original};del original
 controlled=[dict(qid=q,question_hash=qhash[q]) for q in plan['cohort_qids']];excluded=[x['groups'] for x in config['exclusions'].values()]
 atomic(b/'progress.json',dict(stage='ACTUAL_FROZEN_SOURCE_IDENTITIES_LOADED',source_questions=len(groups),old_questions=len(old['groups']),epoch=time.time()))
 result=ns['expand'](groups,old['groups'],controlled,excluded)
 assert all(json.loads(g['qid'])[0]=='nq' for g in result['groups']) and len(old['groups'])==8192
 result.update(raw_complete=t['raw_complete'],prior_identity=t['identity_plan'],controlled_plan=old['controlled6528'],source_code=fp(b/'identity-source.py'),source_commit=COMMIT,domain_scope='NQ-only previously warmup-seen canonical questions; fresh rollout-enrichment',no_original_actor_or_probe_changes=True)
 atomic(b/'identity-plan.json',result);identity_ref=fp(b/'identity-plan.json');loaded=read(checked(identity_ref));assert loaded==result
 summary=dict(result['summary'],identity_plan=identity_ref,raw_complete=t['raw_complete'],prior_identity=t['identity_plan'],source_commit=COMMIT,source_sha256=EXPECTED,exact_identity_file_roundtrip=True,CPU_affinity=list(os.sched_getaffinity(0)),GPUs=0,full_probabilities_or_model_tensors_read=False,untouched_final_labels_or_rollouts_read=False,source_assets_modified=False,elapsed_seconds=time.time()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,epoch=time.time())
 atomic(b/'progress.json',dict(stage='EXPANDED_IDENTITY_FROZEN',questions=summary['questions'],new_questions=summary['new_questions'],epoch=time.time()));atomic(b/'complete.json',summary);print(json.dumps(summary))
'''.replace('BASE',repr(BASE)).replace('ENCODED',repr(encoded)).replace('EXPECTED',repr(expected)).replace('COMMIT',repr(commit))
    script=args.control_root/'outputs/qwen7b-warmup-20261019/operations/freeze-peer-expanded-identity-actual.py';script.write_text(program)
    command='python3 -c '+shlex.quote(program);assert len(command.encode())<16000
    spec=importlib.util.spec_from_file_location('private_expansion_control',args.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
    ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();r=ctl.run(command,account='mou176',source_address=ip,timeout=55)
    if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-1500:]+' '+r.stdout.decode(errors='replace')[-1500:])
    values=[json.loads(x) for x in r.stdout.decode().splitlines() if x.startswith('{')];assert values,'Observation unavailable; inspect existing freeze before retry'
    value=values[-1];target=args.control_root/'outputs/qwen7b-warmup-20261019/reports/peer-expanded-identity-complete.json';target.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');print(json.dumps(value))
if __name__=='__main__':main()
