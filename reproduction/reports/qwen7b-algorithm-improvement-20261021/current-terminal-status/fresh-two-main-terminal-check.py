"""Fresh authoritative terminal metadata only; no rescore/tensor/bulk read."""
import importlib.util,json,pathlib,shlex,subprocess
R=pathlib.Path(__file__).resolve().parents[4];C=R.parents[1];HERE=pathlib.Path(__file__).parent
spec=importlib.util.spec_from_file_location('fresh_terminals_private',C/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
code='''import pathlib,json,hashlib,fcntl,subprocess,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
def read(p):return json.loads(p.read_bytes())
def proof(p):
 data=p.read_bytes();assert len(data)<1024*1024;return dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
def lock(p):
 if not p.exists():return dict(path=str(p),exists=False)
 f=p.open('r');result=dict(path=str(p),exists=True,inode=p.stat().st_ino)
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);result['held']=False;fcntl.flock(f,fcntl.LOCK_UN)
 except BlockingIOError:result['held']=True
 f.close();return result
r=subprocess.run(['sacct','-X','-j','743809,743978','-n','-P','--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);assert r.returncode==0,r.stderr
main={row.split('|')[0]:row for row in r.stdout.splitlines()if row.split('|')[0]in('743809','743978')};assert len(main)==2 and all('|COMPLETED|0:0|'in row for row in main.values()),main
q=subprocess.run(['squeue','-h','-j','743809,743978','-o','%i|%j|%T|%N'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);assert q.returncode==0 and not q.stdout.strip(),q.stdout
out=dict(epoch=time.time(),sacct_main=main,squeue_empty=True,checks='Existing terminal metadata and accepted metric receipts only',rescoring_performed=False,tensor_load_performed=False,new_GPU_submitted=False)
b=A/'direct-hidden-count30016-headlr1e4-v1';c=read(b/'config.json');assert proof(b/'config.json')['sha256']=='71c3ee90f0232b889d4ced2d0cb50d69e852eb1ebbdbb55fdff03eaefdbbcc71';done=read(b/'complete.json');audit=b/'evaluations/cursor-000938/independent-audit.json';a=read(audit);assert proof(audit)['sha256']=='205e78b8a3e1d4b088c8f358e12f5ebc5f9873614ec1a80d7336e051397b992a';assert done['cursor']==938 and a['cursor']==938
out['higher_headLR938']=dict(config=proof(b/'config.json'),complete=proof(b/'complete.json'),accepted_audit=proof(audit),fit_questions=c['fit_questions'],update_budget=c['update_budget'],legacy_Adam=done['legacy_steps'],head_Adam=done['new_head_steps'],metrics=a['metrics']['selection2048FreshG5HF'],training_lock=lock(b/'joint-training.lock'),failure_file_exists=(b/'failure.json').exists())
b=A/'direct-count-frozenE1-fit30016-hold5120-v2';c=read(b/'config.json');assert proof(b/'config.json')['sha256']=='fb637c02812dd070cf392a58df75e0d219bbb10d1a2d24cd6e201f3638147767';done=read(b/'complete.json');assert done['head_Adam']==18760
p=A/'direct-count-large-independent-audit-v3/audits/snapshot-ab2e83b08f781db4.json';a=read(p);assert proof(p)['sha256']=='9552c4666235f182f31980278d3d3e93d135c8030ed80beede3f14efec44b893';assert a['actual_scored_nodes']==12
out['frozen_E1_Count20epoch']=dict(config=proof(b/'config.json'),complete=proof(b/'complete.json'),accepted_all12_audit=proof(p),fit_questions=c['fit_questions'],internal_holdout_questions=c['internal_holdout_questions'],epochs=c['epochs'],update_budget=c['update_budget'],actual_head_Adam=done['head_Adam'],actor_updates=done['actor_updates'],legacy_updates=done['legacy_updates'],max_GPU_allocated_bytes=done['max_GPU_allocated_bytes'],selected_Count12_internal_candidate=a['selected_internal_candidate'],epoch20_nodes=[dict(model=n['model'],NQ=n['metrics']['NQ'],gate=n['screen_gate'])for n in a['nodes']if n['epoch']==20],external_payload_opened=a['external_payload_opened'],external_scoring_permitted=a['external_scoring_permitted'],final_payload_opened=a['final_payload_opened'],training_lock=lock(b/'head-only-training.lock'),failure_file_exists=(b/'failure.json').exists())
p=A/'peer-threebank58432-internal5120-plan-v1/complete.json';d=read(p);out['feature_ready_prepared_data_plan']=dict(complete=proof(p),status=d.get('status'),fit_scope='This is a data/feature plan, not evidence that all58432 were fitted',current_actual_fit_questions=30016,full_feature_ready_questions=58432,prepared_future_fit_questions=53312,internal_holdout_questions=5120)
assert not out['higher_headLR938']['training_lock'].get('held') and not out['frozen_E1_Count20epoch']['training_lock'].get('held')
print(json.dumps(out))'''
cmd='python3 -c '+shlex.quote(code);assert len(cmd.encode())<16000
r=ctl.run(cmd,account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=35)
if r.returncode:raise RuntimeError('Fresh terminal metadata check failed; no new job or state inference: '+r.stderr.decode()[-1000:])
x=json.loads(r.stdout.decode().splitlines()[-1]);(HERE/'actual-two-main-terminal-check.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
