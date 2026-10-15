
import json,sys,pathlib,time,os,signal,hashlib,subprocess
p=json.load(sys.stdin);b=pathlib.Path(p['base']);sys.path.insert(0,p['runtime_repo'])
from reproduction.a800.owned_processes import same,identity
hold=json.loads((b/'logs/human-balanced-sampling-hold-actual.json').read_text());q=hold['process'];child=hold['feature_child']
def live(x):return same(x) and identity(x['pid'])['state'] not in ('Z','X')
assert same(q) and identity(q['pid'])['state']=='T' and not live(child)
done=json.loads((b/'features/features-complete.json').read_text());frozen=json.loads((b/'features/frozen-encoder-proof.json').read_text())
assert done['count']==28160 and done['actor_trained'] is False and done['head_trained'] is False and frozen['exact_equal'] is True
assert not (b/'warmup').exists()
claim=b/'logs/human-balanced-old-supervisor-exit-once.json'
with claim.open('x') as f:json.dump(dict(process=q,scope='only old held supervisor; prefix encoder already exited; no unbalanced fit permitted',claimed_unix=time.time()),f)
os.kill(q['pid'],signal.SIGTERM);os.kill(q['pid'],signal.SIGCONT);time.sleep(.5)
assert not live(q) and not live(child) and not (b/'warmup').exists()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
assert not any(line.startswith(done['gpu']['physical_gpu_uuid']+',') for line in apps.splitlines())
v=dict(role='qualified_prefix_cache_reused_for_balanced20',checked_unix=time.time(),feature_output_dir=str(b/'features'),cache_dir=done['cache_dir'],manifest_sha256=done['manifest_sha256'],count=28160,split_counts=done['split_counts'],feature_bytes=done['feature_bytes'],global_max_tokens=done['global_max_tokens'],encoder_released=True,old_supervisor_released=True,old_unbalanced_fit_never_started=True,actor_and_head_updates=0,frozen_encoder_full_state_exact_equal=True,feature_producer_sha256=hashlib.sha256((b/'qprobe_prefix_correctness_features.py').read_bytes()).hexdigest(),gpu_inventory=gpu,compute_processes=apps)
with (b/'logs/human-balanced-old-supervisor-release-actual.json').open('x') as f:json.dump(v,f)
print(json.dumps(v),flush=True)

