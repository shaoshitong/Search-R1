"""Independent compact CPU evidence read; no model/features/final/GPU access."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

C = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
R = C / 'work/qwen7b-probe-warmup-20261019'
BASE = '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/g20-exact-uniform15504-dp-preparation-v1'
CODE = r'''
import fcntl,hashlib,json,pathlib,time
b=pathlib.Path(BASE)
def read(p):return json.loads(pathlib.Path(p).read_text())
def fp(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def checked(v):
 assert fp(v['path'])==v;return read(v['path'])
pub=read(b/'source-publication.json');root=pathlib.Path(pub['code_root'])
for n,v in pub['files'].items():assert fp(root/n)==dict(v,path=str(root/n)),n
prep=read(b/'preparation.json');config=checked(prep['config']);source=checked(prep['reference_config']);plan=checked(prep['source_plan']);bench=checked(prep['CPU_benchmark'])
for key in ('source_checkpoint','source_config','source_bank','model_config','model_dir','g20_complete','validation_g10_complete','plan','evaluations','micro_trajectories','save_every','global_questions','update_budget','fit_questions','fit_real_trajectories'):
 assert config[key]==source[key],key
assert len(plan['batches'])==136 and all(len(x)==48 for x in plan['batches']) and plan['batch_epochs']==[0]*136
assert config['source_checkpoint']['sha256']=='02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724'
assert config['combinations_per_question']==15504 and config['view_weight_denominator']==15504 and config['original_question_total_weight']==1
assert config['recipe']['loss_kwargs']==source['recipe']['loss_kwargs']==dict(trajectory_weight=1.,three_event_weight=1.)
assert config['explicit_launch_authorized'] is False and config['training_entrypoint'] is None and config['GPU_requested']==0
assert read(b/'cpu-exit.json')['returncode']==0
assert bench['status']=='ACTUAL_CPU2_Q48_G20_ALL15504_DP_BENCHMARK_COMPLETE' and len(bench['variants'])==3
workers=[read(b/'cpu-benchmark'/name) for name in ('old32-chunk64.json','dp-chunk64.json','dp-chunk256.json')]
old=workers[0];comparison=[]
for worker in workers:
 assert worker['questions']==48 and worker['source_G']==20 and worker['combinations_per_question']==15504 and worker['virtual_views']==744192
 assert worker['CPU_affinity']==[0,1] and worker['torch_threads']==2 and worker['peak_RSS_bytes']<3*1024**3 and not worker['CUDA_initialized']
 assert worker['confusion_integer_all_views']==old['confusion_integer_all_views']
 differences=[abs(a-c) for aa,bb in zip(old['gradient'],worker['gradient']) for a,c in zip(aa,bb)]
 comparison.append(dict(mode=worker['mode'],chunk=worker['combination_chunk'],elapsed_seconds=worker['elapsed_seconds'],peak_RSS_bytes=worker['peak_RSS_bytes'],maximum_gradient_absolute_difference=max(differences),loss_absolute_difference=abs(worker['loss']-old['loss'])))
 assert max(differences)<2e-6 and abs(worker['loss']-old['loss'])<2e-5
launch=read(b/'cpu-launch.json')
try:birth=pathlib.Path('/proc/%d/stat'%launch['pid']).read_text().rsplit(')',1)[1].split()[19]
except FileNotFoundError:birth=None
assert birth!=launch['start_ticks'],'Unexpected owned wrapper still active after terminal receipt'
with (b/'cpu-benchmark/benchmark.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not list((b/'cpu-benchmark').glob('failure-*.json'))
log=(b/'actual-cpu-tests.log').read_text();assert 'Ran 6 tests' in log and 'OK' in log
references={name:fp(b/name) for name in ('source-publication.json','actual-cpu-tests.log','cpu-tests.json','cpu-launch.json','cpu-exit.json','preparation.json','config.json','cpu-benchmark/complete.json','cpu-benchmark/old32-chunk64.json','cpu-benchmark/dp-chunk64.json','cpu-benchmark/dp-chunk256.json')}
result=dict(status='INDEPENDENT_ACTUAL_EXACT_G20_DP_CPU_AND_TYPED_BINDING_AUDIT_PASS',epoch=time.time(),references=references,comparisons=comparison,benchmark=bench,preparation=prep,code_files=pub['files'],code_commit=pub['commit'],checked_source_files=len(pub['files']),source_tensor_checks_inherited_not_repeated=True,checkpoint_tensors_read=False,feature_tensors_read=False,evaluation_labels_read=False,final_accessed=False,GPU_requested=0,owned_wrapper_birth_gone=True,benchmark_lock_free=True,error_receipts_absent=True,
 observations=[dict(epoch=1792534183.4709325,completed_variants=1,old32_complete=True,dp64_running=True),dict(epoch=1792534193.4152741,completed_variants=3,exit_returncode=0)],
 qualification_limit='Synthetic finite CPU objective benchmark and toy state restoration only; no learning gain or GPU runtime qualification claimed')
p=b/'independent-cpu-audit.json';temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(result,sort_keys=True)+'\n');temp.replace(p)
print(json.dumps(dict(audit=fp(p),**result)))
'''
spec = importlib.util.spec_from_file_location('dp_actual_control', C / 'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
ctl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctl)
address = subprocess.check_output(['/usr/sbin/ipconfig', 'getifaddr', 'en0'], text=True).strip()
command = 'python3 -c ' + shlex.quote('BASE=' + repr(BASE) + '\n' + CODE)
assert len(command.encode()) < 16000
done = ctl.run(command, account='mou176', source_address=address, timeout=55)
if done.returncode:
    raise RuntimeError(done.stdout.decode()[-2000:] + done.stderr.decode()[-2000:])
result = json.loads(done.stdout.decode().splitlines()[-1])
destination = Path(__file__).parent / 'actual-cpu-independent-audit.json'
destination.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(status=result['status'], epoch=result['epoch'], audit=result['audit'], comparisons=result['comparisons'], config=result['preparation']['config'], GPU_requested=0)))
