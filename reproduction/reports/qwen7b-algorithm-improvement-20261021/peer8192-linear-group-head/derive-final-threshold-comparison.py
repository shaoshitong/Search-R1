"""Read already-scored remote logits once; report prespecified .5 and CV rules.

No model fitting or new feature/model scoring; raw predictions remain remote.
"""
import importlib.util,json,pathlib,shlex,subprocess
CONTROL=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
spec=importlib.util.spec_from_file_location('final_cpu_head_private',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
B='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer8192-linear-group-head-cpu2-v2-20261021'
source="""import gzip,hashlib,json,pathlib,sys,time
b=pathlib.Path(%r);pub=json.loads((b/'source-publication.json').read_text());sys.path.insert(0,pub['code_root'])
from reproduction.operations.qwen7b_warmup.peer_linear_group_head import metric,checked,read,ref
from reproduction.operations.qwen7b_warmup.replay_probe_layers import atomic
c=read(b/'output/complete.json');complete=ref(b/'output/complete.json');assert complete['bytes']==7030 and complete['sha256']=='f87ed1a0b64a7c4bf0dfd0ac6dc7e127a1c38d43e5f8fe86e7c960d4cba4d002'
assert not (b/'output/failure.json').exists()
for n,v in pub['files'].items():
 p=pathlib.Path(pub['code_root'])/n;assert p.stat().st_size==v['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256']
frozen=read(checked(c['frozen_before_external']));checked(frozen['weights']);rows=[]
with gzip.open(checked(c['external_predictions']),'rt') as stream:
 for line in stream:rows.append(json.loads(line))
assert len(rows)==len({r['qid'] for r in rows})==2048
scores=[r['predicted_Mixed_logit'] for r in rows];K=[sum(r['strict_em']) for r in rows];qids=[r['qid'] for r in rows]
cv=metric(scores,K,qids,c['winner']['threshold_logit'],512);assert cv==c['external']
fixed=metric(scores,K,qids,0.,512);assert fixed['AUC']==cv['AUC'] and fixed['Top512']==cv['Top512']
result=dict(status='ALREADY_SCORED_UNIQUE_WINNER_BOTH_PRESPECIFIED_THRESHOLDS_AND_SOURCE_SHA_VERIFIED',source_complete=complete,source_predictions=c['external_predictions'],source_frozen_before_external=c['frozen_before_external'],fixed_probability_half_logit0=fixed,internalCV_frozen_threshold=cv,internal_threshold_probability=1/(1+__import__('math').exp(-c['winner']['threshold_logit'])),same_HF_E4_fixed_probability_half=c['same_HF_E4_baseline'],jointgate_uses='fixed_probability_half_BA_plus_same_AUC_and_Top512',external_features_or_model_rescored=False,external_threshold_tuned=False,production_replacement=False,actor_or_existing_probe_updated=False,final_or_diag512_read=False,internal_split=read(checked(frozen['internal_split'])),internal_grid=read(checked(frozen['grid_results'])),runtime_peak_RSS_bytes=c['peak_observed_RSS_bytes'],runtime_CPU_threads=c['CPU_threads'],runtime_elapsed_seconds=c['elapsed_seconds'],time=time.time())
path=b/'output/independent-fixed-and-CV-threshold-audit.json'
if path.exists():
 old=read(path);assert {k:v for k,v in old.items() if k!='time'}=={k:v for k,v in result.items() if k!='time'};result=old
else:atomic(path,result)
print(json.dumps(dict(reference=ref(path),result=result)))
"""%B
code=B+'/code-a8bdba2e3d25';command='env QWEN7B_CODE_ROOT='+shlex.quote(code)+" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash "+shlex.quote(code+'/reproduction/operations/qwen7b_warmup/runtime.sh')+' -c '+shlex.quote(source)
result=ctl.run(command,account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=50)
if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')[-2000:]+result.stdout.decode(errors='replace')[-1000:])
v=json.loads(result.stdout.decode().splitlines()[-1]);pathlib.Path(__file__).with_name('independent-fixed-and-CV-threshold-audit.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({k:v['result'][k] for k in ('status','fixed_probability_half_logit0','internalCV_frozen_threshold','internal_split')}))
