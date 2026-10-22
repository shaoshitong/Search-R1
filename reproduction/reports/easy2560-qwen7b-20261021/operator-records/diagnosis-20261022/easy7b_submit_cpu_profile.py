# Historical exact operator from2026-10-22. No credentials; do not replay rejected CPU-only sbatch.
import json,subprocess,shlex,time,os
from pathlib import Path
b=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022');commit='665704a6c3a02a4b92e6a441b9543363097edf5a';source=b/'source'/commit
root=b/'diagnostics/candidate-cpu-cause-r1';root.mkdir(exist_ok=True,parents=True)
if (root/'submit-claim.json').exists():
 print(json.dumps({'status':'EXISTING_CLAIM','claim':json.loads((root/'submit-claim.json').read_text()),'receipt':json.loads((root/'submit-receipt.json').read_text()) if (root/'submit-receipt.json').is_file() else None}));raise SystemExit()
old=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp34-oom-r1-20261022');new=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-lean-r1-20261022');cp=old/'checkpoints/search-g1-common8/actor/global_step_52'
pools=[old/'artifacts/mixed-candidate-pools/30f1ef5473e74949b9829c1a1e132e1a-pool-0',new/'artifacts/mixed-candidate-pools/ac76dcefaf5a4cd793dea445f36ecc89-pool-0',old/'artifacts/mixed-candidate-pools/597f7fffb2bf4dffa17d065179a8243b-pool-0']
common=['env','PYTHONDONTWRITEBYTECODE=1','PYTHONUNBUFFERED=1','OMP_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','CUDA_VISIBLE_DEVICES=',str(old/'env/bin/python'),'-B']
ops=source/'reproduction/operations/easy7b_20261021'
commands=[common+[str(ops/'profile_sampling_capture_cpu.py'),'--repo',str(source),'--requests','1024','--repeats','3','--prompt-lengths','128','1024','3072','--generation-config',str(cp/'generation_config.json'),'--output',str(root/'capture.json')],common+[str(ops/'profile_saved_rollout_cpu.py'),'--repo',str(source),'--candidate-file',str(pools[1]/'0.pt'),'--checkpoint',str(cp),'--max-turns','256','--repeats','3','--output',str(root/'prefix.json')],common+[str(ops/'summarize_saved_candidate_work_cpu.py'),'--output',str(root/'actual-work.json')]+sum((['--pool',str(p)] for p in pools),[])]
wrapper=' && '.join(shlex.join(c) for c in commands)
argv=['sbatch','--parsable','--partition=emergency_acd','--qos=formal-user','--account=jxu957','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=32G','--time=00:10:00','--job-name=easy7b-candidate-cpu-profile','--export=NONE','--chdir='+str(source),'--output='+str(root/'slurm-%j.log'),'--error='+str(root/'slurm-%j.log'),'--wrap='+wrapper]
(root/'submit-claim.json').write_text(json.dumps({'epoch':time.time(),'commit':commit,'commands':commands,'argv':argv,'production_modified':False},indent=2)+'\n')
r=subprocess.run(argv,capture_output=True,text=True,timeout=30);out={'epoch':time.time(),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'production_modified':False}
if r.returncode==0:
 job=r.stdout.strip().split(';')[0];out['job']=job;(root/'job.id').write_text(job+'\n')
(root/'submit-receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
