import argparse,hashlib,json,os,pathlib,subprocess,tempfile,time
parser=argparse.ArgumentParser();parser.add_argument('--local-only',action='store_true');options=parser.parse_args()
repo=pathlib.Path(__file__).resolve().parents[3]
branch='codex/q7b-native-joint-root-20261021'
ref='refs/heads/'+branch
def git(*args,env=None):
 return subprocess.check_output(['git',*args],cwd=repo,env=env,text=True).strip()
parent=git('rev-parse',ref)
files=[pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('joint_mixed_direct_labelblind.py','direct_update_labelblind.py','labelblind_question_g5_sampler.py','deploy_mixed_direct_e1_uniform_g5_weighted.py','prepare_frozen_final4096_generation.py','deploy_frozen_final4096_scoring.py','score_frozen_final4096.py','test_score_frozen_final4096.py','accept_online_mixed_pilot_checkpoints.py','deploy_online_pilot_CPU_acceptance.py')]
report=pathlib.Path(__file__).resolve().parent
for folder in ('mixed-direct-e1-unique-g5','online-layer-prefix','online-pilot-CPU-acceptance','uniform-g5-control','unique-vs-labelblind-g5','frozen-final4096-preparation','frozen-final4096-scoring','live-observations'):
 for p in (report/folder).rglob('*'):
  if p.is_file() and p.suffix in ('.py','.md','.json','.jsonl','.txt') and p.stat().st_size<600000:files.append(p.relative_to(repo))
files.extend(pathlib.Path(__file__).resolve().parent.relative_to(repo)/n for n in ('observe-live.py','publish-goal-progress.py','read-current-queue-forecast.py','raise-waiting-current-jobs-medium.py','raise-waiting-current-jobs-third-tier.py','LATEST-ACTIVE-EXPERIMENTS.json'))
files=sorted(set(p for p in files if (repo/p).is_file()))
files.extend(pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('compare_viewed2048_pilot_depth.py','deploy_compare_viewed2048_pilot_depth.py','deploy_online_l21p80_factorial.py','online_mixed_depth_expansion.py','deploy_online_matched_depth_expansion.py'))
files.extend(pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('compare_viewed2048_expanded_depth.py','deploy_compare_viewed2048_expanded_depth.py'))
files.extend(pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('natural_g5_online_control.py','deploy_natural_g5_online_control.py','packed_variance_probe.py','online_packed_variance.py','deploy_online_packed_variance.py','test_packed_variance_probe.py'))
files.append(pathlib.Path('reproduction/operations/qwen7b_warmup/test_packed_variance_full_schema.py'))
files.extend(pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('audit_natural_g5_online_checkpoint.py','compare_natural_g5_online_matched.py','stage_natural_g5_postrun_audits.py'))
files.extend(p.relative_to(repo) for p in (report/'viewed2048-pilot-depth-comparison').rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.txt') and p.stat().st_size<600000)
files.extend(p.relative_to(repo) for p in (report/'online-matched-depth-expansion').rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.txt') and p.stat().st_size<600000)
files.extend(p.relative_to(repo) for p in (report/'viewed2048-expanded-depth-comparison').rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.txt') and p.stat().st_size<600000)
for folder in ('natural-g5-online-8192','packed-variance','packed-variance-M3to10-8192'):
 files.extend(p.relative_to(repo) for p in (report/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.txt') and p.stat().st_size<600000)
files=sorted(set(p for p in files if (repo/p).is_file()))
fd,index=tempfile.mkstemp(prefix='q7b-root-publish-');os.close(fd);os.unlink(index)
env=dict(os.environ,GIT_INDEX_FILE=index)
try:
 git('read-tree',parent,env=env);git('add','--',*[str(p)for p in files],env=env)
 tree=git('write-tree',env=env)
 if tree==git('rev-parse',parent+'^{tree}'):commit=parent
 else:
  commit=subprocess.check_output(['git','commit-tree',tree,'-p',parent],cwd=repo,env=env,text=True,input='Preserve two-GPU pilot progress and isolated G5 control and sealed production\n').strip()
  git('update-ref',ref,commit,parent)
 remote=None
 if not options.local_only:
  git('push','origin',ref+':refs/heads/'+branch)
  remote=git('ls-remote','origin',ref).split()[0];assert remote==commit
 result=dict(time=time.time(),branch=branch,commit=commit,remote_sha=remote,push_verified=remote==commit,local_only=options.local_only,files=[dict(path=str(p),bytes=(repo/p).stat().st_size,sha256=hashlib.sha256((repo/p).read_bytes()).hexdigest())for p in files])
 report_name='root-pending-publication.json'if options.local_only else'root-current-publication.json'
 (report/report_name).write_text(json.dumps(result,sort_keys=True)+'\n')
 print(json.dumps(dict(commit=commit,remote_sha=remote,push_verified=remote==commit,local_only=options.local_only,files=len(files))))
finally:
 if os.path.exists(index):os.unlink(index)
