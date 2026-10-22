import hashlib,json,os,pathlib,subprocess,tempfile,time
repo=pathlib.Path(__file__).resolve().parents[3]
branch='codex/q7b-native-joint-root-20261021'
ref='refs/heads/'+branch
def git(*args,env=None):
 return subprocess.check_output(['git',*args],cwd=repo,env=env,text=True).strip()
parent=git('rev-parse',ref)
files=[pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('joint_mixed_direct_labelblind.py','direct_update_labelblind.py','labelblind_question_g5_sampler.py','deploy_mixed_direct_e1_uniform_g5_weighted.py','prepare_frozen_final4096_generation.py')]
report=pathlib.Path(__file__).resolve().parent
for folder in ('mixed-direct-e1-unique-g5','online-layer-prefix','uniform-g5-control','unique-vs-labelblind-g5','frozen-final4096-preparation','live-observations'):
 for p in (report/folder).rglob('*'):
  if p.is_file() and p.suffix in ('.py','.md','.json','.jsonl','.txt') and p.stat().st_size<600000:files.append(p.relative_to(repo))
files.extend(pathlib.Path(__file__).resolve().parent.relative_to(repo)/n for n in ('observe-live.py','publish-goal-progress.py'))
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
 git('push','origin',ref+':refs/heads/'+branch)
 remote=git('ls-remote','origin',ref).split()[0];assert remote==commit
 result=dict(time=time.time(),branch=branch,commit=commit,remote_sha=remote,files=[dict(path=str(p),bytes=(repo/p).stat().st_size,sha256=hashlib.sha256((repo/p).read_bytes()).hexdigest())for p in files])
 (report/'root-current-publication.json').write_text(json.dumps(result,sort_keys=True)+'\n')
 print(json.dumps(dict(commit=commit,remote_sha=remote,files=len(files))))
finally:
 if os.path.exists(index):os.unlink(index)
