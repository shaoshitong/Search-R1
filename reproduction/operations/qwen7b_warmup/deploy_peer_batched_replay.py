"""Isolated qualified peer8192 batch replay, four normal1GPU shards, no dependencies."""
import argparse,base64,gzip,hashlib,importlib.util,inspect,io,json,shlex,subprocess,tarfile
from pathlib import Path
from . import deploy_peer_newseed_replay as legacy
from .peer_newseed_batched_replay import TARGET
B=legacy.B;A=legacy.A;P=legacy.PACKAGE
FILES=legacy.FILES+tuple(P+n for n in ('peer_newseed_batched_replay.py','replay_batched_context.py','test_replay_batched_context.py','deploy_peer_batched_replay.py'))
GOLDEN=A/'replay-batched-golden-v3/output/complete.json'


def render_sbatch(shard,base,code,selection,selection_sha,preparation_sha):
    assert 0<=shard<4 and Path(base)==TARGET/('shard-%02d'%shard) and Path(code).parent==TARGET
    script='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-peer8192-batch-SHARD
#SBATCH --account=mou176
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=@@BASE@@/logs/replay-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODEQ@@
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec 9>@@BASEQ@@/replay-job.lock
flock -n 9
python3 - @@CODEQ@@ <<'VERIFY'
import pathlib,json,hashlib,sys
root=pathlib.Path(sys.argv[1]);v=json.loads((root.parent/'source-publication.json').read_text());assert str(root)==v['code_root']
for name,item in v['files'].items():
 p=root/name;assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],name
VERIFY
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.peer_newseed_batched_replay --action capture --selection @@SELQ@@ --selection-sha256 @@SELSHA@@ --output @@BASEQ@@/capture --preparation @@PREPQ@@ --preparation-sha256 @@PREPSHA@@
'''
    values={'SHARD':'%02d'%shard,'@@BASE@@':base,'@@BASEQ@@':shlex.quote(base),'@@CODEQ@@':shlex.quote(code),'@@SELQ@@':shlex.quote(selection),'@@SELSHA@@':selection_sha,'@@PREPQ@@':shlex.quote(str(TARGET/'preparation.json')),'@@PREPSHA@@':preparation_sha}
    for name,value in values.items():script=script.replace(name,value)
    return script

CPU=r'''
guard=lock();pub,root=publication();env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
r=subprocess.run(runtime(root,['unittest','reproduction.operations.qwen7b_warmup.test_replay_batched_context','reproduction.operations.qwen7b_warmup.test_peer_newseed_replay']),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=60);(b/'actual-cpu-tests.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-2000:];print(json.dumps(dict(status='NEW_BATCHED_BRANCH_ACTUAL_CPU_TESTS_PASSED',log=fp(b/'actual-cpu-tests.log'))))
'''
START=r'''
guard=lock();pub,root=publication();complete=pathlib.Path(inputs['ingest'])/'complete.json';assert complete.exists(),'Actual raw G10 acceptance required';assert pathlib.Path(inputs['golden']).exists(),'Actual golden GPU completion required'
if (b/'preparation.json').exists():print(json.dumps(dict(status='EXISTING_ACTUAL_BATCHED_PREPARATION_PRESERVED',complete=fp(b/'preparation.json'))))
elif (b/'cpu-features-launch.json').exists():
 v=read(b/'cpu-features-launch.json');print(json.dumps(dict(status='EXISTING_BATCHED_PREPARE_LAUNCH_PRESERVED',launch=v,same_process_birth=ticks(v['pid'])==v['start_ticks'])))
else:
 assert not (b/'cpu-features-launch-intent.json').exists(),'Unknown prior launch; do not repeat'
 atomic(b/'cpu-features-launch-intent.json',dict(raw=fp(complete),golden=fp(inputs['golden']),batch_size=inputs['batch_size'],time=time.time()))
 env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 args=runtime(root,['reproduction.operations.qwen7b_warmup.peer_newseed_batched_replay','--action','prepare','--raw-complete',str(complete),'--code-root',str(root),'--commit',pub['commit'],'--golden',inputs['golden'],'--batch-size',str(inputs['batch_size'])])
 with (b/'cpu-features.log').open('ab') as log:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 v=dict(status='BATCHED_PREFIX_PROJECTION_STARTED_NOT_PROGRESS_VERIFIED',pid=p.pid,start_ticks=ticks(p.pid),source_publication=fp(b/'source-publication.json'),time=time.time());atomic(b/'cpu-features-launch.json',v);print(json.dumps(v))
'''
FINALIZE=r'''
guard=lock();pub,root=publication();done=read(b/'preparation.json');assert done['source_commit']==pub['commit'] and done['questions']==8192 and done['trajectories']==81920
variants={}
for shard,item in enumerate(done['selections']):
 assert fp(item['path'])==item;base=b/('shard-%02d'%shard);(base/'logs').mkdir(exist_ok=True,parents=True);text=render_sbatch(shard,str(base),str(root),item['path'],item['sha256'],fp(b/'preparation.json')['sha256']);p=base/'capture.sbatch'
 if p.exists():assert p.read_text()==text
 else:p.write_text(text)
 command(['bash','-n',str(p)]);variants[str(shard)]=dict(base=str(base),job_name='q7b-peer8192-batch-%02d'%shard,selection=item,sbatch=fp(p))
value=dict(status='FOUR_QUALIFIED_BATCHED_PEER8192_1GPU_SCRIPTS_READY_NOT_SUBMITTED',variants=variants,preparation=fp(b/'preparation.json'),source_publication=fp(b/'source-publication.json'),no_dependency=True);atomic(b/'submission-plan.json',value);print(json.dumps(value))
'''


def bundle(repo):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0];assert commit==remote,'Publish source first'
    data=io.BytesIO();files={}
    with gzip.GzipFile(fileobj=data,mode='wb',mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as tar:
            for name in FILES:
                raw=subprocess.check_output(['git','show',commit+':'+name],cwd=repo);assert raw==(repo/name).read_bytes(),name;files[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());item=tarfile.TarInfo(name);item.size=len(raw);item.mode=0o444;tar.addfile(item,io.BytesIO(raw))
    return commit,data.getvalue(),files


def main():
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path);p.add_argument('--action',choices=('stage','prepare-start','finalize','submit','status'),default='status');p.add_argument('--batch-size',type=int,choices=(16,32),default=16);p.add_argument('--shard',type=int);p.add_argument('--submit-authorized',action='store_true');p.add_argument('--self-test',action='store_true');a=p.parse_args();repo=Path(__file__).resolve().parents[3]
    inputs=dict(base=str(TARGET),target=str(TARGET),receiver=str(legacy.RECEIVER),ingest=str(legacy.INGEST),golden=str(GOLDEN),batch_size=a.batch_size,commit='a'*40,bundle_sha256='b'*64,bundle_bytes=100,chunks=1,files={n:dict(bytes=100,sha256='c'*64) for n in FILES},shard=a.shard)
    blocks=dict(init=legacy.INIT,chunk=legacy.CHUNK,extract=legacy.EXTRACT,cpu=CPU,start=START,finalize=FINALIZE,submit=legacy.SUBMIT,status=legacy.STATUS)
    def source(action,extra=None):
        own=dict(inputs);own.update(extra or {})
        if action not in ('init','extract'):own.pop('files',None)
        if action!='chunk':
            for key in ('chunk_data','chunk_sha256','chunk_index'):own.pop(key,None)
        return 'inputs='+repr(own)+'\n'+legacy.COMMON+'\n'+(('Path=pathlib.Path\nTARGET=pathlib.Path('+repr(str(TARGET))+')\n'+inspect.getsource(render_sbatch)) if action=='finalize' else '')+'\n'+blocks[action]
    if a.self_test:
        lengths={}
        for action in blocks:
            code=source(action,dict(chunk_data=base64.b64encode(b'x'*4096).decode(),chunk_sha256='d'*64,chunk_index=0));compile(code,action,'exec');lengths[action]=len(('python3 -c '+shlex.quote(code)).encode());assert lengths[action]<16000
        text=render_sbatch(0,str(TARGET/'shard-00'),str(TARGET/'code-a'),str(TARGET/'selection.json'),'a'*64,'b'*64);subprocess.run(['bash','-n'],input=text,text=True,check=True);exports='\n'.join(x for x in text.splitlines() if x.startswith('export '));subprocess.run(['bash','-euc',exports+'\ntest "$QWEN7B_CODE_ROOT" = '+shlex.quote(str(TARGET/'code-a'))],check=True);print(json.dumps(dict(status='BATCHED_PEER_RENDER_AND_EXPORT_SELFTEST_PASS',control_command_bytes=lengths)));return
    if a.control_root is None:p.error('--control-root required')
    if a.action=='submit' and (a.shard not in range(4) or not a.submit_authorized):p.error('Explicit shard and --submit-authorized required')
    data=b''
    if a.action=='stage':
        commit,data,files=bundle(repo);inputs.update(commit=commit,files=files,bundle_bytes=len(data),bundle_sha256=hashlib.sha256(data).hexdigest(),chunks=(len(data)+4095)//4096)
    s=importlib.util.spec_from_file_location('batched_peer_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');control=importlib.util.module_from_spec(s);s.loader.exec_module(control);address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    def call(action,extra=None):
        command='python3 -c '+shlex.quote(source(action,extra));assert len(command.encode())<16000;r_=control.run(command,account='mou176',source_address=address,timeout=100 if action=='cpu' else 55);rows=[json.loads(x) for x in r_.stdout.decode(errors='replace').splitlines() if x.startswith('{')];assert r_.returncode==0 and rows,r_.stderr.decode(errors='replace')[-500:]+' '+r_.stdout.decode(errors='replace')[-1500:];return rows[-1]
    if a.action=='stage':
        call('init')
        for i,start in enumerate(range(0,len(data),4096)):call('chunk',dict(chunk_index=i,chunk_data=base64.b64encode(data[start:start+4096]).decode(),chunk_sha256=hashlib.sha256(data[start:start+4096]).hexdigest()))
        call('extract');value=call('cpu')
    else:value=call('start' if a.action=='prepare-start' else a.action)
    out=a.control_root/'outputs/qwen7b-warmup-20261019'/('peer-batched-'+a.action+('-'+str(a.shard) if a.shard is not None else '')+'.json');out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,sort_keys=True))
if __name__=='__main__':main()
