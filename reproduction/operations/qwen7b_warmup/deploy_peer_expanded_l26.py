"""Guarded immutable additional21824 replay, normal four1GPU, no training."""
import argparse,base64,gzip,hashlib,importlib.util,inspect,io,json,shlex,subprocess,tarfile
from pathlib import Path
from . import deploy_peer_newseed_replay as legacy
from . import deploy_peer_batched_replay as batch
from .peer_expanded_l26_replay import TARGET
P=legacy.PACKAGE
FILES=tuple(dict.fromkeys(legacy.FILES+tuple(P+n for n in ('replay_batched_context.py','peer_newseed_batched_replay.py','deploy_peer_batched_replay.py','test_replay_batched_context.py','validation_fixedg5_replay.py','peer_expanded_l26_replay.py','test_peer_expanded_l26.py','deploy_peer_expanded_l26.py'))))


def render_sbatch(shard,base,code,selection,selection_sha,preparation_sha):
    if type(shard) is not int or shard not in range(4) or Path(base)!=TARGET/('shard-%02d'%shard) or Path(code).parent!=TARGET:
        raise ValueError('Four explicit new independent expanded shards required')
    if len(selection_sha)!=64 or len(preparation_sha)!=64:raise ValueError('Bound actual selection/preparation SHA required')
    text='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-expanded-L26-@@SHARD@@
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
python3 - @@CODEQ@@ @@PREPQ@@ @@PREPSHA@@ <<'VERIFY'
import pathlib,json,hashlib,sys
root=pathlib.Path(sys.argv[1]);v=json.loads((root.parent/'source-publication.json').read_text());assert str(root)==v['code_root']
for name,item in v['files'].items():
 p=root/name;assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],name
p=pathlib.Path(sys.argv[2]);assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[3]
VERIFY
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.peer_expanded_l26_replay --action capture --selection @@SELQ@@ --selection-sha256 @@SELSHA@@ --output @@BASEQ@@/capture
'''
    for key,value in {'@@SHARD@@':'%02d'%shard,'@@BASE@@':str(base),'@@BASEQ@@':shlex.quote(str(base)),'@@CODEQ@@':shlex.quote(str(code)),'@@PREPQ@@':shlex.quote(str(TARGET/'preparation.json')),'@@PREPSHA@@':preparation_sha,'@@SELQ@@':shlex.quote(str(selection)),'@@SELSHA@@':selection_sha}.items():text=text.replace(key,value)
    return text


CPU=r'''
guard=lock();pub,root=publication();env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
r=subprocess.run(runtime(root,['unittest','reproduction.operations.qwen7b_warmup.test_peer_expanded_l26']),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=55);(b/'actual-cpu-tests.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-2000:];print(json.dumps(dict(status='EXPANDED_REPLAY_ACTUAL_CPU_TESTS_PASSED',log=fp(b/'actual-cpu-tests.log'))))
'''
START=r'''
guard=lock();pub,root=publication();receipt=b/'cpu-features-launch.json'
if (b/'preparation.json').exists():print(json.dumps(dict(status='EXISTING_PREPARATION_PRESERVED',preparation=fp(b/'preparation.json'))))
elif receipt.exists():
 v=read(receipt);print(json.dumps(dict(status='EXISTING_PREFIX_PROJECTION_PRESERVED',launch=v,same_process_birth=ticks(v['pid'])==v['start_ticks'])))
else:
 assert not (b/'cpu-features-launch-intent.json').exists(),'Unknown prior launch; do not repeat'
 atomic(b/'cpu-features-launch-intent.json',dict(identity_plan=fp(inputs['identity']),source_publication=fp(b/'source-publication.json'),time=time.time()))
 env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 args=runtime(root,['reproduction.operations.qwen7b_warmup.peer_expanded_l26_replay','--action','prepare','--code-root',str(root),'--commit',pub['commit'],'--golden',inputs['golden']])
 with (b/'cpu-features.log').open('ab') as log:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 v=dict(status='PREFIX_PROJECTION_STARTED_NOT_PROGRESS_VERIFIED',pid=p.pid,start_ticks=ticks(p.pid),time=time.time());atomic(receipt,v);print(json.dumps(v))
'''
FINALIZE=r'''
guard=lock();pub,root=publication();done=read(b/'preparation.json');assert done['source_commit']==pub['commit'] and done['questions']==21824 and done['trajectories']==218240 and done['persist_layers']==[26] and len(done['selections'])==4
variants={}
for shard,item in enumerate(done['selections']):
 assert fp(item['path'])==item;base=b/('shard-%02d'%shard);selection=read(item['path']);assert selection['questions']==5456 and len(selection['trajectories'])==54560
 p=base/'capture.sbatch';text=render_sbatch(shard,str(base),str(root),item['path'],item['sha256'],fp(b/'preparation.json')['sha256'])
 if p.exists():assert p.read_text()==text
 else:p.write_text(text)
 command(['bash','-n',str(p)]);variants[str(shard)]=dict(base=str(base),job_name='q7b-expanded-L26-%02d'%shard,selection=item,sbatch=fp(p))
value=dict(status='FOUR_ADDITIONAL21824_LAYER26_SCRIPTS_READY_NOT_SUBMITTED',variants=variants,preparation=fp(b/'preparation.json'),source_publication=fp(b/'source-publication.json'),no_dependency=True,probe_training_submitted=False);atomic(b/'submission-plan.json',value);print(json.dumps(value))
'''


def bundle(repo,commit):
    published=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/peer-expanded-L26-replay-20261021'],cwd=repo,text=True).split()[0]
    if published!=commit:raise ValueError('Publish exact immutable expanded source branch first')
    data=io.BytesIO();files={}
    with gzip.GzipFile(fileobj=data,mode='wb',mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as tar:
            for name in FILES:
                raw_=subprocess.check_output(['git','show',commit+':'+name],cwd=repo)
                if raw_!=(repo/name).read_bytes():raise ValueError('Local code differs from published immutable source: '+name)
                files[name]=dict(bytes=len(raw_),sha256=hashlib.sha256(raw_).hexdigest());entry=tarfile.TarInfo(name);entry.size=len(raw_);entry.mode=0o444;tar.addfile(entry,io.BytesIO(raw_))
    return data.getvalue(),files


def main():
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path);p.add_argument('--commit',required=True);p.add_argument('--action',choices=('stage','prepare-start','finalize','submit','status'),default='status');p.add_argument('--shard',type=int);p.add_argument('--submit-authorized',action='store_true');p.add_argument('--self-test',action='store_true');a=p.parse_args();repo=Path(__file__).resolve().parents[3]
    inputs=dict(base=str(TARGET),target=str(TARGET),receiver=str(legacy.RECEIVER),ingest=str(legacy.INGEST),identity=str(TARGET.parent/'peer-expanded-identity-v1/identity-plan.json'),golden=str(batch.GOLDEN),commit=a.commit,bundle_sha256='b'*64,bundle_bytes=100,chunks=1,files={n:dict(bytes=100,sha256='c'*64) for n in FILES},shard=a.shard)
    blocks=dict(init=legacy.INIT,chunk=legacy.CHUNK,extract=legacy.EXTRACT,cpu=CPU,start=START,finalize=FINALIZE,submit=legacy.SUBMIT,status=batch.STATUS)
    def source(action,extra=None):
        own=dict(inputs);own.update(extra or {})
        if action not in ('init','extract'):own.pop('files',None)
        if action!='chunk':
            for key in ('chunk_data','chunk_sha256','chunk_index'):own.pop(key,None)
        definitions='Path=pathlib.Path\nTARGET=Path('+repr(str(TARGET))+')\n'+inspect.getsource(render_sbatch) if action=='finalize' else ''
        return 'inputs='+repr(own)+'\n'+legacy.COMMON+'\n'+definitions+'\n'+blocks[action]
    if a.self_test:
        lengths={}
        for action in blocks:
            code=source(action,dict(chunk_data=base64.b64encode(b'x'*4096).decode(),chunk_sha256='d'*64,chunk_index=0));compile(code,action,'exec');lengths[action]=len(('python3 -c '+shlex.quote(code)).encode());assert lengths[action]<16000
        text=render_sbatch(0,TARGET/'shard-00',TARGET/'code-test',TARGET/'selection.json','a'*64,'b'*64);subprocess.run(['bash','-n'],input=text,text=True,check=True)
        exports='\n'.join(x for x in text.splitlines() if x.startswith('export '));subprocess.run(['bash','-euc',exports+'\ntest "$QWEN7B_CODE_ROOT" = '+shlex.quote(str(TARGET/'code-test'))],check=True);print(json.dumps(dict(status='EXPANDED_SAFE_RENDER_EXPORT_AND_TRANSPORT_TESTS_PASSED',control_command_bytes=lengths)));return
    if a.control_root is None:p.error('--control-root required')
    if a.action=='submit' and (a.shard not in range(4) or not a.submit_authorized):p.error('Explicit shard + --submit-authorized required')
    data=b''
    if a.action=='stage':
        data,files=bundle(repo,a.commit);inputs.update(files=files,bundle_bytes=len(data),bundle_sha256=hashlib.sha256(data).hexdigest(),chunks=(len(data)+4095)//4096)
    spec=importlib.util.spec_from_file_location('expanded_private_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    def call(action,extra=None):
        command='python3 -c '+shlex.quote(source(action,extra));assert len(command.encode())<16000;result=control.run(command,account='mou176',source_address=ip,timeout=55)
        rows=[json.loads(x) for x in result.stdout.decode(errors='replace').splitlines() if x.startswith('{')]
        if result.returncode or not rows:raise RuntimeError(result.stderr.decode(errors='replace')[-1500:]+' '+result.stdout.decode(errors='replace')[-1500:])
        return rows[-1]
    if a.action=='stage':
        call('init')
        for i,start in enumerate(range(0,len(data),4096)):call('chunk',dict(chunk_index=i,chunk_data=base64.b64encode(data[start:start+4096]).decode(),chunk_sha256=hashlib.sha256(data[start:start+4096]).hexdigest()))
        call('extract');value=call('cpu')
    else:value=call('start' if a.action=='prepare-start' else a.action)
    out=a.control_root/'outputs/qwen7b-warmup-20261019/reports'/('peer-expanded-L26-'+a.action+('-'+str(a.shard) if a.shard is not None else '')+'.json');out.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');print(json.dumps(value))
if __name__=='__main__':main()
