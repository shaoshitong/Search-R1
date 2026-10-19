"""Explicit pilot/full normal-only Easy Selection launch; no CF, training, or download."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

ADAPTER_COMMIT = 'dde6641b'
GPU_INDEX = 5
BASE = '/data/shaoshitong/grounding-eval-20261019'


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--phase', choices=['pilot96', 'full3610'], required=True)
    p.add_argument('--worktree', required=True)
    p.add_argument('--audit-dir', required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--retriever-root', required=True)
    p.add_argument('--retriever-pid', type=int, required=True)
    p.add_argument('--retriever-birth-ticks', type=int, required=True)
    args = p.parse_args()
    origin = subprocess.check_output(['git','-C',args.worktree,'ls-remote','origin','refs/heads/codex/grounding-eval-strict-20261019'],text=True).split()[0]
    subprocess.run(['git','-C',args.worktree,'merge-base','--is-ancestor',ADAPTER_COMMIT,origin],check=True)
    sources = {}
    for name in ['normal_h100.py', 'normal_job.py']:
        rel = 'reproduction/grounding_eval/' + name
        sources[name] = subprocess.check_output(
            ['git', '-C', args.worktree, 'show', ADAPTER_COMMIT + ':' + rel], text=True)
    payload = dict(phase=args.phase, sources=sources, source_commit=ADAPTER_COMMIT,
                   gpu_index=GPU_INDEX, base=BASE, retriever_root=args.retriever_root, retriever_pid=args.retriever_pid, retriever_birth=args.retriever_birth_ticks)
    if not args.execute:
        print(json.dumps(dict(state='PREPARED_NOT_STARTED', phase=args.phase,
            source_commit=ADAPTER_COMMIT, source_sha256={k: hashlib.sha256(v.encode()).hexdigest()
                for k, v in sources.items()}, normal_only=True)))
        return
    remote = r'''
import pathlib,json,sys,subprocess,os,time,hashlib,urllib.request,csv,io
x=json.load(sys.stdin);base=pathlib.Path(x['base']);phase=x['phase']
root=base/('easy-selection-normal-pilot96' if phase=='pilot96' else 'easy-selection-normal3610')
control=root/'control';runtime=base/('runtime-easy-normal-'+x['source_commit'][:8])
control.mkdir(parents=True,exist_ok=True);runtime.mkdir(parents=True,exist_ok=True)
assert not(control/'launch.json').exists(),'Existing task launch must be inspected, never duplicated'
service=pathlib.Path('/proc')/str(x['retriever_pid']);s=(service/'stat').read_text();birth=int(s[s.rfind(')')+2:].split()[19])
assert birth==x['retriever_birth'] and b'a800_retriever.py' in(service/'cmdline').read_bytes()
retriever_root=pathlib.Path(x['retriever_root']);parity=json.loads((retriever_root/'parity.json').read_text());assert parity['count']==16 and parity['all_equal']
with urllib.request.urlopen('http://127.0.0.1:19001/health',timeout=10)as r:health=json.load(r);assert r.status==200
assert health['pid']==x['retriever_pid'] and health['ready'] and health['gpu_shards']==1
asset=base/'assets/easy-selection-iter200';receipt=json.loads((asset/'asset-migration-complete.json').read_text());assert receipt['status']=='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED' and len(receipt['files'])==13
rows=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True))))
selected=[r for r in rows if int(r[0])==x['gpu_index']];assert len(selected)==1;x['gpu_uuid']=selected[0][1].strip()
assert int(selected[0][2])<=512 and int(selected[0][3])>=48*1024 and int(selected[0][4])==0
apps=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True))))
assert not any(r and r[0].strip()==x['gpu_uuid']for r in apps)
if phase=='full3610':
 pilot=base/'easy-selection-normal-pilot96'
 assert json.loads((pilot/'control/exit.json').read_text())['exit_code']==0
 assert json.loads((pilot/'pilot-verification.json').read_text())['state']=='VERIFIED'
for name,source in x['sources'].items():
 p=runtime/name
 if p.exists():assert p.read_text()==source
 else:
  with p.open('x')as f:f.write(source);f.flush();os.fsync(f.fileno())
 assert hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256(source.encode()).hexdigest()
provenance={'original_retriever':'Easy historical CUDA E5FP16/GPUFlatFP16 top3','actual_retriever_pid':x['retriever_pid'],'actual_retriever_birth_ticks':birth,'actual_retriever_health':health,'actual_shards':1,'original_shards':8,'source_spec_match':True,'g1_16query_cropped_tokens_parity':parity,'topology_difference_explicitly_recorded':True,'original_easy_search_token_parity_verified':False,'g1_parity_is_not_easy_specific_query_parity':True}
prov=root/'retrieval-provenance.json'
with prov.open('x')as f:json.dump(provenance,f,indent=2);f.flush();os.fsync(f.fileno())
command=['/data/search-g1/env/bin/python','-B',str(runtime/'normal_h100.py'),'--method','easy','--runtime-repo',str(base/'runtime-easy-legacy-c05602bccc0c/repo'),'--source-config',str(base/'runtime-easy-legacy-c05602bccc0c/source-config.json'),'--data','/data/shaoshitong/Search-G1-a800-fresh200-a20f0475ec824c69ba1554f39e9ff512/data/final.parquet','--checkpoint',str(base/'assets/easy-selection-iter200'),'--output',str(root),'--run-id','15bb15365346401da1d919e9cde8f813','--identity-file',str(base/'runtime-easy-legacy-c05602bccc0c/identity-51713.json'),'--retrieval-url','http://127.0.0.1:19001/retrieve','--gpu-index',str(x['gpu_index']),'--gpu-uuid',x['gpu_uuid'],'--gpu-lock-directory','/data/shaoshitong/.search-g1-gpu-uuid-locks','--batch-size','32' if phase=='pilot96'else'128','--max-batch-size','64'if phase=='pilot96'else'512','--max-num-seqs','512','--max-num-batched-tokens','16384','--gpu-memory-utilization','0.60','--min-start-free-gib','48','--min-running-free-gib','12','--limit','96'if phase=='pilot96'else'3610','--retrieval-provenance',str(prov)]
job=dict(normal_only=True,normal_command=command,control_output=str(control),normal_output=str(root),source_commit=x['source_commit'],source_sha256={k:hashlib.sha256(v.encode()).hexdigest()for k,v in x['sources'].items()},phase=phase,checkpoint_asset_audit='INDEPENDENT_A800_FRESH_ALL_13FILES_SHA_VERIFIED',cf_authorized=False)
jobpath=control/'job.json'
with jobpath.open('x')as f:json.dump(job,f,indent=2);f.flush();os.fsync(f.fileno())
log=(control/'controller.log').open('ab',buffering=0);p=subprocess.Popen(['/data/search-g1/env/bin/python','-B',str(runtime/'normal_job.py'),'--job',str(jobpath)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
stat=pathlib.Path('/proc')/str(p.pid)/'stat';s=stat.read_text();ticks=int(s[s.rfind(')')+2:].split()[19]);receipt=dict(pid=p.pid,birth_ticks=ticks,phase=phase,command=command,root=str(root),source_commit=x['source_commit'],source_sha256=job['source_sha256'],at=time.time(),normal_only=True,cf_started=False,retriever_restarted=False,gpu_uuid=x['gpu_uuid'],gpu_index=x['gpu_index'],initial_batch=32 if phase=='pilot96' else 128,maximum_batch=64 if phase=='pilot96' else 512,admission_gpu=selected[0])
with(control/'launch.json').open('x')as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(receipt))
'''
    compile(remote, 'explicit-normal-only-launch', 'exec')
    spec = importlib.util.spec_from_file_location('existing_control',
        '/Users/shaoshitong/Documents/remote_codex_control/control.py')
    ctl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctl)
    command = ctl.route('baai-a800', '/usr/bin/python3 -c ' + shlex.quote(remote))
    command[1:1] = ['-o','ServerAliveInterval=5','-o','ServerAliveCountMax=6']
    result = subprocess.run(command,
        input=json.dumps(payload), text=True, capture_output=True, timeout=50)
    audit = dict(transport_exit=result.returncode, stderr=result.stderr[-1000:])
    if result.returncode:
        audit['output_tail'] = result.stdout[-1500:]
    else:
        audit['receipt'] = json.loads(result.stdout.strip().splitlines()[-1])
    Path(args.audit_dir).mkdir(parents=True, exist_ok=True)
    dest = Path(args.audit_dir) / ('easy-a800-' + args.phase + '-launch.json')
    dest.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit))
    if result.returncode:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
