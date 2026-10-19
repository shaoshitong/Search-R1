"""Explicit pilot/full normal-only OPSD launch; no CF, training, or download."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

ADAPTER_COMMIT = '1b72456cb13b191a187f47777d6b65bfba937481'
GPU_UUID = 'GPU-9b8498ae-93e5-dee7-545a-25669080cf83'
BASE = '/data/shaoshitong/grounding-eval-20261019'


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--phase', choices=['pilot96', 'full3610'], required=True)
    p.add_argument('--worktree', required=True)
    p.add_argument('--audit-dir', required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    sources = {}
    for name in ['normal_h100.py', 'normal_job.py']:
        rel = 'reproduction/grounding_eval/' + name
        sources[name] = subprocess.check_output(
            ['git', '-C', args.worktree, 'show', ADAPTER_COMMIT + ':' + rel], text=True)
    payload = dict(phase=args.phase, sources=sources, source_commit=ADAPTER_COMMIT,
                   gpu_uuid=GPU_UUID, base=BASE)
    if not args.execute:
        print(json.dumps(dict(state='PREPARED_NOT_STARTED', phase=args.phase,
            source_commit=ADAPTER_COMMIT, source_sha256={k: hashlib.sha256(v.encode()).hexdigest()
                for k, v in sources.items()}, normal_only=True)))
        return
    remote = r'''
import pathlib,json,sys,subprocess,os,time,hashlib,urllib.request,csv,io
x=json.load(sys.stdin);base=pathlib.Path(x['base']);phase=x['phase']
root=base/('opsd3304-normal-pilot96-v4' if phase=='pilot96' else 'opsd3304-normal-v4')
control=root/'control';runtime=base/('runtime-opsd-normal-'+x['source_commit'][:8])
control.mkdir(parents=True,exist_ok=True);runtime.mkdir(parents=True,exist_ok=True)
assert not(control/'launch.json').exists(),'Existing task launch must be inspected, never duplicated'
service=pathlib.Path('/proc/941984');s=(service/'stat').read_text();birth=int(s[s.rfind(')')+2:].split()[19])
assert birth==647096225 and b'a800_retriever.py' in(service/'cmdline').read_bytes()
with urllib.request.urlopen('http://127.0.0.1:19000/health',timeout=10)as r:health=json.load(r);assert r.status==200
assert health['pid']==941984 and health['ready'] and health['gpu_shards']==2
rows=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True))))
selected=[r for r in rows if r[1].strip()==x['gpu_uuid']];assert len(selected)==1 and int(selected[0][0])==0
assert int(selected[0][2])<=512 and int(selected[0][3])>=48*1024 and int(selected[0][4])==0
apps=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True))))
assert not any(r and r[0].strip()==x['gpu_uuid']for r in apps)
if phase=='full3610':
 pilot=base/'opsd3304-normal-pilot96-v4'
 assert json.loads((pilot/'control/exit.json').read_text())['exit_code']==0
 assert json.loads((pilot/'pilot-verification.json').read_text())['state']=='VERIFIED'
for name,source in x['sources'].items():
 p=runtime/name
 if p.exists():assert p.read_text()==source
 else:
  with p.open('x')as f:f.write(source);f.flush();os.fsync(f.fileno())
 assert hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256(source.encode()).hexdigest()
provenance={'original_retriever':'OPSD3304 historical CUDA E5FP16/GPUFlatFP16,8 shards','actual_retriever_pid':941984,'actual_retriever_birth_ticks':birth,'actual_retriever_health':health,'actual_shards':2,'original_shards':8,'source_spec_match':True,'original_opsd_search_token_parity_verified':False,'g1_16query_parity_is_not_opsd_parity':True,'topology_difference_explicitly_recorded':True}
prov=root/'retrieval-provenance.json'
with prov.open('x')as f:json.dump(provenance,f,indent=2);f.flush();os.fsync(f.fileno())
command=['/data/search-g1/env/bin/python','-B',str(runtime/'normal_h100.py'),'--method','opsd','--runtime-repo',str(base/'runtime-opsd-legacy-2ac4463afa45/repo'),'--source-config',str(base/'runtime-opsd-legacy-2ac4463afa45/source-config.json'),'--data','/data/shaoshitong/Search-G1-a800-fresh200-a20f0475ec824c69ba1554f39e9ff512/data/final.parquet','--checkpoint',str(base/'assets/opsd3304-iter200'),'--output',str(root),'--run-id','H100 native training job3304; formal-3304; actor/global_step_200','--retrieval-url','http://127.0.0.1:19000/retrieve','--gpu-index','0','--gpu-uuid',x['gpu_uuid'],'--gpu-lock-directory','/data/shaoshitong/.search-g1-gpu-uuid-locks','--batch-size','32' if phase=='pilot96'else'128','--max-batch-size','64'if phase=='pilot96'else'256','--max-num-seqs','256','--max-num-batched-tokens','8192','--gpu-memory-utilization','0.60','--min-start-free-gib','48','--min-running-free-gib','12','--limit','96'if phase=='pilot96'else'3610','--retrieval-provenance',str(prov)]
job=dict(normal_only=True,normal_command=command,control_output=str(control),normal_output=str(root),source_commit=x['source_commit'],source_sha256={k:hashlib.sha256(v.encode()).hexdigest()for k,v in x['sources'].items()},phase=phase,checkpoint_asset_audit='INDEPENDENT_A800_FRESH_ALL_12FILES_SHA_VERIFIED',cf_authorized=False)
jobpath=control/'job.json'
with jobpath.open('x')as f:json.dump(job,f,indent=2);f.flush();os.fsync(f.fileno())
log=(control/'controller.log').open('ab',buffering=0);p=subprocess.Popen(['/data/search-g1/env/bin/python','-B',str(runtime/'normal_job.py'),'--job',str(jobpath)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
stat=pathlib.Path('/proc')/str(p.pid)/'stat';s=stat.read_text();ticks=int(s[s.rfind(')')+2:].split()[19]);receipt=dict(pid=p.pid,birth_ticks=ticks,phase=phase,command=command,root=str(root),source_commit=x['source_commit'],source_sha256=job['source_sha256'],at=time.time(),normal_only=True,cf_started=False,retriever_restarted=False,gpu_uuid=x['gpu_uuid'],gpu_index=0,initial_batch=32 if phase=='pilot96' else 128,maximum_batch=64 if phase=='pilot96' else 256,admission_gpu=selected[0])
with(control/'launch.json').open('x')as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(receipt))
'''
    compile(remote, 'explicit-normal-only-launch', 'exec')
    spec = importlib.util.spec_from_file_location('existing_control',
        '/Users/shaoshitong/Documents/remote_codex_control/control.py')
    ctl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctl)
    result = subprocess.run(ctl.route('baai-a800', '/usr/bin/python3 -c ' + shlex.quote(remote)),
        input=json.dumps(payload), text=True, capture_output=True, timeout=50)
    audit = dict(transport_exit=result.returncode, stderr=result.stderr[-1000:])
    if result.returncode:
        audit['output_tail'] = result.stdout[-1500:]
    else:
        audit['receipt'] = json.loads(result.stdout.strip().splitlines()[-1])
    Path(args.audit_dir).mkdir(parents=True, exist_ok=True)
    dest = Path(args.audit_dir) / ('opsd-a800-' + args.phase + '-launch.json')
    dest.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit))
    if result.returncode:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
