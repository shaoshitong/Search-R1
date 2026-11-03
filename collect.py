"""Read-only collection of small benchmark evidence, never models or caches."""
import pathlib, subprocess, sys, json, base64, hashlib, time

BASE=pathlib.Path(__file__).resolve().parent.parent
HELPER='/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/hpc3_private_control.py'
REMOTE='/data/user/mou176/experiments/vllm-qwen3-8b-context-debug-20261103'

def main():
    script='''import pathlib,json,base64,hashlib,subprocess,time
root=pathlib.Path(REMOTE_ROOT)
result={'observed_at':time.time(),'files':{},'jobs':{}}
for label in ['.','high-utilization','high-utilization-r2']:
 p=root/label
 patterns=['*.json','*.jsonl','*.log','slurm-*.out','source/*.py','source/config.json','source/*.sbatch','storage-sentinel.txt']
 for pattern in patterns:
  for f in p.glob(pattern):
   if not f.is_file() or f.stat().st_size>2*1024**2:continue
   raw=f.read_bytes();rel=str(f.relative_to(root))
   result['files'][rel]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'data':base64.b64encode(raw).decode()}
for key,cmd in [('squeue',['squeue','-j','795495,795513,795528','-h','-o','%i|%j|%T|%M|%R']),('sacct',['sacct','-j','795495,795513,795528','-n','-P','--format=JobID,State,ExitCode,Elapsed,NodeList,AllocTRES'])]:
 r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=10)
 result['jobs'][key]={'code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
print(json.dumps(result))
'''.replace('REMOTE_ROOT',repr(REMOTE))
    r=subprocess.run([sys.executable,HELPER],input="python3 - <<'REMOTE'\n"+script+'\nREMOTE',capture_output=True,text=True,timeout=45)
    if r.returncode:raise RuntimeError(r.stderr[-1000:])
    data=json.loads(r.stdout)
    for name,f in data['files'].items():
        rel=pathlib.Path(name)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('bad relative path')
        raw=base64.b64decode(f.pop('data'))
        assert hashlib.sha256(raw).hexdigest()==f['sha256']
        p=BASE/'remote-evidence'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    (BASE/'collection-latest.json').write_text(json.dumps(data,indent=2))
    print('FILES_VERIFIED',len(data['files']))
    print('JOBS',json.dumps(data['jobs']))
    for label in ['.','high-utilization','high-utilization-r2']:
        p=BASE/'remote-evidence'/label/'results.json'
        if p.exists():
            for c in json.loads(p.read_text())['cases']:
                print('CASE',json.dumps({k:c.get(k) for k in ['name','memory_utilization','pass','exit_code','capacity_tokens','max_model_len']}))
        p=BASE/'remote-evidence'/label/'complete.json'
        if p.exists():print('COMPLETE',label,json.loads(p.read_text()).get('largest_validated_total_tokens'))

if __name__=='__main__':main()
