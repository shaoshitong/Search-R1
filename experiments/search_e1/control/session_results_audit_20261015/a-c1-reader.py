"""Read only emitted metrics and small receipts; no node mutations or model loading."""
import os,json,re,time,hashlib,subprocess
from pathlib import Path
SPECS = [{'name': 'Recursive A original complete', 'root': '/dataset/shaoshitong/recursive-bran-a-h100-20261012', 'logs': ['/dataset/shaoshitong/recursive-bran-a-h100-20261012/logs/training.log', '/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013/logs/training.log'], 'receipts': ['receipts/*exit*.json', 'receipts/train-start.json', 'recovery-from100-20261013/receipts/train-start.json', 'final*/*.json', 'eval*/*summary*.json', 'recovery-from100-20261013/final51713/84d95ec7600843efb00a5e16b9f13134/*.json', 'recovery-from100-20261013/final51713-batch1024-v1/evaluation-actual-exit.json']}, {'name': 'C1 root-only ablation150 current', 'root': '/dataset/shaoshitong/recursive-ablation150-20261014/C1-root-only', 'logs': ['/dataset/shaoshitong/recursive-ablation150-20261014/C1-root-only/logs/training-attempt-1791985894622564763.log'], 'receipts': ['receipts/train-start.json', 'receipts/*exit*.json', 'evaluations/*.json', 'evaluation/*.json', 'reports/*summary*.json', 'receipts/current-entry.json', 'output/attempt-b977dbd562cf446694bb189a67cbe438/progress.json']}]
def receipt(p):
    try:
        s=p.stat()
        if s.st_size>2_000_000:return None
        b=p.read_bytes(); d=json.loads(b)
        return {'path':str(p),'bytes':s.st_size,'mtime':s.st_mtime,'sha256':hashlib.sha256(b).hexdigest(),'data':d}
    except (ValueError,OSError):return None
out={'at':time.time(),'experiments':[]}
for spec in SPECS:
    r=Path(spec['root']);e={'name':spec['name'],'root':str(r),'exists':r.exists(),'logs':[],'receipts':[],'inventory':[]}
    for path in spec.get('logs',[]):
        p=Path(path)
        if not p.exists():continue
        rows=[];tail=[];num=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
        for i,l in enumerate(p.open(errors='replace')):
            if len(l)<5000:tail.append(l[-1200:]);tail=tail[-25:]
            m=re.search(r'\bstep:\s*(\d+)\s*-',l)
            if m:
                vals={k:float(v) for k,v in re.findall(r'([\w/.-]+):\s*('+num+')',l)}
                vals={k:v for k,v in vals.items() if any(x in k for x in ['em','score/mean','reward','actual_outer','optimizer_step','scheduler_step','refused_attempts']) or k.startswith('val/')}
                if vals:rows.append({'step':int(m[1]),'line':i+1,'metrics':vals})
        s=p.stat();e['logs'].append({'path':str(p),'bytes':s.st_size,'mtime':s.st_mtime,'rows':rows,'tail':tail})
    seen=set()
    for pattern in spec.get('receipts',[]):
        for p in r.glob(pattern):
            if p in seen:continue
            seen.add(p);v=receipt(p)
            if v:e['receipts'].append(v)
    state=r/'checkpoints/training_state'
    if state.exists():
        ms=sorted(state.glob('global_step_*/manifest.json'),key=lambda p:int(p.parent.name.split('_')[-1]))
        if ms:e['latest_manifest']=receipt(ms[-1])
    for dirname, dirs, files in os.walk(r):
        rel=Path(dirname).relative_to(r)
        if len(rel.parts)>3:dirs[:]=[];continue
        dirs[:]=[x for x in dirs if not any(s in x for s in ['rollout','training_state','policy-safety','ray','model','actor','cache','tensor'])]
        for f in files:
            if f.endswith('.json') and any(s in f.lower() for s in ['final','summary','score','eval','result','manifest','progress','exit','complete']):
                p=Path(dirname)/f;e['inventory'].append({'path':str(p),'bytes':p.stat().st_size,'mtime':p.stat().st_mtime})
    if spec.get('unit'):
        e['unit_state']=subprocess.run(['systemctl','show',spec['unit'],'--property=MainPID,ActiveState,SubState,Result'],capture_output=True,text=True).stdout
    for pid in spec.get('pids',[]):
        p=Path('/proc')/str(pid);e.setdefault('processes',[]).append({'pid':pid,'live':p.exists(),'command':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace') if p.exists() else None})
    out['experiments'].append(e)
print(json.dumps(out,ensure_ascii=False))
