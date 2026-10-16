"""Read local experiment receipts; never contact nodes or read tensors/rollouts."""
import json, re, hashlib
from pathlib import Path

OUT = Path(__file__).resolve().parent
CONTROL = Path('/Users/shaoshitong/Documents/Codex/2026-10-02')
PREP = Path('/Users/shaoshitong/Documents/Search-G1/preparation')
ROOTS = [PREP] + [CONTROL / ('users-shaoshitong-documents-remote-codex-control'+s) / t
    for s in ['', '-2', '-5', '-6'] for t in ['work', 'outputs']]
ROOTS += [Path('/Users/shaoshitong/Documents/Search-G1') / n / 'results' for n in
    ['qprobe-prefix10-20-h1008-20261016','qprobe-prefix-only-four-h1008-20261016',
     'qprobe-prefix20-online1000-h1008-20261016']]
KEY = re.compile(r'qprobe|question.?5|question.?20|prefix|dynrpo', re.I)
IDENT = re.compile(r'^(run_id|execution_id|deployment_id|base|job|job_id|slurm_job_id|source_commit|source_branch|status|stage|phase|complete_outers|completed_outers|completed_outer|student_completed_outer|epoch|epochs|head_adam_updates|complete_epochs|actual_head_adam_updates|returncode|exit_code|reason|diagnosis|primary_error|secondary_error|first_error|error|failure_type|failure_reason)$',re.I)

def redact(s):
    s = re.sub(r'hf_[A-Za-z0-9]+', '[HF_TOKEN_REDACTED]', s)
    s = re.sub(r'(?i)(password|token|secret)\s*[=:]\s*[^\s,;]+',r'\1=[REDACTED]',s)
    return s

def walk(v, p='', depth=0):
    if depth > 9: return
    if isinstance(v, dict):
        for k,x in v.items():
            kp=p+'.'+k if p else k
            if IDENT.match(k) and isinstance(x,(str,int,float,bool)):
                yield {'key':kp,'value':redact(str(x))[:1200] if isinstance(x,str) else x}
            if isinstance(x,str) and k in ['stdout','observation','result']:
                for line in x.splitlines():
                    try:
                        z=json.loads(line)
                        if isinstance(z,(dict,list)): yield from walk(z,kp+'.decoded',depth+1)
                    except Exception: pass
            if isinstance(x,str) and any(t in k.lower() for t in ['error','diagnos','failure','tail']):
                for line in x.splitlines():
                    if re.search(r'Error|Exception|Assertion|FAILED|OUT_OF_MEMORY|CUDA|ENOSYS|KeyError|NameError',line):
                        yield {'key':kp+'.errorline','value':redact(line.strip())[:900]}
            if isinstance(x,(dict,list)): yield from walk(x,kp,depth+1)
    elif isinstance(v,list):
        for i,x in enumerate(v):
            if isinstance(x,(dict,list)): yield from walk(x,p+f'[{i}]',depth+1)

rows=[]; seen=set()
for root in ROOTS:
    if not root.exists(): continue
    for f in root.rglob('*.json'):
        if f in seen or not KEY.search(str(f)) or f.stat().st_size>4_000_000: continue
        if any(x in f.parts for x in ['.git','node_modules','env','venv','source','checkpoints']): continue
        if not re.search(r'final|handoff|deployment\.json|submission|terminal|fail|archive|state\.json|acceptance|metrics|summary|human-intent|actual\.json',f.name,re.I):continue
        seen.add(f)
        try:
            raw=f.read_bytes(); d=json.loads(raw); fields=list(walk(d))
        except Exception:continue
        uniq=[]; keys=set()
        for item in fields:
            k=(item['key'],str(item['value']))
            if k not in keys:uniq.append(item);keys.add(k)
        if uniq: rows.append({'source':str(f),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'fields':uniq})
(OUT/'actual-metadata-evidence-index.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'sources':len(rows),'fields':sum(len(r['fields']) for r in rows)},ensure_ascii=False))
