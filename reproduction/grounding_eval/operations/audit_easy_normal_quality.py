"""Read-only partial/full Easy normal QA and actual decode finish audit; no raw export."""
import argparse,importlib.util,json,shlex,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser(__doc__);p.add_argument('--audit-dir',required=True);args=p.parse_args()
remote=r'''
import pathlib,json,sys,os,time,collections,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'/data/shaoshitong/grounding-eval-20261019/runtime-21b9ccf4')
from reproduction.grounding_eval.core import normalize,extract_final_answer
root=pathlib.Path('/data/shaoshitong/grounding-eval-20261019/easy-selection-normal3610')
sha=lambda b:hashlib.sha256(b).hexdigest()
counts=collections.Counter();finishes=collections.Counter();stops=collections.Counter();files={};ordinals=[]
for path in sorted((root/'normalized').glob('*.json')):
 raw=path.read_bytes();d=json.loads(raw);files[path.name]=sha(raw)
 for r in d['rows']:
  answer=extract_final_answer(r['model_spans']);counts['questions']+=1;ordinals.append(r['evaluation_ordinal'])
  counts['nonempty_model_spans']+=int(any(x.strip()for x in r['model_spans']))
  counts['nonempty_legal_final_answers']+=int(bool(answer and answer.strip()))
  counts['correct']+=int(bool(answer and normalize(answer)in{normalize(g)for g in r['gold']}))
  counts['questions_with_search']+=int(r['actual_retrieval_calls']>0)
  counts['all_model_spans_empty']+=int(not any(x.strip()for x in r['model_spans']))
  counts['missing_or_illegal_final_answer']+=int(not answer)
  counts['questions_reaching_final_only_turn']+=int(any(t['final_only']for t in r['turns']))
  for t in r['turns']:
   finishes[str(t['finish_reason'])]+=1;stops[str(t['stop_reason'])]+=1
   counts['turns']+=1;counts['explicit_finish_errors']+=int(t['finish_reason']in('error','abort','cancelled'))
assert ordinals==list(range(len(ordinals)))and counts['questions']>0
errors=[]
for path in sorted(root.glob('normal-error-*.json')):
 e=json.loads(path.read_text());errors.append({k:e.get(k)for k in('state','completed','offset','actual_batch_size','error_type','updated_at')})
r=dict(state='READ_ONLY_QA_FINISH_AUDIT',at=time.time(),target=3610,counts=dict(counts),EM=counts['correct']/counts['questions'],actual_finish_reason_counts=dict(finishes),actual_stop_reason_counts=dict(stops),normal_inference_incidents=errors,normalized_file_sha256=files,complete=(root/'complete.json').exists(),note='Raw decode length at original500-token cap is reported, not called failure; retained terminal parsing is qa_protocol strict. Snapshot only; no full inference or CF started.')
print(json.dumps(r))
'''
s=importlib.util.spec_from_file_location('c','/Users/shaoshitong/Documents/remote_codex_control/control.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
cmd=c.route('baai-a800','/data/search-g1/env/bin/python -B -c '+shlex.quote(remote));cmd[1:1]=['-o','ServerAliveInterval=5','-o','ServerAliveCountMax=6']
r=subprocess.run(cmd,text=True,capture_output=True,timeout=45)
assert r.returncode==0,r.stderr[-1500:]
x=json.loads(r.stdout);out=Path(args.audit_dir);out.mkdir(exist_ok=True)
(out/('easy-normal-quality-'+str(time.time_ns())+'.json')).write_text(json.dumps(x,indent=2));(out/'easy-normal-quality-latest.json').write_text(json.dumps(x,indent=2))
print(json.dumps(x))
