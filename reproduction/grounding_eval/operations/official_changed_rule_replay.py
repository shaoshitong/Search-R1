# CPU scoring replay of existing task outputs; no GPU or training changes.
# Three pure functions transcribed from Rosy0912/Search-G1 revision
# 0870a1794157ae94bedc8691c65a54028686edc1 (qa_em.py, main_ppo_eco.py, cf_judge_server.py).
# Scope: changed-answer rule/parser only; NOT a full official evaluation reproduction.
import json,pathlib,re,string,types,collections,time,os,argparse,hashlib
exec('def normalize_answer(s):\n    def remove_articles(text):\n        return re.sub(r"\\b(a|an|the)\\b", " ", text)\n\n    def white_space_fix(text):\n        return " ".join(text.split())\n\n    def remove_punc(text):\n        exclude = set(string.punctuation)\n        return "".join(ch for ch in text if ch not in exclude)\n\n    def lower(text):\n        return text.lower()\n\n    return white_space_fix(remove_articles(remove_punc(lower(s))))')
exec('def _cf_trust(self, answer_real, cf_res, solution_str):\n        """Map counterfactual outcome to D_trust."""\n        if cf_res is None or not cf_res.get("injected", False):\n            return _causal_trust(solution_str, floor=self.floor)  # fallback\n        changed = qa_em.normalize_answer(answer_real) != qa_em.normalize_answer(cf_res["ans_cf"])\n        if changed:\n            return self.D_TRUE_TOOL          # really used the tool\n        if cf_res.get("em_cf", 0) == 1:\n            return self.floor                # PARAM_HALL: ignored docs, still correct\n        return self.D_AMBIGUOUS')
exec('def _extract_answer(text: str) -> str:\n    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)\n    if m:\n        return m.group(1).strip()\n    if "<answer>" in text:\n        return text.split("</answer>")[0].split("<answer>")[-1].strip()\n    return text[:80].strip()')
qa_em=types.SimpleNamespace(normalize_answer=normalize_answer)
def _causal_trust(*a,**k): raise ValueError('fallback excluded')
manager=types.SimpleNamespace(D_TRUE_TOOL=1.,floor=.3,D_AMBIGUOUS=.6)
parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--root',default='/data/shaoshitong/grounding-eval-20261019')
parser.add_argument('--method');parser.add_argument('--cf-root');parser.add_argument('--output')
args=parser.parse_args()
if bool(args.method)!=bool(args.cf_root):parser.error('--method and --cf-root are required together')
root=pathlib.Path(args.root);out={}
methods=[(args.method,pathlib.Path(args.cf_root))]if args.method else [('Search-G1',root/'search-g1-cf-full-v4'),('Soft-to-Hard',root/'soft-to-hard-cf-full-v4')]
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for method,cfroot in methods:
 records={}
 for file in (cfroot/'results').glob('*.json'):
  r=json.loads(file.read_text());assert r['request_id'] not in records;records[r['request_id']]=r
 counters=[collections.Counter(),collections.Counter()];original_counts=collections.Counter();skips=collections.Counter();parser_diff=0;used=set()
 for line in (cfroot/'prepared.jsonl').open():
  p=json.loads(line)
  if p['status']=='DIRECT_CORRECT':
   for c in counters:c['Direct']+=1
   continue
  if p['status']!='READY':skips[p['status']]+=1;continue
  r=records[p['request_id']];used.add(p['request_id']);assert r['status']=='CLASSIFIED';original_counts[r['classification']]+=1
  answers=[r['cf_answer']or'',_extract_answer(r['cf_response'])];parser_diff+=answers[0]!=answers[1]
  for c,answer in zip(counters,answers):
   em=int(any(normalize_answer(answer)==normalize_answer(v)for v in p['gold']))
   score=_cf_trust(manager,p['original_answer'],dict(injected=True,ans_cf=answer,em_cf=em),'')
   c['True'if score>=.99 else 'Hall'if abs(score-.3)<1e-6 else 'Amb']+=1
 assert used==set(records)
 def metric(c):
  d,t,a,h=(c[k]for k in ['Direct','True','Amb','Hall']);return dict(counts={k:c[k]for k in ['Direct','True','Amb','Hall']},TC=(d+t+.6*a+.3*h)/(d+t+a+h),D_Inv=h/(t+a+h),TC_denominator=d+t+a+h,DInv_denominator=t+a+h)
 out[method]=dict(cf_root=str(cfroot.resolve()),manifest_sha256=sha(cfroot/'manifest.json'),prepared_sha256=sha(cfroot/'prepared.jsonl'),results_content_sha256=hashlib.sha256(''.join(p.name+':'+sha(p)+'\n'for p in sorted((cfroot/'results').glob('*.json'))).encode()).hexdigest(),classification_cache_used=False,source_counts=dict(original_counts),changed_rule_only=metric(counters[0]),official_parser_plus_changed_rule=metric(counters[1]),parser_differences=parser_diff,skipped_status_counts=dict(skips),valid_cf_records=len(used))
result=dict(checked_at=time.time(),official_revision='0870a1794157ae94bedc8691c65a54028686edc1',scope='Same generated responses; only official scoring rule/parser replay; not official replacement/prompt/judge reproduction; no fallback',methods=out)
p=pathlib.Path(args.output)if args.output else root/'official-code-audit/all-complete-same-response-rule-replay.json'
p.parent.mkdir(parents=True,exist_ok=True)
if p.exists():raise FileExistsError('Use a new rule-replay output; no cached classifications are reused')
with p.open('x')as f:json.dump(result,f);f.flush();os.fsync(f.fileno())
print(json.dumps(result))
