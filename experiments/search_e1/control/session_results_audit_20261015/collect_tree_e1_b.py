import json,sys,subprocess,shlex,concurrent.futures,os
from pathlib import Path
D=Path(__file__).resolve().parent;S=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2')
sys.path.insert(0,str(S/'work/tree-speed-repair-20261008'));import ssh_retained18
R=(D/'readonly_scalar_reader.py').read_text()
T=max((S/'outputs/tree-nested-branch-handoff-20261013').glob('NATIVE5_CONTINUE_POLL_[0-9]*.json'),key=lambda p:p.stat().st_mtime);t=json.loads(T.read_text())
br='/home/shaoshitong/search-e1-migration-20261011/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011'
bs=br+'/output-speed-resume3-2e3a9029051d4771bc61c95b28b60e21'
tree=[{'name':'Tree nested current','root':t['run'],'unit':t['unit'],'logs':[t['run']+'/stage.log'],'receipts':['output/validation-cost/validation-*.json','run-result.json','checkpoints/training_state/attempt-results/attempt-*.json']}]
for seg in t['progress']['inherited_native_restore']['effective_ancestor_segments']:
 tree.append({'name':'Tree nested ancestor '+str(seg['first_outer'])+'-'+str(seg['last_outer']),'root':seg['source_run'],'logs':[seg['source_run']+'/stage.log'],'receipts':['output/validation-cost/validation-*.json']})
tree.append({'name':'Tree BRANPO 3360 completed','root':'/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-branpo-fresh200-3360','logs':['/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-branpo-fresh200-3360/stage.log'],'receipts':['stage512-step*.json','output/validation-cost/validation-*.json','run-result.json','output/*summary*.json','output/*final*.json','output/eval-resume/scores/*.json']})
tree.append({'name':'Tree original guarded200 legacy3341','root':'/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-current200-continue-3341','logs':['/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-current200-continue-3341/stage.log'],'receipts':['output/validation-cost/validation-*.json','stage512-step*.json','run-result.json','output/eval*/scores/*.json','output/*final*.json']})
a=[{'name':'Recursive A original complete','root':'/dataset/shaoshitong/recursive-bran-a-h100-20261012','logs':['/dataset/shaoshitong/recursive-bran-a-h100-20261012/logs/training.log','/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013/logs/training.log'],'receipts':['receipts/*exit*.json','receipts/train-start.json','recovery-from100-20261013/receipts/train-start.json','final*/*.json','eval*/*summary*.json','recovery-from100-20261013/final51713/84d95ec7600843efb00a5e16b9f13134/*.json','recovery-from100-20261013/final51713-batch1024-v1/evaluation-actual-exit.json']}, {'name':'C1 root-only ablation150 current','root':'/dataset/shaoshitong/recursive-ablation150-20261014/C1-root-only','logs':['/dataset/shaoshitong/recursive-ablation150-20261014/C1-root-only/logs/training-attempt-1791985894622564763.log'],'receipts':['receipts/train-start.json','receipts/*exit*.json','evaluations/*.json','evaluation/*.json','reports/*summary*.json','receipts/current-entry.json','output/attempt-b977dbd562cf446694bb189a67cbe438/progress.json']}]
b=[{'name':'Recursive B four GPU current','root':bs,'logs':[br+'/logs/train-r2-3da9045.log',br+'/logs/isolated-resume-bc666df-a177507c.log','/home/shaoshitong/rbb5-context2-ray/ray/session_latest/logs/worker-11344cfc20371f65c7cab4989b78e523fe63b02ac0ee8493b64d8895-01000000-1111494.out'],'receipts':['output/validation-cost/validation-*.json','validation-cost/validation-*.json','*result*.json','*summary*.json','receipts/*.json'],'pids':[1109195,3072073]}]
def code(specs):return R.replace("SPECS = json.loads(os.environ.get('AUDIT_SPECS','[]'))",'SPECS = '+repr(specs))
def run(name,specs,n):
 c=code(specs);p=D/(name+'-reader.py');p.write_text(c)
 if name in ('tree','a-c1'):
  sock='/tmp/tree-mgmt-3fw9a1ab/control' if name=='tree' else json.loads(Path('outputs/recursive-ablation150-20261014/control/current-state.json').read_text())['connection']['socket']
  out=D/f'{name}-observation-{n}.json'
  r=subprocess.run(['python3','work/recursive-bran-experiment-a-20261011/release/recursive_bran_a/ablation150_infra/task_transport_stdin_v3.py','--management-socket',sock,'--require-socket','--target','h100-10' if name=='tree' else 'h100-11','--python-file',str(p),'--output',str(out),'--timeout','65'],capture_output=True,text=True,timeout=95)
  if r.returncode==0:print(name,n,'OK',flush=True);return
 else:r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','hkust5','python3 -B -c '+shlex.quote(c)],capture_output=True,text=True,timeout=90)
 out=D/f'{name}-observation-{n}.json'
 if r.returncode:out.write_text(json.dumps({'error':r.stderr[-1500:],'stdout':r.stdout[-1500:],'rc':r.returncode}));print(name,n,'ERROR',r.stderr[-500:],flush=True)
 else:out.write_text(r.stdout);print(name,n,'OK',flush=True)
if __name__=='__main__':
 n=int(sys.argv[1]);tasks=[('tree',tree),('a-c1',a),('b',b)]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as e:
  for f in [e.submit(run,name,specs,n) for name,specs in tasks]:
   try:f.result()
   except Exception as ex:print(type(ex).__name__,str(ex)[-500:],flush=True)
