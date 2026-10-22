import pathlib,json,ast,hashlib,time
REFS={'current64': {'path': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/matched-window-terminal-CPU-acceptance-20261023-v1/current64/epoch-05/accepted.json', 'bytes': 4720, 'sha256': 'e961cdbdac3fc1f37f5c0d71504907a7a1aae2c4fa3a197dc03d81d6448d2f7d'}, 'mse-M5': {'path': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/matched-window-terminal-CPU-acceptance-20261023-v1/mse-M5/epoch-07/accepted.json', 'bytes': 4127, 'sha256': 'cb2212fc882dd92cbce20d2f536af0894110ecb1362b06a3b3abbbb18ada3be5'}, 'window2018': {'path': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/matched-window-terminal-CPU-acceptance-20261023-v1/window2018/epoch-10/accepted.json', 'bytes': 4736, 'sha256': '0857bc597eead3d3bf01f5923060b0a2291bee1948a720c88e9f2b15d1765d73'}}
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def checked(ref):
 p=pathlib.Path(ref['path']);assert proof(p)==ref;return p
out={}
for name,ref in REFS.items():
 accept=json.loads(checked(ref).read_bytes());pubref=accept['source_publication'];pub=json.loads(checked(pubref).read_bytes());relative='reproduction/operations/qwen7b_warmup/'+('online_packed_matched_objective.py'if name=='mse-M5'else'online_packed_window_topk.py');runtime=pathlib.Path(pub['code_root'])/relative;bound=dict(pub['files'][relative],path=str(runtime));source=checked(bound).read_text();tree=ast.parse(source);initial=[];plans=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='SEED'for t in node.targets):
   try:plans.append(ast.literal_eval(node.value))
   except (ValueError,TypeError):pass
  if isinstance(node,ast.Call)and isinstance(node.func,ast.Attribute)and isinstance(node.func.value,ast.Name)and node.func.value.id=='torch'and node.func.attr=='manual_seed'and len(node.args)==1:
   try:initial.append(ast.literal_eval(node.args[0]))
   except (ValueError,TypeError):pass
 assert initial==[9137]and plans==[2026102227]
 binding=accept['binding'];assert binding['seed']==2026102227 and binding['initial_head_sha256']=='777b9a7c77d0402084fcb99929b05c975011e8bf17fe44e4cfe1db016bb02327'
 epoch=accept['epoch'];plan=accept['epoch_plan_SHAs'][str(epoch)];planref=pathlib.Path(accept['checkpoint']['path']).parent/('sampling-plan-epoch-%02d.json'%epoch);saved=json.loads(planref.read_bytes());assert saved['epoch']==epoch and saved['plan_sha256']==plan
 identity=dict(status='ACTUAL_TRAINING_SEED_AND_PLAN_IDENTITY_VERIFIED',time=time.time(),checkpoint=accept['checkpoint'],source_publication=pubref,plan_sha256=plan,init_seed=9137,train_plan_seed=2026102227,epoch=epoch,actual_runtime=bound,source_commit=pub['commit'],actual_initial_head_sha256=binding['initial_head_sha256'],actual_CPU_acceptance=ref,sampling_plan=proof(planref),CP_and_all10_plans_independently_CPU_verified=True,proof_scope='published literal runtime seeds, actual checkpoint binding and independently reconstructed ownG5 plans; no final input read',checkpoint_tensors_opened=False,final4096_read=False)
 dest=pathlib.Path(ref['path']).parent/'actual-seed-identity.json'
 if dest.exists():
  old=json.loads(dest.read_bytes());assert all(old[k]==identity[k]for k in('status','checkpoint','source_publication','plan_sha256','init_seed','train_plan_seed','epoch'));identity=old
 else:dest.write_text(json.dumps(identity,sort_keys=True)+'\n')
 out[name]=dict(identity=identity,proof=proof(dest))
print(json.dumps(dict(time=time.time(),replications=out,final4096_read=False)))
