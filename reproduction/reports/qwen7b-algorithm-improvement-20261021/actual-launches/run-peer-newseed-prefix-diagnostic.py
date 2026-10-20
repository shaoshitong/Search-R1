"""RAM-only exclusion identities, remote source aggregation, local small proof."""
import base64,hashlib,importlib.util,json,pathlib,shlex,subprocess,time
REPO=pathlib.Path(__file__).resolve().parents[4]
CONTROL=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
source=REPO/'reproduction/operations/qwen7b_warmup/peer_newseed_prefix_diagnostic.py'
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
assert source.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(source.relative_to(REPO))],cwd=REPO)
spec=importlib.util.spec_from_file_location('diag_hpc_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');hpc=importlib.util.module_from_spec(spec);spec.loader.exec_module(hpc)
root='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-newseed8192-g10-replay-20261021/code-a0b60649f12f'
code='''import hashlib,json,pathlib
from reproduction.operations.qwen7b_warmup import peer_newseed_ingest as m
b=m.B;c=m.read(b/'config.json');bank=m.read(b/'bank/groups.json');selected=m.read(b/'selection-2048-20261020/bank/groups.json')
final=m.final_identity_projection(m.read(b/'algorithm-improvement-20261021/training-extraG5-matched6528-20261021/final-exclusion-plan.json'))
scopes={role:[{k:g[k] for k in ('qid','question_hash')} for g in values] for role,values in [('diagnostic512',bank['diagnostic']),('selection2048',selected['selection_validation']),('frozen_final4096',final)]}
train=bank['train'];assert len(train)==169028;ids=[g['qid'] for g in train];qids=set(ids);hashes={g['question_hash'] for g in train}
intersections={role:dict(qids=sum(g['qid'] in qids for g in values),hashes=sum(g['question_hash'] in hashes for g in values)) for role,values in scopes.items()}
packet=dict(role='peer_prefix_diagnostic_eval_identity_exclusions',scopes=scopes,final_labels_or_rollouts_read=False,
 canonical_train_proof=dict(questions=169028,bank=m.ref(b/'bank/groups.json'),config=m.ref(b/'config.json'),canonical_train_sha256=c['input_sha256']['train'],
 ordered_qid_canonical_SHA=hashlib.sha256(m.canonical(ids)).hexdigest(),ordered_qid_legacy_SHA=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),evaluation_intersections_with_full_original_train=intersections,labels_consulted=False))
print(json.dumps(packet))
'''
address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
command='QWEN7B_CODE_ROOT='+shlex.quote(root)+' CUDA_VISIBLE_DEVICES= bash '+shlex.quote(root+'/reproduction/operations/qwen7b_warmup/runtime.sh')+' -c '+shlex.quote(code)
r=hpc.run(command,account='mou176',source_address=address,timeout=55)
assert r.returncode==0,r.stderr.decode(errors='replace')[-1000:]
packet=json.loads(r.stdout.decode().strip());assert len(json.dumps(packet).encode())<4*1024*1024
# The packet is never written on Mac: only qid/hash metadata crosses RAM/SSH.
spec=importlib.util.spec_from_file_location('diag_nas_direct',pathlib.Path('/Users/shaoshitong/Documents/remote_codex_control/control.py'));ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
raw=source.read_bytes();digest=hashlib.sha256(raw).hexdigest();remote='/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021/_prefix-diagnostic-'+digest[:12]
for i,start in enumerate(range(0,len(raw),4096)):
 block=raw[start:start+4096]
 py="import pathlib,base64,hashlib; r=pathlib.Path(%r);r.mkdir(exist_ok=True);p=r/%r;x=base64.b64decode(%r);assert not p.exists() or p.read_bytes()==x;p.write_bytes(x) if not p.exists() else None"%(remote,'%05d.part'%i,base64.b64encode(block).decode())
 result=subprocess.run(ctl.route('h100-10','python3 -c '+shlex.quote(py)),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=55);assert result.returncode==0,result.stderr.decode(errors='replace')[-500:]
py="import pathlib,hashlib,json; r=pathlib.Path(%r);x=b''.join(p.read_bytes() for p in sorted(r.glob('*.part')));assert hashlib.sha256(x).hexdigest()==%r;p=r/'peer_newseed_prefix_diagnostic.py';assert not p.exists() or p.read_bytes()==x;p.write_bytes(x) if not p.exists() else None;print(json.dumps(dict(code_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))"%(remote,digest)
result=subprocess.run(ctl.route('h100-10','python3 -c '+shlex.quote(py)),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=55);assert result.returncode==0,result.stderr.decode(errors='replace')[-500:]
snapshot='/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021/newseed-20261021-185747-445fc691'
command='python3 '+shlex.quote(remote+'/peer_newseed_prefix_diagnostic.py')+' --snapshot '+shlex.quote(snapshot)
started=time.monotonic();result=subprocess.run(ctl.route('h100-10',command),input=json.dumps(packet).encode()+b'\n',stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')[-3000:])
v=json.loads(result.stdout);v.update(operator_elapsed_seconds=time.monotonic()-started,operator_source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),remote_diagnostic_source_sha256=digest,source_commit=commit,source_code_root=remote)
p=pathlib.Path(__file__).parent/'peer-newseed-prefix-conflict-aggregate.json';p.write_text(json.dumps(v,indent=2)+'\n')
p=CONTROL/'outputs/qwen7b-warmup-20261019/peer-newseed-prefix-conflict-aggregate.json';p.write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps(v,sort_keys=True))
