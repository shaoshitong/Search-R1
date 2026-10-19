# Task-local operator: execute from the xia workspace root. No model bytes are transferred.
import importlib.util,pathlib,json,subprocess,shlex,hashlib
w=pathlib.Path('work/grounding-eval-strict-20261019');commit=subprocess.check_output(['git','-C',str(w),'rev-parse','HEAD'],text=True).strip();rel='reproduction/grounding_eval/a800_retriever.py';source=subprocess.check_output(['git','-C',str(w),'show',commit+':'+rel],text=True);digest=hashlib.sha256(source.encode()).hexdigest()
code=r'''import pathlib,json,subprocess,time,os,hashlib,shutil
base=pathlib.Path('/data/shaoshitong/grounding-eval-20261019');commit=COMMIT;source=SOURCE;digest=DIGEST
runtime=base/('runtime-retriever-'+commit[:8]);runtime.mkdir(exist_ok=True);script=runtime/'a800_retriever.py'
if script.exists():assert hashlib.sha256(script.read_bytes()).hexdigest()==digest
else:
 with script.open('x')as f:f.write(source);f.flush();os.fsync(f.fileno())
assert hashlib.sha256(script.read_bytes()).hexdigest()==digest
assert not pathlib.Path('/proc/941984').exists(),'Prior task-owned retriever must have exited'
out=base/'retriever-owned-single-c0e14085';out.mkdir(exist_ok=True)
assert not (out/'launch.json').exists(),'Existing launch must be inspected; never duplicated'
src=base/'retriever-owned-51bcf912/data-cache';dst=src;assert dst.is_dir()
for name in ('wiki_dump.jsonl','corpus.json'):
 p=dst/name
 if not p.exists():
  if name.endswith('.jsonl'):os.link(src/name,p)
  else:shutil.copy2(src/name,p)
# Keep extracted corpus as immutable read-only hardlink. Arrow materialization is
# task-owned to prevent writes to another experiment's cache.
cmd=['/data/search-g1/env/bin/python',str(script),'--source-repo','/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-54ac33dcebe64e62a11fe4177f41f116-resume-40389d9b00204a80ba68475fac254f9c/repo','--output',str(out),'--gpu-uuids','GPU-92249a14-a344-f77f-e18c-d1cea5c638a7','--tokenizer','/data/shaoshitong/Search-G1-a800-fresh200-a20f0475ec824c69ba1554f39e9ff512/checkpoints/search-g1-common8/actor/global_step_200','--records',str(base/'search-g1-normal/records'),'--data-cache',str(dst),'--port','19000']
log=(out/'service.log').open('ab',buffering=0);p=subprocess.Popen(cmd,cwd=str(runtime),stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();birth=pathlib.Path('/proc')/str(p.pid)/'stat';s=birth.read_text();ticks=int(s[s.rfind(')')+2:].split()[19]);receipt=dict(pid=p.pid,start_ticks=ticks,command=cmd,source_commit=commit,source_sha256=digest,at=time.time(),original_service_restarted=False,own_independent_service=True,parity_before_listen=True)
with(out/'launch.json').open('x')as f:json.dump(receipt,f);f.flush();os.fsync(f.fileno())
print(json.dumps(receipt))
'''.replace('COMMIT',repr(commit)).replace('SOURCE',repr(source)).replace('DIGEST',repr(digest))
s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=subprocess.run(m.route('baai-a800','/usr/bin/python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=25);v=dict(transport_exit=r.returncode,stderr=r.stderr,receipt=json.loads(r.stdout)if not r.returncode else r.stdout);pathlib.Path('outputs/grounding-eval-20261019/a800-owned-single-retriever-launch.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
