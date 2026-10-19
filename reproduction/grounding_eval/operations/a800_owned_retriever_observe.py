import importlib.util,pathlib,json,subprocess,shlex,time
code=r'''import pathlib,json,time,subprocess,urllib.request
b=pathlib.Path('/data/shaoshitong/grounding-eval-20261019/retriever-owned-51bcf912');out=dict(at=time.time(),files={},process={})
for n in ('admission.json','parity.json','cache/ready.json'):
 p=b/n
 if p.exists():out['files'][n]=json.loads(p.read_text())
l=json.loads((b/'launch.json').read_text());p=pathlib.Path('/proc')/str(l['pid']);out['process']['pid']=l['pid'];out['process']['exists']=p.exists()
if p.exists():
 s=(p/'stat').read_text();out['process']['start_ticks']=int(s[s.rfind(')')+2:].split()[19]);out['process']['command']=(p/'cmdline').read_bytes().decode().replace('\0',' ')
p=b/'service.log';out['log_bytes']=p.stat().st_size;out['log_tail']=p.open('rb').read()[-5000:].decode(errors='replace')
try:
 with urllib.request.urlopen('http://127.0.0.1:19000/health',timeout=2)as f:out['health']=json.loads(f.read())
except Exception as e:out['health_error']=str(e)
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
print(json.dumps(out))
'''
s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=subprocess.run(m.route('baai-a800','/usr/bin/python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=25);v=dict(transport_exit=r.returncode,stderr=r.stderr,observation=json.loads(r.stdout)if not r.returncode else r.stdout);pathlib.Path('outputs/grounding-eval-20261019/a800-owned-retriever-observation-'+str(int(time.time()))+'.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
