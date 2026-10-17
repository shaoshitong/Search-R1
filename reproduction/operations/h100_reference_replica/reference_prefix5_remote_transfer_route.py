"""Remote-to-remote stream over existing SSH identities; no model bytes on Mac."""
from pathlib import Path
import subprocess,json,importlib.util,shlex,datetime
ROOT=Path(__file__).resolve().parent
def controller():
 s=importlib.util.spec_from_file_location('existing_controller','/Users/shaoshitong/Documents/remote_codex_control/control.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def remote_on_mgmt(code, evidence, timeout=240):
 # Forward the already loaded agent for this one connection. No private key,
 # auth.json, global SSH configuration or global known_hosts is copied/edited.
 args=controller().ssh('mgmt-host');args.insert(1,'-A');args+=['python3 -c '+shlex.quote(code)]
 r=subprocess.run(args,capture_output=True,timeout=timeout)
 v={'returncode':r.returncode,'stdout':r.stdout.decode(),'stderr':r.stderr.decode(),'model_payload_via_Mac':False,'private_key_copied':False,'global_ssh_changed':False}
 Path(evidence).write_text(json.dumps(v,indent=2)+'\n');return v
def known():
 q=subprocess.run(['ssh-keygen','-F','[ssh.platform-multi.baai.ac.cn]:2222'],capture_output=True,text=True,check=True)
 return '\n'.join(x for x in q.stdout.splitlines()if x and not x.startswith('#'))+'\n'
if __name__=='__main__':
 hostkey=known();code='''import pathlib,os,subprocess,tempfile,json
d=pathlib.Path(tempfile.mkdtemp(prefix='prefix5-remote-copy-route-'));p=d/'known_hosts';p.write_text(HOSTKEY);p.chmod(0o600)
r=subprocess.run(['ssh','-oBatchMode=yes','-oConnectTimeout=12','-oStrictHostKeyChecking=yes','-oUserKnownHostsFile='+str(p),'-p','2222','job-20e13974-c420-4074-a464-68e93738cfc0-master-0.ldwang.baai-sailing_aquila.dx-calc1.job@ssh.platform-multi.baai.ac.cn','hostname'],capture_output=True,text=True,timeout=25)
print(json.dumps({'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'private_route_dir':str(d)}))
'''.replace('HOSTKEY',repr(hostkey));tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');r=remote_on_mgmt(code,ROOT/('prefix5-remote-copy-route-'+tag+'.json'));print(json.dumps(r))
