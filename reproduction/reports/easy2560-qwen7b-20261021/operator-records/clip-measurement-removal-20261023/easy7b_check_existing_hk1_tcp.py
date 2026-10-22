"""Read-only TCP check via configured HK1; no HPC authentication or credentials."""
import json,shlex,subprocess,time
from pathlib import Path
code="import json,socket; s=socket.create_connection(('10.120.48.31',22),8); print(json.dumps({'hpc_endpoint_tcp_reachable':True})); s.close()"
cmd=['ssh','-T','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=8','hkust1',shlex.join(['python3','-c',code])]
r=subprocess.run(cmd,capture_output=True,text=True,timeout=25)
v={'epoch':time.time(),'route':'configured_hkust1_existing_strict_host_keys','returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'hpc_auth_attempted':False,'credentials_loaded':False,'training_modified':False}
out=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/easy2560-qwen7b-20261021')
p=out/('campus-route-reachability-'+str(time.time_ns())+'.json');p.write_text(json.dumps(v,indent=2)+'\n');print(p);print(json.dumps(v))
