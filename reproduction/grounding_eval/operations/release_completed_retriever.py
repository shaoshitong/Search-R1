"""Release only this task's single-GPU retriever after six-method acceptance."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess


REMOTE = r'''
import json,os,pathlib,signal,subprocess,time,urllib.request
base=pathlib.Path('/data/shaoshitong/grounding-eval-20261019')
pid=1151202;birth=648183650
def read(p):return json.loads(p.read_text())
assert read(base/'easy-selection-final-delivery.json')['state']=='EASY_NORMAL_AND_DUAL_CF_INDEPENDENTLY_VERIFIED'
assert read(base/'easy-selection-normal3610/control/exit.json')['exit_code']==0
assert read(base/'easy-selection-dual-cf-control/complete.json')['state']=='EASY_DUAL_CF_COMPLETE'
proc=pathlib.Path('/proc')/str(pid)
def identity():
 text=(proc/'stat').read_text();fields=text[text.rfind(')')+2:].split()
 command=(proc/'cmdline').read_bytes().split(bytes([0]))
 assert int(fields[19])==birth
 assert b'/data/shaoshitong/grounding-eval-20261019/runtime-retriever-56df9a1d/a800_retriever.py' in command
 assert b'/data/shaoshitong/grounding-eval-20261019/retriever-owned-single-bounded-56df9a1d' in command
 assert b'GPU-92249a14-a344-f77f-e18c-d1cea5c638a7' in command
 return dict(pid=pid,birth_ticks=birth,command=[x.decode() for x in command if x])
def observe():
 current=identity()
 with urllib.request.urlopen('http://127.0.0.1:19001/health',timeout=4)as response:health=json.loads(response.read())
 assert health['pid']==pid and health['run_id']=='grounding-10b412b7c61a4e4cbc5f635fe53f5fa8'
 return dict(at=time.time(),identity=current,health=health)
observations=[observe()];time.sleep(2);observations.append(observe())
for key in ('actual_http_requests','actual_queries','actual_query_batches'):
 assert observations[0]['health'][key]==observations[1]['health'][key], 'Retriever is still serving requests'
identity();os.kill(pid,signal.SIGTERM)
for unused in range(20):
 if not proc.exists():break
 text=(proc/'stat').read_text();fields=text[text.rfind(')')+2:].split()
 if fields[0]=='Z':break
 time.sleep(.5)
else:raise RuntimeError('Task retriever did not terminate after SIGTERM; no escalation was attempted')
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,check=True)
rows=[line.strip()for line in gpu.stdout.splitlines()if line.strip()]
assert not any(row.split(',')[0].strip()==str(pid)for row in rows)
report=dict(state='OWN_RETRIEVER_RELEASED_AFTER_SIX_METHOD_ACCEPTANCE',at=time.time(),
 observations=observations,signal='SIGTERM',pid=pid,birth_ticks=birth,
 owned_gpu_uuid='GPU-92249a14-a344-f77f-e18c-d1cea5c638a7',owned_cuda_context_absent=True,
 other_compute_contexts_after=rows,foreign_processes_modified=False,source_assets_deleted=False)
out=base/'root-owned-retriever-release.json'
with out.open('x')as handle:json.dump(report,handle,indent=2);handle.flush();os.fsync(handle.fileno())
print(json.dumps(report))
'''


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--final-report',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--execute',action='store_true',required=True)
    args=parser.parse_args()
    report=json.loads(Path(args.final_report).read_text())
    assert report['state']=='ALL_SIX_METHODS_COMPLETE_INDEPENDENTLY_VERIFIED'
    assert len(report['methods'])==6
    assert not Path(args.output).exists()
    spec=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py')
    ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
    command=ctl.route('baai-a800','python3 -B -c '+shlex.quote(REMOTE))
    command[1:1]=['-o','ServerAliveInterval=5','-o','ServerAliveCountMax=6']
    completed=subprocess.run(command,capture_output=True,text=True,timeout=35)
    assert completed.returncode==0,completed.stderr[-1500:]
    value=json.loads(completed.stdout)
    with Path(args.output).open('x')as handle:json.dump(value,handle,indent=2);handle.write('\n')
    print(json.dumps(dict(state=value['state'],pid=value['pid'],output=args.output)))


if __name__=='__main__':main()
