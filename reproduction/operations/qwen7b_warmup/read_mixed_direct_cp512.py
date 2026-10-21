"""Bounded pure-CPU status or paired audit of already-scored snapshot outputs."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/';BRANCH='codex/mixed-direct-cp512-eval-20261022';BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/mixed-direct-official-cp512-eval-v1';OUT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-cp512-eval'
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('observe','audit','evidence'),required=True);a=p.parse_args();ref='refs/heads/'+BRANCH;commit=subprocess.check_output(['git','rev-parse',ref],cwd=R).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin',ref],cwd=R).decode().split()[0];assert remote==commit;name=P+'audit_mixed_direct_cp512.py';source=subprocess.check_output(['git','show',commit+':'+name],cwd=R);assert source==(R/name).read_bytes()
 spec=importlib.util.spec_from_file_location('private_cp512_reader',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 if a.action=='audit':body=source.decode()+"\nresult=audit();print(json.dumps(result,sort_keys=True))\n"
 elif a.action=='evidence':body=source.decode()+"\ncp=proof(SNAP/'complete.json');p=SNAP/'E0-CP128-CP512-independent-audit.json';accepted=read(p);assert accepted['complete_refs']['CP512']==cp;print(json.dumps(dict(status='CP512_TERMINAL_COMPLETE_AND_INDEPENDENT_AUDIT_FRESH_METADATA_REVERIFIED',complete=cp,independent_audit=proof(p),initial_complete=accepted['complete_refs']['E0'],source_config=proof(SNAP/'source-config.json'),checkpoint_payload_read=False),sort_keys=True))\n"
 else:body="""import json,pathlib,subprocess,time
b=pathlib.Path(BASE);out=dict(epoch=time.time(),main_job='746247');r=subprocess.run(['sacct','-X','-n','-P','-j','746247','--format=JobIDRaw,State,ExitCode,Elapsed'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);out['main_sacct']=r.stdout.strip();out['sacct_returncode']=r.returncode
for name in('controller.json','evaluation-progress.json','failure.json'):
 try:out[name[:-5]]=json.loads((b/name).read_bytes())
 except FileNotFoundError:pass
p=b/'restore-complete.json'
if p.exists():
 x=json.loads(p.read_bytes());out['restore']=dict(cursor=x['cursor'],full_CP_fresh_SHA_verified=x['full_CP_fresh_SHA_verified'],actual_complete_restore_passed=x['actual_complete_restore_passed'],training_features_reader_constructed=x['training_features_reader_constructed'])
out['prediction_files']={p.name:p.stat().st_size for p in(b/'predictions').glob('*')if p.is_file()};out['complete_exists']=(b/'complete.json').exists();print(json.dumps(out,sort_keys=True))
""".replace('BASE',repr(BASE))
 compile(body,'readonly_snapshot_control','exec');packed=base64.b64encode(zlib.compress(body.encode())).decode();command='env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000;r=ctl.run(command,account='mou176',source_address=ip,timeout=55)
 if r.returncode:raise RuntimeError(r.stdout[-2000:]+r.stderr[-2000:])
 result=json.loads(r.stdout.splitlines()[-1]);OUT.mkdir(parents=True,exist_ok=True);suffix='actual-readonly-'+a.action+'.json';(OUT/suffix).write_text(json.dumps(result,indent=2)+'\n')
 if a.action=='observe':
  with(OUT/'actual-evaluation-observations.jsonl').open('a')as f:f.write(json.dumps(result,sort_keys=True)+'\n')
 print(json.dumps(result))
if __name__=='__main__':main()
