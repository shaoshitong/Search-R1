"""Read-only stdlib CPU acceptance, only after the real milestone receipt."""
import argparse,hashlib,importlib.util,json,pathlib,shlex,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1];OUT=pathlib.Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--cursor',type=int,choices=(512,768,938),required=True);a=p.parse_args()
state=json.loads((OUT/'actual-observer/latest-good.json').read_text())
if 'evaluation_'+str(a.cursor) not in state:raise RuntimeError('Wait for actual completed checkpoint-first evaluation')
if a.cursor==938 and '|COMPLETED|0:0|' not in state.get('sacct',''):raise RuntimeError('Wait for normal terminal Slurm state')
source=(OUT/'milestone-independent-audit.py').read_text();digest=hashlib.sha256(source.encode()).hexdigest()
spec=importlib.util.spec_from_file_location('append_final_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
command='python3 -c '+shlex.quote(source)+' '+str(a.cursor)
if len(command.encode())>=16000:raise ValueError('Audit command exceeds16KB')
r=ctl.run(command,account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=95)
if r.returncode:raise RuntimeError(r.stderr[-2500:])
v=json.loads(r.stdout.splitlines()[-1]);v['published_audit_source_sha256']=digest;(OUT/('actual-cursor-%06d-independent-audit.json'%a.cursor)).write_text(json.dumps(v,indent=2)+'\n')
m=v['evaluations']['selection2048FreshG5HF']['metrics'];print(json.dumps(dict(status=v['status'],cursor=a.cursor,job_id=v['job_id'],audit=v['audit_ref'],BA=m['pool']['mixed_balanced_accuracy'],AUC=m['pool']['mixed_roc_auc'],Top512=m['selected512']['mixed_questions'],TPR=m['pool']['mixed_recall'],TNR=m['pool']['nonmixed_recall'],AW_FPR=m['pool']['allwrong_FPR'],AC_FPR=m['pool']['allcorrect_FPR'])))
