"""Run published stdlib audit only for a real normal-completed owned branch."""
import argparse,hashlib,importlib.util,json,pathlib,shlex,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[4]
CONTROL=ROOT.parents[1]
OUT=pathlib.Path(__file__).parent
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--recipe',choices=('inter1','bothrank1'),action='append');a=p.parse_args()
 source=(OUT/'final-independent-audit.py').read_text();source_sha=hashlib.sha256(source.encode()).hexdigest()
 spec=importlib.util.spec_from_file_location('matched_ranks_final_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
 address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 for recipe in a.recipe or ('inter1','bothrank1'):
  status=json.loads((OUT/('actual-observer/'+recipe+'-latest-good.json')).read_text())
  if not status.get('complete_ref') or '|COMPLETED|0:0|' not in status.get('sacct',''):
   print(json.dumps(dict(recipe=recipe,status='WAITING_REAL_COMPLETE_AND_NORMAL_SLURM_EXIT')));continue
  command='python3 -c '+shlex.quote(source)+' '+shlex.quote(recipe)
  if len(command.encode())>=16000:raise ValueError('Audit source control command exceeds16KB')
  r=ctl.run(command,account='mou176',source_address=address,timeout=90)
  if r.returncode:raise RuntimeError(r.stderr[-2500:])
  result=json.loads(r.stdout.splitlines()[-1]);result['published_audit_source_sha256']=source_sha
  (OUT/('actual-'+recipe+'-final-audit.json')).write_text(json.dumps(result,indent=2)+'\n')
  main=result['evaluations']['selection2048FreshG5HF'];m=main['metrics'];print(json.dumps(dict(recipe=recipe,status=result['status'],job_id=result['job_id'],audit_ref=result['audit_ref'],BA=m['pool']['mixed_balanced_accuracy'],AUC=m['pool']['mixed_roc_auc'],TPR=m['pool']['mixed_recall'],TNR=m['pool']['nonmixed_recall'],allwrong_FPR=m['pool']['allwrong_FPR'],allcorrect_FPR=m['pool']['allcorrect_FPR'],Top512=m['selected512']['mixed_questions'],delta_control_BA=main['delta_BA'],delta_control_Top512=main['delta_Top512'])))
