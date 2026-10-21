"""Reconcile exact failed sbatch gateway response; at most one second attempt."""
import argparse,importlib.util,json,pathlib,shlex,subprocess
R=pathlib.Path(__file__).resolve().parents[4];C=R.parents[1];HERE=pathlib.Path(__file__).parent
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-count-frozenE1-fit30016-hold5120-v2'
BODY='''import pathlib,json,hashlib,fcntl,subprocess,time,datetime
b=pathlib.Path(v['base']);lock=(b/'submission-reconcile.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
training=(b/'head-only-training.lock').open('a');fcntl.flock(training,fcntl.LOCK_EX|fcntl.LOCK_NB)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
if(b/'job.id').exists():print(json.dumps(dict(status='EXISTING_AUTHORITATIVE_JOB_PRESERVED',job_id=(b/'job.id').read_text().strip())));raise SystemExit(0)
response=read(b/'submit-response.json');intent=read(b/'submit-intent.json');prep=read(b/'screen-auditor/entry-preparation.json')
assert response['returncode']==1 and not response['stdout'].strip()and 'default account: timeout: context deadline exceeded'in response['stderr']
assert intent['config']==prep['config']==proof(b/'config.json')and intent['config']['sha256']=='fb637c02812dd070cf392a58df75e0d219bbb10d1a2d24cd6e201f3638147767';assert prep['sbatch']==proof(b/'capacity.sbatch')
out=dict(epoch=time.time(),config=prep['config'],sbatch=prep['sbatch'],original_failed_response=proof(b/'submit-response.json'))
for key,args in [('squeue',['squeue','-h','-u','mou176','-n','q7b-frozenE1-head30016','-o','%i|%j|%T|%Z']),('sacct',['sacct','-X','-S',datetime.datetime.fromtimestamp(intent['epoch']-3600).strftime('%Y-%m-%dT%H:%M:%S'),'-u','mou176','-n','-P','--name=q7b-frozenE1-head30016','--format=JobIDRaw,State,ExitCode,JobName%40,WorkDir%220,Submit'])]:
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);assert r.returncode==0,r.stderr;out[key]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
assert not out['squeue']['stdout'].strip(),'Existing named scheduler job: reconcile/adopt it, no retry'
assert not any(str(b)in line for line in out['sacct']['stdout'].splitlines()),'Previously accepted target allocation: do not duplicate'
out['no_accepted_v2_allocation']=True
if v['action']=='inspect':
 p=b/'submit-noaccepted-inspection.json'
 if not p.exists():p.write_text(json.dumps(out)+chr(10))
 print(json.dumps(out));raise SystemExit(0)
prior=read(b/'submit-noaccepted-inspection.json');assert prior['no_accepted_v2_allocation']and prior['config']==out['config']and out['epoch']-prior['epoch']>=10
work=b/'submissions/attempt-0002';work.mkdir(parents=True,exist_ok=True);assert not(work/'intent.json').exists(),'Unknown second-attempt outcome: reconcile before any further action'
(work/'inspection.json').write_text(json.dumps(out)+chr(10));(work/'intent.json').write_text(json.dumps(dict(config=prep['config'],sbatch=prep['sbatch'],same_resources=True,explicit_CLI_account='mou176',root_GO=True,epoch=time.time()))+chr(10))
r=subprocess.run(['sbatch','--parsable','--account=mou176','--qos=normal','--partition=acd_u',str(b/'capacity.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);(work/'response.json').write_text(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))+chr(10));assert r.returncode==0,r.stderr
jid=r.stdout.strip().split(';')[0];assert jid.isdigit(),r.stdout;(b/'job.id').write_text(jid+chr(10));result=dict(status='ONE_V2_JOB_ACCEPTED_AFTER_CONFIRMED_GATEWAY_FAILURE',job_id=jid,config=prep['config'],sbatch=prep['sbatch'],resources=prep['resources'],source_commit=prep['source_commit'],screen_source=prep['screen_source'],submission_attempt=2,first_failed_response=out['original_failed_response'],two_noaccepted_checks=[proof(b/'submit-noaccepted-inspection.json'),proof(work/'inspection.json')],epoch=time.time());(b/'submission.json').write_text(json.dumps(result)+chr(10));print(json.dumps(result))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('inspect','submit-reconciled'),required=True);p.add_argument('--submit-authorized',action='store_true');args=p.parse_args()
 if args.action=='submit-reconciled'and not args.submit_authorized:p.error('Root GO and explicit authorization flag required')
 spec=importlib.util.spec_from_file_location('largecap_gateway_reconcile_private',C/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
 code='v='+repr(dict(base=BASE,action=args.action))+'\n'+BODY;compile(code,'reconciliation-control','exec');command='python3 -c '+shlex.quote(code)
 if len(command.encode())>=16000:raise ValueError('Bounded command required')
 r=ctl.run(command,account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=55)
 if r.returncode:raise RuntimeError('Reconciliation preserved outcome, no blind retry: '+r.stderr.decode()[-2000:]+r.stdout.decode()[-1000:])
 out=json.loads(next(x for x in reversed(r.stdout.decode().splitlines())if x.startswith('{')));(HERE/('actual-v2-gateway-'+args.action+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
