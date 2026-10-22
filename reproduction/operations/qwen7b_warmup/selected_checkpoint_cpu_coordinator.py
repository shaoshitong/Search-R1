"""Durable serial orchestration of existing terminal-only CPU validators.

The immutable76637 validator owns the shared8GiB-budget mutex and imports Torch
only after acquiring it. This coordinator never imports tensor/model packages.
"""
import argparse,fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path

def read(p):return json.loads(Path(p).read_bytes())
def proof(p):
    p=Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def atomic(p,v):
    p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,sort_keys=True)+'\n');q.replace(p)
def terminal(base):
    if not(base/'complete.json').is_file():return False
    job=str(read(base/'root-submission.json')['job_id'])
    p=subprocess.run(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
    return p.returncode==0 and any(row.split('|')[:3]==[job,'COMPLETED','0:0']for row in p.stdout.splitlines())
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--config-sha',required=True);a=parser.parse_args();assert proof(a.config)['sha256']==a.config_sha
    cfg=read(a.config);out=Path(cfg['out']);assert out.is_dir();guard=(out/'coordinator.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(out/'controller.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],config=proof(a.config),time=time.time()))
    results=[]
    try:
        for item in cfg['targets']:
            base=Path(item['base']);entry=Path(item['entry']['path']);assert proof(entry)==item['entry']
            dest=Path(item['accepted']);epoch=item['epoch']
            while not terminal(base):
                atomic(out/'progress.json',dict(stage='WAITING_ACTUAL_TERMINAL_NO_CP_OR_COMMON_LOCK_HELD',cell=base.name,epoch=epoch,time=time.time()));time.sleep(30)
            if dest.is_file():
                accepted=read(dest);assert accepted['epoch']==epoch and accepted['status']=='TERMINAL_MATCHED_WINDOW_CPU_SHA_MODEL_ADAM_RNG_PLAN_ACCEPTED'
                results.append(dict(cell=base.name,epoch=epoch,status='EXISTING_ACCEPTANCE_REUSED',accepted=proof(dest)));continue
            atomic(out/'progress.json',dict(stage='SERIAL_EXISTING_VALIDATOR_WAITING_FOR_MUTEX_OR_VALIDATING',cell=base.name,epoch=epoch,entry=item['entry'],time=time.time()))
            log=out/(base.name+'-epoch-%02d.log'%epoch)
            with log.open('ab')as stream:
                process=subprocess.Popen(['bash',str(entry),'--epoch',str(epoch)],stdout=stream,stderr=stream,stdin=subprocess.DEVNULL)
                atomic(out/'active-child.json',dict(pid=process.pid,start_ticks=Path('/proc/%d/stat'%process.pid).read_text().rsplit(')',1)[1].split()[19],cell=base.name,epoch=epoch,log=str(log),time=time.time()))
                code=process.wait()  # No short SSH/request timeout may terminate this child.
            if code!=0:raise RuntimeError('CPU validator exited%d; see%s'%(code,log))
            accepted=read(dest);assert accepted['epoch']==epoch and accepted['status']=='TERMINAL_MATCHED_WINDOW_CPU_SHA_MODEL_ADAM_RNG_PLAN_ACCEPTED'
            result=dict(cell=base.name,epoch=epoch,status='ACTUAL_TERMINAL_CPU_ACCEPTANCE_COMPLETE',accepted=proof(dest));results.append(result);atomic(out/'results-so-far.json',dict(results=results,time=time.time()))
        atomic(out/'complete.json',dict(status='ALL_THREE_SELECTED_CHECKPOINT_CPU_VALIDATIONS_COMPLETE',results=results,model_or_optimizer_updated=False,Slurm_mutated=False,time=time.time()))
    except BaseException:
        atomic(out/'failure.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
