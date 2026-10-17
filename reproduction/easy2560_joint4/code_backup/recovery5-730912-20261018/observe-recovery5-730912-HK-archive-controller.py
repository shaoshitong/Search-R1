"""Read the already launched 730912 HK archive birth and progress; never start it."""
import hashlib,json,os,socket,time
from pathlib import Path
root=Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017/archive-controller/job-730912')
assert socket.gethostname()=='xielab3' and os.getuid()==1002
receipt=json.loads((root/'launch-receipt.json').read_text()); birth=receipt['birth']; pid=birth['pid']
assert receipt['source_job']=='730912' and receipt['source_commit']=='d91575dbca221f66c467dccafa8508d9fb703688'
assert receipt['base']=='/data/user/jxu957/experiments/easy2560-corollout-joint4-recovery5-after730226-20261018' and receipt['run_id']=='dfe9c94084bb48b2a9bb7b60f9d882da'
assert receipt['command'][-2:]==['--first-step','50'] and receipt['requested_steps']==[50,100,150,200]
actual=None;same=False
try:
 path=Path('/proc/%d'%pid);fields=(path/'stat').read_text().rsplit(')',1)[1].split()
 actual=dict(pid=pid,start_ticks=int(fields[19]),state=fields[0],uid=path.stat().st_uid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),cwd=os.readlink(str(path/'cwd')))
 same=all(actual[k]==birth[k] for k in ('pid','start_ticks','uid','boot_id')) and actual['state']!='Z'
except (FileNotFoundError,ProcessLookupError):pass
records={}
for name in ('claim.json','progress.json','failure.json','result.json'):
 path=root/name
 if path.is_file():records[name]=json.loads(path.read_text())
log=root/'controller.stdout.log';tail=''
if log.is_file():
 with log.open('rb') as f:f.seek(max(0,log.stat().st_size-2500));tail=f.read().decode(errors='replace')
publication=Path(receipt['source_root'])/'source-publication.json'
print(json.dumps(dict(status='RECOVERY730912_HK_ARCHIVE_READ_ONLY_OBSERVATION',root=str(root),checked_unix=time.time(),same_birth=same,birth=birth,actual=actual,records=records,log_bytes=log.stat().st_size if log.is_file() else 0,log_tail=tail,source_publication_sha256=hashlib.sha256(publication.read_bytes()).hexdigest(),jobs_submitted=0,controllers_started=0)))
