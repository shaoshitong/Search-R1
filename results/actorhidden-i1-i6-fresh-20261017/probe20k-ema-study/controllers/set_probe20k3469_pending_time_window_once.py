from pathlib import Path
import sys,json,datetime
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016')
from reproduction.h1008_qprobe.established_controller_transport import remote
script=r'''
import pathlib,json,subprocess,time
b=pathlib.Path('/mnt/data/shaoshitong/search-g1-prefix-selection200-probe20k-dataset-20261017-0bdddc20840b-r48-resume52');p=b/'preparation/pending-backfill-window-once-v1.json'
if p.exists():print(p.read_text());raise SystemExit
r=subprocess.run(['scontrol','show','job','3469','-o'],capture_output=True,text=True,timeout=20);assert r.returncode==0
before=dict(item.split('=',1)for item in r.stdout.split()if '='in item);assert before['Command']==str(b/'preparation/native-probe20k-data.sh')
change=None
if before['JobState']=='PENDING':
 assert before['RunTime']=='00:00:00'
 change=subprocess.run(['scontrol','update','JobId=3469','TimeLimit=01:00:00'],capture_output=True,text=True,timeout=20);assert change.returncode==0,change.stderr
value=dict(job_id='3469',before=before,after=subprocess.run(['scontrol','show','job','3469','-o'],capture_output=True,text=True).stdout,
 changed_only_own_pending_time_window=change is not None,science_changed=False,foreign_changed=False,expected_remaining_data_blocks=2,measured_first_block_wall_minutes=11.6,window_minutes=60,checked_unix=time.time())
p.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
'''
p=Path('work')/('probe20k3469-backfill-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.transport.json');print(remote(script,str(p)))
