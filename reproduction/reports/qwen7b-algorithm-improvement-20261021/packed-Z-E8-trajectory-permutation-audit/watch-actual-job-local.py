"""Observe Root's audit job through the existing approved direct adapter only."""
from pathlib import Path
import json
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'outputs/qwen7b-warmup-20261019/balanced-control.py').is_file())
command=['python3',str(ROOT/'outputs/qwen7b-warmup-20261019/balanced-control.py'),str(HERE/'read-actual-job-749929.py'),'--timeout','35']
last=None
while True:
    try:
        result=subprocess.run(command,capture_output=True,text=True,timeout=45)
        if result.returncode: raise RuntimeError('Existing direct metadata adapter failed with code%d'%result.returncode)
        value=json.loads(next(line for line in result.stdout.splitlines() if line.startswith('{')))
        (HERE/'actual-live-latest.json').write_text(json.dumps(value,indent=2)+'\n')
        state=(value['accounting']['stdout'], tuple((value.get('progress-rank-%d.json'%i)or{}).get('order') for i in(0,1)), tuple((value.get('progress-rank-%d.json'%i)or{}).get('questions_scored') for i in(0,1)), bool(value.get('complete.json')))
        event=dict(time=time.time(),stage='ACTUAL_READ_ONLY_AUDIT_METADATA',source_time=value['time'],state=state,failures=bool(value['failures']),error=value.get('error_tail'))
        if state!=last:
            (HERE/('actual-live-%d.json'%value['time'])).write_text(json.dumps(value,indent=2)+'\n'); print(json.dumps(event),flush=True);last=state
        with(HERE/'observer-history.jsonl').open('a')as output:output.write(json.dumps(event)+'\n')
        if value.get('complete.json')and '|COMPLETED|0:0|'in value['accounting']['stdout']: break
        if value['failures']or value.get('error_tail'): break
    except Exception as error:
        with(HERE/'observer-history.jsonl').open('a')as output:output.write(json.dumps(dict(time=time.time(),stage='READ_FAILED_CURRENT_STATE_UNKNOWN',error=repr(error)))+'\n')
    time.sleep(60)
