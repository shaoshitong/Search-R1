"""Foreground follow to the native terminal state; no scheduled/background job."""
import json
from pathlib import Path
import subprocess
import time


def main():
    workdir=Path(__file__).resolve().parents[2]
    while True:
        result=subprocess.run(['python3','-m','reproduction.actor50_probe100.local_watch_once'],
                              cwd=workdir,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              text=True,timeout=85)
        if result.returncode:
            print(json.dumps(dict(observer_unknown=True,returncode=result.returncode,
                                  error=result.stderr[-2500:])),flush=True)
            return
        row=json.loads(result.stdout)
        keys=('at','epoch','Adam_EMA','test_BA','train_BA','test_accuracy','stage',
              'rollout_counts','failure','exit','complete','cleanup')
        print(json.dumps({key:row[key] for key in keys}),flush=True)
        if row['failure'] or row['complete'] or row['exit'] or not row['leader_alive']:
            return
        time.sleep(55)


if __name__=='__main__':main()
