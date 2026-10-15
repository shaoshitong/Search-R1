"""On-demand scalar/receipt inspection; never submits or changes a remote job."""
import argparse
import json
from pathlib import Path
import time

from .local_control import run

REMOTE_SCRIPT = "/data/user/jxu957/experiments/topd-20261008/environment/topd-py310/bin/python - <<'PY'\nfrom pathlib import Path\nimport json,subprocess,time\nb=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-native-thread5-20261015');o={'epoch':time.time()}\nfor n in ['preparation/preparation-submission.json','preparation/progress.json','preparation/native-cpu-tests.json','preparation/preparation-controller.json','preparation/features-ready.json','preparation/warmup-features-ready.json','preparation/warmup-feature-cursor.json','preparation/feature-cursor.json','preparation/formal-submission.json','warmup/ready.json']:\n p=b/n\n if p.is_file():o[n]=json.loads(p.read_text())\ns=o.get('preparation/preparation-submission.json',{})\nif s.get('job'):\n j=s['job'];o['accounting']=subprocess.run(['sacct','-nP','-X','-j',j,'--format=JobID,State,ExitCode,User,AllocTRES,NodeList'],text=True,capture_output=True).stdout;o['queue']=subprocess.run(['squeue','-h','-j',j,'-o','%i|%T|%R'],text=True,capture_output=True).stdout\n p=b/'logs'/('preparation-'+j+'.log')\n if p.exists():\n  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-7000));o['log_tail']=f.read().decode(errors='replace')[-5000:]\nformal=o.get('preparation/formal-submission.json',{})\nif formal.get('job'):\n j=formal['job']\n o['formal_accounting']=subprocess.run(['sacct','-nP','-X','-j',j,'--format=JobID,State,ExitCode,User,AllocTRES,NodeList'],text=True,capture_output=True).stdout\n o['formal_queue']=subprocess.run(['squeue','-h','-j',j,'-o','%i|%T|%R'],text=True,capture_output=True).stdout\n for name in ['logs/pipeline-status.json','preparation/initial0-adopted.json','logs/cleanup-status.json']:\n  p=b/name\n  if p.is_file():o[name]=json.loads(p.read_text())\n p=b/'logs'/('formal-'+j+'.log')\n if p.is_file():\n  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-5000));o['formal_log_tail']=f.read().decode(errors='replace')\n for name in ['logs/actual-updates.jsonl']+['logs/optimizer-rank-'+str(rank)+'.jsonl' for rank in range(4)]:\n  p=b/name\n  if p.is_file():\n   with p.open('rb') as f:f.seek(max(0,p.stat().st_size-16384));lines=f.read().decode(errors='replace').splitlines()\n   rows=[]\n   for line in lines[-4:]:\n    try:rows.append(json.loads(line))\n    except ValueError:pass\n   o[name]=rows\nprint(json.dumps(o,ensure_ascii=False))\nPY"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    result = run('sh -s', REMOTE_SCRIPT.encode(), 'mou176')
    record = dict(control_returncode=result.returncode,
                  control_error=result.stderr.decode(errors='replace'),
                  sampled_unix=time.time())
    if result.returncode == 0:
        record.update(json.loads(result.stdout))
    else:
        record['failure_output'] = result.stdout.decode(errors='replace')
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir/'preparation-latest-sample.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        with (args.output_dir/'preparation-samples.jsonl').open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
    view = {key:record.get(key) for key in (
        'control_returncode','control_error','epoch','accounting','formal_accounting',
        'formal_queue','preparation/progress.json','logs/pipeline-status.json',
        'logs/actual-updates.jsonl')}
    view['log_tail'] = record.get('formal_log_tail',record.get('log_tail',''))[-800:]
    print(json.dumps(view, ensure_ascii=False))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

