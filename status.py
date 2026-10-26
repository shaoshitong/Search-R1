"""只读复查两个训练服务；不重启、不停止进程。"""
import concurrent.futures
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
CONTROLLER = '/Users/shaoshitong/Documents/remote_codex_control_cc/control.py'
PROBE = r'''
import pathlib,subprocess,json,datetime
root=pathlib.Path.home()/DIR
unit=UNIT+'.service'
r=subprocess.run(['systemctl','show',unit,'-p','ActiveState','-p','SubState','-p','Result',
                  '-p','MainPID','-p','ExecMainStatus','-p','RuntimeMaxUSec',
                  '-p','ActiveEnterTimestamp','-p','NRestarts','-p','KillMode',
                  '-p','TimeoutStopUSec'],capture_output=True,text=True)
progress=[]
for p in sorted((root/'progress').glob('rank-*.json')):
    try:progress.append(json.loads(p.read_text()))
    except (OSError,ValueError):pass
log=root/'train.log';tail=''
if log.exists():
    with log.open('rb') as f:
        f.seek(max(0,log.stat().st_size-5000));tail=f.read().decode(errors='replace')
g=subprocess.run(['nvidia-smi','-i',GPUS,
                  '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                  '--format=csv,noheader,nounits'],capture_output=True,text=True)
print('STATUS_JSON:'+json.dumps(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      unit=unit,systemd=r.stdout,systemd_exit=r.returncode,progress=progress,
      log_tail=tail,gpu=g.stdout,gpu_exit=g.returncode),ensure_ascii=False))
'''


def check(n, node):
    code = PROBE.replace('DIR', repr(Path(node['directory']).name))
    code = code.replace('UNIT', repr(node['unit'])).replace('GPUS', repr(node['gpus']))
    r = subprocess.run(['python3', CONTROLLER, 'run', '--timeout', '20',
                        'lab_4090_' + str(n), '--', 'python3', '-c', code],
                       capture_output=True, text=True, timeout=30)
    vals = [s[len('STATUS_JSON:'):] for s in r.stdout.splitlines() if s.startswith('STATUS_JSON:')]
    return dict(host=n, exit=r.returncode, status=json.loads(vals[-1]) if vals else None)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--save')
    a = p.parse_args()
    meta = json.loads((ROOT / 'deployment.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda n: check(n, meta['nodes'][str(n)]), [2, 3]))
    if a.save:
        Path(a.save).write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows, ensure_ascii=False))
