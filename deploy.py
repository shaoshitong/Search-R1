"""部署代码并校验 SHA；不启动训练，不读取或保存密码。"""
import argparse
import base64
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
CONTROLLER = '/Users/shaoshitong/Documents/remote_codex_control_cc/control.py'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--python-hkust2', required=True)
    p.add_argument('--python-hkust3', required=True)
    a = p.parse_args()
    files = {n: base64.b64encode((ROOT / n).read_bytes()).decode()
             for n in ['train.py', 'launch.py', 'README.md']}
    sha = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in files}
    runid = datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S')
    meta = dict(run_id=runid, source_commit=subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(), sha256=sha, nodes={})
    for n in [2, 3]:
        dirname = 'resnet50-noise-20261026-hkust%d-%s' % (n, runid)
        meta['nodes'][str(n)] = dict(directory='/home/shaoshitong/' + dirname, unit=dirname,
            python=a.python_hkust2 if n == 2 else a.python_hkust3,
            gpus='4,5' if n == 2 else '4,5,6,7')

    def deploy(n):
        dirname = Path(meta['nodes'][str(n)]['directory']).name
        code = """from pathlib import Path
import json,base64,hashlib
root=Path.home()/NAME
root.mkdir(exist_ok=False)
files=FILES
expected=HASHES
for name,data in files.items():
    p=root/name;p.write_bytes(base64.b64decode(data));p.chmod(0o644)
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected[name]
print('DEPLOY_JSON:'+json.dumps(dict(sha256=expected)))
""".replace('NAME', repr(dirname)).replace('FILES', repr(files)).replace('HASHES', repr(sha))
        r = subprocess.run(['python3', CONTROLLER, 'run', '--timeout', '20',
                            'lab_4090_' + str(n), '--', 'python3', '-c', code],
                           capture_output=True, text=True, timeout=30)
        rows = [s[len('DEPLOY_JSON:'):] for s in r.stdout.splitlines() if s.startswith('DEPLOY_JSON:')]
        assert r.returncode == 0 and rows, '远端部署失败：HKUST%d' % n
        assert json.loads(rows[-1])['sha256'] == sha
        return dict(host=n, verified_sha256=sha)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(deploy, [2, 3]))
    old = ROOT / 'deployment.json'
    if old.exists():
        previous = json.loads(old.read_text())
        history = ROOT / ('deployment-attempt-' + previous['run_id'] + '.json')
        assert not history.exists()
        old.rename(history)
    meta['deployment_verification'] = receipts
    old.write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
