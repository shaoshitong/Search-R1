"""Consumed control: install a CPU-only dispatcher without modifying frozen runtime."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


REMOTE = '''import json,pathlib,subprocess,os,time,hashlib
P=pathlib.Path;meta=json.loads(METADATA);base=P(meta['base']);prep=base/'preparation'
assert hashlib.sha256(meta['source'].encode()).hexdigest()==meta['sha256']
intent=prep/'start-after-predecessor-intent.json';receipt=prep/'start-after-predecessor-receipt.json'
if intent.exists() or receipt.exists():
 print(json.dumps({'existing_intent':json.loads(intent.read_text()) if intent.exists() else None,'existing_receipt':json.loads(receipt.read_text()) if receipt.exists() else None,'action':'inspect only; no replay'}));raise SystemExit
assert not (prep/'submit-intent.json').exists() and not (prep/'submit-receipt.json').exists()
config=json.loads((prep/'h1008-launch-config.json').read_text());terminal_base=P(config.get('predecessor_terminal_evidence_base',str(base)))
watch=json.loads((terminal_base/'preparation/predecessor-watch-receipt.json').read_text());stat=P('/proc')/str(watch['pid'])/'stat'
assert (terminal_base/'preparation/predecessor-exit0.json').exists() or (stat.exists() and stat.read_text().rsplit(')',1)[1].split()[19]==watch['start_ticks']),'Terminal capture must remain available'
def once(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
p=prep/'start-after-predecessor-control.py'
with p.open('x') as f:f.write(meta['source']);f.flush();os.fsync(f.fileno())
assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
once(intent,{'published_control_commit':meta['commit'],'source_sha256':meta['sha256'],'created_unix':time.time(),'gpu_use':False,'frozen_training_source':config['source_commit']})
with (base/'logs/start-after-predecessor.log').open('xb') as log:
 child=subprocess.Popen(['/usr/bin/python3',str(p),'--base',str(base)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),start_new_session=True)
stat=P('/proc')/str(child.pid)/'stat';start=stat.read_text().rsplit(')',1)[1].split()[19]
v={'pid':child.pid,'start_ticks':start,'source_sha256':meta['sha256'],'started_unix':time.time(),'gpu_use':False};once(receipt,v);print(json.dumps(v))
'''


def main():
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo).decode().strip()
    proof = json.loads(Path(sys.argv[1]).read_text())
    assert proof['published_control_commit'] == commit and proof['all_blobs_full_sha_match']
    source = (repo / 'reproduction/h1008_qprobe/start_after_predecessor.py').read_text()
    sha = hashlib.sha256(source.encode()).hexdigest()
    assert proof['files']['reproduction/h1008_qprobe/start_after_predecessor.py'] == sha
    base = sys.argv[2] if len(sys.argv) > 2 else '/mnt/data/shaoshitong/search-g1-qprobe3-h1008-86c80e9-20261015-v2'
    assert base.startswith('/mnt/data/shaoshitong/search-g1-qprobe3-h1008-')
    meta = json.dumps(dict(source=source, sha256=sha, commit=commit, base=base))
    script = REMOTE.replace('METADATA', repr(meta))
    sys.path.insert(0, 'work')
    import search_g1_restore5_transport as transport
    original = subprocess.run
    def direct(args, *a, **kw):
        if isinstance(args, list) and 'mgmt-host' in args:
            args = list(args)
            option = 'ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p'
            if option in args:
                index = args.index(option)
                del args[index-1:index+1]
        return original(args, *a, **kw)
    transport.subprocess.run = direct
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    print(transport.remote('/usr/bin/python3 -c ' + shlex.quote(script),
          'outputs/gigpo-e1-instruct200/QPROBE3-START-AFTER-DEPLOY-' + stamp + '.json', emit=False))


if __name__ == '__main__':
    main()
