"""Build a published code-only CPU analysis deployment; no training actions."""
import argparse
import base64
import gzip
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit,
        'release/recursive_bran_a/ablation150_aggregate_watch_v1',
        'release/recursive_bran_a/ablation150_storage_v1',
        'release/recursive_bran_a/ablation150_closed_storage_v2',
        'release/recursive_bran_a/ablation150_tree_summary_v1',
        'release/recursive_bran_a/ablation150_archive_v1'], text=True).splitlines()
    paths += ['release/recursive_bran_a/ablation150_reports_v1/report.py',
              'release/recursive_bran_a/ablation150_v1/contracts.py',
              'release/recursive_bran_a/ablation150_v1/__init__.py',
              'release/recursive_bran_a/ablation150_infra/analysis_cli.py',
              'release/recursive_bran_a/ablation150_infra/auxiliary_cli.py',
              'search_r1/baselines/rollout.py']
    files = {name: base64.b64encode(subprocess.check_output(
        ['git', 'show', f'{commit}:{name}'])).decode() for name in paths}
    payload = base64.b64encode(gzip.compress(json.dumps(dict(commit=commit, files=files)).encode())).decode()
    code = '''import base64,gzip,hashlib,json,time
from pathlib import Path
r=json.loads(gzip.decompress(base64.b64decode(PAYLOAD)))
b=Path('/dataset/shaoshitong/recursive-ablation150-20261014')
s=json.loads((b/'study-progress.json').read_text())
assert s['source_commit']=='233b0dc95de3f7b797d5b424d3b7dcaafca5328f'
target=b/('source-analysis-'+r['commit'][:8]);target.mkdir(exist_ok=True)
hashes={}
for name,value in r['files'].items():
 p=Path(name);assert not p.is_absolute() and '..' not in p.parts
 path=target/p;data=base64.b64decode(value);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():assert path.read_bytes()==data
 else:path.write_bytes(data)
 hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
 assert hashes[name]==hashlib.sha256(data).hexdigest()
receipt=dict(time=time.time(),source_commit=r['commit'],source=str(target),files=hashes,
 scientific_source_commit=s['source_commit'],training_action='NONE',
 scope='Separate CPU aggregate reports and storage metadata; no model/GPU/inference/signals/production edit')
folder=b/'analysis';folder.mkdir(exist_ok=True)
path=folder/('analysis-publication-'+r['commit'][:8]+'.json')
if path.exists():
 old=json.loads(path.read_text());assert old['source_commit']==r['commit'] and old['files']==hashes
 receipt=old
else:path.write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
'''.replace('PAYLOAD', repr(payload))
    Path(args.output).write_text(code)
    print(json.dumps(dict(commit=commit, files=len(files), request_bytes=len(code.encode()))))


if __name__ == '__main__':
    main()
