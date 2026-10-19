"""Keep full-read implementation acceptance and independently bind shared QA order."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''):
            h.update(data)
    return h.hexdigest()


parser = argparse.ArgumentParser(__doc__)
for key in ('manifest','runtime','legacy-verifier','output'):
    parser.add_argument('--'+key, required=True)
args = parser.parse_args()
output = Path(args.output)
assert not output.exists(), 'A published independent acceptance is immutable'
implementation = output.with_name(output.stem+'.implementation.json')
if not implementation.exists():
    subprocess.run([sys.executable,'-B',args.legacy_verifier,'--manifest',args.manifest,
        '--runtime',args.runtime,'--output',str(implementation)],check=True,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
report = json.loads(implementation.read_text())
assert report['state'] == 'INDEPENDENT_CF_FULL_READ_VERIFIED'
manifest = json.loads(Path(args.manifest).read_text()); source=Path(manifest['source']['path'])
assert sha(source) == report['normal_sha256'] == manifest['source']['sha256']
h = hashlib.sha256(); h.update(b'['); count=0
with source.open() as stream:
    for line in stream:
        row=json.loads(line)
        if count:
            h.update(b',')
        h.update(json.dumps([row['data_source'],int(row['evaluation_ordinal']),row['question'],sorted(row['gold'])],
            ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
        count+=1
h.update(b']'); content=h.hexdigest()
assert count == 3610 and content == '23c0a8307f38a7a8d1cbd9e71fb115b7bd091421a3d3b10952fb71c8210b551c'
report.update(normal_content_order_sha256=content,normal_content_order_rows=count,
    implementation_receipt=str(implementation),implementation_receipt_sha256=sha(implementation))
with output.open('x') as stream:
    json.dump(report,stream); stream.flush(); os.fsync(stream.fileno())
print(json.dumps(report))
