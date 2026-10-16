"""Resume a small source-compacted reference5 archive over the existing routes.

This consumes the CPU producer's pinned transfer manifest, not the original
8.4GB archive. It preserves source-event/EM identities and streams only bytes.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

from reproduction.operations.a800_question5_source_transfer import ROOT,RECEIVE,source_command,target_command


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intent',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--progress',type=Path,required=True)
    args=parser.parse_args();intent=json.loads(args.intent.read_text());bundle=json.loads(args.assets.read_text())
    assert intent['prefix_tokens']==5 and intent['epochs']==200
    assets=bundle['parts'];assert assets and sum(x['bytes'] for x in assets)==bundle['bytes']
    assert bundle['prefix_tokens']==5 and bundle['train_trajectories']==1740800 and bundle['test_trajectories']==2560
    commit=subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()
    for name in ('a800_question5_source_transfer.py',Path(__file__).name):
        raw=subprocess.check_output(['git','-C',str(ROOT/'repo'),'show',commit+':reproduction/operations/'+name])
        assert raw==(Path(__file__).parent/name).read_bytes()
    assert subprocess.check_output(['git','-C',str(ROOT/'repo'),'ls-remote','origin','refs/heads/search-g1'],text=True).split()[0]==commit
    done=[];copied=0
    def publish(**extra):
        value=dict(checked_unix=time.time(),source_commit=commit,base=intent['base'],total_bytes=bundle['bytes'],
            transferred_bytes_this_invocation=copied,verified_parts=done,source_compacted_reference=True,
            local_payload_staging=False,new_rollouts=0,new_em_scoring=0,**extra)
        temp=args.progress.with_name(args.progress.name+'.partial');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(args.progress)
    publish(stage='DERIVED_REFERENCE5_STREAM_STARTING')
    for original in assets:
        entry=dict(original);entry.setdefault('name',Path(entry['path']).name);entry['remote_path']=entry['path']
        assert entry['remote_path'].startswith('/mnt/data/shaoshitong/search-g1-question5-reference5-20261016-')
        assert '/' not in entry['name']
        with subprocess.Popen(target_command(RECEIVE.replace('BASE',repr(intent['base'])).replace('ENTRY',repr(entry))),
                              stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as dst:
            first=json.loads(dst.stdout.readline())
            if first['stage']=='ALREADY_VERIFIED':
                dst.stdin.close();dst.wait(timeout=30);assert dst.returncode==0
                done.append(dict(name=entry['name'],bytes=entry['bytes'],sha256=entry['sha256'],existing_verified=True));publish(stage='DERIVED_PART_ALREADY_VERIFIED');continue
            offset=first['offset']
            code=("import pathlib,sys\np=pathlib.Path("+repr(entry['remote_path'])+")\nassert p.stat().st_size=="+str(entry['bytes'])+
                "\nwith p.open('rb') as f:\n f.seek("+str(offset)+")\n while True:\n  x=f.read(1024**2)\n  if not x:break\n  sys.stdout.buffer.write(x)\nsys.stdout.buffer.flush()\n")
            command,payload=source_command(code)
            with subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as src:
                src.stdin.write(payload);src.stdin.close();count=offset;last=time.monotonic()
                while count<entry['bytes']:
                    raw=src.stdout.read(min(1024**2,entry['bytes']-count))
                    if not raw:raise EOFError('Derived immutable source stream ended: '+src.stderr.read(4096).decode(errors='replace'))
                    dst.stdin.write(raw);copied+=len(raw);count+=len(raw)
                    if time.monotonic()-last>=10:
                        publish(stage='DERIVED_REFERENCE5_BYTES_ADVANCING',current_part=entry['name'],current_part_bytes=count);last=time.monotonic()
                dst.stdin.close();src.wait(timeout=60);assert src.returncode==0,src.stderr.read(4096).decode(errors='replace')
            result=dst.stdout.read(16384);dst.wait(timeout=300);assert dst.returncode==0,dst.stderr.read(4096).decode(errors='replace')
            receipt=json.loads(result);assert receipt['bytes']==entry['bytes'] and receipt['sha256']==entry['sha256']
            done.append(receipt);publish(stage='DERIVED_REFERENCE5_PART_SHA_VERIFIED')
    publish(stage='DERIVED_REFERENCE5_ARCHIVE_COMPLETE',complete=True)
    print(json.dumps(dict(complete=True,parts=len(done),bytes=bundle['bytes'],progress=str(args.progress))))


if __name__=='__main__':main()
