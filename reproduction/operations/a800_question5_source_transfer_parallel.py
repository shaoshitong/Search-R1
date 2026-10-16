"""Four bounded memory streams, resuming immutable archive parts by byte offset."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

from reproduction.operations.a800_question5_source_transfer import (
    ROOT, READY, ARCHIVE, RECEIVE, target_command, source_command)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intent',type=Path,required=True)
    parser.add_argument('--progress',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4,choices=[2,4])
    args=parser.parse_args();intent=json.loads(args.intent.read_text())
    assert intent['prefix_tokens']==5 and intent['epochs']==200
    repo=ROOT/'repo';commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    for name in ('a800_question5_source_transfer.py',Path(__file__).name):
        rel='reproduction/operations/'+name
        assert subprocess.check_output(['git','-C',str(repo),'show',commit+':'+rel])==(Path(__file__).parent/name).read_bytes()
    assert subprocess.check_output(['git','-C',str(repo),'ls-remote','origin','refs/heads/search-g1'],text=True).split()[0]==commit
    assets=sorted((x for x in json.loads(READY.read_text())['assets'] if x['archive']==ARCHIVE),key=lambda x:x['name'])
    assert len(assets)==8 and sum(x['bytes']for x in assets)==8389298946
    lock=threading.Lock();state=dict(checked_unix=time.time(),base=intent['base'],source_commit=commit,
        total_bytes=8389298946,transferred_bytes_this_invocation=0,verified_parts=[],active_parts={},
        model_or_checkpoint_transfer=False,local_payload_staging=False,workers=args.workers,
        stage='STARTING_BOUNDED_PARALLEL_ARCHIVE_TRANSFER')
    def save():
        state['checked_unix']=time.time();temp=args.progress.with_name(args.progress.name+'.partial')
        temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(args.progress)
    with lock:save()
    def transfer(entry):
        target=RECEIVE.replace('BASE',repr(intent['base'])).replace('ENTRY',repr(entry))
        with subprocess.Popen(target_command(target),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as dst:
            first=json.loads(dst.stdout.readline())
            if first['stage']=='ALREADY_VERIFIED':
                dst.stdin.close();dst.wait(timeout=30);assert dst.returncode==0
                receipt=dict(name=entry['name'],bytes=entry['bytes'],sha256=entry['sha256'],existing_verified=True)
            else:
                offset=first['offset']
                source=("import pathlib,sys\np=pathlib.Path("+repr(entry['remote_path'])+")\n"
                    "assert p.stat().st_size=="+str(entry['bytes'])+"\nwith p.open('rb') as f:\n f.seek("+str(offset)+")\n"
                    " while True:\n  x=f.read(1024**2)\n  if not x:break\n  sys.stdout.buffer.write(x)\nsys.stdout.buffer.flush()\n")
                with lock:
                    state['active_parts'][entry['name']]=dict(offset_reused=offset,bytes_written=offset);save()
                command,code=source_command(source)
                with subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as src:
                    src.stdin.write(code);src.stdin.close();count,last=offset,time.monotonic()
                    while count<entry['bytes']:
                        block=src.stdout.read(min(1024**2,entry['bytes']-count))
                        if not block:raise EOFError('Source stream ended: '+src.stderr.read(4096).decode(errors='replace'))
                        dst.stdin.write(block);count+=len(block)
                        with lock:
                            state['transferred_bytes_this_invocation']+=len(block)
                            state['active_parts'][entry['name']]['bytes_written']=count
                            if time.monotonic()-last>=10:
                                state['stage']='STREAMING_BOUNDED_PARALLEL_ARCHIVE';save();last=time.monotonic()
                    dst.stdin.close();src.wait(timeout=60)
                    assert src.returncode==0,src.stderr.read(4096).decode(errors='replace')
                result=dst.stdout.read(16384);dst.wait(timeout=300)
                assert dst.returncode==0,dst.stderr.read(4096).decode(errors='replace')
                receipt=json.loads(result);assert receipt['sha256']==entry['sha256'] and receipt['bytes']==entry['bytes']
            with lock:
                state['active_parts'].pop(entry['name'],None);state['verified_parts'].append(receipt)
                state['stage']='ARCHIVE_PART_SHA_VERIFIED';save()
            return receipt
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            results=list(pool.map(transfer,assets))
        with lock:
            state.update(stage='ALL_EIGHT_DATASET_ARCHIVE_PARTS_VERIFIED',complete=True);save()
        print(json.dumps(dict(complete=True,verified_parts=len(results),progress=str(args.progress))))
    except BaseException as exc:
        with lock:state.update(stage='ARCHIVE_TRANSFER_INTERRUPTED_RETAIN_PARTIALS',error_class=type(exc).__name__);save()
        raise


if __name__=='__main__':main()
