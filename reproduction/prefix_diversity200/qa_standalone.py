"""Isolated same-cluster QA; no trainer control, no changes to training artifacts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import resource
import socket
import subprocess
import sys
import time
import urllib.request
from .actor import read, sha, atomic_json


def gpu_rows():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
    return sorted([(int(a),b.strip(),int(c)) for a,b,c in (line.split(',') for line in lines)])


def own_cgroup(unit):
    raw=Path('/proc/self/cgroup').read_text().splitlines()
    paths=[line.split(':',2)[2] for line in raw if line.startswith('0::')]
    if len(paths)!=1 or paths[0]!='/system.slice/'+unit or not unit.startswith('g1-qa-') or not unit.endswith('.service'):
        raise ValueError('Standalone QA must own its exact systemd cgroup')
    return Path('/sys/fs/cgroup')/paths[0].lstrip('/')


def shared_retriever(config_path):
    """Memory-map the already committed Arrow corpus in original shard order."""
    c=read(config_path);spec=c['qa_standalone'];original=Path(spec['source_cache'])
    source_receipt=read(original/'corpus.json')
    if sha(original/'corpus.json')!=spec['source_corpus_receipt_sha256']:
        raise ValueError('Original prepared corpus receipt changed')
    source_config=Path(spec['source_evaluation_config'])
    if original!=source_config.parents[2]/'retrieval-cache':
        raise ValueError('Arrow cache must be the original actor run corpus cache')
    arrow_directory=Path(spec['arrow_directory'])
    arrow_directory.relative_to(original/'arrow')
    if arrow_directory.resolve()!=arrow_directory or any(p.parent!=arrow_directory for p in map(Path,[i['path'] for i in spec['arrow_shards']])):
        raise ValueError('Arrow shard paths left the committed source cache')
    ready=read(original/'ready.json')
    if (sha(original/'ready.json')!=spec['source_retriever_ready_sha256'] or ready['corpus']!=source_receipt
            or ready['documents']!=spec['corpus_documents']):raise ValueError('Original corpus/index binding changed')
    info=read(arrow_directory/'dataset_info.json')
    if info['splits']['train']['num_examples']!=spec['corpus_documents']:
        raise ValueError('Original Arrow metadata document count changed')
    import datasets
    original_loader=datasets.load_dataset
    expected=str(Path(spec['base'])/'retrieval-cache/wiki_dump.jsonl')
    def load(path,*args,**kwargs):
        if path!='json' or kwargs.get('data_files')!=expected:return original_loader(path,*args,**kwargs)
        parts=[]
        for ordinal,item in enumerate(spec['arrow_shards']):
            p=Path(item['path'])
            if (p.resolve()!=p or p.parent!=Path(spec['arrow_directory']) or p.name!='json-train-%05d-of-%05d.arrow'%(ordinal,len(spec['arrow_shards']))
                    or p.stat().st_size!=item['bytes']):raise ValueError('Original ordered Arrow corpus changed')
            parts.append(datasets.Dataset.from_file(str(p),in_memory=False))
        dataset=datasets.concatenate_datasets(parts)
        if len(dataset)!=spec['corpus_documents']:raise ValueError('Original corpus row count changed')
        return dataset
    datasets.load_dataset=load
    from reproduction.prefix20_online import retriever
    sys.argv=[sys.argv[0],'--index',c['retrieval_index'],'--corpus',c['retrieval_corpus'],
        '--encoder',c['retrieval_encoder'],'--cache',str(Path(spec['base'])/'retrieval-cache'),
        '--run-id',c['run_id'],'--port',str(c['retrieval_port']),'--batch-queries','1024']
    retriever.main()


def run(config_path):
    c=read(config_path);spec=c['qa_standalone'];base=Path(spec['base']);unit=spec['unit']
    if socket.gethostname().split('.')[0]!=c['node'] or c['node'] not in ('h100-5','h100-8','h100-10','h100-11'):
        raise ValueError('Only the configured same-cluster H100 node is authorized')
    if str(base.resolve())!=str(base) or not str(base).startswith('/mnt/'):
        raise ValueError('All QA output must remain on verified remote/shared storage')
    group=own_cgroup(unit)
    if sha(c['validation_questions'])!=c['validation_questions_sha256']:
        raise ValueError('Pinned validation questions changed')
    if spec['expected_count']!=512 or c['actor_version']!=c['run_id']+'-actor%04d'%spec['outer']:
        raise ValueError('Standalone milestone QA actor identity differs')
    if c['search_url']!='http://127.0.0.1:%d/retrieve'%c['retrieval_port']:
        raise ValueError('Retrieval must stay loopback on the inference node')
    if sha(spec['source_evaluation_config'])!=spec['source_evaluation_config_sha256']:
        raise ValueError('Original actor evaluation identity changed')
    if read(spec['source_evaluation_config'])['actor_version']!=c['actor_version']:
        raise ValueError('Original QA actor version differs')
    rows=gpu_rows()
    if [r[1] for r in rows]!=c['physical_gpu_uuids'] or len(rows)!=8 or any(r[2]<40*1024 for r in rows):
        raise ValueError('Fresh eight-GPU UUID/headroom check failed; never clean foreign processes')
    env=dict(os.environ,PYTHONPATH=str(base/'repo')+':'+spec['source_repo']+':'+c['runtime_extra_packages'],CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        HF_HOME=str(base/'cache/hf'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='4',
        XDG_CACHE_HOME=str(base/'cache'),TRITON_CACHE_DIR=str(base/'cache/triton'),TMPDIR=str(base/'tmp'),
        TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',NO_PROXY='127.0.0.1,localhost')
    for key in ('PYTHONHOME','RAY_ADDRESS','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
    for name in ('cache/hf','cache/triton','tmp','logs'):(base/name).mkdir(parents=True,exist_ok=True)
    python=str(Path(c['environment_root'])/'bin/python');handles=[];children=[];peak={r[1]:0 for r in rows};start=time.time();last=0
    def stop(sig,frame):raise InterruptedError('Own standalone QA interrupted: '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(name,args,child_env):
        stream=(base/'logs'/(name+'.log')).open('xb');handles.append(stream)
        p=subprocess.Popen(args,env=child_env,cwd=base/'repo',stdout=stream,stderr=subprocess.STDOUT);children.append(p)
        atomic_json(base/'logs'/('child-'+name+'.json'),dict(pid=p.pid,command=args,created_unix=time.time(),systemd_unit=unit))
        return p
    def guard():
        nonlocal last
        if time.time()-last<5:return
        last=time.time();rows=gpu_rows();pids=set()
        for p in [group/'cgroup.procs',*group.glob('**/cgroup.procs')]:
            try:pids.update(map(int,p.read_text().split()))
            except FileNotFoundError:pass
        memory={uuid:0 for uuid in c['physical_gpu_uuids']}
        lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
        for line in lines:
            uuid,pid,used=[v.strip() for v in line.split(',')]
            if int(pid) in pids:memory[uuid]+=int(used)
        for uuid,value in memory.items():peak[uuid]=max(peak[uuid],value)
        value=dict(observed_unix=last,own_gpu_memory_mib=memory,peak_sampled_own_gpu_memory_mib=peak,
            free_mib={uuid:free for _,uuid,free in rows},owned_pids=sorted(pids),slurm_reserved_gpus=0,physical_gpus=8)
        atomic_json(base/'logs/resource-latest.json',value,replace=True)
        if ([r[1] for r in rows]!=c['physical_gpu_uuids'] or any(r[2]<12*1024 for r in rows)
                or any(v>36*1024 for v in memory.values())):raise RuntimeError('Shared memory/UUID guard; only this systemd unit stops')
    with (base/'uuid-lease.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(base/'logs/uuid-lease.json',dict(node=c['node'],uuids=c['physical_gpu_uuids'],unit=unit,pid=os.getpid(),slurm_reserved_gpus=0))
        result=None;error=None
        try:
            libs=list((Path(c['environment_root'])/'lib/python3.10/site-packages/nvidia').glob('*/lib'))
            libs+=[Path(c['environment_root'])/'lib/python3.10/site-packages/torch/lib']
            retriever=launch('retriever',[python,'-m','reproduction.prefix_diversity200.qa_standalone','--config',str(config_path),'--retriever'],
                dict(env,PYTHONPATH=c['retriever_gpu_lib']+':'+env['PYTHONPATH'],LD_LIBRARY_PATH=':'.join(map(str,libs))))
            deadline=time.monotonic()+1800
            while True:
                guard()
                if retriever.poll() is not None:raise RuntimeError('Own retriever exited before ready: '+str(retriever.returncode))
                try:
                    with urllib.request.urlopen(c['search_url'].replace('/retrieve','/health'),timeout=2) as response:health=json.load(response)
                    if (health['run_id']!=c['run_id'] or health['gpu_shards']!=8 or health['pid']!=retriever.pid
                            or health['documents']!=spec['corpus_documents']
                            or health['index']!=str(Path(c['retrieval_index']).resolve())
                            or health['corpus_archive']!=str(Path(c['retrieval_corpus']).resolve())
                            or health['encoder']!=str(Path(c['retrieval_encoder']).resolve())
                            or health['endpoint']!=c['search_url'] or health['encoder_dtype']!='float16'
                            or health['index_dtype']!='float16'):
                        raise ValueError('Wrong owned loopback retriever identity')
                    break
                except (OSError,TimeoutError):pass
                if time.monotonic()>deadline:raise TimeoutError('Own retriever initialization exceeded1800s; preserve evidence')
                time.sleep(1)
            atomic_json(base/'logs/retriever-ready.json',health)
            evaluation=launch('evaluation',[python,'-m','reproduction.prefix_diversity200.eval','--config',str(config_path),
                '--model',spec['model_dir'],'--questions',c['validation_questions'],'--expected-count','512','--output',str(base/'validation')],env)
            while evaluation.poll() is None:
                guard()
                if retriever.poll() is not None:raise RuntimeError('Own retriever exited during QA')
                time.sleep(.5)
            if evaluation.returncode:raise RuntimeError('Actual QA worker process failed: '+str(evaluation.returncode))
            result=read(base/'validation/complete.json')
        except BaseException as exc:error=repr(exc);raise
        finally:
            cleanup=[];unreaped=[]
            for p in reversed(children):
                try:
                    if p.poll() is None:p.terminate()
                except ProcessLookupError:pass
                except Exception as exc:cleanup.append(dict(pid=p.pid,operation='terminate',error=repr(exc)))
                try:
                    try:p.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        try:p.kill()
                        except ProcessLookupError:pass
                        p.wait(timeout=5)
                except Exception as exc:
                    cleanup.append(dict(pid=p.pid,operation='wait',error=repr(exc)))
                    try:
                        try:p.kill()
                        except ProcessLookupError:pass
                        p.wait(timeout=5)
                    except Exception as last:unreaped.append(p.pid);cleanup.append(dict(pid=p.pid,operation='final-reap',error=repr(last)))
            for stream in handles:
                try:stream.close()
                except Exception as exc:cleanup.append(dict(operation='stream-close',error=repr(exc)))
            usage=[resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)]
            cpu_seconds=sum(u.ru_utime+u.ru_stime for u in usage)
            atomic_json(base/'logs/terminal.json',dict(success=result is not None and not unreaped,error=error,cleanup_errors=cleanup,unreaped_child_pids=unreaped,
                cpu_seconds_self_and_reaped_children=cpu_seconds,wall_gpu8_hours=(time.time()-start)*8/3600,
                started_unix=start,ended_unix=time.time(),outer=spec['outer'],result=result,
                peak_sampled_own_gpu_memory_mib=peak,systemd_unit=unit,slurm_reserved_gpus=0,
                descendant_cleanup='systemd KillMode=control-group; native terminal cgroup readback required'))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--retriever',action='store_true');a=p.parse_args()
    if a.retriever:shared_retriever(a.config)
    else:run(a.config)


if __name__=='__main__':main()
