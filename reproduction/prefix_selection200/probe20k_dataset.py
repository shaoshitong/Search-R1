"""One frozen actor50 dataset: reuse5120 G5, add15360 distinct G5 questions."""
import argparse
import json
import os
from pathlib import Path
import random
import shlex
import signal
import subprocess
import time
import urllib.request

from reproduction.prefix20_online.launch import once, publish, read, sha


def prepare(base):
    import pyarrow.parquet as pq
    from reproduction.prefix20_online.pool import identity
    base=Path(base);c=read(base/'preparation/launch-config.json');s=c['probe20k_study']
    source=Path(s['baseline_data_base']);original=read(source/'outers/outer-000051/questions.json')['rows']
    pool=read(Path(c['source_question_pool']))
    used_ids={r['qid']for r in original};used_text={r['normalized_question_hash']for r in original}
    diagnostic=read(source/'outers/outer-000051/diagnostic-complete.json')
    testrows=[json.loads(line)for line in(Path(diagnostic['scan_path'])/'index.jsonl').open()]
    test_ids={r['qid']for r in testrows};test_hash={r['question_hash']for r in testrows}
    assert len(used_ids)==len(used_text)==5120 and len(test_ids)==512 and not used_ids&test_ids
    eligible=[r for r in pool['records']if r['qid']not in used_ids|test_ids and r['normalized_question_hash']not in used_text and r['question_hash']not in test_hash]
    random.Random(s['dataset_seed']).shuffle(eligible);chosen=eligible[:15360]
    assert len(chosen)==len({r['qid']for r in chosen})==len({r['normalized_question_hash']for r in chosen})==15360
    table=pq.read_table(c['canonical_train'],columns=['prompt','reward_model','data_source','extra_info'])
    for block in range(3):
        outer=52+block;root=base/'outers'/('outer-%06d'%outer);root.mkdir(parents=True,exist_ok=True)
        records=chosen[block*5120:(block+1)*5120]
        actual=table.take([r['source_index']for r in records]).to_pylist();rows=[]
        for ordinal,(record,item)in enumerate(zip(records,actual)):
            key=identity(item);assert all(key[k]==record[k]for k in key)
            rows.append(dict(record,**item,candidate_ordinal=ordinal))
        value=dict(run_id=c['run_id'],outer=outer,questions=5120,rows=rows)
        p=root/'questions.json'
        if p.exists():assert read(p)==value
        else:once(p,value)
        actor=dict(actor_version=c['actor_version'],actor_model_dir=c['actor_model_dir'],actor_model_files_sha256=c['actor_model_files_sha256'])
        p=root/'actor-binding.json'
        if p.exists():assert read(p)==actor
        else:once(p,actor)
    value=dict(status='ORIGINAL5120_PLUS_NEW15360_QUESTION_IDENTITIES_PREPARED',questions=20480,
        original_questions=5120,new_questions=15360,diagnostic_qid_overlap=0,
        actor_version=c['actor_version'],source_pool_sha256=sha(Path(c['source_question_pool'])),dataset_seed=s['dataset_seed'])
    path=base/'preparation/question-selection-complete.json'
    if path.exists():assert read(path)==value
    else:once(path,value)
    return value


def run(base):
    from .local_probe_fit50 import admit_local_allocation
    from reproduction.prefix_only_correctness_ab.launch import torchrun_owned_class, DeferredInterrupt, birth_states
    from reproduction.h1008_qprobe.pipeline import UUIDLocks
    from .probe import candidate_groups
    from .rollout import iter_rows
    base=Path(base);c=read(base/'preparation/launch-config.json');s=c['probe20k_study']
    admit_local_allocation(base,c);lease=UUIDLocks(c['physical_gpu_uuids'])
    owned=torchrun_owned_class()(base/'logs/process-ledger.json',c['run_id'])
    interrupt=DeferredInterrupt();handlers={v:signal.getsignal(v)for v in(signal.SIGTERM,signal.SIGINT)}
    for sig in handlers:signal.signal(sig,interrupt.handler)
    env=dict(os.environ,PYTHONPATH=str(base/'repo'),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        NCCL_NVLS_ENABLE='0',HF_HOME=str(base/'cache/hf'),XDG_CACHE_HOME=str(base/'cache'),
        TRITON_CACHE_DIR=str(base/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(base/'cache/inductor'),
        TMPDIR=str(base/'tmp'),PYTHONUNBUFFERED='1',NO_PROXY='127.0.0.1,localhost')
    if c.get('python_header_cpath'):env['CPATH']=c['python_header_cpath']
    for key in('PYTHONHOME','RAY_ADDRESS','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
    for name in('tmp','cache/hf','cache/triton','cache/inductor'):(base/name).mkdir(parents=True,exist_ok=True)
    python=str(Path(c['environment_root'])/'bin/python');conf=base/'preparation/launch-config.json';services={}
    def start(name,command,child_env=None):
        p=owned.start(name,command,child_env or env,str(base/'logs'/(name+'.log')),str(base/'repo'));owned.observe(name);return p
    def phase(name,command):
        publish(base/'logs/pipeline-status.json',dict(stage=name,actor_version=c['actor_version'],checked_unix=time.time()))
        p=start(name,command)
        while p.poll()is None:
            owned.observe(name);interrupt.check()
            if any(poll.poll()is not None for poll in services.values()):raise RuntimeError('Own retriever actually exited')
            time.sleep(1)
        owned.observe(name);owned.jobs[name]['exit_code']=p.returncode;owned.save();owned.stop(name)
        if p.returncode:raise RuntimeError('Dataset generation phase actually failed: '+name)
    try:
        prepare(base)
        envroot=Path(c['environment_root']);libs=list((envroot/'lib/python3.10/site-packages/nvidia').glob('*/lib'))+[envroot/'lib/python3.10/site-packages/torch/lib']
        services['retriever']=start('retriever',[python,'-m','reproduction.prefix20_online.retriever',
            '--index',c['retrieval_index'],'--corpus',c['retrieval_corpus'],'--encoder',c['retrieval_encoder'],
            '--cache',str(base/'retrieval-cache'),'--run-id',c['run_id'],'--port',str(c['retrieval_port']),
            '--data-cache',c.get('retrieval_data_cache',str(base/'retrieval-cache')),'--batch-queries','1024'],
            dict(env,PYTHONPATH=c['retriever_gpu_lib']+':'+str(base/'repo'),LD_LIBRARY_PATH=':'.join(map(str,libs))))
        while True:
            interrupt.check();owned.observe('retriever')
            if services['retriever'].poll()is not None:raise RuntimeError('Own retriever failed during startup')
            try:
                with urllib.request.urlopen(c['search_url'].replace('/retrieve','/health'),timeout=5)as r:health=json.load(r)
                if health.get('ok')and health.get('run_id')==c['run_id']and health.get('gpu_shards')==8:break
            except(OSError,ValueError):pass
            time.sleep(2)
        source=Path(s['baseline_data_base']);blocks=[dict(root=str(source/'outers/outer-000051'),outer=51,reused=True)]
        for outer in(52,53,54):
            root=base/'outers'/('outer-%06d'%outer)
            if not(root/'prefix/complete.json').exists():
                phase('block%d-prefix'%outer,[python,'-m','reproduction.prefix_selection200.rollout','--config',str(conf),'--outer',str(outer),'--phase','prefix'])
            prefix=read(root/'prefix/complete.json');scan=Path(prefix['scan_path'])
            selection=root/'all-selection.json'
            if not selection.exists():
                index={r['trajectory_id']:r for r in iter_rows(scan/'index.jsonl')}
                once(selection,dict(run_id=c['run_id'],outer=outer,actor_version=c['actor_version'],prefix_scan_sha256=prefix['scan_sha256'],
                    selected=candidate_groups(index,5120),selected_questions=5120,outcome_labels_used=False))
            if not(root/'full/complete.json').exists():
                phase('block%d-full'%outer,[python,'-m','reproduction.prefix_selection200.rollout','--config',str(conf),'--outer',str(outer),'--phase','continue','--selection',str(selection)])
            full=read(root/'full/complete.json');assert full['trajectories']==25600 and full['actor_version']==c['actor_version']
            blocks.append(dict(root=str(root),outer=outer,reused=False))
            publish(base/'logs/dataset-progress.json',dict(complete_questions=len(blocks)*5120,complete_trajectories=len(blocks)*25600,blocks=blocks))
        qids=set();texts=set();entries=[]
        for item in blocks:
            root=Path(item['root']);prefix=read(root/'prefix/complete.json');scan=Path(prefix['scan_path']);rows=[json.loads(v)for v in(scan/'index.jsonl').open()]
            groups=candidate_groups({r['trajectory_id']:r for r in rows},5120);questions=read(root/'questions.json')['rows']
            for row in questions:
                assert row['qid']not in qids and row['normalized_question_hash']not in texts
                qids.add(row['qid']);texts.add(row['normalized_question_hash'])
            entries.append(dict(item,scan_path=str(scan),scan_sha256=prefix['scan_sha256'],own_em=str(root/'full/own-em.jsonl'),
                own_em_sha256=sha(root/'full/own-em.jsonl'),selection_sha256=sha(root/'all-selection.json'),questions=5120,trajectories=25600))
        assert len(qids)==len(texts)==20480
        diagnostic=read(source/'outers/outer-000051/diagnostic-complete.json')
        once(base/'dataset-complete.json',dict(status='FIXED_ACTOR50_20480_DIFFERENT_G5_DATASET_COMPLETE',actor_version=c['actor_version'],
            questions=20480,trajectories=102400,blocks=entries,diagnostic=diagnostic,diagnostic_qid_overlap=0,
            native_hidden_only=True,actor_updates=0,new_rollout_trajectories=76800,reused_rollout_trajectories=25600))
        once(base/'logs/pipeline-complete.json',dict(status='PROBE20K_DATA_ONLY_COMPLETE',actor_updated=False,formal200_resumed=False))
    finally:
        try:
            owned.stop_all();owned.reap();births=[v for j in owned.jobs.values()for v in j['ledger'].values()];present=birth_states(births)
            once(base/'logs/cleanup-status.json',dict(owned_births_all_gone=not present,remaining_owned_births=present,checked_unix=time.time()))
        finally:
            lease.close()
            for sig,handler in handlers.items():signal.signal(sig,handler)


def submit(base):
    base=Path(base);prep=base/'preparation'
    if(prep/'submit-intent.json').exists():return dict(status='EXISTING_INTENT_NO_REPLAY',receipt=read(prep/'submit-receipt.json')if(prep/'submit-receipt.json').exists()else None)
    c=read(prep/'launch-config.json');python=str(Path(c['environment_root'])/'bin/python');script=prep/'native-probe20k-data.sh'
    from .local_probe_fit50 import SHARED_AUTHORIZATION
    assert c['local_probe_fit50']['shared_slurm_physical8_authorization']==SHARED_AUTHORIZATION
    script.write_text('#!/bin/bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\nexport PYTHONPATH='+shlex.quote(str(base/'repo'))+'\nunset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy\nexec '+shlex.join([python,'-m','reproduction.prefix_selection200.probe20k_dataset','run','--base',str(base)])+'\n');script.chmod(0o700)
    command=['sbatch','--parsable','--partition=gpu','--oversubscribe','--nodelist=h100-8','--nodes=1','--ntasks=1',
        '--cpus-per-task=96','--mem=512G','--time=0','--signal=B:TERM@60','--export=NONE','--job-name=g1-probe20k-data-'+c['run_id'][:12],
        '--comment=g1-probe20k-data:'+c['run_id'],'--chdir='+str(base),'--output='+str(base/'logs/native-%j.log'),'--error='+str(base/'logs/native-%j.log'),str(script)]
    once(prep/'submit-intent.json',dict(command=command,config_sha256=sha(prep/'launch-config.json'),created_unix=time.time()))
    try:
        r=subprocess.run(command,capture_output=True,text=True,timeout=30);job=r.stdout.strip().split(';')[0]
        value=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,status='SUBMITTED'if r.returncode==0 and job.isdecimal()else'UNKNOWN_NO_REPLAY')
        if value['status']=='SUBMITTED':value['job_id']=job
    except subprocess.TimeoutExpired as error:value=dict(status='UNKNOWN_NO_REPLAY',error=repr(error))
    once(prep/'submit-receipt.json',value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('run','submit','prepare'));p.add_argument('--base',required=True);a=p.parse_args();print(json.dumps(globals()[a.phase](a.base)))
