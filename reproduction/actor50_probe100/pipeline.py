"""Fifty fresh512-G5 cycles, fixed actor0050, two probe epochs per cycle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import traceback
import urllib.request

from reproduction.prefix20_online.launch import read, sha, once, publish
from reproduction.prefix_only_correctness_ab.launch import torchrun_owned_class, DeferredInterrupt, birth_states
from reproduction.prefix_selection200.a800_runtime import UUIDLocks, native_allocation, PREFIX


def qualify(base,c):
    import torch
    from reproduction.prefix_selection200.prefix5_model import validate_model_state, validate_native_buffers
    path=base/'preparation/source50-qualified.json'
    if path.exists():
        proof=read(path)
        if (proof['joint50_sha256']!=c['joint50_sha256'] or proof['probe50_sha256']!=c['probe50_sha256'] or
            sha(base/'preparation/initial-probe50-model.pt')!=proof['initial_model_sha256']):
            raise ValueError('Existing model-only qualification changed')
        return proof
    if sha(c['joint50_path'])!=c['joint50_sha256']:raise ValueError('Joint50 changed')
    joint=read(c['joint50_path'])
    if joint['outer']!=50 or joint['actor']['actor_version']!=c['actor_version']:
        raise ValueError('Frozen actor50 identity differs')
    for i,(name,item) in enumerate(sorted(joint['actor']['hf_files'].items()),1):
        source=Path(c['actor_model_dir'])/name
        if source.stat().st_size!=item['bytes'] or sha(source)!=item['sha256']:
            raise ValueError('Actor50 actual HF file differs: '+name)
        publish(base/'logs/status.json',dict(stage='QUALIFYING_ACTOR50_HF',files_verified=i,checked_unix=time.time()))
    if sha(c['probe50_snapshot'])!=c['probe50_sha256']:raise ValueError('Actual probe50 payload changed')
    source=torch.load(c['probe50_snapshot'],map_location='cpu',weights_only=False)
    if source['completed_outer']!=50 or source['actual_adam_steps_per_rank']!=[250]*4:
        raise ValueError('Probe model source is not the saved online50 boundary')
    raw=source['model'];buffers=source['model_nonpersistent_buffers']
    validate_model_state(raw);validate_native_buffers(buffers)
    init=dict(role='probe50_model_only_new_balanced_warmup',source_probe50_sha256=c['probe50_sha256'],
        raw_model=raw,raw_buffers=buffers,ema_model=raw,ema_buffers=buffers,
        new_adam_updates=0,ema_updates=0,source_online_adam_updates=250,
        source_optimizer_imported=False,actor_updates=0)
    target=base/'preparation/initial-probe50-model.pt'
    with target.open('xb') as stream:
        torch.save(init,stream);stream.flush();os.fsync(stream.fileno())
    proof=dict(role='actual_actor50_HF_and_probe50_weights_qualified',joint50_sha256=c['joint50_sha256'],
        probe50_sha256=c['probe50_sha256'],actor_version=c['actor_version'],
        actor_hf_files_verified=len(joint['actor']['hf_files']),initial_model_sha256=sha(target),
        initial_new_adam_updates=0,initial_ema_updates=0,source_optimizer_imported=False,
        learned_tensors_finite=True,native_buffers_finite=True,checked_unix=time.time())
    once(path,proof);return proof


def question_bundles(c):
    import pyarrow.parquet as pq
    from reproduction.prefix20_online.pool import identity
    from reproduction.prefix_selection200.pool import fixed_test_identities
    if sha(c['canonical_train'])!=c['canonical_train_sha256']:
        raise ValueError('Canonical training question file changed')
    table=pq.read_table(c['canonical_train'],columns=['prompt','reward_model','data_source','extra_info'])
    fixed=fixed_test_identities(c);wanted={x['qid']:x for x in fixed}
    fixed_text={x['normalized_question_hash'] for x in fixed}
    original=table.to_pylist();test=[]
    for i,row in enumerate(original):
        key=identity(row)
        if key['qid'] in wanted:
            if any(key[k]!=wanted[key['qid']][k] for k in ('question_hash','normalized_question_hash')):
                raise ValueError('Fixed512 content changed')
            test.append(dict(row,**key,source_index=i))
    if len(test)!=512:raise ValueError('Original fixed512 question identity set differs')
    source=read(c['source_question_pool']);eligible=[]
    for row in source['records']:
        if row['qid'] in wanted or row['normalized_question_hash'] in fixed_text:continue
        i=row['source_index'];key=identity(original[i])
        if any(key[k]!=row[k] for k in key):raise ValueError('Approved training-pool identity changed')
        eligible.append(dict(original[i],**key,source_index=i))
    if len({x['qid'] for x in eligible})!=len(eligible) or len(eligible)<50*512:
        raise ValueError('Need enough distinct non-test training questions for50 fresh cycles')
    random.Random(c['seed']).shuffle(eligible)
    return eligible,test


def run(base):
    base=Path(base);c=read(base/'preparation/launch-config.json')
    if (not str(base).startswith(PREFIX) or c['target_probe_epochs']!=100 or c['cycles']!=50 or
        c['epochs_per_cycle']!=2 or c['actor_training_enabled'] is not False or
        c['actor_version']!='54ac33dcebe64e62a11fe4177f41f116-actor0050'):
        raise ValueError('Only the latest authorized frozen actor50/probe100 experiment is supported')
    qualification=qualify(base,c)
    eligible,test_questions=question_bundles(c)
    lease=UUIDLocks(c['physical_gpu_uuids']+c['auxiliary_gpu_uuids'])
    owned=torchrun_owned_class()(base/'logs/process-ledger.json',c['run_id'])
    interrupt=DeferredInterrupt();old_handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    for s in old_handlers:signal.signal(s,interrupt.handler)
    python=str(Path(c['environment_root'])/'bin/python')
    env=dict(os.environ,PYTHONPATH=str(base/'repo'),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        CUDA_VISIBLE_DEVICES=','.join(map(str,c['physical_gpu_indices'])),CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        HF_HOME=str(base/'cache/hf'),XDG_CACHE_HOME=str(base/'cache'),TMPDIR=str(base/'tmp'),
        TRITON_CACHE_DIR=str(base/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(base/'cache/inductor'),
        NCCL_NVLS_ENABLE='0',NO_PROXY='127.0.0.1,localhost')
    for k in ('PYTHONHOME','RAY_ADDRESS','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(k,None)
    if c.get('python_header_cpath'):env['CPATH']=c['python_header_cpath']
    if c.get('cuda_launch_blocking'):env['CUDA_LAUNCH_BLOCKING']='1'
    for name in ('tmp','cache/hf','cache/triton','cache/inductor','cycles','training','probe'):
        (base/name).mkdir(parents=True,exist_ok=True)
    cycle=0;success=False
    def stage(name,**extra):
        publish(base/'logs/status.json',dict(run_id=c['run_id'],cycle=cycle,stage=name,actor_updates=0,
            checked_unix=time.time(),**extra))
    def start(name,command,child_env=None):
        interrupt.check()
        process=owned.start(name,command,child_env or env,str(base/'logs'/(name+'.log')),
                            str(base/'repo'),pass_fds=tuple(lease.fds))
        owned.observe(name);return process
    def phase(name,command):
        stage(name);process=start(name,command)
        while process.poll() is None:
            owned.observe(name);owned.observe('retriever');interrupt.check()
            if retriever.poll() is not None:raise RuntimeError('Own retriever exited')
            time.sleep(2)
        owned.observe(name);owned.jobs[name]['exit_code']=process.returncode;owned.save();owned.stop(name)
        if process.returncode:raise RuntimeError('Own phase failed: '+name)
        interrupt.check()
    def generate(phase_base,outer,questions,is_test):
        bundle_path=base/'fixed-test/complete.json' if is_test else base/'cycles'/('cycle-%04d.json'%outer)
        if bundle_path.exists():return read(bundle_path)
        scoped=dict(c,base=str(phase_base),diagnostic_only=bool(is_test),
                    rollout_candidate_questions=512,rollout_continuation_questions=512,
                    rollout_full_subdirectory='full')
        conf=phase_base/'preparation/rollout-config.json'
        conf.parent.mkdir(parents=True,exist_ok=True)
        if not conf.exists():once(conf,scoped)
        elif read(conf)!=scoped:raise ValueError('Existing rollout configuration differs')
        root=phase_base/'outers'/('outer-%06d'%outer);root.mkdir(parents=True,exist_ok=True)
        rows=[dict(row,candidate_ordinal=i) for i,row in enumerate(questions)]
        if not (root/'questions.json').exists():once(root/'questions.json',dict(run_id=c['run_id'],outer=outer,questions=512,rows=rows))
        if not (root/'actor-binding.json').exists():once(root/'actor-binding.json',dict(actor_model_dir=c['actor_model_dir'],actor_version=c['actor_version']))
        prefix_tag='test' if is_test else 'cycle-%04d'%outer
        phase(prefix_tag+'-prefix',[python,'-m','reproduction.prefix_selection200.rollout','--config',str(conf),'--outer',str(outer),'--phase','prefix'])
        prefix=read(root/'prefix/complete.json')
        from reproduction.prefix_selection200.probe import candidate_groups
        index={x['trajectory_id']:x for x in map(json.loads,(Path(prefix['scan_path'])/'index.jsonl').open())}
        groups=candidate_groups(index,512);selection=root/'all512-selection.json'
        if not selection.exists():once(selection,dict(run_id=c['run_id'],outer=outer,actor_version=c['actor_version'],
            prefix_scan_sha256=prefix['scan_sha256'],selected=groups,selected_questions=512,outcome_labels_used=False))
        phase(prefix_tag+'-continue',[python,'-m','reproduction.prefix_selection200.rollout','--config',str(conf),'--outer',str(outer),'--phase','continue','--selection',str(selection)])
        full=read(root/'full/complete.json')
        bundle=dict(run_id=c['run_id'],cycle=0 if is_test else outer,actor_version=c['actor_version'],
            actor_updates=0,scan_path=prefix['scan_path'],scan_sha256=prefix['scan_sha256'],
            selection_sha256=sha(selection),own_em=full['own_em']['path'],own_em_sha256=full['own_em']['sha256'],
            questions=512,trajectories=2560,strict_EM_binary=True,independent_encoder_forward=False,
            training_questions_used=not is_test,fixed_test_events=is_test,
            fixed_test_identity_sha256=c['fixed_test_identity_sha256'] if is_test else None,
            full_complete_sha256=sha(root/'full/complete.json'))
        once(bundle_path,bundle);return bundle
    try:
        native_allocation(base,c)
        libs=list((Path(c['environment_root'])/'lib/python3.10/site-packages/nvidia').glob('*/lib'))
        retriever=start('retriever',[python,'-m','reproduction.prefix20_online.retriever',
            '--index',c['retrieval_index'],'--corpus',c['retrieval_corpus'],'--encoder',c['retrieval_encoder'],
            '--cache',str(base/'retrieval-cache'),'--run-id',c['run_id'],'--port',str(c['retrieval_port']),
            '--data-cache',c.get('retrieval_data_cache',str(base/'retrieval-cache')),
            '--batch-queries',str(c.get('retrieval_batch_queries',1024)),'--gpu-shards','2'],
            dict(env,CUDA_VISIBLE_DEVICES=','.join(map(str,c['auxiliary_gpu_indices'])),
                PYTHONPATH=c['retriever_gpu_lib']+':'+str(base/'repo'),LD_LIBRARY_PATH=':'.join(map(str,libs))))
        stage('STARTING_OWN_TWO_SHARD_RETRIEVER')
        deadline=time.monotonic()+1800
        while True:
            owned.observe('retriever');interrupt.check()
            if retriever.poll() is not None:raise RuntimeError('Retriever failed while loading')
            try:
                with urllib.request.urlopen(c['search_url'].replace('/retrieve','/health'),timeout=5) as response:
                    health=json.load(response)
                if health.get('ok') and health.get('run_id')==c['run_id'] and health.get('gpu_shards')==2:break
            except (OSError,ValueError):pass
            if time.monotonic()>deadline:raise TimeoutError('Own retriever startup timed out')
            time.sleep(3)
        generate(base/'fixed-test',1,test_questions,True)
        for cycle in range(1,51):
            complete=base/'training'/('cycle-%04d'%cycle)/'complete.json'
            if complete.exists():
                if read(complete)['completed_epochs']!=2*cycle:raise ValueError('Existing complete cycle differs')
                continue
            generate(base,cycle,eligible[(cycle-1)*512:cycle*512],False)
            phase('cycle-%04d-probe-two-epochs'%cycle,[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                '-m','reproduction.actor50_probe100.train','--config',str(base/'preparation/launch-config.json'),'--cycle',str(cycle)])
            latest=read(base/'probe/latest.json')
            if latest['cycle']!=cycle or latest['completed_epochs']!=2*cycle or latest['ema_updates']!=latest['actual_adam_steps_per_rank'][0]:
                raise ValueError('Real cycle Adam/EMA/epoch boundary not closed')
            stage('TWO_EPOCH_CYCLE_COMPLETE',completed_epochs=2*cycle,new_adam_updates=latest['ema_updates'])
        latest=read(base/'probe/latest.json')
        if latest['completed_epochs']!=100 or sha(latest['checkpoint'])!=latest['sha256']:
            raise ValueError('Final100 epoch checkpoint differs')
        once(base/'complete.json',dict(run_id=c['run_id'],role='frozen_actor50_balanced_probe100_complete',
            completed_epochs=100,complete_cycles=50,actor_updates=0,actor_version=c['actor_version'],
            source50_qualification=qualification,latest_probe=latest,checked_unix=time.time()))
        success=True;stage('PROBE100_COMPLETE',completed_epochs=100)
    except BaseException as error:
        publish(base/'logs/failure.json',dict(cycle=cycle,error_type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc(),checked_unix=time.time()))
        raise
    finally:
        try:
            owned.stop_all();owned.reap()
            births=[x for job in owned.jobs.values() for x in job['ledger'].values()]
            present=birth_states(births)
            publish(base/'logs/cleanup-status.json',dict(cleaned=not any(x['state']!='Z' for x in present),
                owned_births_all_gone=not present,remaining_owned_birth_count=len(present),checked_unix=time.time()))
        finally:
            lease.close()
            for s,handler in old_handlers.items():signal.signal(s,handler)
            publish(base/'logs/exit.json',dict(success=success,cycle=cycle,checked_unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True)
    value=parser.parse_args().base
    try:run(value)
    except BaseException as error:
        root=Path(value)
        if not (root/'logs/failure.json').exists():
            publish(root/'logs/failure.json',dict(error_type=type(error).__name__,message=str(error),
                traceback=traceback.format_exc(),checked_unix=time.time()))
        if not (root/'logs/exit.json').exists():
            publish(root/'logs/exit.json',dict(success=False,checked_unix=time.time()))
        raise
