"""Continue the sealed Prefix20 state using forty genuinely new actor50 G5 pools."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import urllib.request

from reproduction.prefix_selection200.a800_runtime import UUIDLocks,native_allocation
from reproduction.prefix_only_correctness_ab.launch import torchrun_owned_class,DeferredInterrupt,birth_states
from reproduction.prefix_selection200.probe import candidate_groups
from reproduction.actor50_prefix20_reuse20.data import read,sha,publish
from .data import fresh_plan,adopt


def run(base):
    base=Path(base);c=read(base/'preparation/config.json')
    assert c['target_probe_epochs']==100 and c['prefix_policy_tokens']==20 and not c['actor_training_enabled']
    assert c['actor_version']=='54ac33dcebe64e62a11fe4177f41f116-actor0050'
    assert 10<=c['resume_completed_cycles']<50
    success=False;owned=None;lease=None;cycle=c['resume_completed_cycles'];interrupt=DeferredInterrupt()
    handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    for s in handlers:signal.signal(s,interrupt.handler)
    def stage(name,**extra):publish(base/'logs/status.json',dict(run_id=c['run_id'],cycle=cycle,stage=name,actor_updates=0,checked_unix=time.time(),**extra))
    try:
        stage('VERIFYING_FRESH_NONREPEATED_QUESTION_PLAN');questions,proof=fresh_plan(base,c)
        lease=UUIDLocks(c['physical_gpu_uuids']+c['auxiliary_gpu_uuids']);native_allocation(base,c)
        owned=torchrun_owned_class()(base/'logs/process-ledger.json',c['run_id'])
        python=str(Path(c['environment_root'])/'bin/python')
        env=dict(os.environ,PYTHONPATH=str(base/'repo'),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
            CUDA_VISIBLE_DEVICES=','.join(map(str,c['physical_gpu_indices'])),CUDA_DEVICE_ORDER='PCI_BUS_ID',
            OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
            HF_HOME=str(base/'cache/hf'),XDG_CACHE_HOME=str(base/'cache'),TMPDIR=str(base/'tmp'),
            TRITON_CACHE_DIR=str(base/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(base/'cache/inductor'),
            NCCL_NVLS_ENABLE='0',NO_PROXY='127.0.0.1,localhost',CUDA_LAUNCH_BLOCKING='1')
        for key in ('PYTHONHOME','RAY_ADDRESS','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
        if c.get('python_header_cpath'):env['CPATH']=c['python_header_cpath']
        for name in ('tmp','cache/hf','cache/triton','cache/inductor','cycles','training','probe'):(base/name).mkdir(parents=True,exist_ok=True)
        def start(name,command,child_env=None):
            interrupt.check();child=owned.start(name,command,child_env or env,str(base/'logs'/(name+'.log')),str(base/'repo'),pass_fds=tuple(lease.fds))
            owned.observe(name);return child
        def phase(name,command):
            stage(name);child=start(name,command)
            while child.poll() is None:
                owned.observe(name);owned.observe('retriever');interrupt.check()
                if retriever.poll() is not None:raise RuntimeError('Own retriever exited')
                time.sleep(2)
            owned.observe(name);owned.jobs[name]['exit_code']=child.returncode;owned.save();owned.stop(name)
            if child.returncode:raise RuntimeError('Own phase failed: '+name)
            interrupt.check()
        libs=list((Path(c['environment_root'])/'lib/python3.10/site-packages/nvidia').glob('*/lib'))
        retriever=start('retriever',[python,'-m','reproduction.prefix20_online.retriever','--index',c['retrieval_index'],
            '--corpus',c['retrieval_corpus'],'--encoder',c['retrieval_encoder'],'--cache',str(base/'retrieval-cache'),
            '--run-id',c['run_id'],'--port',str(c['retrieval_port']),'--data-cache',c.get('retrieval_data_cache',str(base/'retrieval-cache')),
            '--batch-queries',str(c.get('retrieval_batch_queries',1024)),'--gpu-shards','2'],
            dict(env,CUDA_VISIBLE_DEVICES=','.join(map(str,c['auxiliary_gpu_indices'])),
                PYTHONPATH=c['retriever_gpu_lib']+':'+str(base/'repo'),LD_LIBRARY_PATH=':'.join(map(str,libs))))
        stage('STARTING_OWN_TWO_SHARD_RETRIEVER');deadline=time.monotonic()+1800
        while True:
            owned.observe('retriever');interrupt.check()
            if retriever.poll() is not None:raise RuntimeError('Own retriever failed')
            try:
                with urllib.request.urlopen(c['search_url'].replace('/retrieve','/health'),timeout=5) as response:health=json.load(response)
                if health.get('ok') and health.get('run_id')==c['run_id'] and health.get('gpu_shards')==2:break
            except (OSError,ValueError):pass
            if time.monotonic()>deadline:raise TimeoutError('Own retriever startup timed out')
            time.sleep(3)
        conf=base/'preparation/config.json'
        for cycle in range(c['resume_completed_cycles']+1,51):
            root=base/'outers'/('outer-%06d'%cycle);root.mkdir(parents=True,exist_ok=False)
            rows=[dict(row,candidate_ordinal=i) for i,row in enumerate(questions[(cycle-11)*512:(cycle-10)*512])]
            assert len(rows)==512
            publish(root/'questions.json',dict(run_id=c['run_id'],outer=cycle,questions=512,rows=rows))
            publish(root/'actor-binding.json',dict(actor_model_dir=c['actor_model_dir'],actor_version=c['actor_version']))
            publish(base/'preparation/pool-cursor.json',dict(complete_cycles=cycle-1,pending_cycle=cycle,
                old_training_questions_excluded=5120,committed_fresh_questions=(cycle-11)*512,pending_new_questions=512,
                question_plan_sha256=proof['ordered_new_identity_sha256']))
            phase('cycle-%04d-prefix20'%cycle,[python,'-m','reproduction.actor50_prefix20_fresh100.rollout','--config',str(conf),'--outer',str(cycle),'--phase','prefix'])
            prefix=read(root/'prefix/complete.json');assert prefix['head_prefix_tokens']==20 and prefix['trajectories']==2560
            index={r['trajectory_id']:r for r in map(json.loads,(Path(prefix['scan_path'])/'index.jsonl').open())}
            groups=candidate_groups(index,512);selection=root/'all512-selection.json'
            publish(selection,dict(run_id=c['run_id'],outer=cycle,actor_version=c['actor_version'],prefix_scan_sha256=prefix['scan_sha256'],
                selected=groups,selected_questions=512,outcome_labels_used=False))
            phase('cycle-%04d-completeG5'%cycle,[python,'-m','reproduction.actor50_prefix20_fresh100.rollout','--config',str(conf),'--outer',str(cycle),'--phase','continue','--selection',str(selection)])
            full=read(root/'full/complete.json');assert full['trajectories']==2560 and full['questions']==512
            adopt(base,cycle,prefix,full)
            phase('cycle-%04d-probe-two-epochs'%cycle,[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                '-m','reproduction.actor50_prefix20_fresh100.train','--config',str(conf),'--cycle',str(cycle)])
            latest=read(base/'probe/latest.json');assert latest['completed_epochs']==2*cycle and latest['cycle']==cycle
            assert latest['actual_adam_steps_per_rank']==[latest['ema_updates']]*4
            if cycle==c['resume_completed_cycles']+1:
                assert latest['metrics']['before']['test']==read(base/'preparation/resume-qualified.json')['source_latest']['metrics']['after']['test']
            publish(base/'preparation/pool-cursor.json',dict(complete_cycles=cycle,pending_cycle=None,old_training_questions_excluded=5120,
                committed_fresh_questions=(cycle-10)*512,question_plan_sha256=proof['ordered_new_identity_sha256']))
            stage('TWO_NEW_EPOCHS_COMPLETE',completed_epochs=2*cycle,Adam_EMA=latest['ema_updates'])
        latest=read(base/'probe/latest.json');assert latest['completed_epochs']==100 and sha(latest['checkpoint'])==latest['sha256']
        publish(base/'complete.json',dict(run_id=c['run_id'],completed_epochs=100,completed_cycles=50,
            actor_updates=0,fresh_training_cycles_after20=40,fresh_training_questions_after20=20480,
            latest_probe=latest,old20_training_G5_not_reused=True,checked_unix=time.time()))
        success=True;stage('PREFIX20_FRESH100_COMPLETE',completed_epochs=100)
    except BaseException as error:
        publish(base/'logs/failure.json',dict(cycle=cycle,error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),checked_unix=time.time()))
        raise
    finally:
        if owned:
            owned.stop_all();owned.reap();present=birth_states([v for job in owned.jobs.values() for v in job['ledger'].values()])
        else:present=[]
        publish(base/'logs/cleanup-status.json',dict(cleaned=not any(x['state']!='Z' for x in present),owned_births_all_gone=not present,remaining_owned_birth_count=len(present),checked_unix=time.time()))
        if lease:lease.close()
        for s,handler in handlers.items():signal.signal(s,handler)
        publish(base/'logs/exit.json',dict(success=success,cycle=cycle,checked_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);run(p.parse_args().base)
