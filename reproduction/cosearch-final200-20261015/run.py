"""One owned allocation: verify/export both FSDP roles, then native dual inference."""
import fcntl
import datetime
import gc
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request
import traceback
import uuid
from common import BASE, TRAIN, OUT, TOOLS, SOURCE, PYTHON, N, atomic, sha, env


def merge_role(role, name, saved):
    import torch
    from safetensors import safe_open
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger
    from verl.model_merger.base_model_merger import ModelMergerConfig
    original = TRAIN/'checkpoints/global_step_200'/role
    target = OUT/'models'/name
    proofs = [x for x in saved['files'] if pathlib.Path(x['path']).parent == original
              and pathlib.Path(x['path']).name.startswith('model_world_size_')]
    assert len(proofs) == 4
    for item in proofs:
        path = pathlib.Path(item['path'])
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
        print('MODEL_SOURCE_SHA_PASS', role, path.name, flush=True)
    config = ModelMergerConfig(operation='merge', backend='fsdp', local_dir=str(original),
        target_dir=str(target), hf_model_config_path=str(original/'huggingface'),
        use_cpu_initialization=True)
    merger = FSDPModelMerger(config)
    zero = merger._load_rank_zero_state_dict(4)
    mesh, names = merger._extract_device_mesh_info(zero, 4)
    count, shape = merger._calculate_shard_configuration(mesh, names)
    del zero
    gc.collect()
    merged = merger._load_and_merge_state_dicts(4, count, shape, names)
    assert merged and all(torch.isfinite(value).all() for value in merged.values())
    merger.save_hf_model_and_tokenizer(merged)
    # Reopen the real saved weights and compare every exported tensor to native merge.
    keys = set()
    for path in target.glob('*.safetensors'):
        with safe_open(path, framework='pt', device='cpu') as stream:
            for key in stream.keys():
                assert key in merged and torch.equal(stream.get_tensor(key), merged[key])
                keys.add(key)
    missing = set(merged)-keys
    assert not missing or (missing == {'lm_head.weight'} and
                          torch.equal(merged['lm_head.weight'], merged['model.embed_tokens.weight']))
    assert keys
    atomic(OUT/('export-%s-verified.json'%name), dict(epoch=time.time(),
        role=role, checkpoint=200, native_FSDP_merge=True,
        all_exported_tensors_equal=True, exported_tensors=len(keys),
        input_files=proofs, files={p.name:sha(p) for p in target.iterdir() if p.is_file()}))
    del merged, merger
    gc.collect()
    print('MODEL_EXPORT_READBACK_PASS', name, flush=True)


def main():
    assert os.environ.get('SLURM_JOB_ID')
    started_epoch = time.time()
    lock=(OUT/'eval.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((OUT/'source-manifest.json').read_text())
    for rel, digest in manifest['files'].items():
        assert sha(OUT/rel) == digest, rel
    config=json.loads((OUT/'native-config.json').read_text())
    assert config['trainer']['n_gpus_per_node'] == 4
    assert config['actor_rollout_ref']['rollout']['val_kwargs']['temperature'] == 0
    assert config['reranker_actor_rollout_ref']['rollout']['val_kwargs']['temperature'] == 0
    saved=json.loads((TRAIN/'checkpoint-receipt-200.json').read_text())
    assert saved['step'] == 200 and saved['status']=='BOTH_ROLE_FULL_CHECKPOINT_FILES_SHA_PASS'
    for role,name in [('actor','main'),('reranker_actor_rollout','reranker')]:
        receipt=OUT/('export-%s-verified.json'%name)
        if receipt.exists():
            previous=json.loads(receipt.read_text())
            for filename,digest in previous['files'].items():
                assert sha(OUT/'models'/name/filename)==digest
            for item in previous['input_files']:
                assert sha(item['path'])==item['sha256']
        else:
            merge_role(role,name,saved)
    binding=dict(training_job=713379, checkpoint_step=200, rows=N,
        two_roles={role:json.loads((OUT/('export-%s-verified.json'%role)).read_text())
                   for role in ['main','reranker']},
        data_sha256=sha(TRAIN/'data/final.parquet'),
        data_jsonl_sha256=sha(TRAIN/'data/final.jsonl'),
        source_manifest_sha256=sha(OUT/'source-manifest.json'),
        native_main_max_model_len=24576, native_reranker_prompt_limit=16384,
        main_response_retention=4096, native_max_assistant_turns=6,
        no_gold_to_inference=True)
    if (OUT/'binding.json').exists():
        assert json.loads((OUT/'binding.json').read_text()) == binding
    else:
        atomic(OUT/'binding.json',binding)
    visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert len(visible)==4
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid',
        '--format=csv,noheader,nounits'],text=True)
    mapping={line.split(',')[0].strip():line.split(',')[1].strip()
             for line in inventory.splitlines()}
    visible=[item if item.startswith('GPU-') else mapping[item] for item in visible]
    physical={uuid:index for index,uuid in mapping.items()}
    import socket
    for port in [18086,18087,18100,18101,18102,18103]:
        with socket.socket() as sock:
            assert sock.connect_ex(('127.0.0.1',port))!=0, ('Port occupied; do not interfere',port)
    children=[]
    (OUT/'tmp').mkdir(parents=True, exist_ok=True)
    base_env=env()
    # Match the already working native CoSearch launcher: only Unix sockets use
    # node /tmp; model files, package caches and all predictions stay shared.
    rpc=pathlib.Path('/tmp')/('cs-final-rpc-'+str(os.getuid())+'-'+os.environ['SLURM_JOB_ID'])
    rpc.mkdir(mode=0o700,exist_ok=True)
    assert rpc.stat().st_uid==os.getuid()
    assert len((str(rpc)+'/'+'x'*36).encode())<107
    probe=rpc/('probe-'+uuid.uuid4().hex[:8])
    with socket.socket(socket.AF_UNIX) as sock:
        sock.bind(str(probe))
    probe.unlink()
    base_env['VLLM_RPC_BASE_PATH']=str(rpc)
    performance=OUT/'performance.json'
    settings=json.loads(performance.read_text()) if performance.exists() else {}
    concurrency=settings.get('question_concurrency',128)
    main_replicas=settings.get('main_replicas',2)
    assert isinstance(concurrency,int) and 1<=concurrency<=512
    assert isinstance(main_replicas,int) and 1<=main_replicas<=3
    base_env['COSEARCH_EVAL_CONCURRENCY']=str(concurrency)
    base_env['COSEARCH_MAIN_REPLICAS']=str(main_replicas)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def launch(command,name,variables):
        child=subprocess.Popen(command,env=variables,stdout=(OUT/(name+'.log')).open('a'),
                               stderr=subprocess.STDOUT,cwd=SOURCE,start_new_session=True)
        children.append(child)
        return child

    def wait_ready(url,child,deadline=1200):
        start=time.monotonic()
        while time.monotonic()-start<deadline:
            assert child.poll() is None, (url,child.returncode)
            try:
                with opener.open(url,timeout=3) as response:
                    if response.status==200:
                        return
            except OSError:
                pass
            time.sleep(2)
        raise TimeoutError(url)

    def metrics():
        try:
            text=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits'],text=True)
            records=[]
            for port in [18100,18101,18102,18103,18087]:
                try:
                    with opener.open('http://127.0.0.1:%d/health'%port,timeout=2) as response:
                        records.append(dict(port=port,**json.load(response)))
                except OSError:
                    pass
            with (OUT/'utilization.jsonl').open('a') as stream:
                own='\n'.join(line for line in text.splitlines() if line.split(',')[0].strip() in visible)
                stream.write(json.dumps(dict(epoch=time.time(),gpus=own,services=records))+'\n')
        except subprocess.SubprocessError:
            pass

    partial = False
    try:
        # Native retriever is isolated in its verified CUDA FAISS Python environment.
        sys.path.insert(0,str(TRAIN/'tools'))
        from retriever_spec import specification
        command,variables=specification(OUT)
        variables['CUDA_VISIBLE_DEVICES']=','.join(visible)
        variables['COSEARCH_LEASED_UUIDS']=','.join(visible)
        service=launch(command,'retriever',variables)
        wait_ready('http://127.0.0.1:18086/openapi.json',service)
        batch=launch([str(PYTHON),'-B',str(TOOLS/'service.py'),'retrieval','--port','18087'],
                     'retrieval-batcher',base_env)
        wait_ready('http://127.0.0.1:18087/health',batch)
        # Native FAISS shards the index across the same four leased GPUs.
        # Reserve actual remaining memory rather than assuming one card holds the index.
        budgets={}
        memory_rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,memory.total',
            '--format=csv,noheader,nounits'],text=True)
        for line in memory_rows.splitlines():
            uuid,free,total=[x.strip() for x in line.split(',')]
            if uuid in visible:
                fraction=min(.82,float(free)/float(total)-.08)
                assert fraction>=.35, ('Insufficient leased GPU memory after native index sharding',uuid,free,total)
                budgets[uuid]=dict(free_MiB=int(free),total_MiB=int(total),llm_memory_fraction=fraction)
        assert set(budgets)==set(visible)
        atomic(OUT/'gpu-budgets.json',dict(epoch=time.time(),native_faiss_index_shards=4,
            gpus=budgets,headroom_fraction=.08))
        servers=[]
        for index,(name,gpu) in enumerate([
                ('main' if gpu<main_replicas else 'reranker',gpu) for gpu in range(4)]):
            memory=budgets[visible[gpu]]['llm_memory_fraction']
            variables=dict(base_env,CUDA_VISIBLE_DEVICES=physical[visible[gpu]],
                TRITON_CACHE_DIR=str(OUT/('cache/triton%d'%index)))
            servers.append(launch([str(PYTHON),'-B',str(TOOLS/'service.py'),'llm',
                '--model',str(OUT/'models'/name),'--port',str(18100+index),
                '--memory',str(memory)],'server%d'%index,variables))
        for index,child in enumerate(servers):
            wait_ready('http://127.0.0.1:%d/health'%(18100+index),child)
        job_info=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'-o'],text=True)
        end_text=next(item.split('=',1)[1] for item in job_info.split() if item.startswith('EndTime='))
        deadline=datetime.datetime.fromisoformat(end_text).timestamp()-180
        evaluation_env=dict(base_env,COSEARCH_SLICE_DEADLINE=str(deadline))
        evaluation=launch([str(PYTHON),'-B',str(TOOLS/'evaluate.py')],'evaluation',evaluation_env)
        first_progress = None
        startup_checked = False
        while evaluation.poll() is None:
            assert all(child.poll() is None for child in children if child is not evaluation)
            metrics()
            progress_file=OUT/'progress.json'
            if progress_file.exists() and not startup_checked:
                progress=json.loads(progress_file.read_text())
                if progress['epoch']>=started_epoch and progress['rows']>=32 and progress['reranker_calls']>0:
                    if first_progress is None:
                        first_progress=progress
                    elif (progress['rows']>first_progress['rows'] and
                          progress['main_calls']>first_progress['main_calls'] and
                          progress['reranker_calls']>first_progress['reranker_calls']):
                        atomic(OUT/'startup-verified.json',dict(epoch=time.time(),
                            job=int(os.environ['SLURM_JOB_ID']),
                            status='DUAL_MODEL_REAL_QUESTION_PROGRESS_TWO_SAMPLES_PASS',
                            first=first_progress,second=progress,
                            all_owned_services_alive=True,
                            checkpoint_step=200,gpus=4))
                        startup_checked=True
            time.sleep(15)
        if evaluation.returncode==75:
            partial = True
        else:
            assert evaluation.returncode==0
            assert json.loads((OUT/'complete.json').read_text())['rows']==N
    except BaseException as error:
        atomic(OUT/('failure-'+os.environ['SLURM_JOB_ID']+'.json'),
               dict(epoch=time.time(),error=repr(error),training_untouched=True))
        raise
    finally:
        for child in children:
            if child.poll() is None:
                # Process groups contain only children started by this evaluation.
                import signal
                os.killpg(child.pid,signal.SIGTERM)
        cleanup_deadline=time.monotonic()+45
        for child in children:
            try:
                child.wait(timeout=max(.1,cleanup_deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
        atomic(OUT/('cleanup-'+os.environ['SLURM_JOB_ID']+'.json'),
               dict(epoch=time.time(),all_owned_children_reaped=all(c.poll() is not None for c in children),
                    only_exact_owned_child_process_groups=True))
        lock.close()
    if partial:
        assert not (OUT/'complete.json').exists()
        print('DURABLE_QUESTION_SLICE_COMPLETE_REQUEUE_SAME_JOB',flush=True)
        subprocess.run(['scontrol','requeue',os.environ['SLURM_JOB_ID']],check=True)
    else:
        # Submit CPU acceptance only after the full denominator is durably complete.
        # This has no dependency on previous experiments and consumes no GPUs.
        audit_lock=(OUT/'audit-submit.lock').open('a')
        fcntl.flock(audit_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (OUT/'audit-job.id').exists():
            assert not (OUT/'audit-submit-intent.json').exists(), 'Resolve an unknown previous audit submission first'
            assert not subprocess.check_output(['squeue','-h','-u','mou176',
                '--name','cosearch200-final-audit','-o','%i'],text=True).strip()
            atomic(OUT/'audit-submit-intent.json',dict(epoch=time.time(),
                producer_job=int(os.environ['SLURM_JOB_ID']),
                complete_sha256=sha(OUT/'complete.json')))
            audit_job=subprocess.check_output(['sbatch','--parsable',
                '--dependency=afterany:'+os.environ['SLURM_JOB_ID'],
                str(TOOLS/'acceptance.sbatch')],text=True).strip().split(';')[0]
            assert audit_job.isdigit()
            (OUT/'audit-job.id').write_text(audit_job+'\n')
            print('FULL_FINAL_COMPLETE_CPU_ACCEPTANCE_SUBMITTED',audit_job,flush=True)


def await_explicit_repair(error_text):
    """Keep only this failed allocation briefly; never retry an unknown error."""
    info=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'-o'],text=True)
    end=next(x.split('=',1)[1] for x in info.split() if x.startswith('EndTime='))
    deadline=min(time.time()+900,datetime.datetime.fromisoformat(end).timestamp()-180)
    cookie=uuid.uuid4().hex
    atomic(OUT/'repair-wait.json',dict(epoch=time.time(),job=int(os.environ['SLURM_JOB_ID']),
        pid=os.getpid(),wait_id=cookie,deadline_epoch=deadline,error=error_text,
        status='OWNED_FAILED_SERVICES_STOPPED_AWAITING_VERIFIED_REPAIR',
        no_automatic_retry=True))
    command=OUT/'repair-command.json'
    while time.time()<deadline:
        if command.exists():
            approved=json.loads(command.read_text())
            if (approved.get('wait_id')==cookie and approved.get('job')==int(os.environ['SLURM_JOB_ID'])
                and approved.get('manifest_sha256')==sha(OUT/'source-manifest.json')):
                resumed=json.loads((OUT/'repair-wait.json').read_text())
                resumed.update(epoch=time.time(),status='VERIFIED_REPAIR_ACCEPTED_EXECUTING')
                atomic(OUT/'repair-wait.json',resumed)
                print('EXPLICIT_VERIFIED_REPAIR_EXEC_SAME_ALLOCATION',flush=True)
                os.execv(sys.executable,[sys.executable,'-B',str(TOOLS/'run.py')])
        time.sleep(2)
    raise RuntimeError('Verified repair did not arrive before the bounded allocation deadline: '+error_text)


if __name__=='__main__':
    failed=None
    try:
        main()
    except Exception as error:
        traceback.print_exc()
        failed=repr(error)
    if failed is not None:
        await_explicit_repair(failed)
