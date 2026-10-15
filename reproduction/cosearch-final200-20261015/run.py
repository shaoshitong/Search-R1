"""One owned allocation: verify/export both FSDP roles, then native dual inference."""
import fcntl
import gc
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request
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
    import socket
    for port in [18086,18087,18100,18101,18102,18103]:
        with socket.socket() as sock:
            assert sock.connect_ex(('127.0.0.1',port))!=0, ('Port occupied; do not interfere',port)
    children=[]
    (OUT/'tmp').mkdir(parents=True, exist_ok=True)
    base_env=env()
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
                ('main',0),('main',1),('reranker',2),('reranker',3)]):
            memory=budgets[visible[gpu]]['llm_memory_fraction']
            variables=dict(base_env,CUDA_VISIBLE_DEVICES=visible[gpu],
                TRITON_CACHE_DIR=str(OUT/('cache/triton%d'%index)))
            servers.append(launch([str(PYTHON),'-B',str(TOOLS/'service.py'),'llm',
                '--model',str(OUT/'models'/name),'--port',str(18100+index),
                '--memory',str(memory)],'server%d'%index,variables))
        for index,child in enumerate(servers):
            wait_ready('http://127.0.0.1:%d/health'%(18100+index),child)
        evaluation=launch([str(PYTHON),'-B',str(TOOLS/'evaluate.py')],'evaluation',base_env)
        while evaluation.poll() is None:
            assert all(child.poll() is None for child in children if child is not evaluation)
            metrics()
            time.sleep(15)
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
        for child in children:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
        atomic(OUT/('cleanup-'+os.environ['SLURM_JOB_ID']+'.json'),
               dict(epoch=time.time(),all_owned_children_reaped=all(c.poll() is not None for c in children),
                    only_exact_owned_child_process_groups=True))


if __name__=='__main__':
    main()
