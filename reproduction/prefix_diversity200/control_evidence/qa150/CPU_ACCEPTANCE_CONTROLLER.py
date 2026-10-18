"""Invoke the unchanged QA50 saved-output CPU verifier for actual milestone QA only."""
import base64
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

P = {'outer': 150, 'run_id': '4a1cc2d500fe40979ef8ca4792959438', 'base': '/mnt/shared/workspace/training_data_node_10/shaoshitong/search-g1-qa-i1i6-4a1cc2d5-outer0150-22d511fc', 'unit': 'g1-qa-4a1cc2d5-o150-22d511fc.service', 'source_commit': '2b0a9f7b809d63880dcb0e30bebc96ab11267f45', 'config_sha256': '6a87a930ff83bcf0b9404999bbd89ac70a277740724b7f30e4659c1113630c98', 'overlay_sha256': {'__init__.py': '5f804155694951c57dc877257235bc6792040e03a64f7bfe1348f9ddac407424', 'actor.py': '7d97e9c7e190dc680cfe70dd4f8e891541c0e2d825e49ffada8aa1e301c72a86', 'interaction.py': '5a1af508d612842eadf698d1f8ad4570b3b26cf9e3507af5098d66927d428714', 'rollout.py': 'a8e3cea447fbad5abfda07f25b1bb6b3a8a33b96eb983b5ed33f6702f590052f', 'eval.py': 'a19668cd5b63c3ed1e0adfef35dd5bd6678a60828fca8bb9e973c84e1d5f98c4', 'evaluation_initialization.py': '2cf640ac2c6bbf5fcf174007256df722cfcb597e4a7317e766776a7c5ff5ea89', 'qa_standalone.py': '92f05234636af83ff27faf54a4a1ac6ebd2ba2551736c302a049cfde62a9ff3f', 'qa_standalone_identity.py': '576620c10b9cae45d779070c723065590d831b8be8536180c6abe1d1c58b494a', 'qa_acceptance_cpu.py': 'e558437341105a644283790bffc4623c170c0dfffc997975c47fde2bb1be52e7'}, 'joint_sha256': '4b248ce57737c1910502e2e6096ebc121e1f8954bb78d91559f62dd0418bb7b6', 'checkpoint_proof_sha256': '998a5886bc0d0e8ec0cbc546e0114503f25394f5eb2d7e31ebba2c8350c07597', 'model_metadata': [{'name': 'added_tokens.json', 'bytes': 605, 'sha256': '58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b'}, {'name': 'config.json', 'bytes': 870, 'sha256': 'ac7bc598cafe79d5952c8b3047c3fb9afa0c435391f62b2a7e45974ed85940aa'}, {'name': 'generation_config.json', 'bytes': 243, 'sha256': '5cbc3d054b73e70801535a65ffc9b427481257fa29f896e48f68e9ea3c285b96'}, {'name': 'merges.txt', 'bytes': 1671853, 'sha256': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5'}, {'name': 'model.safetensors.index.json', 'bytes': 35640, 'sha256': 'c0139b4a802a6c6f3b4638e46abb152847b2389e24544d7fd2466e1956ea19e3'}, {'name': 'special_tokens_map.json', 'bytes': 613, 'sha256': '76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd'}, {'name': 'tokenizer.json', 'bytes': 11421896, 'sha256': '9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa'}, {'name': 'tokenizer_config.json', 'bytes': 7306, 'sha256': '7e88129d9769a0b14b1587a7d5e829fe93ac0e1511636471fdfc0811951418e6'}, {'name': 'vocab.json', 'bytes': 2776833, 'sha256': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}], 'model_weight_sizes': [{'name': 'model-00001-of-00004.safetensors', 'bytes': 3929249504}, {'name': 'model-00002-of-00004.safetensors', 'bytes': 3917843656}, {'name': 'model-00003-of-00004.safetensors', 'bytes': 3917843680}, {'name': 'model-00004-of-00004.safetensors', 'bytes': 1823527024}]}
b=Path(P['base']);prep=b/'preparation';unit=P['unit']
assert b.parent==Path('/mnt/shared/workspace/training_data_node_10/shaoshitong') and b.resolve()==b
read=lambda path:json.loads(Path(path).read_text())
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
def once(path,value):
    with path.open('x')as stream:json.dump(value,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
def unit_status():
    names=['LoadState','ActiveState','SubState','Result','MainPID','ExecMainStatus','ControlGroup','CPUUsageNSec']
    q=subprocess.run(['systemctl','show',unit]+['--property='+x for x in names],capture_output=True,text=True,timeout=15)
    assert q.returncode==0
    fields=dict(line.split('=',1)for line in q.stdout.splitlines()if'='in line)
    assert fields['ActiveState']=='inactive' and fields['SubState']=='dead' and fields['MainPID']=='0'
    assert fields['ExecMainStatus']=='0' and fields['Result']=='success'
    group=fields.get('ControlGroup')or'/system.slice/'+unit
    path=Path('/sys/fs/cgroup')/group.lstrip('/')
    pids=(path/'cgroup.procs').read_text().split()if(path/'cgroup.procs').exists()else[]
    assert not pids
    return dict(observed_unix=time.time(),systemd=q.stdout,cgroup_exists=path.exists(),cgroup_pids=pids)

c=read(prep/'config.json');spec=c['qa_standalone']
assert sha(prep/'config.json')==P['config_sha256']
assert c['qa_source_commit']==P['source_commit']=='2b0a9f7b809d63880dcb0e30bebc96ab11267f45'
assert c['run_id']==P['run_id']=='4a1cc2d500fe40979ef8ca4792959438' and c['actor_version']==c['run_id']+'-actor%04d'%P['outer']
assert c['node']=='h100-11' and spec['base']==str(b) and spec['unit']==unit and spec['outer']==P['outer'] and spec['expected_count']==512
assert spec['source_joint_receipt_sha256']==P['joint_sha256']
assert read(prep/'source-files.json')==P['overlay_sha256']
for name,digest in P['overlay_sha256'].items():assert sha(b/'repo/reproduction/prefix_diversity200'/name)==digest
assert P['overlay_sha256']['qa_acceptance_cpu.py']=='e558437341105a644283790bffc4623c170c0dfffc997975c47fde2bb1be52e7'
model=Path(spec['model_dir']);model_metadata=[]
for row in P['model_metadata']:
    path=model/row['name'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    model_metadata.append(dict(row,current_metadata_sha_verified=True))
for row in P['model_weight_sizes']:assert(model/row['name']).stat().st_size==row['bytes']
qa_path=b/'validation/complete.json';qa_before=sha(qa_path);qa=read(qa_path)
assert qa['qa_count']==512 and qa['actor_version']==c['actor_version'] and qa['per_source']['nq']['qa_count']==qa['per_source']['hotpotqa']['qa_count']==256
assert qa['actual_worker_exit_codes']=={str(i):0 for i in range(8)}
terminal=read(b/'logs/terminal.json');terminal_sha=sha(b/'logs/terminal.json')
assert terminal['success'] is True and terminal['error'] is None and not terminal['cleanup_errors'] and not terminal['unreaped_child_pids']
assert terminal['systemd_unit']==unit and terminal['outer']==P['outer']
supervisor=read(b/'validation/initialization-attempts/supervisor-terminal.json')
assert supervisor['all_owned_workers_reaped'] and not supervisor['cleanup_errors'] and not supervisor['unreaped_ranks']
before=unit_status()
proof=prep/'independent-qa512-cpu-recompute.json';intent=prep/'independent-qa512-cpu-recompute-intent.json'
assert not proof.exists() and not intent.exists(),'Existing CPU proof/intent: reconcile without repeating raw recompute'
once(intent,dict(run_id=c['run_id'],outer=P['outer'],config_sha256=P['config_sha256'],source_commit=P['source_commit'],joint_sha256=P['joint_sha256'],qa_receipt_sha256=qa_before,created_unix=time.time(),new_generation=False,model_weights_loaded=False))
worker=r'''
import json,os,resource,sys,time
from pathlib import Path
os.nice(19)
available=os.sched_getaffinity(0);os.sched_setaffinity(0,{min(available)})
b=Path(sys.argv[1]);prep=b/'preparation';started=time.monotonic();began=time.time()
def once(path,value):
    with path.open('x')as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
try:
    from reproduction.prefix_diversity200.qa_acceptance_cpu import verify
    value=verify(b,512);value['cpu_recompute_wall_seconds']=time.monotonic()-started
    once(prep/'independent-qa512-cpu-recompute.json',value)
    status='CPU_SAVED_OUTPUT_ACCEPTANCE_PASS';code=0
except BaseException as error:
    status='CPU_SAVED_OUTPUT_ACCEPTANCE_FAILED';code=1
    once(prep/'independent-qa512-cpu-recompute-failure.json',dict(status=status,error_type=type(error).__name__,error=str(error),observed_unix=time.time(),new_generation=False,model_weights_loaded=False))
u=resource.getrusage(resource.RUSAGE_SELF)
once(prep/'independent-qa512-cpu-cost.json',dict(status=status,started_unix=began,ended_unix=time.time(),elapsed_seconds=time.monotonic()-started,cpu_user_seconds=u.ru_utime,cpu_system_seconds=u.ru_stime,max_rss_kib=u.ru_maxrss,nice=os.getpriority(os.PRIO_PROCESS,0),cpu_affinity=sorted(os.sched_getaffinity(0)),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],new_generation=False,model_weights_loaded=False,gpu_forward_performed=False))
raise SystemExit(code)
'''
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(b/'repo')+':'+spec['source_repo']+':'+c['runtime_extra_packages'])
command=[str(Path(c['environment_root'])/'bin/python'),'-u','-c',worker,str(b)]
started=time.monotonic();process=subprocess.run(command,cwd=b/'repo',env=env,capture_output=True,text=True,timeout=120)
execution=dict(returncode=process.returncode,stdout=process.stdout[-12000:],stderr=process.stderr[-12000:],wall_seconds=time.monotonic()-started,observed_unix=time.time())
once(prep/'independent-qa512-cpu-command-exit.json',execution)
after=unit_status();assert qa_before==sha(qa_path) and terminal_sha==sha(b/'logs/terminal.json')
assert sha(prep/'config.json')==P['config_sha256']
value=dict(status='PASS'if process.returncode==0 else'FAILED',execution=execution,cost=read(prep/'independent-qa512-cpu-cost.json'),preflight=before,postflight=after,terminal={k:terminal[k]for k in('success','error','cleanup_errors','unreaped_child_pids','started_unix','ended_unix','outer','systemd_unit','cpu_seconds_self_and_reaped_children','wall_gpu8_hours','slurm_reserved_gpus')},qa_receipt_sha256=qa_before,qa_receipt_unchanged=True,terminal_sha256=terminal_sha,per_source=qa['per_source'],source_commit=P['source_commit'],source_overlay_sha256=P['overlay_sha256'],model_metadata=model_metadata,model_weight_sizes_stat_only=P['model_weight_sizes'],reused_checkpoint_cpu_proof_sha256=P['checkpoint_proof_sha256'],full_hf_weights_read=False,model_weights_loaded=False,new_generation=False,training_modified=False,instances_started_stopped_or_restarted=False,observed_unix=time.time())
if process.returncode==0:
    acceptance=read(proof)
    assert acceptance['config_sha256']==P['config_sha256'] and acceptance['qa_receipt_sha256']==qa_before
    assert acceptance['qa_count']==spec['expected_count']==512 and acceptance['correct']/512==acceptance['em']==qa['em'] and abs(acceptance['f1']-qa['f1'])<1e-12
    value.update(acceptance=acceptance,proof_path=str(proof),proof_sha256=sha(proof),proof_base64=base64.b64encode(proof.read_bytes()).decode())
else:value['failure']=read(prep/'independent-qa512-cpu-recompute-failure.json')
print(json.dumps(value))
