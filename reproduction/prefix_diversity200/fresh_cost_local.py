"""CPU-only bounded cost sync for the actual current same-run continuation chain."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from reproduction.h1008_qprobe.management_transport import remote, connection_options, local_connection_path

SCRIPT=r'''
import os,sys,json,pathlib,hashlib,time,resource
os.nice(19);os.environ['CUDA_VISIBLE_DEVICES']='';resource.setrlimit(resource.RLIMIT_CPU,(90,90))
p=PAYLOAD;b=pathlib.Path(p['base']);config=b/'preparation/launch-config.json'
assert hashlib.sha256(config.read_bytes()).hexdigest()==p['config_sha256']
chain=[];seen=set();current=b
for _ in range(16):
 assert current.parent==pathlib.Path('/mnt/data/shaoshitong') and current.resolve()==current
 assert current.name.startswith('search-g1-prefix-diversity200-') and str(current) not in seen
 c=json.loads((current/'preparation/launch-config.json').read_text());assert c['run_id']==p['run_id']
 chain.append(str(current));seen.add(str(current));parent=c.get('resume_joint',{}).get('source_base')
 if not parent:break
 current=pathlib.Path(parent)
else:raise ValueError('Continuation chain exceeds the bounded audit')
chain.reverse();sys.path.insert(0,str(b/'repo'))
from reproduction.prefix_diversity200.cost_snapshot import collect as generation
from reproduction.prefix_diversity200.cost_retrieval import collect as retrieval
started=time.time();cpu_started=resource.getrusage(resource.RUSAGE_SELF);result={}
for name,callback in [('generation',lambda:generation(chain,p['run_id'])),
 ('retrieval',lambda:retrieval(chain,p['run_id'],maximum_new_outers=p['maximum_new_outers']))]:
 try:result[name]=callback()
 except Exception as error:result[name]={'error':repr(error)}
u=resource.getrusage(resource.RUSAGE_SELF)
result['reader']=dict(observed_unix=time.time(),bases=chain,wall_seconds=time.time()-started,
 cpu_seconds=u.ru_utime+u.ru_stime-cpu_started.ru_utime-cpu_started.ru_stime,
 maxrss_kib=u.ru_maxrss,model_tensors_read=False,new_gpu_forward=False,training_modified=False)
print(json.dumps(result,allow_nan=False))
'''


def collect(root,maximum_new_outers=4,connection=None):
    if not 1<=maximum_new_outers<=4:raise ValueError('At most4 new raw-shard proofs per invocation')
    root=Path(root);out=root/'outputs/search-g1-i1i6-fresh200-20261017'
    config_path=out/'CURRENT_CONFIG.json';c=json.loads(config_path.read_text())
    connection=json.loads(Path(connection or local_connection_path(root)).read_text())
    payload=dict(base=c['base'],run_id=c['run_id'],config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),maximum_new_outers=maximum_new_outers)
    transport=out/('fresh-cost-'+str(time.time_ns())+'-transport.json')
    value=json.loads(remote(SCRIPT.replace('PAYLOAD',repr(payload)),transport,target=c['node'], **connection_options(connection)))
    actual=transport.with_name(transport.name.replace('-transport','-actual'));actual.write_text(json.dumps(value,indent=2)+'\n')
    (out/'COST_CURRENT_PROGRESS.json').write_text(json.dumps(value,indent=2)+'\n')
    gen,ret=value['generation'],value['retrieval']
    summary=dict(observed_unix=value['reader']['observed_unix'],reader=value['reader'],evidence=str(actual),
        generation_complete_outers=sorted(map(int,gen.get('complete_outer_rows',{}))),
        sampled_policy_tokens_committed=gen.get('total_completed_outer_sampled_tokens'),
        retrieval_complete_outers=sorted(map(int,ret.get('complete_outer_rows',{}))),
        new_retrieval_proofs=ret.get('new_complete_outer_proofs'),
        logical_retrieval_calls_committed=ret.get('total_distinct_logical_retrieval_calls'),
        retrieval_pending=ret.get('pending_complete_outers'),errors={k:v['error']for k,v in value.items()if 'error'in v},
        scope='Committed native generation/retrieval only; no inherited-prefix double count. Failed attempts, QA, physical HTTP/cache/FAISS calls, and GPU/CPU reservation are separate.')
    (out/'COST_COVERAGE_CURRENT.json').write_text(json.dumps(summary,indent=2)+'\n')
    for target in (root/'work/tree-grpo/reproduction/baselines/tree-validation-20261002'/out.name,
                   Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo')/out.name):
        target.mkdir(parents=True,exist_ok=True)
        for name in ('COST_CURRENT_PROGRESS.json','COST_COVERAGE_CURRENT.json'):shutil.copyfile(out/name,target/name)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--connection');p.add_argument('--maximum-new-outers',type=int,default=4);a=p.parse_args()
    print(json.dumps(collect(a.root,a.maximum_new_outers,a.connection)))
