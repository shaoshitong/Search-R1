"""Actual first sealed own batch only; reuse accepted CPU causal/seed audit.

No model tensor, reference GPU or generation is loaded. Tokenizer-only strict EM
is recomputed. Full per-trajectory rows remain remote. Not a whole-tail audit.
"""
import argparse,collections,fcntl,gzip,hashlib,inspect,json,math,os,sys,time
from pathlib import Path
from types import SimpleNamespace
REVISION='a09a35458c702b33eeacc393d103063234e8bc28'
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
BASE=B/'algorithm-improvement-20261021/hpc3-tail-newseedG5-20261021-v1'
OUTPUT=B/'algorithm-improvement-20261021/hpc3-tail-firstbatch-audit-v1'
CONFIG_SHA='368ad5d8df9e0238bdee8033964b56f0f4a6426e606f9514114fac0dcb05d835'
INHERITED_SOURCE_BINDING={'commit': '1864f7246e9642b958deed401dd4797be80c099b', 'functions': {'canonical': {'source': 'reproduction/operations/qwen7b_warmup/peer_newseed_audit_reference.py', 'sha256': '0299bd6afe1ddbd8b04d2105dd3d780d53749f6390a8eb43afb354a74eafe29d'}, 'require': {'source': 'reproduction/operations/qwen7b_warmup/peer_newseed_audit_reference.py', 'sha256': '570ace48effbcffbb325b4bdc4424b575500114187d28c59c86b8870081e9f76'}, 'token_ids': {'source': 'reproduction/operations/qwen7b_warmup/peer_newseed_audit_reference.py', 'sha256': '39697d74a6178fc85777bdc4782e0cd767ba01a7fa0f472654e59dc8bdb11046'}, 'log_probs': {'source': 'reproduction/operations/qwen7b_warmup/peer_newseed_audit_reference.py', 'sha256': 'c2d1cbd611e9dab9622c54c56569e68f1840b569caef6a155c006aff51a85573'}, 'seed_for': {'source': 'reproduction/operations/qwen7b_warmup/peer_newseed_audit_reference.py', 'sha256': '23588ba858aed9e07c05d78ec10e65935284c942f2cddf8b45bbdf07a56c7e64'}, 'audit_trajectory': {'source': 'reproduction/operations/qwen7b_warmup/peer_newseed_ingest.py', 'sha256': '524f2166bc4dbdbc875a13a8884bb429174d8f0024f2a75aed9cdd9d31eeada5'}}}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def token_ids(value, name, nonempty=False):
    require(isinstance(value, list) and (value or not nonempty)
            and all(type(x) is int and x >= 0 for x in value), name + ': invalid token IDs')
    return value

def log_probs(value, name):
    require(isinstance(value, list) and all(type(x) in (int, float)
            and math.isfinite(x) and x <= 1e-6 for x in value), name + ': invalid log probabilities')
    return value

def seed_for(run_id, trajectory_id, turn, chunk, seed, attempt):
    # Exact CPU formula in prefix_selection200.rollout.seed_for; no GPU imports.
    value = [run_id, 1, trajectory_id, int(turn), int(chunk), int(seed)]
    if attempt is not None:
        value.append(attempt)
    return int.from_bytes(hashlib.sha256(canonical(value)).digest()[:8], 'big') % (2 ** 31 - 1)

def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
audit=SimpleNamespace(token_ids=token_ids,log_probs=log_probs,seed_for=seed_for)

def audit_trajectory(prefix,full,source,config,own_label,score,ledger,event):
    """Replay actual causal token/context/chunk positions; never infer a label."""
    tid=full['trajectory_id'];qid=source['qid'];run=config['run_id'];attempt=config.get('sampling_attempt_id')
    actor='official7b-'+REVISION;index=full['response_index']
    if type(index) is not int or not 0<=index<5: raise ValueError('Each real event has actual slots0..4')
    wanted='%s-outer000001-c%04d-g%d'%(run,source['candidate_ordinal'],index)
    for row in (prefix,full):
        for key,value in dict(trajectory_id=wanted,qid=qid,question_hash=source['question_hash'],run_id=run,outer=1,
                split='train',response_index=index,actor_version=actor,candidate_ordinal=source['candidate_ordinal'],
                initial_prompt_token_ids=source['initial_prompt_token_ids']).items():
            if row.get(key)!=value: raise ValueError('Actual own trajectory/prompt identity differs: '+key)
    claimed=prefix['prefix_sha256']
    # Native capture appends feature_* locators after sealing the causal prefix.
    # Qualified replay normalizes the same exact sealed body; never hash locators.
    material={key:value for key,value in prefix.items() if key!='prefix_sha256' and not key.startswith('feature_')}
    if digest(material)!=claimed or prefix.get('future_information_included') is not False:
        raise ValueError('Prefix SHA/future-information contract differs')
    if (full.get('prefix_sha256')!=claimed or full.get('policy_snapshot_id')!=actor or
            full.get('reward_model')!=source['reward_model'] or full.get('data_source')!=source['data_source'] or
            type(full.get('strict_em')) is not int or full['strict_em']!=own_label or own_label not in (0,1)):
        raise ValueError('Full own EM/actor/gold/prefix binding differs')
    prompt=audit.token_ids(source['initial_prompt_token_ids'],'prompt',nonempty=True)
    rebuilt=[];mask=[];logs=[];journal=[];retained=[];positions=[];segments=[];seeds=[]
    for turn in full['behavior_turns']:
        request=audit.token_ids(turn['actual_request_token_ids'],'request',nonempty=True)
        raw=audit.token_ids(turn['raw_generated_token_ids'],'raw')
        lp=audit.log_probs(turn['selected_logprobs'],'raw');kept=turn['kept_token_count']
        if (request!=prompt+rebuilt or type(kept) is not int or not 0<=kept<=len(raw)==len(lp) or
                turn['trajectory_id']!=tid or turn['policy_snapshot_id']!=actor or
                turn['prefix_cut'].get('kept_token_ids')!=raw[:kept] or turn['prefix_cut'].get('kept_token_count')!=kept):
            raise ValueError('Actual causal turn request/cut differs')
        chunkids=[];chunklp=[];chunks=turn['actual_sampling_chunks']
        if not chunks: raise ValueError('Actual native chunks required')
        for chunk in chunks:
            t,c=chunk['logical_turn_index'],chunk['native_chunk_index']
            expected_seed=audit.seed_for(run,tid,t,c,config['seed'],attempt)
            if (chunk['actual_request_token_ids']!=request+chunkids or t!=turn['turn_index'] or
                    chunk['request_id']!='%s-t%d-chunk%d'%(tid,t,c) or chunk['seed']!=expected_seed or
                    chunk.get('sampling_attempt_id')!=attempt or chunk.get('sampling',{}).get('seed')!=expected_seed):
                raise ValueError('Actual native chunk context/seed differs')
            ids=audit.token_ids(chunk['raw_generated_token_ids'],'chunk');pl=audit.log_probs(chunk['selected_logprobs'],'chunk')
            if len(ids)!=len(pl): raise ValueError('Actual native chunk logprobs differ')
            chunkids.extend(ids);chunklp.extend(pl);seeds.append(expected_seed)
        if chunkids!=raw or chunklp!=lp: raise ValueError('Native chunks do not reconstruct turn')
        journal.extend(chunks);take=min(kept,20-len(retained))
        if take:
            segments.append(dict(turn_index=turn['turn_index'],policy_start=len(retained),policy_length=take,
                request_length=len(request),position_start=len(request),context_tokens=request+raw[:take]))
            retained.extend(raw[:take]);positions.extend(range(len(request),len(request)+take))
        observation=audit.token_ids(turn['observation_token_ids'],'observation');correction=audit.token_ids(turn['correction_token_ids'],'correction')
        if observation and correction: raise ValueError('Both environment suffixes present')
        env=observation+correction;rebuilt.extend(raw[:kept]+env);mask.extend([1]*kept+[0]*len(env));logs.extend(lp[:kept]+[0.]*len(env))
    if (not full['behavior_turns'] or journal!=full['sampling_journal'] or rebuilt!=full['response_token_ids'] or
            mask!=full['response_info_mask'] or logs!=full['response_log_probs']):
        raise ValueError('Actual full response/mask/journal does not reconstruct')
    if (prefix['policy_token_ids']!=retained or prefix['policy_position_ids']!=positions or
            prefix['prefix_segments']!=segments or prefix['retained_policy_length']!=len(retained) or
            len(retained)>20 or (len(retained)<20 and prefix['terminal'] is not True)):
        raise ValueError('Prefix20 causal contexts/positions differ')
    if full['terminal_reason'] not in ('natural_eos','answer','final_action_budget'):
        raise ValueError('Missing actual full terminal')
    if prefix['terminal'] and prefix['terminal_reason']!=full['terminal_reason']:
        raise ValueError('Prefix natural terminal differs from continuation')
    if score(full)!=own_label: raise ValueError('Tokenizer strict EM recomputation differs')
    own=ledger.setdefault(qid,{})
    for seed in seeds:
        if seed in own: raise ValueError('Actual same-question original/new native seed overlap')
        own[seed]=dict(event=event,trajectory_id=tid)
    return dict(seed_chunks=len(seeds),policy_tokens=sum(mask),response_tokens=len(rebuilt),
        multi_segment_prefix=int(len(segments)>1),short_terminal_prefix=int(len(retained)<20))

def read(p):return json.loads(Path(p).read_bytes())
def proof(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(ref):
 actual=proof(ref['path']);require(actual['sha256']==ref['sha256']and actual['bytes']==ref['bytes'],'Artifact SHA/bytes changed');return Path(actual['path'])
def records(p):
 opener=gzip.open if str(p).endswith('.gz')else open
 with opener(p,'rt')as f:
  for line in f:yield json.loads(line)
def run(worker):
 require(sys.platform!='darwin'and not os.environ.get('CUDA_VISIBLE_DEVICES'),'Remote CPU-only tokenizer audit required')
 OUTPUT.mkdir(exist_ok=True);lock=(OUTPUT/('worker-%d.lock'%worker)).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 main=read(BASE/'config.json');require(proof(BASE/'config.json')['sha256']==CONFIG_SHA,'Frozen main config differs')
 job=(BASE/('worker-%d.job.id'%worker)).read_text().strip();seals=[]
 for path in(BASE/'sampling').glob('shard-*/batches/batch*/complete.json'):
  value=read(path)
  if value['execution_job']==job:seals.append((path,value))
 if not seals:return dict(status='NOT_READY_NO_ACTUAL_OWN_SEALED_BATCH',worker=worker,job_id=job,accepted=False)
 path,seal=min(seals,key=lambda item:item[1]['epoch']);done=OUTPUT/('worker-%d-first.json'%worker)
 if done.exists():
  old=read(done);require(old['seal']==proof(path),'Previously audited actual first seal changed');return old
 confpath=path.parent/'config.json';conf=read(confpath);outer=path.parent/'outers/outer-000001';source_input=BASE/'inputs'/path.parent.parent.parent.name/(path.parent.name+'.json');batch=read(source_input)
 require(seal['input_sha256']==proof(source_input)['sha256']and seal['questions']==len(batch['rows'])and seal['trajectories']==5*len(batch['rows']),'Actual wholeG5 input binding differs')
 require(seal['run_id']==seal['sampling_protocol_source_run_id']==main['sampling_protocol_source_run_id']and seal['execution_run_id']==main['execution_run_id']and seal['execution_host_cluster']=='HPC3'and seal['execution_source_commit']==main['execution_source_commit']and seal['potentialOverlap']and seal['donor_original_receipt_claim']is False and seal['actor_optimizer_updates']==0,'Own execution/source provenance differs')
 require(conf['seed']==20261021 and conf['sampling_attempt_id']=='independent-seed20261021-g5-20261020'and conf['actor_optimizer_updates']==0 and conf['actor_hidden_native']is False and seal['model_revision']==REVISION,'Seed/revision/frozen actor contract differs')
 prefix,full=seal['prefix'],seal['full'];scan=outer/'prefix/scan.json';selection=outer/'probe-score/selection.json'
 require(prefix['config_sha256']==full['config_sha256']==proof(confpath)['sha256'],'Rank config binding differs')
 require(full['prefix_scan_sha256']==proof(scan)['sha256']and full['selection_sha256']==proof(selection)['sha256'],'Current prefix/full receipt chain differs')
 require(read(selection)['kind']=='ALL_G5_NO_SELECTION'and read(scan)['ranks']==[prefix],'No-selection/scan own rank contract differs')
 for key in ['sampling_contract','action_boundary','interaction_protocol','actual_compute_dtype']:require(prefix[key]==full[key],'Prefix/full protocol differs')
 require(prefix['actual_compute_dtype']=='torch.bfloat16'and prefix['sampling_attempt_id']==full['sampling_attempt_id']==conf['sampling_attempt_id'],'Actual BF16/attempt differs')
 artifacts={k:dict(v)for k,v in [('states',prefix['states']),('prefix',prefix['prefix']),('full',full['full']),('own_em',full['own_em'])]}
 for value in artifacts.values():checked(value)
 before={x['path']:(Path(x['path']).stat().st_size,Path(x['path']).stat().st_mtime_ns)for x in artifacts.values()}
 byqid={g['qid']:g for g in batch['rows']};expected={('%s-outer000001-c%04d-g%d'%(conf['run_id'],g['candidate_ordinal'],j)):(g,j)for g in batch['rows']for j in range(5)}
 prefix_rows=list(records(artifacts['prefix']['path']));em_rows=list(records(artifacts['own_em']['path']));pref={g['trajectory_id']:g for g in prefix_rows};labels={g['trajectory_id']:g['strict_em']for g in em_rows};require(len(prefix_rows)==len(em_rows)==len(expected)==len(pref)==len(labels)and all(type(v)is int and v in(0,1)for v in labels.values())and set(pref)==set(labels)==set(expected),'Actual ownG5 prefix/EM count or ID set differs')
 model=Path(main['model_dir']);snapshot=read(checked(main['model_snapshot']));require(snapshot['revision']==REVISION,'Official tokenizer revision differs')
 tokenizer_refs={}
 for name,item in snapshot['files'].items():
  if name.endswith(('.json','.txt','.model')):tokenizer_refs[name]=proof(checked(dict(path=str(model/name),bytes=item['bytes'],sha256=item['sha256'])))
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
 from reproduction.prefix20_online.rollout import strict_label
 score=lambda row:strict_label(tokenizer,row['response_token_ids'],row['response_info_mask'],row['reward_model']['ground_truth'])
 ledger={};seen=set();counts=collections.Counter();groups={}
 for row in records(artifacts['full']['path']):
  tid=row['trajectory_id'];require(tid in expected and tid not in seen,'Repeated/foreign actual full ID');seen.add(tid);source,index=expected[tid]
  result=audit_trajectory(pref[tid],row,source,conf,labels[tid],score,ledger,'HPC3_OWN_TAIL')
  counts.update(result);groups.setdefault(source['qid'],[]).append(labels[tid])
 require(seen==set(expected)and len(groups)==len(byqid)and all(len(x)==5 for x in groups.values()),'Whole authentic ownG5 incomplete')
 require(before=={x['path']:(Path(x['path']).stat().st_size,Path(x['path']).stat().st_mtime_ns)for x in artifacts.values()},'Sealed payload changed during independent audit')
 import torch
 require(not torch.cuda.is_initialized(),'Audit must not initialize CUDA')
 value=dict(status='HPC3_OWN_FIRST_BATCH_P20_G5_STRICT_EM_CAUSAL_NATIVE_SEEDS_INDEPENDENTLY_PASSED',worker=worker,job_id=job,seal=proof(path),input=proof(source_input),config=proof(confpath),main_config=proof(BASE/'config.json'),execution_run_id=main['execution_run_id'],sampling_protocol_source_run_id=main['sampling_protocol_source_run_id'],artifacts=artifacts,questions=len(groups),trajectories=len(seen),K5_histogram={str(k):sum(sum(x)==k for x in groups.values())for k in range(6)},strict_EM_recomputed=len(seen),strict_EM_mismatches=0,**dict(counts),tokenizer_files=tokenizer_refs,tokenizer_strict_scorer=proof(inspect.getfile(strict_label)),auditor_source=proof(Path(__file__)),inherited_function_binding=INHERITED_SOURCE_BINDING,source_commit=main['execution_source_commit'],actor_updates=0,Prefix20_causal_positions_verified=True,real_G5_ownIDs_and_seeds_verified=True,potentialOverlap=True,CUDA_initialized=False,model_tensors_loaded=False,whole_tail_accepted=False,epoch=time.time())
 tmp=done.with_name(done.name+'.tmp');tmp.write_bytes(canonical(value)+b'\n');os.replace(str(tmp),str(done));return value
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int,choices=(0,1),required=True);a=p.parse_args();print(json.dumps(run(a.worker),sort_keys=True))
