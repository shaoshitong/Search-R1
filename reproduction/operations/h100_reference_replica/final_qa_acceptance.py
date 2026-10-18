"""Once CPU readback of actual final51713 outputs, never generation/admission.

Original compressed shards stay remote. Recompute strict EM/F1 from actual
tokens, canonical golds and the frozen scorer; this does not prove native exit
or replace actor/probe checkpoint and owned-resource acceptance.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time

CANONICAL_SHA='55d515e116a4c6aa63b45282f9bc87202b030156be41be60420c38a9d3ff0ed1'
SOURCES={
    '2wikimultihopqa':12576,'bamboogle':125,'hotpotqa':7405,'musique':2417,
    'nq':3610,'popqa':14267,'triviaqa':11313}
FIELDS=('qid','data_source','evaluation_ordinal','em','f1','answer_present','answer_sha256',
    'actual_retrieval_calls','valid_actions','turns','response_token_count','terminal_reason')


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())
def stat_key(value):return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns)
def file_sha(path):
    path=Path(path);before=stat_key(path.stat());h=hashlib.sha256()
    with path.open('rb')as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    need(before==stat_key(path.stat()),'Immutable input changed during SHA readback')
    return h.hexdigest()


class DigestReader:
    """Hash compressed bytes in the very same pass used for gzip decoding."""
    def __init__(self,stream):self.stream=stream;self.digest=hashlib.sha256();self.bytes=0
    def read(self,size=-1):
        value=self.stream.read(size);self.digest.update(value);self.bytes+=len(value);return value
    def tell(self):return self.stream.tell()
    def seekable(self):return False


def read_shard(producer,worker,questions,rank,prompt_for,score):
    path=Path(producer['path']);before=stat_key(path.stat())
    need(path.is_absolute() and path.is_file() and before[2]==producer['bytes'],'Missing/changed original QA shard')
    expected={i for i in range(rank,len(questions),4)}
    recorded={row['evaluation_ordinal']:row for row in worker['rows']}
    need(len(recorded)==len(worker['rows']) and set(recorded)==expected,'Worker output identities missing/duplicated')
    seen={};
    with path.open('rb')as raw:
        digest=DigestReader(raw)
        with gzip.GzipFile(fileobj=digest,mode='rb')as compressed,io.TextIOWrapper(compressed,encoding='utf8')as stream:
            for line in stream:
                item=json.loads(line);ordinal=item['evaluation_ordinal']
                need(type(ordinal)is int and ordinal in expected and ordinal not in seen,'QA ordinal missing/duplicated/in another rank')
                question=questions[ordinal]
                need(item['qid']==question['qid'] and item['data_source']==question['data_source'],'Native output question identity differs')
                need({key:item[key]for key in FIELDS}==recorded[ordinal],'Worker metric row differs from original full output')
                prompt=item['initial_prompt_token_ids'];response=item['response_token_ids'];mask=item['response_info_mask']
                need(prompt==prompt_for(question),'Actual canonical prompt/template token IDs differ')
                need(len(response)==len(mask)==item['response_token_count'] and all(v in (0,1)for v in mask),'Actual response/mask identity differs')
                need(len(item['behavior_turns'])==item['turns'],'Actual recorded turns differ')
                measured=score(question,prompt,response,mask)
                need(type(item['em'])is int and item['em']in (0,1) and measured['em']==item['em'],'Original strict EM differs from canonical-gold rescore')
                need(math.isfinite(item['f1']) and 0<=item['f1']<=1 and math.isclose(measured['f1'],item['f1'],rel_tol=0,abs_tol=1e-12),'Original F1 differs from canonical-gold rescore')
                need(all(measured[key]==item[key]for key in ('answer_present','answer_sha256')),'Original answer identity differs from rescore')
                seen[ordinal]={key:item[key]for key in ('qid','data_source','em','f1')}
        need(digest.bytes==before[2] and digest.digest.hexdigest()==producer['sha256'],'Actual compressed shard full SHA differs')
    need(seen.keys()==expected and before==stat_key(path.stat()),'Original shard incomplete or changed while reading')
    return seen,dict(path=str(path),bytes=before[2],sha256=digest.digest.hexdigest(),
        dev=before[0],inode=before[1],mtime_ns=before[3],actual_questions=len(seen),
        actual_original_full_sha_and_cpu_rescore=True,compressed_file_read_passes=1)


def aggregates(rows):
    ordered=[rows[i]for i in range(len(rows))]
    def metrics(items):return dict(qa_count=len(items),em=sum(r['em']for r in items)/len(items),f1=sum(r['f1']for r in items)/len(items))
    sources=sorted({row['data_source']for row in ordered})
    return dict(metrics(ordered),per_source={source:metrics([r for r in ordered if r['data_source']==source])for source in sources})


def matching_aggregate(actual,recorded):
    need(actual['qa_count']==recorded['qa_count'],'Aggregate QA count differs')
    for key in ('em','f1'):need(math.isclose(actual[key],recorded[key],rel_tol=0,abs_tol=1e-12),'Aggregate '+key+' differs')
    need(actual['per_source'].keys()==recorded['per_source'].keys(),'Aggregate source set differs')
    for source,values in actual['per_source'].items():
        expected=recorded['per_source'][source]
        need(values['qa_count']==expected['qa_count'],'Per-source QA count differs')
        for key in ('em','f1'):need(math.isclose(values[key],expected[key],rel_tol=0,abs_tol=1e-12),'Per-source '+key+' differs')


def once(path,value):
    with path.open('x')as stream:json.dump(value,stream,sort_keys=True,allow_nan=False);stream.flush();os.fsync(stream.fileno())


def scoring_publication(base,config,paths):
    if config.get('source_compatibility'):
        proof=Path(config['source_compatibility']);need(file_sha(proof)==config['source_compatibility_sha256'],'Runtime compatibility proof changed')
        publication=read(proof)
        need(publication['runtime_source_commit']==config['source_commit'],'Scoring runtime commit differs')
        expected=publication['target_publication']['files_sha256']
    else:
        candidates=[base/'preparation'/name for name in ('assigned-source-full-sha.json','source-full-sha.json')]
        proof=next((path for path in candidates if path.is_file()),None)
        need(proof is not None,'Missing pinned actual runtime source publication')
        publication=read(proof);need(publication['commit']==config['source_commit'],'Scoring source publication commit differs')
        expected={row['path']:row['sha256']for row in publication['files']}
    actual={name:file_sha(path)for name,path in paths.items()}
    need(all(name in expected and value==expected[name]for name,value in actual.items()),'Actual imported scorer/prompt code differs from frozen publication')
    return dict(proof=str(proof),proof_sha256=file_sha(proof),source_commit=config['source_commit'],actual_imported_sources=actual)


def inspect_final(base):
    base=Path(base).resolve();prep=base/'preparation';output=prep/'final51713-original-output-cpu-once.json'
    names=('logs/pipeline-complete.json','latest-joint.json','final-evaluation/complete.json','preparation/final-evaluation-config.json')
    if any(not (base/name).is_file()for name in names):return dict(status='PENDING_FINAL_OUTPUT_NOT_COMMITTED_NO_AUDIT',base=str(base))
    config_path=base/names[3];complete_path=base/names[2];config=read(config_path)
    if output.exists():
        proof=read(output)
        need(proof['run_id']==config['run_id'] and proof['evaluation_config_sha256']==file_sha(config_path)
             and proof['final_receipt_sha256']==file_sha(complete_path),'Existing final QA proof belongs to changed output')
        for file in proof['original_compressed_shards']:
            need(stat_key(Path(file['path']).stat())==(file['dev'],file['inode'],file['bytes'],file['mtime_ns']),
                 'Previously sealed final QA output changed; do not reuse old proof')
        return dict(status='EXISTING_FINAL_QA_PROOF_NO_OUTPUT_RESCAN',proof=str(output),sha256=file_sha(output))
    intent=output.with_suffix('.intent.json')
    need(not intent.exists(),'Existing final QA audit intent requires reconciliation, never replay')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Final QA acceptance is CPU-only')
    os.environ['SEARCH_G1_B_PREFIX_TOKENS']=str(config['prefix_policy_tokens'])
    # The caller places the arm's frozen repo first on PYTHONPATH; these are
    # its actual unchanged generation/scoring helpers, not a second QA recipe.
    from reproduction.prefix_selection200 import eval as native
    from reproduction.prefix_selection200 import rollout as native_rollout
    from reproduction.prefix_selection200 import interaction_protocol
    from reproduction.common8 import calibration_evaluation
    from reproduction.common8 import context_budget
    from reproduction.prefix20_online import rollout as prompt_assembly
    from search_g1 import signals
    from search_r1 import qa_protocol
    from verl.utils.reward_score import qa_em
    from transformers import AutoTokenizer
    config_sha=file_sha(config_path);complete_sha=file_sha(complete_path)
    joint=read(base/'latest-joint.json');terminal=read(base/names[0]);complete=read(complete_path)
    need(joint['outer']==200 and terminal['complete_outers']==200 and terminal['latest_joint']==joint
         and terminal['final_evaluation']==complete,'No complete native200/final-evaluation marker')
    need(complete['run_id']==terminal['run_id']==config['run_id'] and complete['actor_version']==config['actor_version']==joint['actor']['actor_version']
         and complete['actor_model_dir']==joint['actor']['hf_model_dir'],'Final QA is not the actual final actor')
    need(complete['qa_count']==51713 and config['final_questions_sha256']==CANONICAL_SHA
         and complete['question_source_sha256']==CANONICAL_SHA,'Canonical final51713 binding differs')
    need(complete['evaluation_contract']==native.CONTRACT and complete['interaction_protocol']==native.INTERACTION_CONTRACT
         and complete['actual_compute_dtype']==native.ACTUAL_COMPUTE_DTYPE,'Final QA protocol differs from frozen runtime')
    exits=complete['actual_worker_exit_codes'];need(set(exits)==set(map(str,range(4)))and all(type(v)is int and v==0 for v in exits.values()),'Actual four QA worker exit0 records missing')
    publication=scoring_publication(base,config,{
        'reproduction/prefix_selection200/eval.py':native.__file__,
        'reproduction/prefix_selection200/rollout.py':native_rollout.__file__,
        'reproduction/prefix_selection200/interaction_protocol.py':interaction_protocol.__file__,
        'reproduction/common8/calibration_evaluation.py':calibration_evaluation.__file__,
        'reproduction/common8/context_budget.py':context_budget.__file__,
        'reproduction/prefix20_online/rollout.py':prompt_assembly.__file__,
        'search_g1/signals.py':signals.__file__,
        'search_r1/qa_protocol.py':qa_protocol.__file__,
        'verl/utils/reward_score/qa_em.py':qa_em.__file__})
    once(intent,dict(run_id=config['run_id'],actor_version=config['actor_version'],started_unix=time.time(),cpu_only=True,whole_goal_completion_claimed=False))
    need(file_sha(config['final_questions'])==CANONICAL_SHA,'Actual complete canonical input SHA differs')
    questions=native.load_questions(config['final_questions'],51713)
    need(dict(Counter(q['data_source']for q in questions))==SOURCES,'Actual seven source identities/counts differ')
    tokenizer=AutoTokenizer.from_pretrained(joint['actor']['hf_model_dir'],local_files_only=True,trust_remote_code=False)
    prompt_for=lambda q:native_rollout.initial_state(config,0,dict(q,candidate_ordinal=q['evaluation_ordinal']),0,tokenizer)['prompt']
    score=lambda q,p,r,m:native.score_tokens(tokenizer,p,r,m,q['reward_model']['ground_truth']['target'])
    need(len(complete['rank_receipts'])==len(complete['output_shards'])==4,'Four original QA shards required')
    rows={};files=[];receipt_seals=[]
    for rank,reference in enumerate(complete['rank_receipts']):
        info=stat_key(Path(reference['path']).stat());receipt_seals.append((reference['path'],info))
        need(file_sha(reference['path'])==reference['sha256'],'Original worker receipt SHA differs')
        worker=read(reference['path'])
        for key in ('run_id','actor_version','actor_model_dir','question_source_sha256','evaluation_contract','interaction_protocol','actual_compute_dtype'):
            need(worker[key]==complete[key],'Worker/aggregate binding differs: '+key)
        need(worker['rank']==rank and worker['qa_count']==len(range(rank,51713,4))
             and worker['seed']==int(config.get('seed',1729))+rank*104729
             and worker['full_output']==complete['output_shards'][rank],'Original worker rank/count/seed/shard differs')
        part,file=read_shard(worker['full_output'],worker,questions,rank,prompt_for,score)
        need(not rows.keys()&part.keys(),'QA ordinal repeats between native workers')
        rows.update(part);files.append(file)
    unique_qids=len({r['qid']for r in rows.values()});coverage=set(rows)==set(range(51713))
    need(coverage and unique_qids==51713,'Actual final QA identity coverage differs')
    actual_counts=dict(Counter(row['data_source']for row in rows.values()))
    need(actual_counts==SOURCES,'Actual output source counts differ')
    aggregate=aggregates(rows);matching_aggregate(aggregate,complete)
    for file in files:
        need(stat_key(Path(file['path']).stat())==(file['dev'],file['inode'],file['bytes'],file['mtime_ns']),
             'Original final QA output changed before proof commit')
    need(all(stat_key(Path(path).stat())==seal for path,seal in receipt_seals),'Worker receipt changed before proof commit')
    need(file_sha(config_path)==config_sha and file_sha(complete_path)==complete_sha,'Final QA configuration/receipt changed during rescore')
    result=dict(status='FINAL51713_ORIGINAL_OUTPUT_SCORE_READBACK_PASS',run_id=config['run_id'],actor_version=config['actor_version'],
        evaluation_config_sha256=config_sha,final_receipt_sha256=complete_sha,canonical_input_sha256=CANONICAL_SHA,
        actual_source_counts=actual_counts,actual_qa_count=len(rows),actual_unique_qids=unique_qids,actual_full_ordinal_coverage=coverage,
        aggregate=aggregate,original_compressed_shards=files,actual_worker_exit_codes=exits,
        frozen_eval_source_sha256=file_sha(native.__file__),scoring_publication=publication,completed_unix=time.time(),
        model_weights_loaded=False,gpu_forward_performed=False,new_generation_performed=False,
        original_outputs_copied_to_mac=False,native_leader_exit_or_owned_cleanup_verified=False,whole_goal_complete=False)
    once(output,result);return dict(status=result['status'],proof=str(output),sha256=file_sha(output))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args();print(json.dumps(inspect_final(a.base)))
