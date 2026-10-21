"""Independent real CP128 next-batch IO+compute profile; never writes the main.

Only the accepted metadata descriptors and requested320 retained rows are read.
No complete276GB feature-payload SHA pass, training checkpoint save, or sampler
commit is published. Every diagnostic step restores all CP128 components first.
"""
import argparse,copy,gc,hashlib,json,os,re,statistics,time
from collections import OrderedDict
from pathlib import Path
import torch
from . import probe_ablation_train as u,mixed_group_checkpoint as ck

A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
SNAP=A/'mixed-direct-official-cp128-eval-v1'
PIN=dict(path=str(SNAP/'inputs/checkpoint-cp128.pt'),bytes=3777515802,sha256='f9d6860eb01f0f76051319f53d78bd6012e47292268a00456f15549464ef2b39')
PIN_RECEIPT=dict(path=str(SNAP/'pinned-checkpoint.json'),bytes=1675,sha256='a68b54cd5ff3b703f462b0b703c7a18979cf8f9abaf8102c0108fc4cef4a8ab1')


def proof(path):
    p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=u.sha(p))


def reader_shell(cls,records):
    reader=cls.__new__(cls);reader.layer=26;reader.dimension=3584;reader.synthetic_fixture=False;reader.closed=False
    reader.records=dict(records);reader.cache=OrderedDict();reader.prototypes={};reader.manifests=[];reader.files=[]
    return reader


def selected_records(manifests,wanted):
    """Validate index descriptors, parse only requested IDs, stop when found."""
    from .replay_feature_reader import ReplayFeatureReader,FILES
    from .replay_probe_layers import CAPTURE_KIND,REVISION
    requested=set(wanted)
    if len(requested)!=320:raise ValueError('Frozen next batch must contain320 unique own trajectories')
    if any(not t.isascii()or '"'in t or '\\'in t for t in requested):raise ValueError('Use exact ASCII protocol trajectory IDs for the bounded index lookup')
    needle={t.encode():t for t in requested};pattern=re.compile(rb'"trajectory_id"\s*:\s*"([^"\\]*)"');found={};refs=[];scanned=lines=0;started=time.perf_counter()
    validator=reader_shell(ReplayFeatureReader,{})
    for descriptor in manifests:
        path=u.checked(descriptor);manifest=u.read(path);scanned+=1
        if(manifest.get('status')!='LOSSLESS_HF_REPLAY_FEATURES_COMPLETE'or manifest.get('source_runtime')!='huggingface_frozen_causal_prefill'or manifest.get('layout')!='lossless_question_base_xor_policy_v1'or manifest.get('capture_layer')!=26 or manifest.get('feature_layer')!=26 or manifest.get('dimension')!=3584 or manifest.get('question_max_tokens')!=273 or manifest.get('prefix_tokens')!=20 or manifest.get('capture_kind')!=CAPTURE_KIND or manifest.get('model_revision')!=REVISION or manifest.get('actor_optimizer_updates')!=0 or set(manifest.get('files',{}))!=FILES):raise ValueError('Actual qualified layer26 manifest geometry/source differs')
        for name,ref in manifest['files'].items():
            p=Path(ref['path'])
            if p!=path.parent/name:raise ValueError('Packed source locator differs')
        index=manifest['files']['index.jsonl'];p=u.checked(index);raw=p.read_bytes();line_count=raw.count(b'\n')+(bool(raw)and not raw.endswith(b'\n'))
        if line_count!=manifest['count']:raise ValueError('Complete sealed index row count differs')
        selected=0
        for line in raw.splitlines():
            lines+=1;match=pattern.search(line)
            if match is None:raise ValueError('Index row lacks canonical trajectory ID locator')
            tid=needle.get(match.group(1))
            if tid is None:continue
            row=json.loads(line);validator.validate_row(row,manifest)
            if row['trajectory_id']!=tid or tid in found:raise ValueError('Repeated/different selected index identity')
            found[tid]=(str(path.parent),row);selected+=1
        if selected:
            for ref in manifest['files'].values():
                if Path(ref['path']).stat().st_size!=ref['bytes']:raise ValueError('Selected packed source size differs')
        refs.append(dict(manifest=descriptor,index=index,selected_rows=selected))
        if len(found)==len(requested):break
    if set(found)!=requested:raise ValueError('Selected next-batch retained rows not fully found')
    return found,dict(setup_seconds=time.perf_counter()-started,requested_rows=320,found_rows=len(found),manifests_scanned=scanned,index_rows_scanned=lines,manifest_and_index_descriptors_fresh_SHA=True,selected_row_schema_and_span_validated=True,bulk_feature_payload_SHA_repeated=False,each_get_reconstructs_full_retained_row_SHA=True,scanned_metadata_refs=refs)


class TimedReader:
    def __init__(self,reader):self.reader=reader;self.calls=[]
    def get(self,ids,device):
        start=time.perf_counter();value=self.reader.get(ids,device);self.calls.append(dict(rows=len(ids),wall_seconds=time.perf_counter()-start));return value
    def close(self):self.reader.close()


def process_io():
    try:return {line.split(':')[0]:int(line.split(':')[1])for line in Path('/proc/self/io').read_text().splitlines()}
    except OSError:return {}


def cpu_gradients(model):return {name:p.grad.detach().float().cpu().clone()for name,p in model.named_parameters()}


def gradient_difference(model,reference):
    error=norm=maximum=0.
    for name,p in model.named_parameters():
        r=reference[name].to(p.device);d=p.grad.detach().float()-r;error+=float(d.square().sum());norm+=float(r.square().sum());maximum=max(maximum,float(d.abs().max()));del r,d
    return dict(relative_L2=(error/max(norm,1e-30))**.5,max_abs=maximum,BF16_cross_microbatch_bitexact_claim=False)


def run(config,output,device=None):
    if not os.environ.get('SLURM_JOB_ID')or not torch.cuda.is_available():raise ValueError('Independent owned GPU profile allocation required')
    source=Path(config);c=u.read(source);device=torch.device(device or'cuda:0');torch.cuda.set_device(device);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    if 'H100'not in torch.cuda.get_device_name(device):raise ValueError('Actual H100 profile allocation required')
    from .joint_mixed_direct_pretrain import own_groups,warm_gradient_buckets,direct_update
    from .direct_update_fast import direct_update_fast
    from .balanced_hard_g10_sampler import BalancedHardG10Sampler
    from .model import Probe
    from .native_mixed_group_classifier import fresh_official_training_state,block_state_hash
    from .replay_feature_reader import ReplayFeatureReader
    from .replay_feature_reader_fast import FastReplayFeatureReader,ParallelFastReplayFeatureReader
    overall=time.perf_counter();u.checked(PIN_RECEIPT);p=u.checked(PIN);saved=torch.load(p,map_location='cpu',mmap=True,weights_only=False);ck.validate(saved)
    if saved['cursor']!=128 or saved['native_steps']!=128 or saved['head_steps']!=128 or saved['config_sha256']!=u.sha(source)or saved['matched_binding_sha256']!=u.digest(c):raise ValueError('Fixed CP128/source-config binding differs')
    binding={k:saved[k]for k in('config_sha256','matched_binding_sha256','training_groups','features','sampler_source','initialization_source','official_revision','official_inventory')}
    groups,blocked,hashes=own_groups(c);lookup={g['qid']:g for g in groups}
    sampler=BalancedHardG10Sampler(groups,seed=c['seed'],batch_questions=32,steps_per_epoch=c['steps_per_epoch'],forbidden_qids=blocked,forbidden_hashes=hashes)
    legacy=Probe(c['model_dir'],273,20);state=fresh_official_training_state(legacy,official_inventory=c['official_inventory'],expected_block_hash=block_state_hash(legacy.block),head_seed=c['head_seed'])
    model=state['model'].to(device);ema=state['ema'].to(device).requires_grad_(False);block_opt=torch.optim.Adam(model.block.parameters(),lr=c['lr']);head_opt=torch.optim.Adam(model.group_head.parameters(),lr=c['lr']);del state,legacy;gc.collect()
    ck.restore(saved,model,ema,block_opt,head_opt,device,sampler,binding);warm_gradient_buckets(model)
    frozen=sampler.next_batch();sampler.load_state_dict(saved['sampler_state']);qids=frozen['source_qids'];wanted=[t for q in qids for t in lookup[q]['trajectory_ids']]
    records,index_setup=selected_records(u.read(u.checked(c['features'])),wanted)
    if any(records[t][1]['qid']!=q for q in qids for t in lookup[q]['trajectory_ids']):raise ValueError('Selected row qid/ownG10 association differs')
    # Exact decoded bits/masks/positions gate for the actual random next batch,
    # done in ten-row pieces so GPU RAM and comparison stay bounded.
    old=reader_shell(ReplayFeatureReader,records);fast=reader_shell(ParallelFastReplayFeatureReader,records);equal_rows=0
    try:
        for start in range(0,320,10):
            a=old.get(wanted[start:start+10],torch.device('cpu'));b=fast.get(wanted[start:start+10],torch.device('cpu'))
            for key in('features','attention_mask','position_ids'):
                if not torch.equal(a[key],b[key]):raise ValueError('Old/parallel reconstructed feature bits/mask/positions differ')
            equal_rows+=10
    finally:old.close();fast.close()
    setup_seconds=time.perf_counter()-overall;results=[];reference=None;settings=[('old_scalar',ReplayFeatureReader,1),('parallel_fast8',ParallelFastReplayFeatureReader,8)]
    if c.get('fullstep_profile',{}).get('include_fast_micro1',True):settings.append(('parallel_fast1',ParallelFastReplayFeatureReader,1))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    for repeat in range(3):
        for name,reader_type,micro in settings:
            restore_start=time.perf_counter();ck.restore(saved,model,ema,block_opt,head_opt,device,sampler,binding);warm_gradient_buckets(model);torch.cuda.synchronize();restore_seconds=time.perf_counter()-restore_start
            reader=TimedReader(reader_shell(reader_type,records));before_io=process_io();start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True);start_event.record();started=time.perf_counter()
            try:
                if name=='old_scalar':metrics,batch=direct_update(model,ema,block_opt,head_opt,reader,lookup,sampler,device,c['ema_decay'])
                else:metrics,batch=direct_update_fast(model,ema,block_opt,head_opt,reader,lookup,sampler,device,c['ema_decay'],micro_questions=micro)
                end_event.record();end_event.synchronize();wall=time.perf_counter()-started;after_io=process_io()
                if batch!=frozen or sampler.accepted_updates!=129 or ck.optimizer_steps(block_opt.state_dict(),True)!=129 or ck.optimizer_steps(head_opt.state_dict(),True)!=129:raise ValueError('Diagnostic step batch/twoAdam/sampler differs from frozen CP128 next update')
                if reference is None:reference=cpu_gradients(model);difference=dict(reference=True)
                else:difference=gradient_difference(model,reference)
                item=dict(variant=name,repeat=repeat,micro_questions=micro,actual_batch_equal=True,cursor_before=128,diagnostic_cursor_after=129,wall_seconds=wall,cuda_event_stream_seconds=start_event.elapsed_time(end_event)/1000,restore_full_nine_components_seconds_excluded=restore_seconds,reader_get_wall_seconds=sum(x['wall_seconds']for x in reader.calls),reader_get_calls=reader.calls,process_io_delta={k:after_io.get(k,0)-v for k,v in before_io.items()},metrics=metrics,gradient_difference_from_first_old=difference)
                results.append(item)
            finally:reader.close()
            result=dict(status='INDEPENDENT_CP128_SAME_RANDOM_BATCH_FULLSTEP_PROFILE_IN_PROGRESS',source_config=proof(source),pinned_checkpoint=PIN,pin_receipt=PIN_RECEIPT,source_component_hashes=saved['accepted_hashes'],frozen_next_batch_token=frozen['token'],frozen_qids_sha256=u.digest(qids),frozen_own_trajectory_ids_sha256=u.digest(wanted),questions=32,rows=320,old_parallel_exact_decoded_rows=equal_rows,index_setup=index_setup,setup_total_seconds_excluded=setup_seconds,results=results,official_model_initialization_only_no_whole_bank_SHA=True,main_state_or_pinned_checkpoint_written=False,formal_training_update_count=0,external_or_final_outcomes_read=False,job_id=os.environ['SLURM_JOB_ID'],epoch=time.time())
            tmp=output.with_suffix(output.suffix+'.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(output)
    ck.restore(saved,model,ema,block_opt,head_opt,device,sampler,binding)
    result['status']='INDEPENDENT_CP128_SAME_RANDOM_BATCH_FULLSTEP_PROFILE_COMPLETE';result['final_state_restored_to_CP128']=True
    result['summary']={name:dict(repeats=3,median_wall_seconds=statistics.median(x['wall_seconds']for x in results if x['variant']==name),median_reader_get_seconds=statistics.median(x['reader_get_wall_seconds']for x in results if x['variant']==name))for name,_,_ in settings}
    result['epoch']=time.time();output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-config',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=run(a.source_config,a.output);print(json.dumps(dict(status=r['status'],output=a.output,summary=r['summary'])))
