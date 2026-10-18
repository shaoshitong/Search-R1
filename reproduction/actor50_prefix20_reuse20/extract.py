"""Replay only stored causal tokens through frozen actor0050 layer27."""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .data import QMAX,WIDTH,DIM,read,sha,publish


class Captured(Exception):pass


def run(base):
    base=Path(base);c=read(base/'preparation/config.json');rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    from transformers import AutoModelForCausalLM
    actor=AutoModelForCausalLM.from_pretrained(c['actor_model_dir'],local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2',low_cpu_mem_usage=False)
    actor.to(device).eval().requires_grad_(False)
    assert len(actor.model.layers)==36 and actor.config.hidden_size==DIM
    captured=[]
    def capture(_module,_args,output):
        value=output[0] if isinstance(output,tuple) else output
        assert value.ndim==3 and value.shape[-1]==DIM and torch.isfinite(value).all()
        captured.append(value.detach().to(device='cpu',dtype=torch.bfloat16))
        raise Captured()
    handle=actor.model.layers[27].register_forward_hook(capture)
    try:
        for number in range(11):
            name='test' if number==0 else 'cycle-%04d'%number;target=base/'data'/name
            meta=read(target/'prepared.json');assert meta['actor_version']==c['actor_version']
            reqpath=target/('rank-%d-requests.jsonl'%rank)
            assert sha(reqpath)==meta['request_sha256'][str(rank)]
            requests=[json.loads(line) for line in reqpath.open()]
            rows={r['feature_row']:r for r in map(json.loads,(target/'index.jsonl').open()) if r['feature_rank']==rank}
            source=np.load(meta['source_native_feature_shards'][rank]['path'],mmap_mode='r',allow_pickle=False)
            assert source.shape==(640,278,DIM) and source.dtype==np.uint16
            output_path=target/('rank-%d.npy'%rank)
            if output_path.exists():raise FileExistsError('Do not overwrite extracted native inputs')
            output=np.lib.format.open_memmap(output_path,mode='w+',dtype=np.uint16,shape=(640,WIDTH,DIM))
            output[:]=0;output[:,:278]=source[:]
            # Replayed old slots are deliberately discarded: retain exact Prefix5 control inputs.
            filled=np.zeros((640,20),dtype=np.bool_)
            for address,row in rows.items():filled[address,:min(5,row['policy_length'])]=True
            work={}
            for request in requests:
                for segment in request['segments']:
                    start=segment['policy_start'];take=segment['policy_length'];skip=max(0,5-start)
                    if take<=skip:continue
                    key=tuple(segment['input_ids'])
                    assert len(key)==segment['request_length']+take and len(key)<=4096
                    work.setdefault(key,[]).append((request['feature_row'],start,take,skip,segment['request_length']))
            tasks=sorted(work,key=len);completed=0;micro=int(c.get('feature_micro_batch',8))
            while completed<len(tasks):
                keys=tasks[completed:completed+micro];width=max(map(len,keys))
                tokens=torch.zeros((len(keys),width),dtype=torch.long,device=device)
                mask=torch.zeros_like(tokens);positions=torch.zeros_like(tokens)
                for i,key in enumerate(keys):
                    tokens[i,:len(key)]=torch.tensor(key,device=device);mask[i,:len(key)]=1
                    positions[i,:len(key)]=torch.arange(len(key),device=device)
                captured.clear()
                try:
                    with torch.no_grad():actor.model(input_ids=tokens,attention_mask=mask,position_ids=positions,use_cache=False)
                except Captured:pass
                assert len(captured)==1
                hidden=captured.pop()
                for i,key in enumerate(keys):
                    for address,start,take,skip,req in work[key]:
                        raw=hidden[i,req+skip:req+take].contiguous().view(torch.uint16).numpy()
                        output[address,QMAX+start+skip:QMAX+start+take]=raw
                        filled[address,start+skip:start+take]=True
                completed+=len(keys)
                publish(target/('rank-%d-progress.json'%rank),dict(dataset=name,rank=rank,
                    replayed_unique_segments=completed,total_unique_segments=len(tasks),rows=640,
                    new_rollout_tokens=0,actor_updates=0,checked_unix=time.time()))
                del tokens,mask,positions,hidden
            for address,row in rows.items():
                assert filled[address,:row['policy_length']].all() and not filled[address,row['policy_length']:].any()
            assert np.array_equal(output[:,:278],source[:])
            output.flush();del output,source
            receipt=dict(dataset=name,rank=rank,path=str(output_path),bytes=output_path.stat().st_size,
                sha256=sha(output_path),rows=640,width=WIDTH,dim=DIM,source_actor=c['actor_version'],
                source_index_sha256=meta['index_sha256'],preserved_question_and_first5_bitwise=True,
                replayed_actor_layer=27,boundary='post_block_hidden',new_rollouts=0,new_sampled_tokens=0,
                gold_or_final_outcomes_passed_to_actor=False,checked_unix=time.time())
            publish(target/('rank-%d-complete.json'%rank),receipt)
    finally:handle.remove()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);run(p.parse_args().base)
