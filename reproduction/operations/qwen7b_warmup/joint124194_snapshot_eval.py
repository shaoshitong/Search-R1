"""One immutable accepted main checkpoint, read-only fixed512/2048 inference.

This standalone entry imports the already published qualified main runtime;
it never invokes the main trainer or writes inside its namespace.
"""
import argparse,copy,fcntl,gzip,hashlib,json,math,os,time,traceback
from pathlib import Path

MAIN_SHA='5aaa63c9348f410357be92ce491ba6d18286d618ab86dae6e63f193f2b4367b6'

def class_errors(path):
    matrix=[[0]*3 for _ in range(3)];hist=[0]*6
    with gzip.open(path,'rt') as stream:
        for line in stream:
            r=json.loads(line);p=r['p_j'];ys=r['strict_em'];k=sum(ys)
            q=1-math.prod(p)-math.prod(1-x for x in p)
            if abs(q-r['predicted_mixed'])>3e-6:raise ValueError('Saved probability formula mismatch')
            scores=[math.prod(1-x for x in p),q,math.prod(p)]
            actual=0 if k==0 else 2 if k==5 else 1
            predicted=max(range(3),key=lambda j:scores[j]);matrix[actual][predicted]+=1;hist[k]+=1
    return dict(order=['allwrong','Mixed','allcorrect'],argmax_three_event_confusion=matrix,K_histogram=hist)

def run(config_path,expected_sha):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Only the separately authorized Slurm GPU allocation may load snapshot tensors')
    from reproduction.operations.qwen7b_warmup import probe_ablation_train as u,evaluate_frozen_validation as ev
    from reproduction.operations.qwen7b_warmup.joint_balanced_native_evaluation import accepted_HF_validation
    from reproduction.operations.qwen7b_warmup.replay_feature_reader import ReplayFeatureReader
    from reproduction.operations.qwen7b_warmup.model import Probe
    import torch
    config_path=Path(config_path);assert u.sha(config_path)==expected_sha
    cfg=u.read(config_path);base=Path(cfg['base']);base.mkdir(exist_ok=True,parents=True)
    with (base/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (base/'complete.json').exists():raise ValueError('Already completed; do not score twice')
        assert cfg['role']=='accepted_current_checkpoint_read_only_snapshot' and cfg['source_config']['sha256']==MAIN_SHA
        assert cfg['actor_updates']==cfg['probe_updates']==0 and cfg['final4096_scored'] is False
        root=Path(cfg['qualified_code_root']);pub=u.read(u.checked(cfg['qualified_publication']))
        assert pub['commit']=='3b56e583efe2469ac9108873c31de7144a5201e6'
        for name,ref in pub['files'].items():assert u.sha(root/name)==ref['sha256']
        assert u.sha(__file__)==cfg['worker_sha256']
        c=u.read(u.checked(cfg['source_config']));snap=cfg['snapshot'];cp=snap['checkpoint']
        assert cp['config_sha256']==MAIN_SHA and cp['steps']==14418+cp['cursor'] and cp['full_CPU_six_state_reload_passed']
        assert cp['sampler_full_state_and_cursor_reload_passed'] and cp['data_cursor']['cursor']==cp['cursor']
        assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(0)
        device=torch.device('cuda',0);torch.cuda.set_device(device);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
        os.environ['QWEN7B_QUESTION_MAX_TOKENS']='273';os.environ['SEARCH_G1_B_PREFIX_TOKENS']='20'
        u.atomic(base/'controller.json',dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().split()[21],job_id=os.environ['SLURM_JOB_ID'],snapshot_cursor=cp['cursor'],phase='loading_accepted_snapshot',epoch=time.time()))
        blob=torch.load(u.checked(snap['pinned_checkpoint']),map_location='cpu',mmap=True,weights_only=False)
        assert blob['cursor']==cp['cursor'] and blob['steps']==cp['steps'] and blob['config_sha256']==MAIN_SHA
        assert blob['accepted_hashes']==cp['accepted_hashes'] and len(blob['source_all_rank_states'])==4
        model=Probe(c['model_dir'],273,20).to(device);ema=copy.deepcopy(model).eval().requires_grad_(False)
        opt=torch.optim.Adam(model.parameters(),lr=c['lr']);u.restore_blob(blob,model,ema,opt,device)
        assert u.state_hashes(model,ema,opt,device)==cp['accepted_hashes']
        spec=c['evaluation'];diagnostic=u.read(u.checked(spec['diagnostic_groups']))['diagnostic'];dr=ReplayFeatureReader(u.read(u.checked(spec['diagnostic_features'])),26)
        validation,vr,vproof=accepted_HF_validation(spec);assert len(diagnostic)==512 and len(validation)==2048
        excluded={r['qid']:r['question_hash'] for r in u.read(u.checked(c['exclusions']))['groups']}
        assert all(g['qid'] in excluded and (not g.get('question_hash') or excluded[g['qid']]==g['question_hash']) for g in diagnostic+validation)
        # Reader/model construction may consume RNG: restore the exact source again.
        u.restore_blob(blob,model,ema,opt,device);before=u.state_hashes(model,ema,opt,device)
        assert before==cp['accepted_hashes'];rng=u.rng_state(device);saved_buffers=(u.buffers(model),u.buffers(ema))
        u.atomic(base/'restoration-acceptance.json',dict(snapshot_cursor=cp['cursor'],steps=cp['steps'],six_components=before,checkpoint=cp,pinned_checkpoint=snap['pinned_checkpoint'],source_all_four_rank_states_preserved=True,epoch=time.time()))
        metrics={}
        try:
            for label,probe in [('raw',model),('EMA',ema)]:
                metrics[label]={}
                for name,reader,groups in [('diagnostic512',dr,diagnostic),('fixed_selection2048',vr,validation)]:
                    dest=base/(label+'-'+name+'.jsonl.gz')
                    result=ev.evaluate(probe,reader,groups,device,dest,base/'progress.json',label+'-'+name,micro_groups=2)
                    metrics[label][name]=dict(result,predictions=ev.proof(dest),three_class=class_errors(dest))
        finally:
            u.restore_rng(rng,device);u.restore_buffers(model,saved_buffers[0]);u.restore_buffers(ema,saved_buffers[1]);dr.close();vr.close()
        after=u.state_hashes(model,ema,opt,device);assert before==after,'Read-only inference changed full six state'
        result=dict(status='ACCEPTED_CURRENT_NATIVE_JOINT_SNAPSHOT_RAW_EMA_FIXED512_2048_COMPLETE',snapshot_cursor=cp['cursor'],completed_training_epoch_claim=False,checkpoint=cp,pinned_checkpoint=snap['pinned_checkpoint'],source_config=cfg['source_config'],config=ev.proof(config_path),metrics=metrics,six_components_before=before,six_components_after=after,training_six_components_unchanged=True,actor_updates=0,probe_updates=0,final4096_scored=False,validation_proof=vproof,epoch=time.time())
        u.atomic(base/'complete.json',result)
        print(json.dumps(dict(status=result['status'],snapshot_cursor=cp['cursor'],complete=ev.proof(base/'complete.json'))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha256',required=True);a=p.parse_args()
    try:run(a.config,a.config_sha256)
    except BaseException:
        try:
            cfg=json.loads(Path(a.config).read_text());dest=Path(cfg['base'])/'failure.json';dest.write_text(json.dumps(dict(epoch=time.time(),traceback=traceback.format_exc())))
        except BaseException:pass
        raise
