"""CPU audit only fixed missing262 seals from the accepted donor971 observation.

Uses the byte-identical peer auditor in a separate own proof directory. Does
not write the producer run, discover new members, export payloads, or use GPU.
"""
import argparse,fcntl,hashlib,importlib.util,json,os,resource,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
from . import peer_newseed_delta as d


def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f:f.write(d.canonical(v)+b'\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)


def run(reference_path,reference_sha,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    if str(root).startswith(str(d.BASE)+'/') or not root.is_absolute() or root.is_symlink():raise ValueError('Separate own real proof directory required')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(1560,1560))
    ref,rref=d.load(Path(reference_path))
    if rref['sha256']!=reference_sha or d.digest(ref['delta_batches_metadata'])!=ref['delta_records_sha256'] or ref['delta_batches']!=505 or ref['delta_questions']!=64546:raise ValueError('Exact parent505 reference required')
    selected=[r for r in ref['delta_batches_metadata'] if r['existing_audit']['status']!='VERIFIED']
    if len(selected)!=262 or sum(r['questions']for r in selected)!=33536:raise ValueError('Exact missing262 finite partition required')
    config,cref=d.load(d.BASE/'config.json');d.contract(config)
    if cref['sha256']!=ref['source_config']['sha256']:raise ValueError('Source config changed')
    source=Path(__file__).with_name('peer_newseed_audit_reference.py')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=d.AUDITOR:raise ValueError('Pinned CPU auditor differs')
    spec=importlib.util.spec_from_file_location('pinned_donor_auditor',source);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    with (root/'controller.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'complete.json').exists():
            complete=json.loads((root/'complete.json').read_bytes())
            if complete['source_reference']!=rref:raise ValueError('Existing proof belongs to another parent')
            return complete
        owner=dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split(') ')[1].split()[19]),source_reference=rref,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_auditor_sha256=d.AUDITOR,epoch=time.time())
        atomic(root/'controller.json',owner);atomic(root/'intent.json',dict(owner,status='STARTED_FIXED262_CPU_PROOF_ONLY'))
        auditor=a.Audit(SimpleNamespace(base=str(d.BASE),audit_dir=str(root/'audit'),path_map=[],max_new_batches=0))
        auditor.root_config=config;auditor.root_config_hash=cref['sha256'];questions=0;proofs=[];started=time.time()
        try:
            for index,item in enumerate(selected):
                inp=d.regular(d.BASE,item['input']);batch,iref=d.load(inp);seal,sref=d.load(d.regular(d.BASE,item['seal']))
                if iref['sha256']!=item['input_sha256'] or sref['sha256']!=item['seal_sha256']:raise ValueError('Fixed sealed input/seal changed')
                proof=auditor.batch(inp,batch,iref['sha256'])
                if proof is None or proof.get('deferred') or proof['questions']!=item['questions'] or proof['labels']!=5*item['questions']:raise ValueError('Batch proof incomplete')
                _,after=d.load(d.regular(d.BASE,item['seal']))
                if after!=sref:raise ValueError('Producer seal changed across proof')
                questions+=proof['questions'];pp=root/'audit/batches'/inp.parent.name/(inp.stem+'.json');_,pref=d.load(pp);proofs.append(pref)
                atomic(root/'progress.json',dict(status='FIXED262_FRESH_SHA_CAUSAL_OWN_ID_PROOF_ADVANCING',checked_batches=index+1,checked_questions=questions,checked_trajectories=5*questions,expected_batches=262,expected_questions=33536,source_reference=rref,source_auditor_sha256=d.AUDITOR,bulk_payload_SHA_recomputed=True,strict_EM_tokenizer_recomputed=False,producer_modified=False,elapsed_seconds=time.time()-started,epoch=time.time(),pid=os.getpid()))
            result=dict(status='FIXED262_SEALED_FRESH_PAYLOAD_SHA_CAUSAL_OWN_ID_PROOFS_COMPLETE',source_reference=rref,proofs=proofs,batches=len(proofs),questions=questions,trajectories=questions*5,source_auditor_sha256=d.AUDITOR,proof_root=str(root/'audit'),all_current_source_fstats_held=True,bulk_payload_SHA_recomputed=True,strict_EM_tokenizer_recomputed=False,producer_modified=False,payload_exported=False,GPU_used=False,elapsed_seconds=time.time()-started,epoch=time.time())
            atomic(root/'complete.json',result);return result
        except BaseException as e:
            atomic(root/'failure.json',dict(error=type(e).__name__,message=str(e),checked_batches=len(proofs),checked_questions=questions,source_reference=rref,epoch=time.time()))
            raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--reference-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    if sys.platform=='darwin':p.error('Remote CPU source storage only')
    r=run(a.reference,a.reference_sha256,a.output);print(json.dumps({k:v for k,v in r.items()if k!='proofs'}),flush=True)
if __name__=='__main__':main()
