"""Bind P80 pilot's physical full sources and exact HPC3_first9344 overlap."""
import pathlib,json,hashlib,collections,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
def read(p):
 assert not any(s in str(p)for s in('final4096','confirmation','stability2048'));return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);r=p.read_bytes();return dict(path=str(p),bytes=len(r),sha256=hashlib.sha256(r).hexdigest())
def checked(r):
 assert proof(r['path'])=={k:r[k]for k in('path','bytes','sha256')};return r['path']
href=dict(path=str(A/'hpc3-tail9344-independent-L26-prepared-v2/bank/groups.json'),bytes=13005705,sha256='670593b4a58207b55170089d96be2a243fd076da43c5b748877ddf04377c3a7c');raw=read(checked(href));hs=raw['train']if'train'in raw else raw
if isinstance(hs,dict):hs=list(hs.values())
assert len(hs)==9344 and len({g['qid']for g in hs})==9344;hq={g['qid']for g in hs}
pf=A/'online-packed-matched-L15P80-8192-20261023-v1/fixture/fixture-complete.json';p=read(pf);t=read(checked(p['train_groups']));v=read(checked(p['validation_groups']));b=read(checked(p['fixed_train_G5']));c=read(checked(p['source_fixture']));contract=A/'hpc3-tail9344-old5-qualified-g10-v1/raw-g10/source-resolution-contract.json';cp=proof(contract);assert cp['sha256']=='229710cbfccc53c92be96081535eeeee308cc52b969b07713258212c2b0d86c2'
three=['["hotpotqa", "hotpotqa:train:70604"]','["hotpotqa", "hotpotqa:train:2094"]','["hotpotqa", "hotpotqa:train:81793"]'];pools={}
for name,gs in [('pilot_train8192',list(t.values())),('viewed_dev2048',list(v.values())),('fixed_trainB2048',b)]:
 q={g['qid']for g in gs};o=sorted(q&hq);pools[name]=dict(questions=len(q),HPC3_first9344_overlap=len(o),overlap_examples=o[:8],three_case_presence={x:x in q for x in three},domains=dict(collections.Counter(json.loads(g['qid'])[0]for g in gs)))
paths=collections.Counter()
for r in c['source_full']:
 x=r['path'];paths['original'if'/sampling/'in x and'peer-newseed-h100'not in x and'validation-g10'not in x else('viewed_validation'if'validation-g10'in x else'peer_newseed')]+=1
out=dict(time=time.time(),status='EXACT_9344_QUESTION_INTERSECTION_CHECK_COMPLETE',HPC3_first9344=href,HPC3_source_resolution_contract=cp,pilot_fixture=proof(pf),causal_fixture=p['source_fixture'],selected_physical_full_source_refs_SHA256=hashlib.sha256(json.dumps(c['source_full'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),selected_physical_full_source_count=len(c['source_full']),selected_physical_full_path_families=dict(paths),pools=pools,claim='Question intersection is exact; actual selected source ownEM+seal verification is a separate audit, and protocol trajectory ID alone is not payload identity',Slurm_mutated=False,final4096_read=False)
print(json.dumps(out))
