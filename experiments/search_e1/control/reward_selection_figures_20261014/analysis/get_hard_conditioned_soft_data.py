from pathlib import Path
import json, subprocess, sys, shlex
D = Path(__file__).resolve().parent
code = r'''
from pathlib import Path
import os,json,hashlib,datetime
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
p=Path('/data/shaoshitong/Search-G1-reward-reconstruction-20261010-1ec36152611a/group-statistics-selected-ff98085b6918-v1/original-group-statistics.jsonl')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
h=sha(p);assert h=='2fadf346927a6ed1ca634b5f691e38049971f4fb48b1c6d40c3b4d5814708f17',h
RE=np.linspace(-.11,1.61,87);AE=np.linspace(-1.81,1.81,75);SE=np.linspace(0,1.01,102)
def hist(a,edges):
 a=np.asarray(a).ravel();counts=np.histogram(a,bins=edges)[0]
 assert int(counts.sum())==a.size,('hist out of range',float(a.min()) if a.size else None,float(a.max()) if a.size else None)
 return {'edges':edges.tolist(),'counts':counts.tolist(),'n':int(a.size)}
def moments(a):
 a=np.asarray(a).ravel()
 if not a.size:return {'n':0,'mean':None,'median':None,'min':None,'max':None,'quantile_levels':[0,.01,.1,.25,.5,.75,.9,.99,1],'quantiles':[]}
 return {'n':int(a.size),'mean':float(a.mean()),'median':float(np.median(a)),'min':float(a.min()),'max':float(a.max()),'quantile_levels':[0,.01,.1,.25,.5,.75,.9,.99,1],'quantiles':np.quantile(a,[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist()}
def summarize(C,R):
 n=len(C);delta=R-R.mean(axis=1,keepdims=True);sd=R.std(axis=1,ddof=1);A=delta/(sd[:,None]+1e-6);active=sd>0
 return {'groups':n,'trajectories':n*5,'hard_em':float(C.mean()) if n else None,
 'soft_reward':moments(R),'soft_reward_hist':hist(R,RE),'soft_centered_delta':moments(delta),'soft_centered_delta_hist':hist(delta,AE),'soft_advantage':moments(A),'soft_advantage_hist':hist(A,AE),
 'soft_group_std':moments(sd),'soft_group_std_hist':hist(sd,SE),'soft_nonzero_group_std':moments(sd[active]),
 'soft_group_std_segments':{'eq_0':int((sd==0).sum()),'gt_0_le_1e-6':int(((sd>0)&(sd<=1e-6)).sum()),'gt_1e-6_le_1e-4':int(((sd>1e-6)&(sd<=1e-4)).sum()),'gt_1e-4_le_0.01':int(((sd>1e-4)&(sd<=.01)).sum()),'gt_0.01':int((sd>.01).sum())},
 'absolute_soft_advantage_thresholds':{str(t):{'count':int((np.abs(A)>t).sum()),'fraction':float((np.abs(A)>t).mean()) if n else None} for t in [.1,.5,1]},
 'inverse_std_plus_eps':moments(1/(sd+1e-6)), 'inverse_std_plus_eps_nonzero_std':moments(1/(sd[active]+1e-6)),
 'active_soft_groups':int(active.sum()),'active_soft_trajectories':int(active.sum())*5,
 'active_soft_advantage_hist':hist(A[active],AE),
 'small_raw_delta_nontrivial_advantage':{'definition':'abs(delta)<0.0001 and abs(A)>0.1','count':int(((np.abs(delta)<1e-4)&(np.abs(A)>.1)).sum()),'fraction_all_trajectories':float(((np.abs(delta)<1e-4)&(np.abs(A)>.1)).mean()) if n else None},
 }
grouped={}
for line in p.open():
 g=json.loads(line)
 if g['step'] in [20,100,140]:grouped.setdefault(g['step'],[]).append(g)
out={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':str(p),'sha256':h,'provenance':'same immutable five-response groups; strict binary EM C and historical frozen-probe reconstructed soft reward R; not original bitwise online R','group_size':5,'std_correction':1,'epsilon':1e-6,'formula':'A_R=(R-mean_group(R))/(sample_std_group(R)+1e-6)','weighting':'histograms use equal trajectory weights; group std uses equal group weights; condition is fixed by Hard Reward C only','steps':[],'examples':[]}
for step,gs in sorted(grouped.items()):
 C=np.asarray([g['C_raw'] for g in gs],dtype=float);R=np.asarray([g['R_reconstructed'] for g in gs],dtype=float)
 assert C.shape==R.shape==(512,5);assert set(np.unique(C))<={0.,1.};assert np.isfinite(R).all()
 cs=C.std(1,ddof=1);k=C.sum(1).astype(int);assert np.array_equal(cs==0,(k==0)|(k==5))
 masks={'hard_std_zero':cs==0,'hard_std_positive':cs>0,'all_correct':k==5,'all_wrong':k==0}
 row={'step':step,'conditions':{name:summarize(C[m],R[m]) for name,m in masks.items()}}
 zz=row['conditions']['hard_std_zero'];pp=row['conditions']['hard_std_positive'];ac=row['conditions']['all_correct'];aw=row['conditions']['all_wrong']
 assert zz['groups']+pp['groups']==512
 assert zz['trajectories']+pp['trajectories']==2560
 assert ac['groups']+aw['groups']==zz['groups']
 for v in row['conditions'].values():assert sum(v['soft_group_std_segments'].values())==v['groups']
 row['validation']={'groups_partition_sum':512,'trajectories_partition_sum':2560,'hard_std_zero_eq_all_correct_plus_all_wrong':True,'all_histogram_bins_cover_all_values':True}
 out['steps'].append(row)
 if step==100:
  sd=R.std(1,ddof=1);eligible=(k==5)&(sd>1e-5)&(sd<=1e-4)
  ix=int(np.flatnonzero(eligible)[np.argmin(sd[eligible])]);delta=R[ix]-R[ix].mean();A=delta/(sd[ix]+1e-6)
  out['examples'].append({'step':step,'anonymous_group_hash':gs[ix]['upstream_group_sha256'],'C':C[ix].tolist(),'hard_std':float(cs[ix]),'R_reconstructed':R[ix].tolist(),'R_mean':float(R[ix].mean()),'R_sample_std':float(sd[ix]),'centered_delta':delta.tolist(),'A_R_recomputed':A.tolist(),'inverse_std_plus_eps':float(1/(sd[ix]+1e-6)),'interpretation':'hard correctness gives zero group advantage; tiny reconstructed soft differences produce nonzero normalized advantages; this is not proof of an erroneous gradient'})
assert set(grouped)=={20,100,140}
print(json.dumps(out,allow_nan=False,separators=(',',':')))
'''
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control')
import control
py='/data/shaoshitong/Search-G1-a800-fresh200-5e1f65d8e7d44363b02938c8d19852ab/env/bin/python'
route=control.route('baai-a800',py+' -B -c '+shlex.quote(code));route[1:1]=['-o','UpdateHostKeys=no']
p=subprocess.run(route,text=True,capture_output=True,timeout=100)
assert p.returncode==0,p.stderr[-2000:]
v=json.loads(p.stdout)
out=D/'hard-conditioned-soft-data-20261015.json';out.write_text(json.dumps(v,ensure_ascii=False,separators=(',',':')))
print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'sha256':v['sha256'],'steps':[{'step':x['step'],'groups':{k:z['groups'] for k,z in x['conditions'].items()},'validation':x['validation']} for x in v['steps']],'examples':v['examples']},ensure_ascii=False,indent=2))
