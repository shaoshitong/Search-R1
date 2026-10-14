from pathlib import Path
import json, subprocess, sys, shlex
D=Path(__file__).resolve().parent
code=r'''
from pathlib import Path
import os,json,hashlib,datetime
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
p=Path('/data/shaoshitong/Search-G1-reward-reconstruction-20261010-1ec36152611a/group-statistics-selected-ff98085b6918-v1/original-group-statistics.jsonl')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
h=sha(p);assert h=='2fadf346927a6ed1ca634b5f691e38049971f4fb48b1c6d40c3b4d5814708f17'
steps={20:[],100:[],140:[]}
for line in p.open():
 v=json.loads(line)
 if v['step'] in steps and sum(v['C_raw'])==5:steps[v['step']].append(v)
def hist(a,lo,hi,bins):
 a=np.asarray(a).ravel();e=np.linspace(lo,hi,bins+1);c=np.histogram(a,e)[0];assert int(c.sum())==a.size
 ix=int(np.argmax(c));return {'edges':e.tolist(),'counts':c.tolist(),'n':int(a.size),'peak_bin':{'index':ix,'lo':float(e[ix]),'hi':float(e[ix+1]),'count':int(c[ix]),'fraction':float(c[ix]/a.size),'density':float(c[ix]/a.size/(e[ix+1]-e[ix]))},'occupied_bins':int((c>0).sum())}
def distribution(a,lo,hi):
 a=np.asarray(a).ravel();values,counts=np.unique(a,return_counts=True);order=np.argsort(-counts,kind='stable')[:8]
 grid=np.unique(np.r_[np.linspace(lo,hi,241),0]);sorted_values=np.sort(a);cdfcounts=np.searchsorted(sorted_values,grid,side='right')
 return {'n':int(a.size),'min':float(a.min()),'max':float(a.max()),'unique_exact_values':int(len(values)),'exact_zero_count':int((a==0).sum()),'largest_exact_atom_count':int(counts.max()),'largest_exact_atom_fraction':float(counts.max()/a.size),'top_exact_atoms':[{'value':float(values[i]),'count':int(counts[i])} for i in order],
 'histograms':{str(n):hist(a,lo,hi,n) for n in [30,40,60]},'ecdf_grid':{'thresholds':grid.tolist(),'counts_le':cdfcounts.tolist(),'n':int(a.size),'method':'exact ECDF queried on fixed 241-threshold grid including zero; no smoothing'},
 'near_zero_counts':{str(t):{'count':int((np.abs(a)<=t).sum()),'fraction':float((np.abs(a)<=t).mean())} for t in [0,1e-6,1e-4,.001,.01,.05,.1]}}
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':str(p),'source_sha256':h,'provenance':'historical frozen-probe soft reward reconstruction; not original bitwise online rewards','condition':'C=[1,1,1,1,1] only; all-wrong groups excluded','group_size':5,'sample_std_correction':1,'advantage_epsilon':1e-6,'steps':[]}
for step,gs in steps.items():
 R=np.asarray([g['R_reconstructed'] for g in gs],dtype=float);sd=R.std(1,ddof=1);delta=R-R.mean(1,keepdims=True);A=delta/(sd[:,None]+1e-6)
 row={'step':step,'groups':len(gs),'trajectories':R.size,'soft_reward':distribution(R,-.11,1.61),'soft_centered_delta':distribution(delta,-1.81,1.81),'soft_advantage':distribution(A,-1.81,1.81),
 'std_segments':{'eq_0':int((sd==0).sum()),'gt0_le1e-6':int(((sd>0)&(sd<=1e-6)).sum()),'gt1e-6_le1e-4':int(((sd>1e-6)&(sd<=1e-4)).sum()),'gt1e-4_le0.01':int(((sd>1e-4)&(sd<=.01)).sum()),'gt0.01':int((sd>.01).sum())},
 'soft_std':{'mean':float(sd.mean()),'median':float(np.median(sd)),'min':float(sd.min()),'max':float(sd.max())},'per_group_normalized_std':{'min':float((sd/(sd+1e-6)).min()),'median':float(np.median(sd/(sd+1e-6))),'max':float((sd/(sd+1e-6)).max())}}
 assert sum(row['std_segments'].values())==len(gs);assert abs(float(A.mean()))<1e-11
 out['steps'].append(row)
print(json.dumps(out,allow_nan=False,separators=(',',':')))
'''
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
py='/data/shaoshitong/Search-G1-a800-fresh200-5e1f65d8e7d44363b02938c8d19852ab/env/bin/python'
route=control.route('baai-a800',py+' -B -c '+shlex.quote(code));route[1:1]=['-o','UpdateHostKeys=no']
p=subprocess.run(route,text=True,capture_output=True,timeout=100);assert p.returncode==0,p.stderr[-1000:]
v=json.loads(p.stdout);out=D/'all-correct-soft-binning-20261015.json';out.write_text(json.dumps(v,separators=(',',':')))
print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'steps':[{'step':r['step'],'groups':r['groups'],'summary':{k:{'unique_exact_values':r[k]['unique_exact_values'],'largest_exact_atom_count':r[k]['largest_exact_atom_count'],'peak_bins':{b:d['peak_bin'] for b,d in r[k]['histograms'].items()}} for k in ['soft_reward','soft_centered_delta','soft_advantage']}} for r in v['steps']]},indent=2))
