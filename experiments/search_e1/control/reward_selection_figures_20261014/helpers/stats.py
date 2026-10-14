from pathlib import Path
import json,math,hashlib,datetime,sys,os,re,importlib.util
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def loadmod(n,p):
 s=importlib.util.spec_from_file_location(n,str(p));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def hist(a,edges):return {'edges':list(map(float,edges)),'counts':np.histogram(np.asarray(a).ravel(),bins=edges)[0].tolist()}
def moments(a):
 a=np.asarray(a).ravel();return {'n':int(a.size),'mean':float(a.mean()),'std':float(a.std()),'q':np.quantile(a,[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist()}
def stat(C,R=None):
 C=np.asarray(C,dtype=float);assert C.ndim==2 and C.shape[1]==5 and set(np.unique(C))<={0,1}
 n=len(C);k=C.sum(1).astype(int);st=C.std(1,ddof=1);A=(C-C.mean(1)[:,None])/(st[:,None]+1e-6)
 # Probability a uniformly chosen pair of siblings disagrees = k(5-k)/10.
 z={'groups':n,'trajectories':n*5,'correct_count_hist':np.bincount(k,minlength=6).tolist(),'em':float(C.mean()),'mixed_fraction':float(((k>0)&(k<5)).mean()),'pair_disagreement':float((k*(5-k)/10).mean()),'mean_binary_entropy_bits':float(np.mean([-p*math.log2(p)-(1-p)*math.log2(1-p) if 0<p<1 else 0 for p in k/5])),'C_std':moments(st),'C_std_hist':hist(st,np.linspace(0,.56,29)),'C_advantage':moments(A),'C_A_hist':hist(A,np.linspace(-1.81,1.81,75)),'C_zero_group_fraction':float((st==0).mean()),'C_active_A_hist':hist(A[st>0],np.linspace(-1.81,1.81,75)) if (st>0).any() else None}
 if R is not None:
  R=np.asarray(R,dtype=float);assert R.shape==C.shape and np.isfinite(R).all();rs=R.std(1,ddof=1);AR=(R-R.mean(1)[:,None])/(rs[:,None]+1e-6);active=rs>0;ca=st>0;extra=~ca&active
  sig=np.sign(A)*np.sign(AR);cos=np.sum(A*AR,1)/(np.linalg.norm(A,axis=1)*np.linalg.norm(AR,axis=1)+1e-30)
  z.update(R= moments(R),R_std=moments(rs),R_range=moments(np.ptp(R,axis=1)),R_std_hist=hist(rs,np.linspace(0,1.01,52)),R_hist=hist(R,np.linspace(-.11,1.61,87)),R_advantage=moments(AR),R_A_hist=hist(AR,np.linspace(-1.81,1.81,75)),R_active_A_hist=hist(AR[active],np.linspace(-1.81,1.81,75)),R_zero_group_fraction=float((rs==0).mean()),R_tiny_std_fraction=float(((rs>0)&(rs<.01)).mean()),C_zero_R_active_count=int(extra.sum()),extra_all_correct=int((extra&(k==5)).sum()),extra_all_wrong=int((extra&(k==0)).sum()),R_signal_allcorrect_fraction=float((active&(k==5)).sum()/max(1,active.sum())),paired_advantage_cosine=moments(cos[ca&active]),sign_reversal_count=int((sig<0).sum()),R_A_by_class={name:hist(AR[k==kk],np.linspace(-1.81,1.81,75)) for name,kk in [('all_wrong',0),('all_correct',5)]},group_std_2d={'xedges':np.linspace(0,.56,15).tolist(),'yedges':np.linspace(0,1.01,27).tolist(),'counts':np.histogram2d(st,rs,bins=[np.linspace(0,.56,15),np.linspace(0,1.01,27)])[0].astype(int).tolist()})
 return z

