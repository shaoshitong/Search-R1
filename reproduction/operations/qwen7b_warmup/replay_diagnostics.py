"""CPU snapshots for diagnosing exact GPU replay differences, never weakening gates."""
import torch

def snapshot(model,ema,optimizer):
 def own(v):
  if torch.is_tensor(v):return v.detach().to(device='cpu',copy=True)
  if isinstance(v,dict):return {k:own(x) for k,x in v.items()}
  if isinstance(v,list):return [own(x) for x in v]
  if isinstance(v,tuple):return tuple(own(x) for x in v)
  return v
 return own(dict(raw=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict()))

def compare(reference,current):
 out=[];tensor_count=0;different=0
 def walk(a,b,path):
  nonlocal tensor_count,different
  if torch.is_tensor(a):
   assert a.shape==b.shape and a.dtype==b.dtype,(path,a.shape,b.shape)
   tensor_count+=1
   if not torch.equal(a,b):
    different+=1;delta=(a.double()-b.double());out.append(dict(path=path,shape=list(a.shape),changed_elements=int((a!=b).sum()),numel=a.numel(),max_absolute_difference=float(delta.abs().max()),rms_difference=float(delta.square().mean().sqrt()),reference_rms=float(a.double().square().mean().sqrt())))
  elif isinstance(a,dict):
   assert a.keys()==b.keys(),path
   for k in a:walk(a[k],b[k],path+'/'+str(k))
  elif isinstance(a,(list,tuple)):
   assert len(a)==len(b),path
   for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'/'+str(i))
  else:assert a==b,(path,a,b)
 walk(reference,current,'state')
 return dict(tensors=tensor_count,different_tensors=different,bit_exact=different==0,largest_differences=sorted(out,key=lambda x:x['max_absolute_difference'],reverse=True)[:30],all_differences=out)
