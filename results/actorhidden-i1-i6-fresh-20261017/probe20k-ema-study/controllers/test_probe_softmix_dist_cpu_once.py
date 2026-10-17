from pathlib import Path
import json,sys,datetime,hashlib
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016')
from reproduction.h1008_qprobe.established_controller_transport import remote
repo=Path('/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016');shared=(repo/'reproduction/prefix_selection200/probe_group_vector_dist.py').read_text();source=(repo/'reproduction/prefix_selection200/probe_softmix_dist.py').read_text()
inner=r'''
import torch,sys,types,json
sys.path.insert(0,'/mnt/data/shaoshitong/search-g1-prefix-selection200-probe-dist-divkl-resplit5120-15360-20261018-0bdddc20840b-eval15360fix/repo')
name='reproduction.prefix_selection200.probe_group_vector_dist';m=types.ModuleType(name);m.__package__='reproduction.prefix_selection200';sys.modules[name]=m;exec(SHARED,m.__dict__)
n=types.ModuleType('reproduction.prefix_selection200.probe_softmix_dist');n.__package__='reproduction.prefix_selection200';exec(SOURCE,n.__dict__)
c=((torch.arange(32).unsqueeze(1)>>torch.arange(5).unsqueeze(0))&1).float();k=c.sum(-1);t=n.soft_mixed_target(c)
expected=torch.tensor([0,.9,1,1,.9,0])[k.long()];assert torch.equal(t,expected)
torch.manual_seed(42);z=torch.randn((32,5),requires_grad=True);p=z.sigmoid();q=1-p.prod(-1)-(1-p).prod(-1);h=n.mixed_event_logit(z).sigmoid();assert torch.allclose(h,q,atol=1e-6)
v=n.softmix_loss(z,c);manual=torch.xlogy(t,t/q)+torch.xlogy(1-t,(1-t)/(1-q));assert abs(float(v['soft_mixed_kl'])-float(manual.mean()))<2e-6
assert abs(float(v['total_loss']-v['dist_loss']-50*v['soft_mixed_kl']))<2e-6
v['loss'].backward();assert torch.isfinite(z.grad).all()
for sign in (-1,1):
 zz=torch.full((32,5),float(sign*1000),requires_grad=True);vv=n.softmix_loss(zz,c);vv['loss'].backward();assert torch.isfinite(vv['loss'])and torch.isfinite(zz.grad).all()
from reproduction.prefix_selection200.actor_hidden import WIDTH,DIM
features=torch.cat([torch.ones((5,WIDTH,DIM)),torch.zeros((5,WIDTH,DIM))]).to(torch.bfloat16);labels=torch.tensor([[1.,0,0,0,0],[1.,1,0,0,0]])
b,r,gt,aug=n.mix_batch(dict(features=features),labels,[(0,1)],2026,.2);lam=aug['lambda_sum'];assert lam>.5
assert abs(float(gt[0])-(lam*.9+1-lam))<1e-6 and abs(float(gt[1])-(lam+(1-lam)*.9))<1e-6
assert abs(float(b['features'][0,0,0])-lam)<.005
print(json.dumps(dict(status='SAME_H_SOFT_MIXED_KL_CPU_FIXTURE_PASS',target_by_K=[0,.9,1,1,.9,0],h_product_and_logdomain_equal=True,KL_matches_manual=True,total_DIST_plus50KL=True,extreme_loss_and_gradients_finite=True,dominant_lambda=lam,same_lambda_hidden_vector_group_target=True,individual_BCE=False,mean_KL=False,D_intensity_KL=False,extra_GPU_test=False)))
'''.replace('SHARED',repr(shared)).replace('SOURCE',repr(source))
script='import os,subprocess\nr=subprocess.run(["/dataset/shaoshitong/Search-G1-common8/env/bin/python","-c",'+repr(inner)+'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),text=True,capture_output=True,timeout=90);assert r.returncode==0,r.stderr;print(r.stdout)'
p=Path('work')/('probe-softmix50-lr1e5-cpu-fixture-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');v=json.loads(remote(script,str(p.with_suffix('.transport.json'))));p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(dict(actual=str(p.resolve()),result=v)))
