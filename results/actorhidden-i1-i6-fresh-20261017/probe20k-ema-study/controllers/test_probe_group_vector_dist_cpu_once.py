from pathlib import Path
import sys,json,datetime
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016')
from reproduction.h1008_qprobe.established_controller_transport import remote
source=Path('/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/prefix_selection200/probe_group_vector_dist.py').read_text()
fixture=r'''
import sys,json,torch,math
sys.path.insert(0,'/mnt/data/shaoshitong/search-g1-prefix-selection200-probe20k-fit-20261017-0bdddc20840b-r48-q5120-g5-mixup-lr5e5/repo')
ns=dict(__name__='group_vector_dist_cpu_fixture',__package__='reproduction.prefix_selection200');exec(SOURCE,ns)
labels=((torch.arange(32).unsqueeze(1)>>torch.arange(5).unsqueeze(0))&1).float()
d=ns['diversity_target'](labels);k=labels.sum(-1);expected=4*(k/5)*(1-k/5)
assert torch.allclose(d,expected,atol=1e-6)
realz=(labels*2-1)*18;am,ad=ns['group_logits'](realz)
assert torch.allclose(torch.sigmoid(am),k/5,atol=1e-6)
assert torch.allclose(torch.sigmoid(ad),d,atol=1e-6)
r=torch.tensor([[.9,.1,.8,.2,.65],[.9,.9,.9,.9,.9],[.1,.1,.1,.1,.1]])
z=torch.logit(r).requires_grad_();loss=ns['vector_dist_loss'](z,r)
assert abs(float(loss['mean_correctness_kl']))<1e-6 and abs(float(loss['diversity_kl']))<1e-6
loss['loss'].backward();assert torch.isfinite(z.grad).all() and z.grad.abs().max()<1e-4
for value in (-1000.,1000.):
 zz=torch.full((2,5),value,requires_grad=True);v=ns['vector_dist_loss'](zz,torch.tensor([[0.,0,0,0,0],[1.,1,1,1,1]]));v['loss'].backward()
 assert torch.isfinite(v['loss'])and torch.isfinite(zz.grad).all()
from reproduction.prefix_selection200.actor_hidden import WIDTH,DIM
batch=dict(features=torch.cat([torch.ones((5,WIDTH,DIM)),torch.zeros((5,WIDTH,DIM))]).to(torch.bfloat16))
x,soft,aug=ns['dominant_hidden_mixup'](batch,torch.tensor([[1.,1,1,1,1],[0.,0,0,0,0]]),[(0,1)],2026,.2)
lam=float(soft[0,0]);assert lam>.5 and torch.all(soft[0]==soft[0,0])and torch.all(soft[1]==soft[1,0])
assert abs(float(ns['diversity_target'](soft)[0])-3.2*lam*(1-lam))<1e-6
assert abs(float(x['features'][0,0,0])-lam)<.004
print(json.dumps(dict(status='CPU_ONLY_GROUP_VECTOR_OBJECTIVE_FIXTURE_PASS',all32_hard_targets=True,hard_mean_and_diversity_mapping=True,
 soft_teacher_common_optimum=True,constant_group_correlation_omission=True,endpoint_KL_and_gradients_finite=True,
 primary_hidden_coefficient=lam,soft_mean_target=lam,soft_diversity_target=float(ns['diversity_target'](soft)[0]),
 individual_BCE_used=False,extra_GPU_test=False)))
'''.replace('SOURCE',repr(source))
script='import os,subprocess\nr=subprocess.run(["/dataset/shaoshitong/Search-G1-common8/env/bin/python","-c",'+repr(fixture)+'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),text=True,capture_output=True,timeout=90);assert r.returncode==0,r.stderr;print(r.stdout)'
p=Path('work')/('probe-group-vector-dist-cpu-fixture-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');v=json.loads(remote(script,str(p.with_suffix('.transport.json'))));p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(dict(actual=str(p.resolve()),result=v)))
