"""Regression: repeated Adam restore must not mutate the source checkpoint."""
import copy,hashlib,json
import torch

def digest(state):
 h=hashlib.sha256()
 for key,value in state['state'].items():
  for name,tensor in sorted(value.items()):h.update(str(key).encode()+name.encode()+tensor.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def main():
 torch.manual_seed(1729);model=torch.nn.Linear(3,1).double();opt=torch.optim.Adam(model.parameters(),lr=5e-6)
 x=torch.tensor([[1.,2.,3.],[-1.,0.,2.]],dtype=torch.float64);y=torch.tensor([[1.],[0.]],dtype=torch.float64)
 def update():
  opt.zero_grad();torch.nn.functional.binary_cross_entropy_with_logits(model(x),y).backward();opt.step()
 update();saved=dict(raw=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()))
 initial=digest(saved['optimizer']);answers=[]
 for i in range(2):
  model.load_state_dict(saved['raw']);opt.load_state_dict(copy.deepcopy(saved['optimizer']));update()
  assert digest(saved['optimizer'])==initial
  answers.append((copy.deepcopy(model.state_dict()),digest(opt.state_dict())))
 assert answers[0][1]==answers[1][1]
 assert all(torch.equal(answers[0][0][k],answers[1][0][k]) for k in answers[0][0])
 # Demonstrate that unprotected same-device Adam loading aliases the checkpoint.
 model.load_state_dict(saved['raw']);opt.load_state_dict(saved['optimizer']);update()
 assert digest(saved['optimizer'])!=initial
 print(json.dumps(dict(status='ADAM_RESTORE_ALIAS_REGRESSION_AND_PROTECTED_REPLAY_PASSED',torch=torch.__version__,source_state_unchanged_with_copy=True,replay_model_and_optimizer_bit_exact=True,unprotected_alias_reproduced=True)))
if __name__=='__main__':main()
