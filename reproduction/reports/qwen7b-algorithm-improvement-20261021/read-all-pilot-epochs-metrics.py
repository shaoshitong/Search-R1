import pathlib,json,time,subprocess
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
x={'time':time.time(),'pilots':{},'uniform':{},'final4096_read':False}
for layer,prefix in [(15,20),(18,20),(21,20),(24,20),(27,20),(21,80),(27,80)]:
 for ep in range(4):
  b=A/'online-mixed-layer-prefix-20261022-2gpu-v1'/('layer%d-prefix%d'%(layer,prefix));e=json.loads((b/('evaluation-epoch-%02d.json'%ep)).read_bytes());cell={}
  for pool,states in e['metrics'].items():
   cell[pool]={st:{'BA':m['balanced_accuracy'],'Top512_Mixed':m['selected512']['Mixed'],'BCE':m['binary_Mixed_BCE']}for st,m in states.items()}
  x['pilots']['L%dP%d_E%d'%(layer,prefix,ep)]={'epoch':e['epoch'],'cursor':e['cursor'],'metrics':cell,'actual_complete':(b/'complete.json').is_file()}
b=A/'mixed-direct-e1-uniform-g5-weighted-20261022-v1'
for epoch in [1,2,3]:
 for folder,label in [('evaluations','independent'),('fixed-training-evaluations','training')]:
  p=b/folder/('epoch-%02d'%epoch)/'complete.json'
  if p.exists():
   e=json.loads(p.read_bytes());cell={}
   for st,sets in e['metrics'].items():
    if'pool'in sets:sets={'fixed_train2048':sets}
    cell[st]={pool:{'BA':m['pool']['mixed_balanced_accuracy'],'Top512_Mixed':m['selected512']['mixed_questions'],'BCE':m['binaryMixedBCE']}for pool,m in sets.items()}
   x['uniform'][label+str(epoch)]={'cursor':e['cursor'],'metrics':cell}
p=b/'progress.json';x['uniform']['progress']=json.loads(p.read_bytes())['cursor'];x['uniform']['complete']=(b/'complete.json').is_file()
print(json.dumps(x))
