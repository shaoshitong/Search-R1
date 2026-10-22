import pathlib,json,gzip,statistics,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-M3to10-P20-variance-8192-20261022-v1')
assert json.loads((b/'complete.json').read_text())['cursor']==192
out={'time':time.time(),'final4096_read':False,'all_rows_from_viewed_pilot_only':True,'metrics':{}}
train=[]
with gzip.open(b/'epoch-00-fixed_train2048_G5-predictions.jsonl.gz','rt')as f:train=[json.loads(z)for z in f]
trainmean=statistics.mean(r['target_variance']for r in train)
out['constant_prediction_from_fixed_training2048_G5_target_mean']=trainmean
for epoch in range(4):
 for role in ['fixed_train2048_G5','fixed_validation2048_G5']:
  with gzip.open(b/('epoch-%02d-%s-predictions.jsonl.gz'%(epoch,role)),'rt')as f:rows=[json.loads(z)for z in f]
  ys=[r['target_variance']for r in rows];mean=statistics.mean(ys)
  cell={'questions':len(rows),'true_variance_mean':mean,'oracle_constant_MSE':statistics.mean((y-mean)**2 for y in ys),'training_mean_constant_MSE':statistics.mean((y-trainmean)**2 for y in ys),'states':{}}
  for state in ['raw','EMA']:
   ps=[r[state+'_predicted_variance']for r in rows]
   cell['states'][state]={'prediction_mean':statistics.mean(ps),'prediction_stdev':statistics.pstdev(ps),'MSE':statistics.mean((p-y)**2 for p,y in zip(ps,ys)),'MAE':statistics.mean(abs(p-y)for p,y in zip(ps,ys))}
  out['metrics']['epoch%d_%s'%(epoch,role)]=cell
print(json.dumps(out))
