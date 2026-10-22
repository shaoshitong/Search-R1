import pathlib,json
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-natural-g5-8192-20261022-v1');x=json.loads((b/'natural-vs-conditional-matched-metrics.json').read_text());out=[]
for layer,epochs in x['metrics'].items():
 for epoch,pools in epochs.items():
  for pool,value in pools.items():
   out.append(dict(layer=layer,epoch=int(epoch),pool=pool,models={label:{scheme:{k:row[k]for k in('binary_Mixed_BCE','balanced_accuracy','Mixed_recall','nonMixed_false_positive_rate','selected512','confusion','classwise')}for scheme,row in data.items()if scheme!='delta'}for label,data in value['metrics'].items()},deltas={label:data['delta']for label,data in value['metrics'].items()}))
print(json.dumps(dict(status=x['status'],rows=out,initial_logits=x['initial_raw_EMA_max_logit_difference_by_layer_pool'],missing=len(x['missing']),final4096_read=False)))
