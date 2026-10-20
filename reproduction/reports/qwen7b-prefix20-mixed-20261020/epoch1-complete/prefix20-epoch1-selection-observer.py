import pathlib,json,gzip,hashlib,math,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/mixed-only-ft-cp2-prefix20-20261020');metric=json.loads((b/'probe/metrics-latest.json').read_text());assert metric['epoch']==1 and metric['steps']==10464
r=metric['metrics']['selection2048'];path=pathlib.Path(r['predictions']['path']);raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==r['predictions']['sha256'];pred=[json.loads(x) for x in gzip.decompress(raw).decode().splitlines()]
held=pathlib.Path(json.loads((b/'config.json').read_text())['selection_base']);proof=json.loads((held/'bank/complete.json').read_text());groupsfile=held/'bank/groups.json';assert hashlib.sha256(groupsfile.read_bytes()).hexdigest()==proof['groups_sha256'];groups=json.loads(groupsfile.read_text())['selection_validation'];truth={x['qid']:x for x in groups};assert len(pred)==len(truth)==2048 and {x['qid'] for x in pred}==set(truth)
for row in pred:
 assert set(row)=={'qid','trajectory_ids','p_j','predicted_mixed'} and row['trajectory_ids']==truth[row['qid']]['trajectory_ids']
 ar=aw=1.
 for p in row['p_j']:assert math.isfinite(p) and 0<=p<=1;ar*=p;aw*=1-p
 assert abs(1-ar-aw-row['predicted_mixed'])<1e-12
selected=sorted(pred,key=lambda r:(-r['predicted_mixed'],r['qid']))[:512];hist=[0]*6
for row in selected:hist[sum(truth[row['qid']]['labels'])]+=1
assert hist==r['selected512']['K_histogram'] and sum(hist[1:5])==r['selected512']['true_mixed']==182
receipt=json.loads((b/'probe/epoch-0001.json').read_text());assert receipt['completed_epoch'] and receipt['steps']==10464 and receipt['metrics']==metric['metrics'] and receipt['independent_cpu_reload']
out={'status':'EPOCH1_METRIC_RECEIPT_AND_UNSEEN2048_PREDICTED_TOP512_INDEPENDENTLY_RECOMPUTED','epoch':1,'steps':10464,'metrics':metric['metrics'],'checkpoint':{k:v for k,v in receipt.items() if k!='metrics'},'selected_histogram':hist,'pool_mixed':sum(0<sum(g['labels'])<5 for g in groups),'selected_mixed':182,'selected_fraction':182/512,'ranked_without_truth':True,'time':time.time()}
print(json.dumps(out))
