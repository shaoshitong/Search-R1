"""Read-only training-window summary; no model, scoring or trajectory metric."""
import hashlib,json,math,time
from pathlib import Path
CURVE=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-b128-v1/curve.jsonl')
FIRST=1676;WIDTH=100
REQUIRED=('cursor','source_cursor','native_steps','head_steps','questions','effective_batch_questions','views','mixed_views','nonmixed_views','confusion','loss','binary_Mixed_BCE','time')
def main():
 with CURVE.open('rb')as stream:
  st=__import__('os').fstat(stream.fileno());raw=stream.read(st.st_size)
 complete=raw if raw.endswith(b'\n')else raw[:raw.rfind(b'\n')+1];rows=[json.loads(line)for line in complete.splitlines()if line.strip()];missing={};invalid_clock=invalid_batch=invalid_confusion=invalid_loss=0;windows={};previous=FIRST-1;previous_time=None
 for r in rows:
  absent=[k for k in REQUIRED if k not in r]
  for key in absent:missing[key]=missing.get(key,0)+1
  cursor=r.get('cursor');clock=type(cursor)is int and cursor==previous+1 and r.get('native_steps')==cursor==r.get('head_steps')and r.get('source_cursor')==FIRST-1 and isinstance(r.get('time'),(float,int))and(previous_time is None or r['time']>=previous_time)
  invalid_clock+=not clock
  if type(cursor)is not int:continue
  previous=cursor;previous_time=r.get('time',previous_time)
  shape=r.get('questions')==r.get('effective_batch_questions')==128 and r.get('views')==512 and r.get('mixed_views')==r.get('nonmixed_views')==256;invalid_batch+=not shape
  c=r.get('confusion',{});valid_c=set(c)=={'tp','tn','fp','fn'}and all(type(v)is int and v>=0 for v in c.values())and c.get('tp',0)+c.get('fn',0)==256 and c.get('tn',0)+c.get('fp',0)==256;invalid_confusion+=not valid_c
  loss=r.get('loss');bce=r.get('binary_Mixed_BCE');valid_loss=isinstance(loss,(int,float))and isinstance(bce,(int,float))and math.isfinite(loss)and loss>=0 and abs(loss-bce)<1e-12;invalid_loss+=not valid_loss
  start=FIRST+((cursor-FIRST)//WIDTH)*WIDTH;w=windows.setdefault(start,dict(expected_first=start,expected_last=start+WIDTH-1,first_cursor=cursor,last_cursor=cursor,updates=0,question_visits=0,views=0,mixed_views=0,nonmixed_views=0,confusion=dict(tp=0,tn=0,fp=0,fn=0),weighted_loss_sum=0.,loss_weight=0,invalid_rows=0,micro_questions=set()))
  w['last_cursor']=cursor;w['updates']+=1;w['invalid_rows']+=not(clock and shape and valid_c and valid_loss);w['micro_questions'].add(r.get('micro_questions'))
  if shape:
   w['question_visits']+=128;w['views']+=512;w['mixed_views']+=256;w['nonmixed_views']+=256
  if valid_c:
   for k in c:w['confusion'][k]+=c[k]
  if valid_loss and shape:w['weighted_loss_sum']+=loss*512;w['loss_weight']+=512
 out=[]
 for start,w in sorted(windows.items()):
  c=w['confusion'];pos=c['tp']+c['fn'];neg=c['tn']+c['fp'];valid=w['invalid_rows']==0;w['complete_100_updates']=w['updates']==100 and w['first_cursor']==w['expected_first']and w['last_cursor']==w['expected_last'];w['loss_weighted_mean']=w.pop('weighted_loss_sum')/w['loss_weight']if w['loss_weight']else None;w['mixed_recall']=c['tp']/pos if valid and pos else None;w['nonmixed_recall']=c['tn']/neg if valid and neg else None;w['Mixed_BA']=(w['mixed_recall']+w['nonmixed_recall'])/2 if valid and pos and neg else None;w['micro_questions']=sorted(w['micro_questions'],key=str);out.append(w)
 result=dict(status='READONLY_B128_TRAINING_SAMPLED_WINDOWS_SUMMARIZED',snapshot_epoch=time.time(),curve=dict(path=str(CURVE),snapshot_bytes=len(raw),complete_prefix_bytes=len(complete),complete_prefix_sha256=hashlib.sha256(complete).hexdigest(),trailing_incomplete_bytes=len(raw)-len(complete),inode=st.st_ino,uid=st.st_uid),checks=dict(rows=len(rows),first_cursor=rows[0].get('cursor')if rows else None,last_cursor=rows[-1].get('cursor')if rows else None,missing_fields=missing,invalid_clock_rows=invalid_clock,invalid_batch128_512views_256perclass_rows=invalid_batch,invalid_confusion_rows=invalid_confusion,invalid_loss_rows=invalid_loss,all_rows_valid=not(missing or invalid_clock or invalid_batch or invalid_confusion or invalid_loss)),window_width=100,windows=out,loss_weight='actual_G5_views; each accepted row512 equally weighted views',scope='training_sampler_windows_not_independent_test; repeated question visits are not unique questions',no_trajectory_accuracy=True,no_new_scoring=True,model_or_threshold_changed=False)
 print(json.dumps(result,sort_keys=True,separators=(',',':')))
if __name__=='__main__':main()
