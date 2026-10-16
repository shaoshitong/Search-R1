"""Read each pinned history in full; store redacted compact research excerpts only."""
from pathlib import Path
import json,re,hashlib,datetime,collections
D=Path(__file__).resolve().parent
sources=json.loads((D/'history-source-index.json').read_text())
MATCH=re.compile(r'question[ _-]*(?:diversity|probe|selection)|q[ _-]?probe|diversity[ _-]?prob|prefix(?:5|20|1024)|prefix[-_ ]selection|teacher[-_ ]warmup|探针|暖启动|预热实验|混合组|mixed[_ -](?:group|probability)|dynrpo',re.I)
EXPERIMENT=re.compile(r'q[ _-]?probe|question[ _-]*(?:diversity|probe)|teacher[-_ ]warmup|prefix(?:5|20|1024)|prefix[-_ ]selection|diversity[ _-]?prob|dynrpo|混合组|问题.*探针|探针.*(?:训练|warmup|预热|拟合|epoch)',re.I)
JOB=re.compile(r'(?:JobId|job_id|job|作业|Slurm)[^\d\n]{0,24}(\d{4,6})',re.I)
PATH=re.compile(r'(?:/data/[^\s\"\'<>]+|/mnt/[^\s\"\'<>]+|/Users/[^\s\"\'<>]+|outputs/[^\s\"\'<>]+|preparation/[^\s\"\'<>]+)')
def redact(s):
 s=re.sub(r'hf_[A-Za-z0-9]{12,}', '[HF_TOKEN_REDACTED]',s)
 s=re.sub(r'-----BEGIN[^\n]*PRIVATE KEY-----[\s\S]*?-----END[^\n]*PRIVATE KEY-----','[PRIVATE_KEY_REDACTED]',s)
 s=re.sub(r'(?i)(password|passwd|token|api[_ -]?key)(\s*[=:]\s*[\"\']?)([^\s,;\"\']{8,})',r'\1\2[SECRET_REDACTED]',s)
 return s
index=[];all_messages=[]
for src in sources:
 p=Path(src['rollout_path']);limit=p.stat().st_size;h=hashlib.sha256();counts=collections.Counter();hits=[];messages=[];line_no=0;offset=0;first=None;last=None;seen=set()
 with p.open('rb') as f:
  while offset<limit:
   raw=f.readline(min(limit-offset,64*1024*1024))
   if not raw:break
   start=offset;offset+=len(raw);h.update(raw);line_no+=1
   try:x=json.loads(raw)
   except ValueError:counts['invalid_or_partial_line']+=1;continue
   typ=x.get('type');counts[typ]+=1;ts=x.get('timestamp');first=first or ts;last=ts
   if typ!='response_item':continue
   pay=x.get('payload',{});kind=pay.get('type');role=pay.get('role');s=''
   if kind=='message' and role in ('user','assistant'):
    s='\n'.join(c.get('text','') for c in pay.get('content',[]) if c.get('type') in ('input_text','output_text','text'))
   elif kind in ('function_call','custom_tool_call'):
    s=str(pay.get('arguments',pay.get('input','')))
   elif kind in ('function_call_output','custom_tool_call_output'):
    out=pay.get('output','');s=out if isinstance(out,str) else json.dumps(out,ensure_ascii=False)
   if not s or not MATCH.search(s):continue
   matches=list(MATCH.finditer(s));excerpt=' | '.join(s[max(0,m.start()-80):min(len(s),m.end()+230)].replace('\n',' ') for m in matches[:3]);excerpt=redact(excerpt)[:1000]
   v=dict(session=src['id'],title=src['title'],timestamp=ts,ordinal=x.get('ordinal'),line=line_no,byte_offset=start,record_bytes=len(raw),kind=kind,role=role,record_sha256=hashlib.sha256(raw).hexdigest(),keywords=sorted(set(m.group(0).lower() for m in matches))[:15],jobs=sorted(set(JOB.findall(s))),paths=[redact(q[:350]) for q in dict.fromkeys(PATH.findall(s)) if MATCH.search(q)][:12],excerpt=excerpt)
   hits.append(v)
   if kind=='message' and EXPERIMENT.search(s):
    digest=hashlib.sha256(s.encode()).hexdigest()
    if digest not in seen:
     seen.add(digest);messages.append(dict(v,text=redact(s)[:18000],truncated=len(s)>18000))
 (D/(src['id']+'-history-hits.json')).write_text(json.dumps(hits,ensure_ascii=False,separators=(',',':'))+'\n')
 # Keep human/assistant relevant records as necessary research documentation, not full history or raw tool outputs.
 (D/(src['id']+'-research-messages.json')).write_text(json.dumps(messages,ensure_ascii=False,separators=(',',':'))+'\n')
 rec=dict(**src,scanned_prefix_bytes=offset,scanned_prefix_sha256=h.hexdigest(),source_bytes_after=p.stat().st_size,total_records=line_no,event_counts=dict(counts),matching_records=len(hits),research_message_records=len(messages),first_timestamp=first,last_timestamp=last,all_existing_records_in_start_snapshot_scanned=(offset==limit),credentials_redacted=True)
 index.append(rec);print(json.dumps({k:rec[k] for k in ['title','total_records','matching_records','research_message_records','last_timestamp']},ensure_ascii=False),flush=True)
(D/'full-history-scan-audit.json').write_text(json.dumps(dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=index,total_scanned_bytes=sum(x['scanned_prefix_bytes'] for x in index),total_scanned_records=sum(x['total_records'] for x in index),all16_snapshots_scanned=all(x['all_existing_records_in_start_snapshot_scanned'] for x in index)),ensure_ascii=False,indent=2)+'\n')
