from pathlib import Path
import json,re,html,hashlib
D=Path(__file__).resolve().parent
out=[]
for p in D.glob('*-research-messages.json'):
 seen=set();rows=[]
 for r in json.loads(p.read_text()):
  text=r['text'];kind='human' if r['role']=='user' else 'assistant'
  if '<heartbeat>' in text:continue
  if text.startswith('<codex_internal_context'):
   m=re.search(r'<objective>([\s\S]*?)</objective>',text)
   if not m:continue
   text=m[1].strip();kind='goal_snapshot'
  if text.startswith('<environment_context>')and len(text)<10000:continue
  if 'AGENTS.md instructions' in text and '## ' in text:continue
  normalized=re.sub(r'\s+',' ',html.unescape(text)).strip();digest=hashlib.sha256(normalized.encode()).hexdigest()
  if digest in seen:continue
  seen.add(digest);rows.append(dict(r,text=text,record_role=kind))
 rows.sort(key=lambda r:r['timestamp']);(D/(p.name.replace('-research-messages','-chronology'))).write_text(json.dumps(rows,ensure_ascii=False,separators=(',',':'))+'\n');out.append(dict(session=rows[0]['session']if rows else p.name[:36],records=len(rows),humans=sum(r['record_role']=='human' for r in rows),goals=sum(r['record_role']=='goal_snapshot' for r in rows)))
(D/'chronology-index.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
