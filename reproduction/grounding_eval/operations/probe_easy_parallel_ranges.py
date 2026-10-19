"""Two small RAM-only private shard ranges; existing asset writer stays unchanged."""
import pathlib,json,subprocess,shlex,sys
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
D=pathlib.Path('outputs/grounding-eval-20261019');status=json.loads((D/'easy-hf-verified-status.json').read_text());revision=status['final_revision']
code=r'''import sys,json,subprocess,os,time,hashlib,concurrent.futures
x=json.load(sys.stdin);url='https://huggingface.co/Sutoonq/TMP_MODEL/resolve/'+x['revision']+'/grounding-eval-20261019/easy-selection-iter200/model-00001-of-00003.safetensors';env={k:v for k,v in os.environ.items()if k.lower()not in ['http_proxy','https_proxy','all_proxy']}
def snapshot():
 out={}
 for line in open('/proc/net/dev'):
  if ':'not in line:continue
  name,rest=line.split(':');v=rest.split();out[name.strip()]={'rx':int(v[0]),'tx':int(v[8])}
 return out
before=snapshot();started=time.monotonic()
def sample(start):
 end=start+(32<<20)-1;t=time.monotonic();cfg='url = "'+url+'"\nheader = "Authorization: Bearer '+x['token']+'"\n';command=['curl','--silent','--fail','--location','--proto','=https','--proto-redir','=https','--noproxy','*','--max-redirs','5','--resolve','huggingface.co:443:3.167.112.25','--connect-timeout','8','--max-time','45','--range',str(start)+'-'+str(end),'--max-filesize',str(32<<20),'--config','-'];p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,env=env);p.stdin.write(cfg.encode());p.stdin.close();h=hashlib.sha256();first=hashlib.sha256();n=0
 while True:
  b=p.stdout.read(1<<20)
  if not b:break
  h.update(b)
  if n<(8<<20):first.update(b[:(8<<20)-n])
  n+=len(b)
 rc=p.wait(timeout=5);elapsed=time.monotonic()-t
 return {'start':start,'bytes':n,'curl_exit':rc,'seconds':elapsed,'MBps_including_handshakes':n/elapsed/1e6,'sha256':h.hexdigest(),'first8MiB_sha256':first.hexdigest()if start==0else None}
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:rows=list(pool.map(sample,[0,32<<20]))
elapsed=time.monotonic()-started;after=snapshot();print(json.dumps({'at':time.time(),'ranges':rows,'total_seconds':elapsed,'total_RAM_probe_bytes':sum(r['bytes']for r in rows),'aggregate_MBps_including_handshakes':sum(r['bytes']for r in rows)/elapsed/1e6,'network_byte_deltas':{n:{k:after[n][k]-v[k]for k in v}for n,v in before.items()if n in after},'existing_asset_writer_modified':False,'model_bytes_to_Mac':0,'model_bytes_to_disk':0}))'''
r=subprocess.run(control.route('mgmt-host','python3 -B -c '+shlex.quote("from pathlib import Path;import sys;sys.stdout.write(Path('/mnt/lustre/shaoshitong/hf-checkpoint-upload-20261005/task-token').read_text().strip())")),capture_output=True,text=True,timeout=20);assert r.returncode==0;token=r.stdout.strip();a=control.route('baai-a800','python3 -B -c '+shlex.quote(code));a[1:1]=['-o','ServerAliveInterval=5','-o','ServerAliveCountMax=6'];r=subprocess.run(a,input=json.dumps({'revision':revision,'token':token}),capture_output=True,text=True,timeout=55);assert r.returncode==0,'Range diagnosis transport failed';x=json.loads(r.stdout);(D/'easy-hf-two-ranges-diagnostic.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
