"""Serve this task's existing result viewer at its established loopback URL."""
from pathlib import Path
import datetime,json,socket,subprocess,sys,time,urllib.request
D=Path(__file__).resolve().parent
served=D.parent/'training-and-fixed512-em-20261012-2245'
assert (served/'ranking.html').is_file()
sock=socket.socket();occupied=sock.connect_ex(('127.0.0.1',18766))==0;sock.close()
receipt={'url':'http://127.0.0.1:18766/ranking.html','directory':str(served),'already_served':occupied}
if not occupied:
    with (D/'viewer-server.log').open('ab') as out:
        proc=subprocess.Popen([sys.executable,'-m','http.server','18766','--bind','127.0.0.1','--directory',str(served)],stdin=subprocess.DEVNULL,stdout=out,stderr=out,start_new_session=True)
    receipt['pid']=proc.pid
checks=[]
for _ in range(20):
    try:
        with urllib.request.urlopen(receipt['url'],timeout=3) as response:
            raw=response.read()
            assert raw==(D/'ranking.html').read_bytes(), 'Existing port serves a different result page'
            checks.append({'status':response.status,'bytes':len(raw),'time':time.time()})
        if len(checks)>=2:break
    except urllib.error.URLError:pass
    time.sleep(.25)
assert len(checks)>=2, 'Viewer did not return two verified responses'
receipt.update(checks=checks,verified_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
(D/'viewer-server-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
