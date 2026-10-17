"""Single diagnostic du observation; no credentials, training or source edits."""
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import time

WS = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6')
OUT = Path(__file__).parent
TAG = 'easy2560-personal-storage-metadata-1792265074'
REMOTE = '/tmp/' + TAG
BASE = '/data/user/jxu957/experiments'
OLD = BASE + '/easy2560-corollout-joint4-20261017'
NEW = BASE + '/easy2560-corollout-joint4-micro16-after730080-20261018'
SUPERVISOR = '''import json,os,subprocess,time
from pathlib import Path
root=Path('/tmp/easy2560-personal-storage-metadata-1792265074')
def birth(pid):
 p=Path('/proc')/str(pid);f=(p/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,start_ticks=int(f[19]),uid=p.stat().st_uid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),pid_namespace=os.readlink(p/'ns/pid'))
def put(name,x):
 with (root/name).open('x') as f:json.dump(x,f,sort_keys=True);f.write('\\n');f.flush();os.fsync(f.fileno())
start=time.time()
with (root/'du.stdout').open('xb') as out,(root/'du.stderr').open('xb') as err:
 p=subprocess.Popen(['du','-s','-B1','/data/user/jxu957/experiments'],stdin=subprocess.DEVNULL,stdout=out,stderr=err)
 put('du-claim.json',dict(process_identity=birth(p.pid),command=['du','-s','-B1','/data/user/jxu957/experiments'],started_unix=start,observing_uid=os.getuid(),symlinks_followed=False))
 rc=p.wait()
put('result.json',dict(returncode=rc,started_unix=start,finished_unix=time.time(),observing_uid=os.getuid(),scope='/data/user/jxu957/experiments'))
'''


def call(account, code, payload=''):
    p = WS/'work/qprobe-sigma3-loss-20261015/reproduction/hpc3_qprobe/local_account_bridge.py'
    spec = importlib.util.spec_from_file_location('modern_bridge', p)
    bridge = importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)
    result = bridge.run(account, shlex.join(['/data/user/jxu957/experiments/easy2560-corollout-joint4-20261017/env/bin/python', '-B', '-c', code]), payload, str(WS/'work'),
        login_host='10.120.48.31', expected_host_identity=dict(hostname='ACD-Login-1',
        boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a', pid_namespace='pid:[4026531836]'), observation_timeout=55)
    record = dict(account=account, checked_local_unix=time.time(), returncode=result.returncode,
                  stdout=result.stdout, stderr=result.stderr)
    path = OUT/('personal-storage-refresh-'+sys.argv[1]+'-'+str(int(time.time()))+'.json')
    path.write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
    print(path)
    if sys.argv[1] == 'poll' and result.returncode == 0:
        remote = json.loads(result.stdout)
        print(json.dumps(dict(result=remote['result.json'], du_stdout=remote['du.stdout'],
            permission_denied_lines=(remote['du.stderr'] or '').count('Permission denied'),
            active_directory_fds=remote.get('active_directory_fds'), du_cpu_ticks=remote.get('du_cpu_ticks'),
            supervisor_error=remote['supervisor.log']), sort_keys=True))
    else: print(json.dumps(record, sort_keys=True))


def main():
    mode = sys.argv[1]
    if mode == 'launch':
        call('mou176', '''import json,os,subprocess,sys,time
from pathlib import Path
root=Path('''+repr(REMOTE)+''');assert os.getuid()==21930
root.mkdir(mode=0o700)
request={'command':['du','-s','-B1','/data/user/jxu957/experiments'],'started_unix':time.time(),'uid':os.getuid()}
with (root/'start-request.json').open('x') as f:json.dump(request,f);f.flush();os.fsync(f.fileno())
log=(root/'supervisor.log').open('xb')
p=subprocess.Popen([sys.executable,'-B','-c','''+repr(SUPERVISOR)+'''],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'status':'STARTED_ONCE','root':str(root),'supervisor_pid':p.pid,'request':request},sort_keys=True))''')
    elif mode == 'poll':
        call('mou176', '''import json,os,time
from pathlib import Path
root=Path('''+repr(REMOTE)+''');assert root.stat().st_uid==os.getuid()==21930
out={'checked_unix':time.time(),'root':str(root)}
for name in ['start-request.json','du-claim.json','result.json']:
 p=root/name;out[name]=json.loads(p.read_text()) if p.exists() else None
c=out['du-claim.json']
if c:
 p=Path('/proc')/str(c['process_identity']['pid']);out['du_process_exists']=p.exists()
 if p.exists():
  f=(p/'stat').read_text().rsplit(')',1)[1].split();assert int(f[19])==c['process_identity']['start_ticks']
  out['du_cpu_ticks']={'utime':int(f[11]),'stime':int(f[12])};out['du_process_state']=f[0]
  out['active_directory_fds']=[]
  for fd in list((p/'fd').iterdir()):
   try:t=os.readlink(fd)
   except OSError:continue
   if t.startswith('/data/user/jxu957/experiments'):out['active_directory_fds'].append(t)
for name in ['du.stdout','du.stderr','supervisor.log']:
 p=root/name;out[name]=p.read_text() if p.exists() else None
print(json.dumps(out,sort_keys=True))''')
    elif mode in ('targets', 'complements'):
        paths = [NEW, OLD] if mode == 'targets' else json.loads(Path(sys.argv[2]).read_text())
        code = '''import json,os,subprocess,sys,time
from pathlib import Path
paths=json.load(sys.stdin);assert os.getuid()==14090
assert all(isinstance(x,str) and (x=='/data/user/jxu957/experiments' or x.startswith('/data/user/jxu957/experiments/')) and '..' not in Path(x).parts and Path(x).resolve()==Path(x) for x in paths)
start=time.time();r=subprocess.run(['du','-s','-B1','--']+paths,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=45)
print(json.dumps({'started_unix':start,'finished_unix':time.time(),'observing_uid':os.getuid(),'paths':paths,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'symlinks_followed':False},sort_keys=True))'''
        call('jxu957', code, json.dumps(paths))
    else: raise SystemExit('launch, poll, targets or complements only')


if __name__ == '__main__': main()
