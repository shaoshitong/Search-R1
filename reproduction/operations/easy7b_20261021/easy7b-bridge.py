import importlib.util,pathlib,sys,shlex,json,time
cwd=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6')
spec=importlib.util.spec_from_file_location('bridge',cwd/'work/qprobe-sigma3-loss-20261015/reproduction/hpc3_qprobe/local_account_bridge.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
code=pathlib.Path(sys.argv[1]).read_text();cmd=shlex.join(['/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python','-B','-c',code]);r=b.run('jxu957',cmd,pathlib.Path(sys.argv[3]).read_text() if len(sys.argv)>3 else '',str(cwd/'work'),login_host='10.120.48.31',expected_host_identity=dict(hostname='ACD-Login-1',boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a',pid_namespace='pid:[4026531836]'),observation_timeout=55)
p=cwd/'outputs/easy2560-qwen7b-20261021'/(sys.argv[2]+'-'+str(int(time.time()))+'.json');p.write_text(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n');print('receipt',p,'rc',r.returncode);
if not r.returncode and sys.argv[2]=='observation':
 d=json.loads(r.stdout);out={k:v for k,v in d.items() if k in ('checked_unix','job','queue','accounting','control/progress.json','control/failure.json','normal/progress.json','cf-v4-results','cf-v5-results','control/complete.json')}
 out['summaries']={k:{z:v.get(z) for z in ('status','counts','coverage','TC','D_Inv')} for k,v in d.items() if k.endswith('/summary.json')}
 out['latest_log_tails']={k:dict(bytes=v['bytes'],tail=v['tail'][-700:]) for k,v in d['logs'].items() if k==d.get('control/progress.json',{}).get('phase','')+'.log'}
 print(json.dumps(out))
else:print(r.stdout[:15000] if not r.returncode else r.stderr[-3000:])
sys.exit(bool(r.returncode))
