import importlib.util,pathlib,subprocess,shlex,json,time
root=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
s=importlib.util.spec_from_file_location('c',root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
a=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
probe="""import pathlib,os,json,time
out=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().decode().split(chr(0))
  if '-m' not in args or args[args.index('-m')+1]!='reproduction.operations.qwen7b_warmup.balanced_finetune':continue
  if '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/balanced-rank1-from-epoch1-prefix20-20261020' not in args:continue
  stat=(p/'stat').read_text().split(') ',1)[1].split();io=dict(line.split(': ',1) for line in (p/'io').read_text().splitlines())
  out.append(dict(pid=int(p.name),start_ticks=int(stat[19]),cpu_ticks=int(stat[11])+int(stat[12]),state=stat[0],rss_bytes=int(stat[21])*os.sysconf('SC_PAGE_SIZE'),read_bytes=int(io['read_bytes']),rchar=int(io['rchar'])))
 except (OSError,ValueError,IndexError):continue
assert len(out)==4,out
print(json.dumps({'time':time.time(),'ticks_per_second':os.sysconf('SC_CLK_TCK'),'workers':sorted(out,key=lambda p:p['pid'])}))
"""
c='import subprocess\nr=subprocess.run('+repr(['srun','--jobid=742276','--overlap','--ntasks=1','--cpus-per-task=1','--gres=gpu:0','--mem=1G','--time=00:01:00','python3','-c',probe])+',stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=35)\nprint(r.stdout)\nassert r.returncode==0,r.stderr'
r=m.run('python3 -c '+shlex.quote(c),account='mou176',source_address=a,timeout=45);assert r.returncode==0,r.stderr.decode()
v=json.loads(next(x for x in r.stdout.decode().splitlines() if x.startswith('{')))
d=root/'outputs/qwen7b-warmup-20261019';(d/'balanced-worker-vitals-latest.json').write_text(json.dumps(v,indent=2));(d/('balanced-worker-vitals-%d.json'%v['time'])).write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
