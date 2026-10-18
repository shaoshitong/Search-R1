"""Direct authenticated B-only publication/CP50 preparation/once launch control."""
import argparse,base64,datetime,hashlib,importlib.util,io,json,subprocess,tarfile
from pathlib import Path
from .follow_h10011_ab_normal_boundary import ROOT,PATH_LAYOUTS
REPO=Path(__file__).resolve().parents[3]

def ctl():
    spec=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);return c

def remote(inner,label,*,namespace=True,timeout=100):
    script=inner
    if namespace:
        script='import subprocess,os\np=subprocess.run('+repr(['unshare','--mount','--propagation','private','bash',PATH_LAYOUTS['retained'],'/dataset/shaoshitong/Search-G1-common8/env/bin/python','-c',inner])+',env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",SEARCH_G1_B_PREFIX_TOKENS="20"),text=True,capture_output=True,timeout='+str(timeout-10)+')\nassert p.returncode==0,p.stderr\nprint(p.stdout)'
    p=subprocess.run(ctl().route('h100-11','/usr/bin/python3 -'),input=script,text=True,capture_output=True,timeout=timeout)
    tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/('work/b-progressive-'+label+'-'+tag+'.json');out.write_text(json.dumps(dict(rc=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2)+'\n')
    if p.returncode:raise RuntimeError(str(out)+' '+p.stderr[-5000:])
    value=json.loads(p.stdout);print(json.dumps(dict(evidence=str(out),result=value)),flush=True);return value

def stage(commit):
    raw=subprocess.check_output(['git','archive',commit],cwd=REPO);expected={}
    with tarfile.open(fileobj=io.BytesIO(raw))as t:
        for item in t:
            if item.isfile():expected[item.name]=hashlib.sha256(t.extractfile(item).read()).hexdigest()
    physical='/mnt/shared/workspace/training_data_node_10/shaoshitong/search-g1-b-progressive-code-20261018-'+commit
    inner='''import base64,hashlib,io,json,tarfile\nfrom pathlib import Path\nb=Path(PHYSICAL);b.mkdir(exist_ok=True)\nraw=base64.b64decode(PAYLOAD)\nexpected=EXPECTED\nwith tarfile.open(fileobj=io.BytesIO(raw))as archive:\n for item in archive:\n  if not item.isfile():continue\n  path=b/item.name\n  if not path.is_relative_to(b)or item.name.startswith('/')or'..'in Path(item.name).parts:raise ValueError('Archive path escapes publication')\n  content=archive.extractfile(item).read();path.parent.mkdir(parents=True,exist_ok=True)\n  if not path.exists():\n   with path.open('xb')as f:f.write(content)\n  if hashlib.sha256(path.read_bytes()).hexdigest()!=expected[item.name]:raise ValueError('Staged source SHA mismatch')\nprint(json.dumps(dict(physical=str(b),files=len(expected),commit=COMMIT,full_target_source_sha=True)))\n'''.replace('PHYSICAL',repr(physical)).replace('PAYLOAD',repr(base64.b64encode(raw).decode())).replace('EXPECTED',repr(expected)).replace('COMMIT',repr(commit))
    return remote(inner,'code-stage',namespace=False,timeout=100)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','prepare','submit','observe'));p.add_argument('--commit');p.add_argument('--base');a=p.parse_args()
    if a.action=='stage':return stage(a.commit)
    source=json.loads((ROOT/'work/prefix5-ab-assigned-arms.json').read_text())['B']['base']
    b=a.base
    if a.action=='prepare':
        staged='/mnt/shared/workspace/training_data_node_10/shaoshitong/search-g1-b-progressive-code-20261018-'+a.commit
        inner='import sys,json;sys.path.insert(0,'+repr(staged)+');from reproduction.prefix_selection200.b_progressive_resume import prepare;print(json.dumps(prepare('+','.join(map(repr,[source,b,staged,a.commit,json.loads((ROOT/'work/prefix5-b-progressive-human-authorization.json').read_text())]))+')))' 
    elif a.action=='submit':
        inner='import sys,json;sys.path.insert(0,'+repr(b+'/repo')+');from reproduction.prefix_selection200.launch import submit;print(json.dumps(submit('+repr(b)+')))'
    else:
        inner='''import sys,json,time\nfrom pathlib import Path\nb=Path(BASE);sys.path.insert(0,str(b/'repo'))\nfrom reproduction.prefix_selection200.ab_control import observe\nv=observe(b,50);j=json.loads((b/'latest-joint.json').read_text());v['probe_migrations']=[str(p)for p in (b/'preparation').glob('prefix-migration-*.json')];log=b/'logs/native-pipeline.log'\nwith log.open('rb')as f:f.seek(max(0,log.stat().st_size-3000));v['log_tail']=f.read().decode(errors='replace')\nprint(json.dumps(v))'''.replace('BASE',repr(b))
    value=remote(inner,a.action,timeout=130)
    if a.action=='prepare':(ROOT/'work/b-progressive-prepared.json').write_text(json.dumps(value,indent=2)+'\n')
if __name__=='__main__':main()
