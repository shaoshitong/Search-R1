"""Direct authenticated B-only publication/CP50 preparation/once launch control."""
import argparse,base64,datetime,gzip,hashlib,importlib.util,io,json,subprocess,tarfile
from pathlib import Path
from .follow_h10011_ab_normal_boundary import ROOT,PATH_LAYOUTS
REPO=Path(__file__).resolve().parents[3]

def ctl():
    spec=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);return c

def remote(inner,label,*,namespace=True,timeout=100):
    script=inner
    if namespace:
        script='import subprocess,os\np=subprocess.run('+repr(['unshare','--mount','--propagation','private','bash',PATH_LAYOUTS['source8'],'/dataset/shaoshitong/Search-G1-common8/env/bin/python','-'])+',input='+repr(inner)+',env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",SEARCH_G1_B_PREFIX_TOKENS="20"),text=True,capture_output=True,timeout='+str(timeout-10)+')\nassert p.returncode==0,p.stderr\nprint(p.stdout)'
    p=subprocess.run(ctl().route('h100-11','/usr/bin/python3 -'),input=script,text=True,capture_output=True,timeout=timeout)
    tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/('work/b-progressive-'+label+'-'+tag+'.json');out.write_text(json.dumps(dict(rc=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2)+'\n')
    if p.returncode:raise RuntimeError(str(out)+' '+p.stderr[-5000:])
    value=json.loads(p.stdout);print(json.dumps(dict(evidence=str(out),result=value)),flush=True);return value

def stage(commit):
    source=json.loads((ROOT/'work/prefix5-b-progressive-human-authorization.json').read_text())['source']['base']+'/repo'
    parent='2e3976230ac9425d66d8553afa9962821e964932'
    def tree(revision):
        data=subprocess.check_output(['git','ls-tree','-rz',revision],cwd=REPO);result={}
        for row in data.split(b'\0'):
            if row:
                meta,name=row.split(b'\t',1);result[name.decode()]=meta.split()[2].decode()
        return result
    parent_tree=tree(parent);current=tree(commit)
    expected={};delta={}
    archive=subprocess.check_output(['git','archive',commit],cwd=REPO)
    with tarfile.open(fileobj=io.BytesIO(archive))as tar:
        for item in tar:
            name=item.name
            if not item.isfile():continue
            # Runtime includes version files, YAML/templates and package data.
            # Published plots/full metric reports are not runtime dependencies.
            if name.startswith('reproduction/reports/') and not name.endswith(('.py','.sh')):continue
            raw=tar.extractfile(item).read();expected[name]=hashlib.sha256(raw).hexdigest()
            if parent_tree.get(name)!=current[name]:delta[name]=base64.b64encode(raw).decode()
    physical='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/search-g1-b-progressive-code-retained-20261018-'+commit
    inner="""import base64,hashlib,json,os
from pathlib import Path
b=Path(DEST);b.mkdir(exist_ok=True);source=Path(SOURCE);expected=EXPECTED;delta=DELTA
for name,checksum in expected.items():
 path=b/name;path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():
  if name in delta:
   with path.open('xb')as f:f.write(base64.b64decode(delta[name]))
  else:os.link(source/name,path)
 if hashlib.sha256(path.read_bytes()).hexdigest()!=checksum:raise ValueError('Runtime source SHA differs: '+name)
print(json.dumps(dict(physical=str(b),files=len(expected),changed_files=len(delta),commit=COMMIT,full_target_source_sha=True,only_code_deployed=True,source_hardlinks_never_mutated=True)))
""".replace('DEST',repr(physical)).replace('SOURCE',repr(source)).replace('EXPECTED',repr(expected)).replace('DELTA',repr(delta)).replace('COMMIT',repr(commit))
    return remote(inner,'code-stage',timeout=180)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','prepare','submit','observe'));p.add_argument('--commit');p.add_argument('--base');a=p.parse_args()
    if a.action=='stage':return stage(a.commit)
    source=json.loads((ROOT/'work/prefix5-b-progressive-human-authorization.json').read_text())['source']['base']
    b=a.base
    if a.action=='prepare':
        staged='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/search-g1-b-progressive-code-retained-20261018-'+a.commit
        inner='import sys,json;sys.path.insert(0,'+repr(staged)+');from reproduction.prefix_selection200.b_progressive_resume import prepare;print(json.dumps(prepare('+','.join(map(repr,[source,b,staged,a.commit,json.loads((ROOT/'work/prefix5-b-progressive-human-authorization.json').read_text())]))+')))' 
    elif a.action=='submit':
        inner='import sys,json;sys.path.insert(0,'+repr(b+'/repo')+');from reproduction.prefix_selection200.launch import submit;print(json.dumps(submit('+repr(b)+')))'
    else:
        inner='''import sys,json,time\nfrom pathlib import Path\nb=Path(BASE);sys.path.insert(0,str(b/'repo'))\nfrom reproduction.prefix_selection200.ab_control import observe\nv=observe(b,50);j=json.loads((b/'latest-joint.json').read_text());v['probe_migrations']=[str(p)for p in (b/'preparation').glob('prefix-migration-*.json')];log=b/'logs/native-pipeline.log'\nwith log.open('rb')as f:f.seek(max(0,log.stat().st_size-3000));v['log_tail']=f.read().decode(errors='replace')\nprint(json.dumps(v))'''.replace('BASE',repr(b))
    value=remote(inner,a.action,timeout=130)
    if a.action=='prepare':(ROOT/'work/b-progressive-prepared.json').write_text(json.dumps(value,indent=2)+'\n')
if __name__=='__main__':main()
