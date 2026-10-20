"""Run the published read-only audit remotely; return only small summaries."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--base',required=True);p.add_argument('--epoch',type=int,required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[3]
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();assert subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]==commit
    raw=Path(__file__).with_name('audit_epoch_predictions.py').read_bytes();digest=hashlib.sha256(raw).hexdigest()
    header='payload='+repr(base64.b64encode(raw).decode())+'\ndigest='+repr(digest)+'\nbase='+repr(a.base)+'\nepoch='+repr(a.epoch)+'\ncommit='+repr(commit)+'\n'
    code=header+'''import pathlib,hashlib,base64,json,runpy\nb=pathlib.Path(base);assert b.parent.name=='dynrpo-qwen25-7b-probe-warmup-20261019'\nroot=b/'control-tools';root.mkdir(exist_ok=True);path=root/('epoch-prediction-audit-'+digest[:12]+'.py');own=base64.b64decode(payload);assert hashlib.sha256(own).hexdigest()==digest\nif path.exists():assert hashlib.sha256(path.read_bytes()).hexdigest()==digest\nelse:path.write_bytes(own)\nmodule=runpy.run_path(str(path));result=module['accept'](b,epoch);result.update(audit_source_sha256=digest,audit_source_commit=commit);print(json.dumps(result))\n'''
    spec=importlib.util.spec_from_file_location('control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();r=m.run('python3 -c '+shlex.quote(code),account='mou176',source_address=address,timeout=50)
    if r.returncode:print(r.stdout.decode());print(r.stderr.decode());raise SystemExit(r.returncode)
    rows=[json.loads(x) for x in r.stdout.decode().splitlines() if x.strip().startswith('{')];assert len(rows)==1
    path=a.control_root/'outputs/qwen7b-warmup-20261019'/('%s-epoch%02d-prediction-audit.json'%(Path(a.base).name,a.epoch));path.write_text(json.dumps(rows[0],indent=2));print(json.dumps(rows[0]))


if __name__=='__main__':main()
