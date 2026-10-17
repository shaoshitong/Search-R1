from pathlib import Path
import importlib.util,json,subprocess,datetime,hashlib,sys
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reference_a800_route','/Users/shaoshitong/Documents/Search-G1/actorhidden-prefix5-prune-aware-resume-20261017/reproduction/operations/a800_actorhidden_prefix5_control.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
code=r'''
import pathlib,json,hashlib,socket,os
b=pathlib.Path('/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-54ac33dcebe64e62a11fe4177f41f116-resume-a4a0c2a18cca40bc8c3850d40eed8554')
p=b/'preparation/launch-config.json';raw=p.read_bytes();c=json.loads(raw)
out={'host':socket.gethostname(),'config_path':str(p),'config_sha256':hashlib.sha256(raw).hexdigest(),'config':c,'assets':{}}
keys=['probe_source_checkpoint','canonical_train','validation_questions','final_questions','common_manifest_path','composite_manifest_path','source_question_pool','fixed_test_identity_path','grounding_manifest','grounding_calibration_questions','probe_source_cpu_proof']
for key in keys:
 if key not in c:continue
 p=pathlib.Path(c[key]);s=p.stat();out['assets'][key]={'path':str(p),'bytes':s.st_size,'is_file':p.is_file(),'inode':s.st_ino}
 if p.is_file() and s.st_size<512*1024 and p.suffix=='.json':
  raw=p.read_bytes();out['assets'][key].update(sha256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
print(json.dumps(out))
'''
r=subprocess.run(m.route(code),capture_output=True,timeout=65);tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');p=ROOT/('prefix5-reference-source-assets-'+tag+'.json');v=dict(returncode=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode());p.write_text(json.dumps(v,indent=2)+'\n');assert r.returncode==0,v['stderr'];v=json.loads(r.stdout);q=ROOT/'prefix5-reference-source-assets.json';q.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(dict(actual=str(p),source_config_sha256=v['config_sha256'],keys=list(v['assets']),bytes={k:x['bytes']for k,x in v['assets'].items()})))
