"""Read-only exact vanished temporary-gap lstat follow-up."""
import importlib.util,json,sys,time
from pathlib import Path
p=Path(__file__).parent
s=importlib.util.spec_from_file_location('request',p/'personal_storage_cp5_metadata_request_1792265074.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
sys.argv=['request','gap-absence']
remote="import json,os,stat,time\nfrom pathlib import Path\npaths=['/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018/tmp/tmpj16r1flk', '/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018/tmp/tmpje60wuf1', '/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018/tmp/tmpndtrbvjl', '/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018/tmp/tmpnx92ne6z']\nassert os.getuid()==14090\nrows=[]\nfor text in paths:\n p=Path(text);assert p.parent==Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018/tmp')\n try:\n  s=p.lstat();rows.append({'path':text,'exists':True,'mode':s.st_mode,'is_symlink':stat.S_ISLNK(s.st_mode),'allocated_bytes':s.st_blocks*512,'device':s.st_dev,'inode':s.st_ino})\n except FileNotFoundError:rows.append({'path':text,'exists':False})\nprint(json.dumps({'checked_unix':time.time(),'uid':os.getuid(),'role':'exact_four_vanished_tmp_gaps_lstat','paths':rows,'other_paths_scanned':False,'symlink_followed':False},sort_keys=True))\n"
m.call('jxu957',remote)
