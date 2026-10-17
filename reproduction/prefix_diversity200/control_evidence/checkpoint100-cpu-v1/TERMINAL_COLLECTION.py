"""Collect small CP100 terminal evidence; verify the owned unit has exited."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time

BASE = Path('/mnt/data/shaoshitong/search-g1-prefix-diversity200-20261017-4a1cc2d500fe-calibration1')
ROOT = BASE/'acceptance100-cpu-control-v1'
HELPER = '759d28c45aed0f18ab4f42ad6be17f5f690029d5'
UNIT = 'prefix-diversity-4a1cc2d5-cal1-checkpoint100-cpu-v1'
PREFIX = 'joint-milestone-0100-independent-full-sha-cpu-once-v1'
PROOF_SHA = 'd5b612106cc942ebfcfab978704cda740fe5f3403831e952af086e9e89d1eb1b'
files = {}

def small(path):
    assert path.resolve(strict=True) == path and BASE in path.parents
    before = path.stat(); assert before.st_size <= 1024**2
    raw = path.read_bytes(); after = path.stat()
    assert (before.st_ino,before.st_size,before.st_mtime_ns) == (after.st_ino,after.st_size,after.st_mtime_ns)
    files[str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),base64=base64.b64encode(raw).decode())
    return raw

def read(path): return json.loads(small(path))

deploy = read(ROOT/'deploy-receipt.json'); c = deploy['config']
assert deploy['helper_commit'] == HELPER and c['outer'] == 100 and c['audit_version'] == 'v1'
assert c['unit'] == UNIT and c['base'] == str(BASE) and c['control_name'] == ROOT.name
exit_ = read(ROOT/'reader-exit.json'); start = read(ROOT/'reader-start.json')
proof_path = BASE/'preparation'/(PREFIX+'.json'); proof = read(proof_path)
assert files[str(proof_path)]['sha256'] == PROOF_SHA == exit_['proof_sha256']
assert proof['status'] == 'PREFIX_SELECTION200_JOINT_MILESTONE_FULL_SHA_CPU_PASS'
assert proof['run_id'] == c['run_id'] and proof['outer'] == 100 and proof['audit_version'] == 'v1'
assert proof['source_commit'] == c['runtime_source_commit'] and proof['config_sha256'] == c['runtime_config_sha256']
assert proof['joint_complete_sha256'] == 'b1db2307b6a4170f0ab91b7aeb6d037f99ee8e7c6a000673794b907e7d2f00b4'
assert exit_['helper_commit'] == start['helper_commit'] == HELPER
assert exit_['status'] == 'CPU_ACCEPTANCE_PASS' and exit_['subprocess_exit'] == exit_['exit_code'] == 0
assert start['CUDA_VISIBLE_DEVICES'] == '' and start['actual_systemd_limits'] == dict(MemoryMax=str(12*2**30),CPUQuotaPerSecUSec='1s',Nice='19')
intent = read(BASE/'preparation'/(PREFIX+'-intent.json'))
assert intent['joint_complete_sha256'] == proof['joint_complete_sha256'] and intent['audit_version'] == 'v1'
small(ROOT/'reader.log'); read(ROOT/'systemd-submit.json')
fields = ['LoadState','ActiveState','SubState','Result','ExecMainPID','ExecMainStatus','ControlGroup']
query = subprocess.run(['systemctl','--user','show',UNIT]+['--property='+x for x in fields],capture_output=True,text=True,timeout=15)
assert query.returncode == 0
systemd = dict(line.split('=',1)for line in query.stdout.splitlines()if'='in line)
assert systemd['ActiveState'] == 'inactive' and systemd['SubState'] == 'dead' and systemd['ExecMainPID'] == '0'
assert systemd['ExecMainStatus'] == '0' and systemd['Result'] == 'success'
pid_path = Path('/proc')/str(start['pid'])/'stat'
old_birth = start['proc_stat'].rsplit(')',1)[1].split()[19]
same_process_alive = pid_path.exists() and pid_path.read_text().rsplit(')',1)[1].split()[19] == old_birth
assert not same_process_alive
cg = systemd.get('ControlGroup')
cg_members = None
if cg:
    path = Path('/sys/fs/cgroup')/cg.lstrip('/')/'cgroup.procs'
    cg_members = path.read_text().split() if path.exists() else []
    assert not cg_members
cleanup = dict(status='OWNED_CPU_UNIT_EXITED_NO_ACTIVE_WORKERS',observed_unix=time.time(),helper_commit=HELPER,
    proof_sha256=PROOF_SHA,systemd=systemd,reader_pid=start['pid'],reader_birth_ticks=old_birth,
    same_reader_process_alive=False,owned_cgroup_members=cg_members,
    action='Successful transient unit already exited; no stop/reset/removal required. Preserve proof, intent, logs and deployed source.',
    tensor_payload_read=False,large_payload_repeated=False,training_modified=False,qa_or_grounding_modified=False,
    gpu_forward_performed=False)
p = ROOT/'terminal-cleanup-receipt.json'
with p.open('x') as stream: json.dump(cleanup,stream,sort_keys=True);stream.flush()
read(p)
print(json.dumps(dict(status=cleanup['status'],cleanup=cleanup,files=files)))
