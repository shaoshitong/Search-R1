#!/usr/bin/env python3
"""Read the completed fixed trace200 handle after an exact scalar size qualification.
Never submits a unit or rereads experiment JSON inputs. Original scanner unchanged.
"""
import hashlib,importlib.util,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'work/tree-speed-repair-20261008/tree200_trace_retention_durable.py'
EXPECTED_SOURCE='58083e5e19343411ccb6fad9dfc3e784a79346f64734e0c684770b870e32a361'
DIAGNOSTIC=ROOT/'outputs/tree-nested-branch-handoff-20261013/TREE200_TRACE_DURABLE_PAYLOAD_SIZE_DIAGNOSTIC.json'
EXPECTED_DIAGNOSTIC='d9ccb46f2ad491ebd137c3948aede1d82f23b32d6cdb532f60176f20e044fa7f'
PROGRAM='fc7dcb42a53ddce0f752e50bdb7ad32693cb462dc243d074100af53a2c860767'
CREDIT_BYTES=204976
CREDIT_SHA='fc2b7a29f3d43373128c1eb133688035d099ffe54ee01206424972afc8276793'

def main():
    assert __debug__ and hashlib.sha256(SOURCE.read_bytes()).hexdigest()==EXPECTED_SOURCE
    assert hashlib.sha256(DIAGNOSTIC.read_bytes()).hexdigest()==EXPECTED_DIAGNOSTIC
    spec=importlib.util.spec_from_file_location('frozen_trace200_durable_reader_only',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    diagnostic=m.bound(dict(path=str(DIAGNOSTIC),bytes=DIAGNOSTIC.stat().st_size,sha256=EXPECTED_DIAGNOSTIC))
    assert diagnostic['returncode']==0
    previous=json.loads(diagnostic['stdout'])
    assert previous['systemd']['MainPID']=='0' and previous['systemd']['ExecMainStatus']=='0' and previous['systemd']['Result']=='success'
    assert previous['once']['program_sha256']==PROGRAM and previous['result.json']['returncode']==0
    assert previous['result.json']['files']['shared-prefix-credit200.json']==dict(bytes=CREDIT_BYTES,sha256=CREDIT_SHA)
    assert previous['files']['shared-prefix-credit200.json']['bytes']==CREDIT_BYTES
    assert CREDIT_BYTES<358400
    code=m.read_handle(PROGRAM)
    old="('shared-prefix-credit200.json',200000)";assert code.count(old)==1
    code=code.replace(old,"('shared-prefix-credit200.json',204976)")
    route_spec=importlib.util.spec_from_file_location('retained_trace_handle_only',m.ROUTE);route=importlib.util.module_from_spec(route_spec);route_spec.loader.exec_module(route)
    os.environ['TREE_MGMT_CONTROL_PATH']='/tmp/tree-mgmt-3fw9a1ab/control'
    start=m.publish(m.NAME+'_TERMINAL_SIZE_REPAIR_ATTEMPT.json',m.enc(dict(status='EXACT_DURABLE_SCALAR_SIZE_QUALIFIED_READ_ONCE',diagnostic=m.pin(DIAGNOSTIC),reader_source=dict(path=str(Path(__file__).resolve()),bytes=Path(__file__).stat().st_size,sha256=m.sha(Path(__file__).read_bytes())),program_sha256=PROGRAM,credit_exact_bytes=CREDIT_BYTES,credit_exact_sha256=CREDIT_SHA,scanner_resubmitted=False,input_JSON_rescanned=False,automatic_retry=False)))
    result=route.run(m.remote_command(code),target='h100-10',timeout=60)
    response=m.preserve_response('TERMINAL_SIZE_REPAIR',result)
    assert result.returncode==0,'Preserved exact response; no automatic retry'
    observed=json.loads(result.stdout);payloads=observed.pop('payloads')
    assert observed['unit']==m.AUDIT_UNIT and observed['once']['program_sha256']==PROGRAM
    assert observed['systemd']['InvocationID']==previous['systemd']['InvocationID'] and observed['systemd']['MainPID']=='0' and observed['systemd']['SubState']=='exited' and observed['systemd']['ExecMainStatus']=='0' and observed['systemd']['Result']=='success'
    assert observed['result.json']==previous['result.json']
    for name,raw in payloads.items():
        record=observed['result.json']['files'][name]
        assert len(raw.encode())==record['bytes'] and m.sha(raw.encode())==record['sha256']
    assert observed['files']['shared-prefix-credit200.json']==dict(bytes=CREDIT_BYTES,sha256=CREDIT_SHA)
    saved={}
    for key,suffix in [('stdout.json','_DURABLE_STDOUT.json'),('stderr.txt','_DURABLE_STDERR.txt'),('shared-prefix-credit200.json','_SHARED_PREFIX_CREDIT200.json')]:saved[key]=m.publish(m.NAME+suffix,payloads[key].encode())
    observed.update(terminal=True,original_response=response,saved_payloads=saved,exact_scalar_size_repair=start)
    product=m.publish(m.NAME+'_TERMINAL_READBACK.json',m.enc(observed))
    print(json.dumps(dict(status='COMPLETED_EXISTING_DURABLE_TRACE_HANDLE_READ',terminal=product,systemd=observed['systemd'],files=saved,program_resubmitted=False,input_JSON_rescanned=False)))
if __name__=='__main__':main()
