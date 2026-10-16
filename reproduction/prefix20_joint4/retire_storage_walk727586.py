"""Preserved once-only CPU operation after job727586 left the queue.

Already executed on the login node at1792172638.693056. Do not replay.
Only the isolated own du child was signalled, never a training process.
"""


def retire():
    import os,json,time,signal
    from pathlib import Path
    assert os.getuid()==21930
    pid=496581
    row=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
    assert int(row[1])==496506 and int(row[19])==88171688
    argv=Path('/proc/'+str(pid)+'/cmdline').read_bytes().split(bytes([0]))[:-1]
    assert argv==[b'du',b'-s',b'-B1',b'/data/user/jxu957/experiments'],repr(argv)
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='a27512a3-24a0-4ca3-aa92-98d2a5256d9a'
    os.kill(pid,signal.SIGTERM)
    print(json.dumps(dict(checked_unix=time.time(),stopped_only_own_isolated_storage_walk=pid,training_job_changed=False)))


if __name__=='__main__':retire()
