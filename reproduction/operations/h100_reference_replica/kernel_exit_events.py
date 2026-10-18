"""Passive Linux proc-connector exit decoding; never ptrace/signal a trainer.

ABI source: the execution host's /usr/include/linux/{cn_proc,connector}.h,
Linux5.15. PROC_EVENT_EXIT.exit_code is a raw wait status; exit_signal is the
parent notification signal, not the process's termination signal.
"""
import base64
import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time

NL=struct.Struct('=IHHII')
CN=struct.Struct('=IIIIHH')
COMMON=struct.Struct('=IIQ')
EXIT=0x80000000


def decode_status(raw):
    if os.WIFEXITED(raw):return dict(raw_wait_status=raw,exit_code=os.WEXITSTATUS(raw),termination_signal=None)
    if os.WIFSIGNALED(raw):return dict(raw_wait_status=raw,exit_code=None,termination_signal=os.WTERMSIG(raw))
    raise ValueError('Kernel status is not an actual exit/termination')


def decode_packet(packet):
    events=[];offset=0
    while offset<len(packet):
        if len(packet)-offset<NL.size:raise ValueError('Truncated netlink header')
        length,kind,flags,seq,port=NL.unpack_from(packet,offset)
        if length<NL.size or offset+length>len(packet):raise ValueError('Truncated netlink message')
        raw=packet[offset:offset+length];body=raw[NL.size:]
        if kind==2:
            error=struct.unpack_from('=i',body)[0]
            if error:raise OSError(-error,os.strerror(-error))
        elif kind==4:raise OSError(errno.ENOBUFS,'Kernel netlink overrun; do not infer an exit')
        elif kind==3:
            if len(body)<CN.size:raise ValueError('Truncated connector header')
            idx,val,cseq,ack,size,cflags=CN.unpack_from(body)
            if (idx,val)==(1,1):
                data=body[CN.size:CN.size+size]
                if len(data)!=size or len(data)<COMMON.size:raise ValueError('Truncated proc event')
                what,cpu,stamp=COMMON.unpack_from(data)
                if what==0:
                    error=struct.unpack_from('=I',data,COMMON.size)[0]
                    events.append(dict(kind='ack',error=error))
                elif what==EXIT:
                    if len(data)<COMMON.size+16:raise ValueError('Truncated process exit')
                    pid,tgid,status,notification=struct.unpack_from('=IIII',data,COMMON.size)
                    events.append(dict(kind='exit',pid=pid,tgid=tgid,cpu=cpu,kernel_monotonic_ns=stamp,
                        parent_notification_signal=notification,**decode_status(status),
                        kernel_message_base64=base64.b64encode(raw).decode(),kernel_message_sha256=hashlib.sha256(raw).hexdigest()))
        offset+=(length+3)&~3
    return events


class Connector:
    def __init__(self):self.sock=None;self.subscribed=False
    def request(self,operation):
        payload=CN.pack(1,1,1,0,4,0)+struct.pack('=I',operation)
        self.sock.sendto(NL.pack(NL.size+len(payload),3,0,1,self.sock.getsockname()[0])+payload,(0,0))
    def receive(self,timeout):
        self.sock.settimeout(timeout);raw,address=self.sock.recvfrom(1024*1024)
        if address[0]!=0:raise ValueError('Only actual kernel-origin proc events are accepted')
        return decode_packet(raw)
    def __enter__(self):
        self.sock=socket.socket(socket.AF_NETLINK,socket.SOCK_DGRAM,11)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,1024*1024)
            self.sock.bind((0,1));self.request(1);self.subscribed=True
            end=time.monotonic()+3
            while time.monotonic()<end:
                for event in self.receive(max(.01,end-time.monotonic())):
                    if event['kind']=='ack':
                        if event['error']:raise OSError(event['error'],os.strerror(event['error']))
                        return self
            raise TimeoutError('Kernel did not acknowledge proc-connector subscription')
        except BaseException:
            self.close();raise
    def close(self):
        if self.sock is not None:
            try:
                if self.subscribed:self.request(2)
            finally:self.sock.close();self.sock=None;self.subscribed=False
    def __exit__(self,*unused):self.close()


def qualification():
    """Two tiny owned CPU children provide independent wait() ground truth."""
    children={};captured={}
    try:
        with Connector()as connector:
            for code in (0,7):
                child=subprocess.Popen([sys.executable,'-S','-c',
                    'import time,sys;time.sleep(.4);sys.exit('+str(code)+')'],
                    stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                children[child.pid]=(child,code)
            end=time.monotonic()+6
            while len(captured)<len(children)and time.monotonic()<end:
                try:events=connector.receive(max(.01,end-time.monotonic()))
                except socket.timeout:break
                for event in events:
                    if event['kind']=='exit'and event['pid']==event['tgid']and event['pid']in children:
                        captured[event['pid']]=event
            waits={pid:child.wait(timeout=2)for pid,(child,code)in children.items()}
            if set(captured)!=set(children):raise RuntimeError('Owned CPU child kernel exit events missing')
            for pid,(child,expected)in children.items():
                if waits[pid]!=expected or captured[pid]['exit_code']!=expected or captured[pid]['termination_signal']is not None:
                    raise ValueError('Kernel raw status differs from actual owned-child wait()')
            return dict(status='OWNED_CPU_KERNEL_EXIT_CAPTURE_0_AND_7_PASS',
                tests=[dict(expected_exit=children[pid][1],actual_parent_wait=waits[pid],event=event)for pid,event in captured.items()],
                actual_training_process_touched=False,gpu_or_model_test=False,kernel_policy_changed=False)
    finally:
        for child,code in children.values():
            # These are only the tiny owned test children, never a trainer.
            child.wait(timeout=3)


def qualify_once(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);result=root/'RESULT.json';intent=root/'INTENT.json'
    source=Path(__file__);source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    host=socket.gethostname();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if result.exists():
        value=json.loads(result.read_text())
        if value['source_sha256']!=source_sha or value['host']!=host or value['boot_id']!=boot:
            raise ValueError('Existing kernel qualification belongs to another source/host/boot')
        return dict(value,existing_result_reused_without_child_rerun=True)
    if intent.exists():raise RuntimeError('Existing qualification intent: inspect original outcome, never replay')
    common=dict(host=host,boot_id=boot,kernel=os.uname().release,source=str(source),source_sha256=source_sha,
        pid=os.getpid(),started_unix=time.time(),training_modified=False)
    with intent.open('x')as stream:json.dump(common,stream);stream.flush();os.fsync(stream.fileno())
    try:value=dict(common,**qualification())
    except (OSError,RuntimeError,ValueError,TimeoutError)as error:
        value=dict(common,status='KERNEL_PROC_EXIT_QUALIFICATION_UNAVAILABLE_NOT_TRAINING_FAILURE',error=repr(error),errno=getattr(error,'errno',None))
    value['completed_unix']=time.time()
    with result.open('x')as stream:json.dump(value,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
    return value


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--qualify-root',required=True);args=parser.parse_args()
    print(json.dumps(qualify_once(args.qualify_root)))
