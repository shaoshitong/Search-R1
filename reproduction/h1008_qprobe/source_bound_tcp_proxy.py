"""Task-only SSH byte transport with explicit source IP and bounded TCP segments."""
import argparse
import os
import selectors
import socket
import errno
import sys

p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--host', required=True)
p.add_argument('--port', type=int, default=22)
p.add_argument('--mss', type=int, default=1200)
a = p.parse_args()
assert 536 <= a.mss <= 1400
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, a.mss)
except OSError as e:
    if e.errno not in (errno.EINVAL, errno.ENOPROTOOPT):
        raise
    print('TCP_MAXSEG override unsupported; using source-bound system TCP', file=sys.stderr)
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
s.bind((a.source, 0))
s.settimeout(10)
s.connect((a.host, a.port))
s.settimeout(None)
events = selectors.DefaultSelector()
events.register(s, selectors.EVENT_READ)
events.register(0, selectors.EVENT_READ)
try:
    while True:
        ready = events.select(30)
        if not ready:
            raise TimeoutError('SSH transport idle for 30 seconds')
        for event, _ in ready:
            if event.fileobj == s:
                data = s.recv(65536)
                if not data:
                    raise SystemExit(0)
                offset = 0
                while offset < len(data):
                    offset += os.write(1, data[offset:])
            else:
                data = os.read(0, 65536)
                if not data:
                    events.unregister(0)
                    s.shutdown(socket.SHUT_WR)
                else:
                    s.sendall(data)
finally:
    s.close()
