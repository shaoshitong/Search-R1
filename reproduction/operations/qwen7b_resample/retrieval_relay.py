"""Bounded stdlib HTTP transport for the existing original retrieval service.

Upstream binds its host-network listener before entering the verified source
network namespace. Downstream stays in its own host network and listens only
on loopback. Neither role imports a GPU library or owns the source retriever.
Request/response bodies are opaque bytes; no queries, scores or JSON are edited.
"""

import argparse
import ctypes
import hashlib
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import socketserver
import stat
import sys
import threading
import time
from urllib.parse import urlsplit


AUTH_HEADER = 'X-Search-G1-Relay-Token'
HOP_HEADERS = frozenset(('connection', 'keep-alive', 'proxy-authenticate',
    'proxy-authorization', 'te', 'trailer', 'transfer-encoding', 'upgrade'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_token(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600,
                'Token must be a regular 0600 file')
        require(info.st_uid == os.geteuid(), 'Token must belong to the current effective UID')
        value = os.read(fd, 4097).strip()
    finally:
        os.close(fd)
    require(32 <= len(value) <= 512 and all(chr(x).isalnum() and x < 128 or x in b'._~-' for x in value),
            'Token must contain 32..512 ASCII header-safe characters')
    return value


def birth(pid):
    return Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()[19]


def source_namespace(pid, expected_birth, expected_inode):
    """One startup identity check, including a second birth read around open."""
    require(birth(pid) == str(expected_birth), 'Source PID birth differs')
    fd = os.open('/proc/%d/ns/net' % pid, os.O_RDONLY | os.O_CLOEXEC)
    try:
        require(os.fstat(fd).st_ino == expected_inode, 'Source network namespace inode differs')
        require(birth(pid) == str(expected_birth), 'Source PID changed while opening its namespace')
        proof = dict(pid=pid, birth=str(expected_birth), netns_inode=expected_inode,
            cgroup=Path('/proc/%d/cgroup' % pid).read_text(), verified_epoch=time.time())
        return fd, proof
    except BaseException:
        os.close(fd)
        raise


def enter_network_namespace(fd):
    libc = ctypes.CDLL(None, use_errno=True)
    libc.setns.argtypes = (ctypes.c_int, ctypes.c_int)
    libc.setns.restype = ctypes.c_int
    if libc.setns(fd, 0x40000000) != 0:  # Linux CLONE_NEWNET; Python 3.10 has no os.setns.
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


class RelayHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = False

    def __init__(self, address, target, *, expected_token=None, outgoing_token=None,
                 max_workers=16, timeout=60, max_request_bytes=8 * 1024 ** 2,
                 max_response_bytes=64 * 1024 ** 2):
        require(max_workers > 0 and timeout > 0 and max_request_bytes > 0 and max_response_bytes > 0,
                'Relay bounds must be positive')
        self.target = target
        self.expected_token, self.outgoing_token = expected_token, outgoing_token
        self.timeout_seconds = timeout
        self.max_request_bytes, self.max_response_bytes = max_request_bytes, max_response_bytes
        self.slots = threading.BoundedSemaphore(max_workers)
        self.request_queue_size = max_workers * 2
        super().__init__(address, RelayHandler)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(self.timeout_seconds)
        return connection, address

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            try:
                request.sendall(b'HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        # Never dump request headers or bodies (including the private token).
        pass


class RelayHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_):
        pass

    def send_bytes(self, status, body, headers=(), reason=None):
        self.close_connection = True
        self.send_response(status, reason)
        for name, value in headers:
            if name.lower() not in HOP_HEADERS | {'content-length', 'server', 'date'}:
                self.send_header(name, value)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)

    def error_bytes(self, status, code):
        self.send_bytes(status, json.dumps(dict(error=code), separators=(',', ':')).encode(),
                        [('Content-Type', 'application/json')])

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def forward(self):
        server = self.server
        if server.expected_token is not None:
            try:
                supplied = self.headers.get(AUTH_HEADER, '').encode('ascii')
            except UnicodeEncodeError:
                supplied = b''
            if not hmac.compare_digest(supplied, server.expected_token):
                self.error_bytes(403, 'relay_auth_denied')
                return
        path = urlsplit(self.path)
        if path.scheme or path.netloc or (self.command, path.path) not in (('GET', '/health'), ('POST', '/retrieve')):
            self.error_bytes(404, 'relay_route_not_found')
            return
        if self.headers.get('Transfer-Encoding') is not None:
            self.error_bytes(400, 'relay_requires_content_length')
            return
        lengths = self.headers.get_all('Content-Length', [])
        try:
            require(len(lengths) <= 1, 'Ambiguous content length')
            length = int(lengths[0]) if lengths else 0
            require(length >= 0, 'Negative content length')
        except (ValueError, TypeError):
            self.error_bytes(400, 'relay_invalid_content_length')
            return
        if length > server.max_request_bytes:
            self.error_bytes(413, 'relay_request_too_large')
            return
        if self.command == 'POST' and not lengths:
            self.error_bytes(411, 'relay_requires_content_length')
            return
        connection = None
        try:
            body = self.rfile.read(length)
            require(len(body) == length, 'Truncated request body')
            headers = {key: self.headers[key] for key in ('Content-Type', 'Content-Encoding', 'Accept') if key in self.headers}
            if server.outgoing_token is not None:
                headers[AUTH_HEADER] = server.outgoing_token.decode('ascii')
            connection = http.client.HTTPConnection(*server.target, timeout=server.timeout_seconds)
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            expected_length = response.length
            output = response.read(server.max_response_bytes + 1)
            require(len(output) <= server.max_response_bytes, 'Backend response exceeds relay bound')
            require(expected_length is None or len(output) == expected_length, 'Truncated backend response')
            # No response JSON parsing, score conversion, document sorting or tokenization.
            self.send_bytes(response.status, output, response.getheaders(), response.reason)
        except (OSError, ValueError, http.client.HTTPException):
            self.error_bytes(502, 'relay_backend_failure')
        finally:
            if connection is not None:
                connection.close()


def process_proof():
    return dict(pid=os.getpid(), birth=birth(os.getpid()),
        cgroup=Path('/proc/self/cgroup').read_text(),
        netns_inode=os.stat('/proc/self/ns/net').st_ino)


def publish_ready(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('x') as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)  # A previous launch proof is never silently overwritten.
    finally:
        temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=('upstream', 'downstream'))
    parser.add_argument('--token-file', required=True)
    parser.add_argument('--ready-file', required=True)
    parser.add_argument('--log-file', help='Optional remote-only log opened after entering the retained mount view')
    parser.add_argument('--listen-host', help='Required internal IPv4 address for upstream; downstream is 127.0.0.1')
    parser.add_argument('--listen-port', type=int, help='Upstream default 0; downstream default 28831')
    parser.add_argument('--source-pid', type=int)
    parser.add_argument('--source-birth', '--source-start-ticks')
    parser.add_argument('--source-netns-inode', type=int)
    parser.add_argument('--backend-port', type=int, default=28831)
    parser.add_argument('--upstream-url', help='Downstream target http://INTERNAL_IP:ACTUAL_PORT')
    parser.add_argument('--upstream-ready-file', help='Optional downstream: inherit exact source/listener proof on shared NAS10')
    parser.add_argument('--max-workers', type=int, default=16)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--max-request-bytes', type=int, default=8 * 1024 ** 2)
    parser.add_argument('--max-response-bytes', type=int, default=64 * 1024 ** 2)
    args = parser.parse_args()
    if args.log_file:
        fd = os.open(args.log_file, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)
    require(sys.platform == 'linux', 'Relay processes run on remote Linux only')
    token = read_token(args.token_file)
    original_netns = os.stat('/proc/self/ns/net').st_ino
    shared = dict(max_workers=args.max_workers, timeout=args.timeout,
                  max_request_bytes=args.max_request_bytes, max_response_bytes=args.max_response_bytes)
    source, upstream_proof = None, None
    if args.role == 'upstream':
        require(args.listen_host and args.listen_host not in ('0.0.0.0', '127.0.0.1'),
                'Upstream requires an explicit internal host address')
        socket.inet_aton(args.listen_host)
        require(args.source_pid and args.source_birth and args.source_netns_inode,
                'Upstream requires exact source PID, birth and namespace inode')
        server = RelayHTTPServer((args.listen_host, args.listen_port or 0),
            ('127.0.0.1', args.backend_port), expected_token=token, **shared)
        try:
            fd, source = source_namespace(args.source_pid, args.source_birth, args.source_netns_inode)
            try:
                enter_network_namespace(fd)
            finally:
                os.close(fd)
            require(os.stat('/proc/self/ns/net').st_ino == source['netns_inode'], 'setns did not enter source network namespace')
        except BaseException:
            server.server_close()
            raise
    else:
        require(args.listen_host in (None, '127.0.0.1'), 'Downstream listener must be 127.0.0.1')
        require(args.upstream_ready_file or args.upstream_url, 'Downstream requires an upstream URL or ready proof')
        if args.upstream_ready_file:
            raw = Path(args.upstream_ready_file).read_bytes()
            upstream_proof = json.loads(raw)
            require(upstream_proof['role'] == 'upstream' and upstream_proof['status'] == 'RELAY_LISTENING',
                    'Invalid upstream listener proof')
            source = upstream_proof['source']
            target = upstream_proof['listen']
            if args.upstream_url:
                require(args.upstream_url.rstrip('/') == upstream_proof['listen_url'], 'Upstream URL differs from its ready proof')
            upstream_proof = dict(path=args.upstream_ready_file, sha256=hashlib.sha256(raw).hexdigest(),
                                  process=upstream_proof['process'], listen=target)
        else:
            parsed = urlsplit(args.upstream_url)
            require(parsed.scheme == 'http' and parsed.hostname and parsed.port
                    and parsed.path in ('', '/') and not parsed.query and not parsed.fragment
                    and parsed.username is None and parsed.password is None, 'Invalid upstream HTTP URL')
            target = dict(host=parsed.hostname, port=parsed.port)
            upstream_proof = dict(listen=target, source_identity_inherited=False)
        require(target['host'] not in ('0.0.0.0', '127.0.0.1') and 0 < int(target['port']) < 65536,
                'Invalid upstream internal listener address')
        server = RelayHTTPServer(('127.0.0.1', 28831 if args.listen_port is None else args.listen_port),
            (target['host'], int(target['port'])), outgoing_token=token, **shared)
    try:
        proof = dict(status='RELAY_LISTENING', role=args.role,
            listen=dict(host=server.server_address[0], port=server.server_address[1]),
            listen_url='http://%s:%d' % server.server_address,
            source_pid=source['pid'] if source else None,
            source_birth=source['birth'] if source else None,
            source_netns_inode=source['netns_inode'] if source else None,
            source=source, process=process_proof(), listener_netns_inode=original_netns,
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            bounds=shared, upstream_proof=upstream_proof, source_process_owned=False, epoch=time.time())
        publish_ready(args.ready_file, proof)
        print(json.dumps(proof, sort_keys=True), flush=True)

        def stop(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
