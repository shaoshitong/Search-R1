"""New-stage bounded FIFO admission for the unchanged opaque HTTP relay.

Workers start after upstream setns, never before it. CPU-only; no data fallback.
Default downstream concurrency 8 leaves headroom in the retained upstream16.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import socket
import socketserver
import sys
import threading
import time
try:
    import retrieval_relay as core
except ModuleNotFoundError:
    from . import retrieval_relay as core

_CORE_PATH = Path(core.__file__).resolve()
_HTTP_CONNECTION = core.http.client.HTTPConnection
_REQUEST = threading.local()
_CONNECTION_LOCK = threading.Lock()
_CONNECTION_USERS = 0


class DeadlineHTTPConnection(_HTTP_CONNECTION):
    """Preserve HTTP bytes while every socket operation shares one deadline."""
    def __init__(self, *args, **kwargs):
        self._budget = getattr(_REQUEST, 'value', None)
        self._limit = kwargs.get('timeout', 50)
        super().__init__(*args, **kwargs)

    def _timeout(self):
        if self._budget is None:return self._limit
        remaining = self._budget['deadline'] - self._budget['clock']()
        if remaining <= 0:raise TimeoutError('Relay absolute request deadline exhausted')
        return min(self._limit, remaining)

    def connect(self):
        self.timeout = self._timeout()
        super().connect()
        if self._budget is not None:
            with self._budget['lock']:
                self._budget['sockets'].append(self.sock)
                if self._budget['expired']:
                    try:self.sock.shutdown(socket.SHUT_RDWR)
                    except OSError:pass

    def request(self, *args, **kwargs):
        self.timeout = self._timeout()
        if self.sock is not None:self.sock.settimeout(self.timeout)
        return super().request(*args, **kwargs)

    def getresponse(self):
        if self.sock is not None:self.sock.settimeout(self._timeout())
        return super().getresponse()


class ObservedRelayHandler(core.RelayHandler):
    def send_bytes(self, status, body, headers=(), reason=None):
        self.server.record_status(status)
        return super().send_bytes(status, body, headers, reason)


class QueuedRelayHTTPServer(core.RelayHTTPServer):
    def __init__(self, address, target, *, max_queued=128, queue_wait=45.0, listen_backlog=256,
                 request_budget=59.0, stats_file=None, clock=time.monotonic, **kwargs):
        core.require(max_queued > 0 and 0 < queue_wait < request_budget < 60 and listen_backlog > 0, 'Invalid relay queue bounds')
        self.pending = queue.Queue(maxsize=max_queued)
        self.max_queued, self.queue_wait, self.listen_backlog = max_queued, queue_wait, listen_backlog
        self.request_budget, self.clock = request_budget, clock
        self.max_workers = kwargs.get('max_workers', 16)
        self.stats_file = Path(stats_file) if stats_file else None
        self._closing = threading.Event()
        self._stats_lock = threading.Lock()
        self._stats = dict(accepted=0, completed=0, active=0, peak_active=0, peak_queued=0,
                           rejected_full=0, rejected_wait=0, rejected_shutdown=0, peak_queue_wait_seconds=0,
                           minimum_backend_timeout_seconds=kwargs.get('timeout', 60), http_status_counts={})
        self._next_stats = 0
        self._workers = []
        super().__init__(address, target, **kwargs)
        self.RequestHandlerClass = ObservedRelayHandler

    @property
    def timeout_seconds(self):
        value = self._configured_timeout
        state = getattr(_REQUEST, 'value', None)
        if state is not None and state['server'] is self:
            value = min(value, max(.001, state['deadline'] - state['clock']()))
            with self._stats_lock:
                self._stats['minimum_backend_timeout_seconds'] = min(self._stats['minimum_backend_timeout_seconds'], value)
        return value

    @timeout_seconds.setter
    def timeout_seconds(self, value):self._configured_timeout = value

    def server_activate(self):
        self.request_queue_size = self.listen_backlog
        super().server_activate()

    def stats(self):
        with self._stats_lock:
            return dict(self._stats, queued=self.pending.qsize(), max_workers=self.max_workers,
                        max_queued=self.max_queued, queue_wait_seconds=self.queue_wait, request_budget_seconds=self.request_budget,
                        http_status_counts=dict(self._stats['http_status_counts']))

    def record_status(self, status):
        with self._stats_lock:
            key = str(status)
            self._stats['http_status_counts'][key] = self._stats['http_status_counts'].get(key, 0) + 1

    def _change(self, **values):
        with self._stats_lock:
            for name, amount in values.items():self._stats[name] += amount
            self._stats['peak_active'] = max(self._stats['peak_active'], self._stats['active'])
            self._stats['peak_queued'] = max(self._stats['peak_queued'], self.pending.qsize())

    def _reject(self, request, reason):
        self.record_status(503)
        body = json.dumps(dict(error='relay_' + reason), separators=(',', ':')).encode()
        try:
            request.sendall(b'HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: '
                + str(len(body)).encode() + b'\r\nConnection: close\r\n\r\n' + body)
        except OSError:
            pass
        finally:
            self.shutdown_request(request)

    def process_request(self, request, client_address):
        if self._closing.is_set():
            self._change(rejected_shutdown=1);self._reject(request, 'shutdown');return
        # The accept thread may wait, but no thread is created for a queued
        # socket. Userspace is bounded by W active + Q queued + 1 admission.
        try:
            self.pending.put((request, client_address, self.clock()), timeout=self.queue_wait)
            self._change(accepted=1)
        except queue.Full:
            self._change(rejected_full=1);self._reject(request, 'queue_full')

    def _worker(self):
        while not self._closing.is_set():
            try:item = self.pending.get(timeout=.1)
            except queue.Empty:continue
            request, address, accepted = item
            try:
                if self._closing.is_set():
                    self._change(rejected_shutdown=1);self._reject(request, 'shutdown')
                elif self.clock() - accepted >= self.queue_wait:
                    self._change(rejected_wait=1);self._reject(request, 'queue_timeout')
                else:
                    elapsed = self.clock() - accepted
                    with self._stats_lock:
                        self._stats['peak_queue_wait_seconds'] = max(self._stats['peak_queue_wait_seconds'], elapsed)
                    state = dict(server=self, deadline=accepted + self.request_budget, clock=self.clock,
                                 sockets=[], lock=threading.Lock(), expired=False)
                    _REQUEST.value = state
                    request.settimeout(max(.001, state['deadline'] - self.clock()))
                    def expire(budget=state, client=request):
                        with budget['lock']:
                            budget['expired'] = True
                            targets = list(budget['sockets']) or [client]
                        for target in targets:
                            try:target.shutdown(socket.SHUT_RDWR)
                            except OSError:pass
                    timer = threading.Timer(max(.001, state['deadline'] - self.clock()), expire)
                    timer.daemon = True;timer.start()
                    self._change(active=1)
                    try:
                        # Run the original RelayHandler byte-for-byte forwarding
                        # and stdlib cleanup, without its nonblocking semaphore.
                        socketserver.ThreadingMixIn.process_request_thread(self, request, address)
                    finally:
                        timer.cancel();del _REQUEST.value
                        self._change(active=-1, completed=1)
            finally:
                self.pending.task_done()

    def _write_stats(self):
        if self.stats_file is not None:
            self.stats_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.stats_file.with_name(self.stats_file.name + '.tmp-' + str(os.getpid()))
            temporary.write_text(json.dumps(dict(self.stats(), epoch=time.time())))
            os.replace(temporary, self.stats_file)

    def service_actions(self):
        if time.monotonic() >= self._next_stats:
            self._write_stats();self._next_stats = time.monotonic() + 5

    def serve_forever(self, poll_interval=.2):
        global _CONNECTION_USERS
        core.require(not self._workers, 'A relay server may serve only once')
        with _CONNECTION_LOCK:
            if _CONNECTION_USERS == 0:core.http.client.HTTPConnection = DeadlineHTTPConnection
            _CONNECTION_USERS += 1
        # Upstream core.main has now entered the exact backend namespace.
        self._workers = [threading.Thread(target=self._worker, name='relay-fifo-%d' % i, daemon=True)
                         for i in range(self.max_workers)]
        for worker in self._workers:worker.start()
        try:
            super().serve_forever(poll_interval=poll_interval)
        finally:
            self._closing.set()
            with _CONNECTION_LOCK:
                _CONNECTION_USERS -= 1
                if _CONNECTION_USERS == 0:core.http.client.HTTPConnection = _HTTP_CONNECTION

    def server_close(self):
        self._closing.set()
        while True:
            try:request, _, _ = self.pending.get_nowait()
            except queue.Empty:break
            self._change(rejected_shutdown=1);self._reject(request, 'shutdown');self.pending.task_done()
        super().server_close()
        self._write_stats()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--max-queued', type=int, default=128)
    parser.add_argument('--queue-wait', type=float, default=45.0)
    parser.add_argument('--request-budget', type=float, default=59.0)
    parser.add_argument('--listen-backlog', type=int, default=256)
    parser.add_argument('--stats-file')
    options, remainder = parser.parse_known_args()
    role = remainder[0] if remainder else None
    if '--max-workers' not in remainder:remainder += ['--max-workers', '8' if role == 'downstream' else '16']
    if '--timeout' not in remainder:remainder += ['--timeout', '50' if role == 'downstream' else '40']
    server_class, publisher, original_path = core.RelayHTTPServer, core.publish_ready, core.__file__
    bounds = dict(max_queued=options.max_queued, queue_wait=options.queue_wait, listen_backlog=options.listen_backlog,
                  request_budget=options.request_budget)
    def server(*args, **kwargs):return QueuedRelayHTTPServer(*args, **kwargs, **bounds, stats_file=options.stats_file)
    def publish(path, value):
        value.update(transport_core_sha256=hashlib.sha256(_CORE_PATH.read_bytes()).hexdigest(),
                     backpressure=dict(bounds, fixed_worker_pool=True, overload_status=503, no_empty_retrieval_fallback=True))
        publisher(path, value)
    try:
        core.RelayHTTPServer = server
        core.publish_ready = publish
        core.__file__ = __file__
        with_args = [sys.argv[0], *remainder]
        old_argv, sys.argv = sys.argv, with_args
        try:core.main()
        finally:sys.argv = old_argv
    finally:
        core.RelayHTTPServer, core.publish_ready, core.__file__ = server_class, publisher, original_path


if __name__ == '__main__':main()
