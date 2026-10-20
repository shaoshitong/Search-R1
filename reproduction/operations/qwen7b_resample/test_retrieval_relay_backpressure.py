"""Real CPU HTTP bursts: reproduce retained 503, then verify bounded FIFO."""
import concurrent.futures
import contextlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import unittest
from . import retrieval_relay as core
from . import retrieval_relay_backpressure as queued


@contextlib.contextmanager
def serving(server):
    thread = threading.Thread(target=server.serve_forever, kwargs=dict(poll_interval=.01), daemon=True)
    thread.start()
    try:yield server
    finally:server.shutdown();server.server_close();thread.join(timeout=2)


def request(address, body=b'{"queries":["q"],"topk":3,"return_scores":true}'):
    connection = http.client.HTTPConnection(*address, timeout=4)
    try:
        connection.request('POST', '/retrieve', body=body, headers={'Content-Type': 'application/json'})
        response = connection.getresponse()
        return response.status, response.getheader('Content-Type'), response.read()
    finally:connection.close()


def wait_for(predicate):
    end = time.monotonic() + 3
    while not predicate():
        if time.monotonic() > end:raise AssertionError('CPU HTTP fixture failed to reach barrier')
        time.sleep(.005)


class Backend(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


class BackpressureTests(unittest.TestCase):
    def test_retained_nonblocking_limit_can_exhaust_all_three_attempts(self):
        gate = threading.Event();mutex = threading.Lock();entered = [0]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                with mutex:entered[0] += 1
                gate.wait(3)
                self.send_response(200);self.send_header('Content-Length', '2');self.end_headers();self.wfile.write(b'{}')
        with serving(Backend(('127.0.0.1', 0), Handler)) as backend, \
             serving(core.RelayHTTPServer(('127.0.0.1', 0), backend.server_address, max_workers=16, timeout=3)) as old, \
             concurrent.futures.ThreadPoolExecutor(max_workers=16) as clients:
            held = [clients.submit(request, old.server_address) for _ in range(16)]
            try:
                wait_for(lambda: entered[0] == 16)
                # Empty body avoids an unrelated client write race: this old
                # rejection occurs before parsing any HTTP bytes or query.
                statuses = [request(old.server_address, b'')[0] for _ in range(3)]
                self.assertEqual(statuses, [503, 503, 503])
            finally:gate.set()
            self.assertTrue(all(f.result()[0] == 200 for f in held))

    def test_96_client_two_hop_burst_is_opaque_and_bounded_at_eight(self):
        mutex = threading.Lock();seen = [];active = [0];peak = [0]
        answer = b'{"result":[[{"id":1,"score":0.12345678901234567},{"id":2,"score":-0.22},{"id":3,"score":-2.0}]]}'
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):pass
            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                with mutex:
                    seen.append(body);active[0] += 1;peak[0] = max(peak[0], active[0])
                time.sleep(.08)
                self.send_response(207);self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(answer)));self.end_headers();self.wfile.write(answer)
                with mutex:active[0] -= 1
        token = b'test-only-opaque-token-012345678901234567890'
        with serving(Backend(('127.0.0.1', 0), Handler)) as backend, \
             serving(core.RelayHTTPServer(('127.0.0.1', 0), backend.server_address, expected_token=token, max_workers=16, timeout=3)) as old_upstream:
            server = queued.QueuedRelayHTTPServer(('127.0.0.1', 0), old_upstream.server_address,
                outgoing_token=token, max_workers=8, max_queued=128, queue_wait=45, request_budget=59,
                listen_backlog=256, timeout=50, clock=lambda: time.monotonic() * 20)
            self.assertEqual(server._workers, [], 'No worker may start before upstream setns')
            with serving(server) as downstream, concurrent.futures.ThreadPoolExecutor(max_workers=96) as clients:
                barrier = threading.Barrier(96)
                payloads = [b'{"queries":["opaque\\u0000-' + str(i).encode() + b'"],"topk":3,"return_scores":true}' for i in range(96)]
                def send(body):barrier.wait(timeout=3);return request(downstream.server_address, body)
                results = list(clients.map(send, payloads))
                self.assertEqual(results, [(207, 'application/json; charset=utf-8', answer)] * 96)
                self.assertCountEqual(seen, payloads)
                stats = downstream.stats()
                self.assertEqual((stats['rejected_full'], stats['rejected_wait']), (0, 0))
                self.assertLessEqual(stats['peak_active'], 8)
                self.assertLessEqual(stats['peak_queued'], 128)
                self.assertGreater(stats['peak_queue_wait_seconds'], 8, 'Scaled queue must exceed the rejected short 5s gate')
                self.assertLess(stats['minimum_backend_timeout_seconds'], 50, 'Queue time must reduce the remaining backend budget')
                self.assertLessEqual(peak[0], 8)
                self.assertEqual(len(downstream._workers), 8)

    def test_queue_full_and_queue_expiry_are_real_503_not_empty_success(self):
        gate = threading.Event();entered = threading.Event()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']));entered.set();gate.wait(3)
                self.send_response(200);self.send_header('Content-Length', '4');self.end_headers();self.wfile.write(b'real')
        with serving(Backend(('127.0.0.1', 0), Handler)) as backend, \
             serving(queued.QueuedRelayHTTPServer(('127.0.0.1', 0), backend.server_address,
                 max_workers=1, max_queued=1, queue_wait=.15, timeout=3)) as server, \
             concurrent.futures.ThreadPoolExecutor(max_workers=2) as clients:
            first = clients.submit(request, server.server_address)
            try:
                self.assertTrue(entered.wait(2))
                second = clients.submit(request, server.server_address)
                wait_for(lambda: server.stats()['accepted'] == 2)
                full = request(server.server_address)
                self.assertEqual(full[0], 503);self.assertIn(b'relay_queue_full', full[2])
            finally:gate.set()
            self.assertEqual(first.result()[2], b'real')
            expired = second.result()
            self.assertEqual(expired[0], 503);self.assertIn(b'relay_queue_timeout', expired[2])
            self.assertEqual((server.stats()['rejected_full'], server.stats()['rejected_wait']), (1, 1))

    def test_absolute_budget_interrupts_slow_drip_body_without_success(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200);self.send_header('Content-Length', '1000');self.end_headers()
                try:
                    for _ in range(20):self.wfile.write(b'partial');self.wfile.flush();time.sleep(.15)
                except OSError:pass
        with serving(Backend(('127.0.0.1', 0), Handler)) as backend, \
             serving(queued.QueuedRelayHTTPServer(('127.0.0.1', 0), backend.server_address,
                 max_workers=1, max_queued=2, queue_wait=.5, request_budget=1, timeout=50)) as server:
            start = time.monotonic();status, _, body = request(server.server_address)
            self.assertLess(time.monotonic() - start, 1.8)
            self.assertEqual(status, 502)
            self.assertIn(b'relay_backend_failure', body)


if __name__ == '__main__':unittest.main()
