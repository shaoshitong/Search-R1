"""CPU-only real HTTP checks for opaque, authenticated retrieval forwarding."""

import contextlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest

from reproduction.operations.qwen7b_resample.retrieval_relay import AUTH_HEADER, RelayHTTPServer


@contextlib.contextmanager
def serving(server):
    thread = threading.Thread(target=server.serve_forever, kwargs=dict(poll_interval=.01), daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(server, method, path, body=None, headers=None):
    client = http.client.HTTPConnection(*server.server_address, timeout=3)
    try:
        client.request(method, path, body=body, headers=headers or {})
        response = client.getresponse()
        return response.status, response.getheader('Content-Type'), response.read()
    finally:
        client.close()


class RetrievalRelayTests(unittest.TestCase):
    def test_raw_body_scores_status_and_auth(self):
        seen = []
        health = b'{"ready":true,"pid":123,"encoder_dtype":"float16"}\n'
        answer = b'{ "result": [[{"id": 17, "score": 0.12345678901234567, "text": "\\u4e2d"}]] }\n'
        content_type = 'application/json; charset=utf-8'

        class Backend(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(health)))
                self.end_headers()
                self.wfile.write(health)

            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                seen.append((self.path, body, self.headers.get('Content-Type'), self.headers.get(AUTH_HEADER)))
                self.send_response(207)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(answer)))
                self.end_headers()
                self.wfile.write(answer)

        token = b'private-test-only-token-1234567890123456789'
        with serving(ThreadingHTTPServer(('127.0.0.1', 0), Backend)) as backend:
            with serving(RelayHTTPServer(('127.0.0.1', 0), backend.server_address,
                    expected_token=token, max_workers=2, timeout=2)) as upstream:
                with serving(RelayHTTPServer(('127.0.0.1', 0), upstream.server_address,
                        outgoing_token=token, max_workers=2, timeout=2)) as downstream:
                    payload = b' {"queries":["caf\xc3\xa9\\u0000\\u4e2d"], "topk":3,"return_scores":true}\n'
                    status, actual_type, body = request(downstream, 'POST', '/retrieve', payload,
                        {'Content-Type': content_type, AUTH_HEADER: 'incoming-header-must-not-pass'})
                    self.assertEqual((status, actual_type, body), (207, content_type, answer))
                    self.assertEqual(seen, [('/retrieve', payload, content_type, None)])
                    self.assertEqual(request(downstream, 'GET', '/health'), (200, content_type, health))
                    denied = request(upstream, 'POST', '/retrieve', payload,
                        {'Content-Type': content_type, AUTH_HEADER: 'wrong-token'})
                    self.assertEqual(denied[0], 403)
                    self.assertEqual(len(seen), 1, 'Unauthorized request must not reach the backend')

    def test_backend_failure_is_502(self):
        # A bound, non-listening socket supplies a deterministic refused port.
        import socket
        with socket.socket() as unavailable:
            unavailable.bind(('127.0.0.1', 0))
            with serving(RelayHTTPServer(('127.0.0.1', 0), unavailable.getsockname(), timeout=.2)) as relay:
                status, content_type, body = request(relay, 'GET', '/health')
                self.assertEqual(status, 502)
                self.assertEqual(content_type, 'application/json')
                self.assertIn(b'relay_backend_failure', body)
                self.assertNotIn(b'"ready":true', body)


if __name__ == '__main__':
    unittest.main()
