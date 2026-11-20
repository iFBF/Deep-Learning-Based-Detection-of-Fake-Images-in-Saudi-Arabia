"""Local V3.1 research portal; uses the frozen client and original loopback API."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
import argparse
import hashlib
import json
import os
import sys
import threading
import time

PORTAL_VERSION = 'real-abshar-v3.1-portal-1'
BASELINE_ID = 'real-abshar-m8-fbaa1c95818d'
MAX_BYTES = 10 * 1024 * 1024


def check_baseline_url(url):
    p = urlsplit(url)
    if (p.scheme != 'http' or p.hostname not in {'localhost', '127.0.0.1', '::1'}
            or p.username or p.password or p.path not in {'', '/'} or p.query or p.fragment):
        raise ValueError('Use the original local service URL, for example http://127.0.0.1:8000.')
    return url.rstrip('/')


def baseline_health(url):
    with urlopen(check_baseline_url(url) + '/health', timeout=5) as response:
        value = json.load(response)
    if value.get('status') != 'ready' or value.get('release_id') != BASELINE_ID:
        raise ValueError('The original service is not the expected ready M8 release.')
    return value


def verify_portal_files(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'portal_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files_sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Unsafe portal file path.')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError('Portal integrity check failed: ' + name)
    return manifest


class Application:
    def __init__(self, release, baseline_url):
        self.release = Path(release).resolve()
        self.baseline_url = check_baseline_url(baseline_url)
        baseline_health(self.baseline_url)
        sys.path.insert(0, str(self.release))
        import predict_v3
        self.client = predict_v3
        self.runtime = predict_v3.load_runtime(self.release)
        self.manifest = self.runtime['manifest']
        self.policy = self.runtime['policy']
        if (self.policy.get('version') != 'real-abshar-v3.1-development'
                or self.policy.get('swin_threshold') != 0.99
                or self.policy.get('supplemental_raw_logit_boundary') != 4.25
                or self.policy.get('baseline_release_id') != BASELINE_ID):
            raise ValueError('This portal requires the frozen V3.1 policy: Swin 0.99, supplemental 4.25.')
        self.lock = threading.Lock()

    def health(self):
        # Readiness is checked live, so a stopped original API is not shown as ready.
        baseline_health(self.baseline_url)
        return {'status': 'ready', 'portal_version': PORTAL_VERSION,
                'research_release_id': self.manifest['release_id'],
                'baseline_release_id': BASELINE_ID, 'research_only': True,
                'swin_threshold': self.policy['swin_threshold'],
                'supplemental_raw_boundary': self.policy['supplemental_raw_logit_boundary']}

    def analyze(self, data):
        started = time.perf_counter()
        # Reuse the exact locked model/preprocessing without modifying candidate files.
        result = self.client.predict(self.release, data, supplemental_only=True, runtime=self.runtime)
        requested = time.perf_counter()
        original = self.client.baseline_predict(self.baseline_url, data)
        baseline_ms = 1000 * (time.perf_counter() - requested)
        result['decision'] = self.client.fuse(original, result['supplemental_classifier']['raw_logit'],
                                               self.policy, result['image'])
        result['baseline_summary'] = {name: original[name]
                                      for name in ('release_id', 'classifier', 'detections', 'mask', 'decision')}
        result['scope'] = 'V3.1 four-model research screening through the unchanged original local service'
        result['selected_swin_threshold'] = self.policy['swin_threshold']
        result['portal_version'] = PORTAL_VERSION
        result['timing_ms'].update(baseline_http_roundtrip=baseline_ms,
                                  client_total=1000 * (time.perf_counter() - started),
                                  includes_supplemental_model_load=False)
        return result


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != 'nt'


class Handler(BaseHTTPRequestHandler):
    server_version = 'RealAbsharLocal/1'

    def setup(self):
        super().setup()
        self.connection.settimeout(180)

    def log_message(self, fmt, *args):
        # No filenames, image bytes, scores or personal data are written to logs.
        return

    def send(self, status, payload, content_type='application/json; charset=utf-8'):
        data = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode() if isinstance(payload, dict) else payload
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def allowed_origin(self):
        port = self.server.server_address[1]
        hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
        host = self.headers.get('Host', '')
        origin = self.headers.get('Origin')
        return host in hosts and (not origin or origin in {'http://' + h for h in hosts})

    def do_GET(self):
        if not self.allowed_origin():
            return self.send(403, {'error': 'Open the portal through its local address.'})
        if self.path in ('/', '/index.html'):
            return self.send(200, self.server.page, 'text/html; charset=utf-8')
        if self.path == '/favicon.ico':
            return self.send(204, b'', 'image/x-icon')
        if self.path == '/health':
            try:
                return self.send(200, self.server.app.health())
            except (HTTPError, URLError, OSError, ValueError):
                return self.send(503, {'error': 'The original service is unavailable. Keep it running on its configured port.'})
        return self.send(404, {'error': 'Route not found.'})

    def do_OPTIONS(self):
        return self.send(403, {'error': 'Use this portal from its own local browser page.'})

    def do_POST(self):
        if not self.allowed_origin():
            return self.send(403, {'error': 'Open the portal through its local address.'})
        if self.path != '/predict':
            return self.send(404, {'error': 'Route not found.'})
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/octet-stream':
            return self.send(415, {'error': 'Upload an image through the portal.'})
        if self.headers.get('Transfer-Encoding'):
            return self.send(400, {'error': 'A fixed upload size is required.'})
        try:
            size = int(self.headers.get('Content-Length', ''))
        except ValueError:
            return self.send(411, {'error': 'A valid upload size is required.'})
        if not 0 < size <= MAX_BYTES:
            return self.send(413, {'error': 'Choose an image of at most 10 MiB.'})
        if not self.server.app.lock.acquire(blocking=False):
            self.close_connection = True
            return self.send(409, {'error': 'An image is already being analyzed. Wait for it to finish.'})
        try:
            data = self.rfile.read(size)
            if len(data) != size:
                return self.send(400, {'error': 'The image upload was incomplete.'})
            return self.send(200, self.server.app.analyze(data))
        except (HTTPError, URLError, TimeoutError):
            return self.send(503, {'error': 'The original local service did not respond. Check its PowerShell window and retry.'})
        except (ValueError, OSError) as error:
            return self.send(400, {'error': str(error) or 'This image could not be processed.'})
        except Exception:
            return self.send(500, {'error': 'Analysis failed. Check the original service window and try another supported image.'})
        finally:
            self.server.app.lock.release()


def create_server(app, page, port, host='127.0.0.1', handler=Handler):
    server = Server((host, port), handler)
    server.app = app
    server.page = page
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--baseline-url', default='http://127.0.0.1:8000')
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Choose a port between 1 and 65535.')
    root = Path(__file__).resolve().parent
    try:
        verify_portal_files(root)
        app = Application(args.release, args.baseline_url)
        server = create_server(app, (root / 'portal_index.html').read_bytes(), args.port)
    except OSError as error:
        if getattr(error, 'winerror', None) == 10048 or getattr(error, 'errno', None) == 98:
            parser.exit(1, f'Portal port {args.port} is already in use. Check http://127.0.0.1:{args.port}/health or use --port 8002. Keep the original service URL unchanged.\n')
        parser.exit(1, f'Could not start the local portal: {error}\n')
    except Exception as error:
        parser.exit(1, f'Could not start the local portal: {error}\n')
    url = f'http://127.0.0.1:{args.port}'
    print('V3.1 local research portal ready:', url, flush=True)
    print('Original service:', app.baseline_url, '| Policy: Swin 0.99 OR supplemental raw logit 4.25 + original local flags.', flush=True)
    print('Keep both PowerShell windows open. Press Ctrl+C here to stop only this comparison portal.', flush=True)
    if args.open_browser:
        import webbrowser
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
