"""Serve only public assets, on this computer. Inference runs in the browser."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, '.mjs': 'text/javascript', '.wasm': 'application/wasm'}

    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cache-Control', 'public, max-age=3600' if self.path.startswith(('/models/', '/runtime/')) else 'no-cache')
        super().end_headers()


if __name__ == '__main__':
    root = Path(__file__).resolve().parent / 'public'
    server = ThreadingHTTPServer(('127.0.0.1', 8765), partial(Handler, directory=str(root)))
    print('Mobile-O browser test: http://127.0.0.1:8765', flush=True)
    print('Press Ctrl+C to stop. No inference server or external API is used.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
