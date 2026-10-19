"""Exercise real Edge fetch/WebSocket against a local Jupyter protocol fixture."""
import base64
import hashlib
import io
import json
import os
import signal
import struct
import subprocess
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from playwright.sync_api import sync_playwright

from dcu import BridgeError, Jupyter, base_url, file_bytes, remote_path


class Fixture(BaseHTTPRequestHandler):
    files = {}
    terminals = {}
    failures = False

    def log_message(self, *_):
        pass

    def reply(self, data, code=200):
        payload = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        if self.command == "GET":
            self.send_header("Cache-Control", "public, max-age=3600")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def body(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))

    def file_path(self):
        return unquote(urlsplit(self.path).path.split("/api/contents/", 1)[1])

    def do_GET(self):
        if "/terminals/websocket/" in self.path:
            return self.websocket()
        if self.path.endswith("/lab"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Set-Cookie", "_xsrf=fixture-token; Path=/")
            self.end_headers()
            self.wfile.write(b"<html>Jupyter fixture</html>")
        elif self.failures:
            self.reply({}, 403)
        elif "/api/contents/" in self.path:
            path = self.file_path()
            if not path:
                self.reply({"type": "directory", "content": []})
            elif path not in self.files:
                self.reply({}, 404)
            else:
                self.reply(self.files[path])
        elif self.path.endswith("/api/terminals"):
            self.reply([])
        else:
            self.reply({}, 404)

    def do_PUT(self):
        if self.headers.get("X-XSRFToken") != "fixture-token":
            return self.reply({}, 403)
        path = self.file_path()
        model = self.body()
        model["path"] = path
        self.files[path] = model
        self.reply(model, 201)

    def do_POST(self):
        if self.headers.get("X-XSRFToken") != "fixture-token":
            return self.reply({}, 403)
        name = str(len(self.terminals) + 1)
        self.terminals[name] = {"stop": threading.Event(), "process": None}
        self.reply({"name": name}, 201)

    def do_DELETE(self):
        name = self.path.rsplit("/", 1)[-1]
        terminal = self.terminals[name]
        terminal["stop"].set()
        proc = terminal["process"]
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=3)
        self.reply(None, 204)

    def frame(self, data):
        payload = json.dumps(data, ensure_ascii=False).encode()
        length = len(payload)
        header = bytes([0x81, length]) if length < 126 else b"\x81\x7e" + struct.pack("!H", length)
        self.wfile.write(header + payload)
        self.wfile.flush()

    def read_exact(self, count):
        data = self.rfile.read(count)
        if len(data) != count:
            raise EOFError()
        return data

    def read_frame(self):
        first, second = self.read_exact(2)
        length = second & 127
        if length == 126:
            length = struct.unpack("!H", self.read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self.read_exact(8))[0]
        mask = self.read_exact(4) if second & 128 else None
        payload = self.read_exact(length)
        if first & 15 == 8:
            raise EOFError()
        if mask:
            payload = bytes(x ^ mask[i % 4] for i, x in enumerate(payload))
        return json.loads(payload)

    def websocket(self):
        key = self.headers["Sec-WebSocket-Key"]
        digest = hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
        self.send_response(101)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", base64.b64encode(digest).decode())
        self.end_headers()
        terminal = self.terminals[self.path.rsplit("/", 1)[-1]]
        try:
            self.frame(["setup", {}])
            data = self.read_frame()
            if data[0] == "set_size":
                data = self.read_frame()
            self.frame(["stdout", "root@fixture# " + data[1]])
            proc = subprocess.Popen(["/bin/bash", "-c", data[1].rstrip("\r")],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, start_new_session=True)
            terminal["process"] = proc
            # Every marker is split across frames, including the exit code.
            for char in iter(lambda: proc.stdout.read(1), ""):
                if terminal["stop"].is_set():
                    break
                self.frame(["stdout", char])
            proc.stdout.close()
            proc.wait(timeout=3)
            terminal["stop"].wait(5)
        except (BrokenPipeError, ConnectionResetError, EOFError):
            pass


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(channel="msedge", headless=True)
        cls.page = cls.browser.new_page()
        root = f"http://127.0.0.1:{cls.server.server_port}/jupyter-forward/fixture/"
        cls.config = {"lab_url": root + "lab", "base_url": root}
        cls.page.goto(cls.config["lab_url"])
        cls.api = Jupyter(cls.page, cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def test_01_upload_binary_unicode_and_overwrite(self):
        source = Path(self.tmp.name) / "input.bin"
        source.write_bytes(bytes(range(256)) + "中文\n".encode())
        result = self.api.upload(source, "目录/a b.bin")
        self.assertTrue(result["verified"])
        self.assertEqual(file_bytes(self.api.contents("目录/a b.bin")), source.read_bytes())
        with self.assertRaisesRegex(BridgeError, "已存在"):
            self.api.upload(source, "目录/a b.bin")
        source.write_bytes(b"new")
        self.api.upload(source, "目录/a b.bin", overwrite=True)
        self.assertEqual(file_bytes(self.api.contents("目录/a b.bin")), b"new")

    def test_02_expired_login(self):
        Fixture.failures = True
        try:
            with self.assertRaisesRegex(BridgeError, "403"):
                self.api.probe()
        finally:
            Fixture.failures = False

    def test_03_exec_fragmented_unicode_and_nonzero(self):
        log = Path(self.tmp.name) / "run.log"
        with redirect_stdout(io.StringIO()) as output:
            code = self.api.execute("printf '中文 and quotes: \\\"x\\\"\\n'; exit 7", "/tmp", 15, log)
        self.assertEqual(code, 7)
        self.assertIn("中文", output.getvalue())
        self.assertEqual(log.read_text(), output.getvalue())
        self.assertNotIn("__DCU_", output.getvalue())
        self.assertEqual(json.loads(log.with_suffix(".json").read_text())["status"], "completed")

    def test_04_timeout_cleans_owned_terminal(self):
        log = Path(self.tmp.name) / "timeout.log"
        with redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(BridgeError, "超过"):
                self.api.execute("printf 'before timeout\\n'; sleep 10", "/tmp", 0.5, log)
        self.assertEqual(json.loads(log.with_suffix(".json").read_text())["status"], "timeout")
        self.assertIn("before timeout", log.read_text())
        self.assertTrue(list(Fixture.terminals.values())[-1]["stop"].is_set())
        self.assertIsNotNone(list(Fixture.terminals.values())[-1]["process"].poll())

    def test_05_directory_listing_refreshes_after_remote_change(self):
        Fixture.files["refresh"] = {"type": "directory", "content": [{"name": "before"}]}
        self.assertEqual(self.api.contents("refresh")["content"][0]["name"], "before")
        Fixture.files["refresh"] = {"type": "directory", "content": [{"name": "after"}]}
        self.assertEqual(self.api.contents("refresh")["content"][0]["name"], "after")


class PathTests(unittest.TestCase):
    def test_proxy_url_and_path_encoding(self):
        self.assertEqual(base_url("https://example.org/proxy/id/lab/tree/a?x=1"),
                         "https://example.org/proxy/id/")
        self.assertEqual(remote_path("/a b/中文.py"), "a%20b/%E4%B8%AD%E6%96%87.py")
        for path in ("a/../secret", "a/./b", "bad\x00name", "a\\b"):
            with self.assertRaises(BridgeError):
                remote_path(path)


if __name__ == "__main__":
    unittest.main()
