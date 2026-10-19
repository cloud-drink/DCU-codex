#!/usr/bin/env python3
"""Local Playwright bridge for an authenticated Jupyter Server."""
from __future__ import annotations

import argparse
import base64
import json
import math
import posixpath
import re
import shlex
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".dcu"


class BridgeError(RuntimeError):
    pass


def base_url(lab_url: str) -> str:
    url = urlsplit(lab_url)
    if url.scheme not in {"http", "https"} or not url.netloc or url.username:
        raise BridgeError("lab_url 必须是完整的 HTTP(S) 地址，且不能包含用户名。")
    path = re.sub(r"/lab(?:/.*)?$", "/", url.path)
    if not path.endswith("/"):
        path += "/"
    return urlunsplit((url.scheme, url.netloc, path, "", ""))


def remote_path(path: str) -> str:
    # Contents paths are relative to the Jupyter file-browser root.
    parts = path.strip("/").split("/")
    if any(p in {".", ".."} for p in parts) or "\x00" in path or "\\" in path:
        raise BridgeError("远端文件路径不能包含 .、..、反斜线或空字符。")
    return "/".join(quote(p, safe="") for p in parts)


def load_config(path: Path) -> dict:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BridgeError(f"无法读取配置 {path}: {exc}") from exc
    if not isinstance(config, dict) or not isinstance(config.get("lab_url"), str):
        raise BridgeError("配置需要包含字符串 lab_url。")
    config["base_url"] = base_url(config["lab_url"])
    return config


@contextmanager
def browser_page(config: dict, *, visible: bool = False, cdp: str | None = None):
    from playwright.sync_api import sync_playwright

    STATE.mkdir(mode=0o700, exist_ok=True)
    with sync_playwright() as playwright:
        context = None
        owned_page = None
        try:
            if cdp:
                browser = playwright.chromium.connect_over_cdp(cdp, timeout=15_000)
                if not browser.contexts:
                    raise BridgeError("CDP 浏览器没有可用的会话。")
                context = browser.contexts[0]
                page = next((p for p in context.pages
                             if p.url.startswith(config["base_url"])), None)
                if page is None:
                    owned_page = context.new_page()
                    page = owned_page
            else:
                context = playwright.chromium.launch_persistent_context(
                    str(STATE / "browser-profile"),
                    channel=config.get("channel", "msedge"),
                    headless=not visible,
                    accept_downloads=True,
                )
                page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(20_000)
            if not page.url.startswith(config["base_url"]):
                page.goto(config["lab_url"], wait_until="domcontentloaded", timeout=45_000)
            yield page
        finally:
            if cdp:
                # Never close an attached user's browser or an existing tab.
                if owned_page and not owned_page.is_closed():
                    owned_page.close()
            elif context:
                context.close()


FETCH = r"""async ({url, method, body}) => {
    const target = new URL(url);
    if (target.origin !== location.origin) {
        throw new Error('当前页面不在 Jupyter 站点；请先运行 login 完成登录。');
    }
    const cookie = document.cookie.split(';').map(x => x.trim())
        .find(x => x.startsWith('_xsrf='));
    const headers = {'Accept': 'application/json'};
    if (cookie) headers['X-XSRFToken'] = decodeURIComponent(cookie.slice(6));
    if (body !== null) headers['Content-Type'] = 'application/json';
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 20000);
    try {
        const response = await fetch(url, {method, headers, credentials: 'same-origin', cache: 'no-store',
            redirect: 'error', signal: controller.signal,
            body: body === null ? undefined : JSON.stringify(body)});
        const text = await response.text();
        let data = null;
        try { data = text ? JSON.parse(text) : null; } catch (_) {}
        return {status: response.status, data, isJson: !text || data !== null};
    } finally { clearTimeout(timer); }
}"""


class Jupyter:
    def __init__(self, page, config):
        self.page = page
        self.config = config

    def request(self, method: str, endpoint: str, body=None, *, allow_missing=False):
        result = self.page.evaluate(FETCH, {
            "url": self.config["base_url"] + endpoint,
            "method": method, "body": body,
        })
        status = result["status"]
        if status == 404 and allow_missing:
            return None
        if status in {401, 403}:
            raise BridgeError(f"Jupyter 返回 HTTP {status}，请运行 login；若已登录，检查权限或 XSRF。")
        if not 200 <= status < 300:
            raise BridgeError(f"Jupyter {method} {endpoint} 返回 HTTP {status}。")
        if not result["isJson"]:
            raise BridgeError("接口返回了网页，可能已跳转到登录页，或转发路径不正确。")
        return result["data"]

    def contents(self, path, *, content=True, allow_missing=False):
        return self.request("GET", "api/contents/" + remote_path(path)
                            + ("?content=1" if content else "?content=0"),
                            allow_missing=allow_missing)

    def probe(self):
        model = self.contents("", content=False)
        if not isinstance(model, dict) or model.get("type") != "directory":
            raise BridgeError("未取得 Jupyter 根目录，登录状态尚不可用。")
        return model

    def upload(self, local: Path, remote: str, overwrite=False):
        if not remote.strip("/"):
            raise BridgeError("上传目标必须是文件路径。")
        content = base64.b64encode(local.read_bytes()).decode("ascii")
        existing = self.contents(remote, content=False, allow_missing=True)
        if existing:
            if existing.get("type") == "directory":
                raise BridgeError("上传目标是目录，请填写完整远端文件名。")
            if not overwrite:
                raise BridgeError("远端文件已存在；确认需要替换时添加 --overwrite。")
        saved = self.request("PUT", "api/contents/" + remote_path(remote), {
            "type": "file", "format": "base64", "content": content,
        })
        # Read back: successful HTTP alone does not verify the saved contents.
        actual = file_bytes(self.contents(remote))
        if actual != base64.b64decode(content):
            raise BridgeError("上传后的文件校验失败。")
        return {"path": saved["path"], "bytes": len(actual), "verified": True}

    def execute(self, command: str, cwd: str, timeout: float, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch()
        terminal = self.request("POST", "api/terminals", {})
        name = terminal.get("name") if isinstance(terminal, dict) else None
        if not name:
            raise BridgeError("终端接口未返回终端名称。")
        token = uuid.uuid4().hex
        start, end = "__DCU_START_" + token + "__", "__DCU_END_" + token + "__"
        # Split markers in the input so shell echo cannot mimic actual results.
        wrapper = (
            "stty -echo; printf '\\n%s%s\\n' '__DCU_START_' " + shlex.quote(token + "__")
            + "; (cd " + shlex.quote(cwd) + " && /bin/bash -lc " + shlex.quote(command)
            + "); __dcu_rc=$?; printf '\\n%s%s:%s\\n' '__DCU_END_' "
            + shlex.quote(token + "__") + ' "$__dcu_rc"\r'
        )
        status = "error"
        exit_code = None
        elapsed_start = time.monotonic()
        try:
            self.page.evaluate(START_TERMINAL, {
                "url": self.config["base_url"] + "terminals/websocket/" + quote(name, safe=""),
                "command": wrapper, "start": start, "end": end,
            })
            with log_path.open("w", encoding="utf-8") as log:
                while True:
                    update = self.page.evaluate("""() => {
                        const s = window.__dcuTerminal;
                        const chunk = s.pending; s.pending = '';
                        return {chunk, done:s.done, code:s.code, error:s.error};
                    }""")
                    if update["chunk"]:
                        print(update["chunk"], end="", flush=True)
                        log.write(update["chunk"])
                        log.flush()
                    if update["done"]:
                        exit_code = update["code"]
                        status = "completed"
                        return exit_code
                    if update["error"]:
                        raise BridgeError(update["error"])
                    if time.monotonic() - elapsed_start > timeout:
                        status = "timeout"
                        raise BridgeError(f"命令超过 {timeout:g} 秒，已请求中断；日志：{log_path}")
                    self.page.wait_for_timeout(100)
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        finally:
            try:
                tail = self.page.evaluate("""() => {
                    const s = window.__dcuTerminal;
                    let tail = '';
                    if (s) tail = s.pending + (s.started && !s.done ? s.buffer : '');
                    if (s && s.ws.readyState === WebSocket.OPEN) {
                        if (!s.done) s.ws.send(JSON.stringify(['stdin', '\\x03']));
                        s.ws.close();
                    }
                    delete window.__dcuTerminal;
                    return tail;
                }""")
                if tail:
                    print(tail, end="", flush=True)
                    with log_path.open("a", encoding="utf-8") as log:
                        log.write(tail)
            except Exception:
                pass
            try:
                self.request("DELETE", "api/terminals/" + quote(name, safe=""))
            except Exception as exc:
                print(f"警告：专用终端 {name} 清理失败：{exc}", file=sys.stderr)
            log_path.with_suffix(".json").write_text(json.dumps({
                "status": status, "exit_code": exit_code, "terminal": name,
                "elapsed_seconds": round(time.monotonic() - elapsed_start, 3),
                "log": str(log_path),
            }, ensure_ascii=False, indent=2), encoding="utf-8")


def file_bytes(model: dict) -> bytes:
    if model.get("type") == "notebook" and model.get("format") == "json":
        return (json.dumps(model["content"], ensure_ascii=False, indent=1) + "\n").encode()
    if model.get("type") != "file":
        raise BridgeError("指定路径不是文件。")
    if model.get("format") == "base64":
        return base64.b64decode(model["content"])
    if model.get("format") == "text":
        return model["content"].encode("utf-8")
    raise BridgeError("服务器返回了不支持的文件格式。")


START_TERMINAL = (ROOT / "terminal_bridge.js").read_text(encoding="utf-8")


def parser():
    p = argparse.ArgumentParser(description="通过本地浏览器连接比赛 JupyterLab")
    p.add_argument("--config", type=Path, default=ROOT / "config.example.json")
    p.add_argument("--cdp", help="可选：连接已经启用远程调试的浏览器地址")
    p.add_argument("--visible", action="store_true", help="显示自动化浏览器窗口")
    sub = p.add_subparsers(dest="action", required=True)
    login = sub.add_parser("login", help="打开独立 Edge 窗口，完成登录并保存会话")
    login.add_argument("--timeout", type=float, default=600)
    sub.add_parser("doctor", help="验证登录和文件/终端接口")
    ls = sub.add_parser("ls", help="列出远端目录")
    ls.add_argument("path", nargs="?", default="/")
    read = sub.add_parser("read", help="读取 UTF-8 文本文件")
    read.add_argument("path")
    upload = sub.add_parser("upload", help="上传文件并回读校验")
    upload.add_argument("local", type=Path)
    upload.add_argument("remote")
    upload.add_argument("--overwrite", action="store_true")
    download = sub.add_parser("download", help="下载文件")
    download.add_argument("remote")
    download.add_argument("local", type=Path)
    download.add_argument("--overwrite", action="store_true")
    mkdir = sub.add_parser("mkdir", help="创建单层目录（父目录需要已存在）")
    mkdir.add_argument("path")
    run = sub.add_parser("exec", help="在专用终端执行 bash 命令，实时输出并记录退出码")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("command", nargs="?")
    source.add_argument("--file", type=Path, help="执行本地 UTF-8 shell 脚本内容")
    run.add_argument("--cwd", help="服务器绝对工作目录")
    run.add_argument("--timeout", type=float, default=300)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    config = load_config(args.config)
    if hasattr(args, "timeout") and (not math.isfinite(args.timeout) or args.timeout <= 0):
        raise BridgeError("timeout 必须大于 0。")
    with browser_page(config, visible=args.visible or args.action == "login", cdp=args.cdp) as page:
        api = Jupyter(page, config)
        if args.action == "login":
            print("请在打开的 Edge 自动化窗口中完成比赛平台登录，并进入 JupyterLab。", flush=True)
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                try:
                    # Some platform login flows open Jupyter in another tab.
                    for candidate in reversed(page.context.pages):
                        if candidate.url.startswith(config["base_url"]):
                            api.page = candidate
                            break
                    api.probe()
                    print("登录验证成功，会话已保存在 .dcu/browser-profile/。")
                    return 0
                except Exception:
                    page.wait_for_timeout(1000)
            raise BridgeError("等待登录超时，请重新运行 login。")
        api.probe()
        if args.action == "doctor":
            terminals = api.request("GET", "api/terminals")
            print(json.dumps({"authenticated": True, "contents_api": True,
                              "terminals_api": isinstance(terminals, list),
                              "cwd": config.get("remote_cwd")}, ensure_ascii=False, indent=2))
        elif args.action == "ls":
            model = api.contents(args.path)
            if model.get("type") != "directory":
                raise BridgeError("ls 路径不是目录。")
            print(json.dumps([{k: item.get(k) for k in ("name", "path", "type", "size")}
                              for item in model["content"]], ensure_ascii=False, indent=2))
        elif args.action == "read":
            sys.stdout.write(file_bytes(api.contents(args.path)).decode("utf-8"))
        elif args.action == "upload":
            print(json.dumps(api.upload(args.local, args.remote, args.overwrite), ensure_ascii=False))
        elif args.action == "download":
            if args.local.exists() and not args.overwrite:
                raise BridgeError("本地目标已存在；如需替换请添加 --overwrite。")
            data = file_bytes(api.contents(args.remote))
            with args.local.open("wb" if args.overwrite else "xb") as dest:
                dest.write(data)
            print(f"已下载 {len(data)} 字节到 {args.local.resolve()}")
        elif args.action == "mkdir":
            if not args.path.strip("/"):
                raise BridgeError("不能创建根目录。")
            existing = api.contents(args.path, content=False, allow_missing=True)
            if existing:
                if existing.get("type") != "directory":
                    raise BridgeError("目标已存在且不是目录。")
                print("目录已存在。")
            else:
                api.request("PUT", "api/contents/" + remote_path(args.path), {"type": "directory"})
                print("目录已创建。")
        elif args.action == "exec":
            command = args.file.read_text(encoding="utf-8") if args.file else args.command
            cwd = args.cwd or config.get("remote_cwd", "/")
            if not isinstance(cwd, str) or not posixpath.isabs(cwd) or "\x00" in cwd:
                raise BridgeError("cwd 必须是服务器绝对路径。")
            log = STATE / "runs" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8] + ".log")
            code = api.execute(command, cwd, args.timeout, log)
            print(f"\n退出码：{code}；日志：{log}", file=sys.stderr)
            return code if 0 <= code <= 255 else 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
