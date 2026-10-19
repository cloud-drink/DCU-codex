"""Verify the real server using a new isolated directory on every run."""
import argparse
import json
import posixpath
import shlex
import time
import uuid
from pathlib import Path

from dcu import ROOT, STATE, BridgeError, Jupyter, browser_page, file_bytes, load_config, remote_path


def main():
    parser = argparse.ArgumentParser(description="在独立目录验证真实服务器的文件和终端通道")
    parser.add_argument("--config", type=Path, default=ROOT / "config.example.json")
    args = parser.parse_args()
    config = load_config(args.config)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    remote_dir = posixpath.join(config.get("remote_cwd", "/tmp"), "dcu_bridge_test_" + run_id)
    local_dir = STATE / "verification" / run_id
    local_dir.mkdir(parents=True)
    report = {"remote_dir": remote_dir, "passed": False}
    try:
        with browser_page(config) as page:
            api = Jupyter(page, config)
            api.probe()
            parent = api.contents(config.get("remote_cwd", "/tmp"), content=False)
            if parent.get("type") != "directory":
                raise BridgeError("配置中的 remote_cwd 不是 Jupyter 可访问的目录。")
            if api.contents(remote_dir, content=False, allow_missing=True) is not None:
                raise BridgeError("测试目录已存在，拒绝覆盖。")
            api.request("PUT", "api/contents/" + remote_path(remote_dir), {"type": "directory"})
            source = ROOT / "examples" / "smoke_test.py"
            target = posixpath.join(remote_dir, source.name)
            report["upload"] = api.upload(source, target)
            downloaded = file_bytes(api.contents(target))
            (local_dir / "downloaded_smoke_test.py").write_bytes(downloaded)
            report["download_verified"] = downloaded == source.read_bytes()
            if not report["download_verified"]:
                raise BridgeError("文件下载后与本地源文件不同。")
            print(f"文件上传下载校验通过，开始在 {remote_dir} 执行测试。", flush=True)
            report["exit_code"] = api.execute(
                "python3 " + shlex.quote(source.name), remote_dir, 30, local_dir / "execution.log")
            report["passed"] = report["exit_code"] == 0
            if not report["passed"]:
                raise BridgeError(f"远端测试退出码：{report['exit_code']}")
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        (local_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"验证报告：{local_dir / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
