#!/usr/bin/env python3
"""Run on the DCU server: install the starter without replacing vendor torch.

Only the Python standard library is imported by this entry point. Progress is
written to START_ROOT/setup_status.json and subprocess output to logs/setup.log.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import tarfile
import traceback


DEFAULT_ROOT = str(Path(__file__).resolve().parent)
ROOT = Path(os.environ.get("START_ROOT", DEFAULT_ROOT)).expanduser().resolve()
PROXY = os.environ.get("START_PROXY", "")
SYSTEM_PACKAGES = [
    "git", "ffmpeg", "build-essential", "pkg-config", "libosmesa6",
    "libosmesa6-dev", "libgl1-mesa-dev", "libegl1-mesa-dev",
    "libglib2.0-0", "libglfw3", "libglew-dev",
]
PROBE = r'''
import importlib.metadata as md
import json
from pathlib import Path
import sys
import torch
packages = {}
for name in ("torch", "torchvision", "torchaudio", "numpy", "Pillow",
             "transformers", "peft", "lerobot", "hf-libero", "mujoco", "av"):
    try:
        packages[name] = md.version(name)
    except md.PackageNotFoundError:
        packages[name] = None
result = {"python": sys.version, "executable": sys.executable,
          "packages": packages, "torch_version": torch.__version__,
          "torch_file": str(Path(torch.__file__).resolve()),
          "torch_hip": torch.version.hip}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
'''
HARDWARE_CHECK = r'''
import torch
import torchvision
from torchvision.transforms import v2
x = torch.rand(3, 256, 256)
assert v2.Resize((224, 224))(x).shape == (3, 224, 224)
assert torch.version.hip, "This interpreter is not using the vendor HIP torch build"
assert torch.cuda.is_available(), "DCU is not visible to torch.cuda"
assert torch.cuda.device_count() >= 1, "No visible DCU devices"
x = torch.randn(256, 256, device="cuda", requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss).item() and torch.isfinite(x.grad).all().item()
print("torch:", torch.__version__, torch.__file__)
print("torchvision:", torchvision.__version__, torchvision.__file__)
print("HIP:", torch.version.hip, "Device:", torch.cuda.get_device_name(0))
print("CPU Resize and DCU forward/backward OK; loss:", loss.item())
'''


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Setup:
    def __init__(self) -> None:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "logs").mkdir(exist_ok=True)
        self.log = (ROOT / "logs/setup.log").open("a", encoding="utf-8", buffering=1)
        self.phase = "starting"
        self.started_at = now()
        self.env = os.environ.copy()
        self.env.update({
            "START_ROOT": str(ROOT),
            "NO_PROXY": "localhost,127.0.0.1", "no_proxy": "localhost,127.0.0.1",
            "HF_HOME": str(ROOT / "cache/huggingface"),
            "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
            "HF_LEROBOT_HOME": str(ROOT / "cache/lerobot"),
            "LIBERO_CONFIG_PATH": str(ROOT / "cache/libero-config"),
            "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa",
            "TOKENIZERS_PARALLELISM": "false", "ACCELERATE_MIXED_PRECISION": "no",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_DEFAULT_TIMEOUT": "60",
            "PIP_INDEX_URL": "https://pypi.tuna.tsinghua.edu.cn/simple",
            "PIP_PROGRESS_BAR": "off",
            "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4",
            "HF_HUB_DOWNLOAD_TIMEOUT": "120", "HF_HUB_ETAG_TIMEOUT": "120",
        })
        if PROXY:
            self.env.update({"HTTP_PROXY": PROXY, "HTTPS_PROXY": PROXY,
                             "http_proxy": PROXY, "https_proxy": PROXY})
        if not any(self.env.get(k) for k in
                   ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
            self.env["HIP_VISIBLE_DEVICES"] = "0"

    def status(self, state: str, error: str | None = None) -> None:
        payload = {"phase": self.phase, "status": state, "started_at": self.started_at,
                   "updated_at": now(), "pid": os.getpid(), "root": str(ROOT),
                   "log": str(ROOT / "logs/setup.log")}
        if error:
            payload["error"] = error
        temporary = ROOT / "setup_status.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(ROOT / "setup_status.json")

    def begin(self, phase: str) -> None:
        self.phase = phase
        self.status("running")
        message = f"[{now()}] {phase}"
        print(message, flush=True)
        self.log.write("\n" + message + "\n")

    def run(self, args: list[str], *, timeout: int = 600, capture: bool = False) -> str:
        # Commands are fixed package/source operations; never log the environment.
        command = " ".join(args)
        if "@" in PROXY:
            command = command.replace(PROXY, "http://[authenticated-proxy]:3128")
        self.log.write("Command: " + command + "\n")
        self.log.flush()
        process = subprocess.Popen(
            args, cwd=ROOT, env=self.env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture else self.log,
            stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
        output = ""
        try:
            if capture:
                output, _ = process.communicate(timeout=timeout)
                self.log.write(output or "")
            else:
                process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            if capture:
                remainder, _ = process.communicate(timeout=10)
                self.log.write(remainder or "")
            self.log.flush()
            raise RuntimeError(f"{self.phase}: interrupted or exceeded {timeout}s; see logs/setup.log")
        self.log.flush()
        if process.returncode:
            raise RuntimeError(
                f"{self.phase}: {Path(args[0]).name} exited {process.returncode}; "
                "see logs/setup.log (proxy access, package resolution and permissions may need attention)"
            )
        return output or ""

    def probe(self, python: Path | str, filename: str) -> dict:
        target = ROOT / filename
        self.run([str(python), "-c", PROBE, str(target)])
        return json.loads(target.read_text())

    @staticmethod
    def verify_vendor(baseline: dict, current: dict) -> None:
        for field in ("torch_version", "torch_file", "torch_hip"):
            if current[field] != baseline[field]:
                raise RuntimeError(f"Vendor torch changed: {field}; compare environment-system.json and probe logs")

    def install(self) -> None:
        self.begin("record_system_environment")
        baseline = self.probe(sys.executable, "environment-system.json")
        if not baseline["torch_hip"]:
            raise RuntimeError("The starting interpreter has no HIP torch build; use the platform's DCU interpreter")

        self.begin("create_venv")
        venv = ROOT / ".venv"
        python = venv / "bin/python"
        if not venv.exists():
            self.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)])
        if not python.is_file():
            raise RuntimeError("Existing .venv has no bin/python; inspect it before rerunning")
        cfg = (venv / "pyvenv.cfg").read_text().lower()
        if not re.search(r"include-system-site-packages\s*=\s*true", cfg):
            raise RuntimeError("Existing .venv does not inherit system packages; preserve it and choose another START_ROOT")
        current = self.probe(python, "environment-before.json")
        self.verify_vendor(baseline, current)

        self.begin("install_system_libraries")
        if not shutil.which("apt-get") or not shutil.which("dpkg-query"):
            raise RuntimeError("apt-get/dpkg-query are unavailable; this setup requires the Ubuntu server container")
        query = self.run(["dpkg-query", "-W", "-f=${binary:Package} ${db:Status-Status}\\n"], capture=True)
        installed = {line.split()[0].split(":")[0] for line in query.splitlines()
                     if len(line.split()) >= 2 and line.split()[1] == "installed"}
        missing = [package for package in SYSTEM_PACKAGES if package not in installed]
        if missing:
            prefix = [] if os.geteuid() == 0 else ["sudo", "-n"]
            if prefix and not shutil.which("sudo"):
                raise RuntimeError("Missing system libraries require root; sudo is unavailable. Ask the platform to install the listed packages")
            apt = prefix + ["apt-get"]
            if PROXY:
                apt += ["-o", f"Acquire::http::Proxy={PROXY}",
                        "-o", f"Acquire::https::Proxy={PROXY}"]
            self.env["DEBIAN_FRONTEND"] = "noninteractive"
            self.run(apt + ["update"])
            self.run(apt + ["install", "-y", "--no-install-recommends", *missing])

        self.begin("install_cpu_torchvision")
        if not current["packages"].get("Pillow"):
            self.run([str(python), "-m", "pip", "install", "Pillow>=10,<13"])
        if current["packages"].get("torchvision") != "0.24.0+cpu":
            self.run([str(python), "-m", "pip", "install", "--no-deps",
                      "torchvision==0.24.0+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"])
        self.run([str(python), "-c", HARDWARE_CHECK])
        self.verify_vendor(baseline, self.probe(python, "environment-vision.json"))

        self.begin("prepare_lerobot_source")
        source = ROOT / "lerobot"
        if not (source / "pyproject.toml").is_file():
            if source.exists():
                source.rename(ROOT / f"lerobot-incomplete-{int(time.time())}")
            url = "https://api.github.com/repos/huggingface/lerobot/tarball/v0.4.4"
            archive = ROOT / "lerobot-v0.4.4.tar.gz"
            code = "import urllib.request, sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])"
            self.run([sys.executable, "-c", code, url, str(archive)], timeout=600)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            expected = "ba3df1aea437c88b77d7ee01efae4092d04ca7c3926e8259e19b9d0c18de0e00"
            if digest != expected:
                raise RuntimeError("Official source archive differs from the locally inspected v0.4.4 archive")
            extracted = ROOT / "source-unpack"
            extracted.mkdir(exist_ok=True)
            with tarfile.open(archive) as tar:
                tar.extractall(extracted, filter="data")
            (extracted / "huggingface-lerobot-8fff0fd").rename(source)
            (ROOT / "source_provenance.json").write_text(json.dumps({"url": url, "sha256": digest}, indent=2))
        project_file = source / "pyproject.toml"
        if not project_file.is_file():
            raise RuntimeError("Existing lerobot directory has no pyproject.toml; it was left unchanged")
        if tomllib.loads(project_file.read_text())["project"]["version"] != "0.4.4":
            raise RuntimeError("Existing LeRobot checkout is not v0.4.4; it was left unchanged")

        self.begin("generate_dependencies")
        generator = ROOT / "make_requirements.py"
        if not generator.is_file():
            generator = Path(__file__).resolve().parent / "make_requirements.py"
        if not generator.is_file():
            raise RuntimeError("make_requirements.py is missing; upload the complete starter bundle")
        self.run([str(python), str(generator), str(source)])
        # Retain an already compatible Transformers release from the server.
        version = baseline["packages"].get("transformers")
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version or "")
        if match and (4, 57, 1) <= tuple(map(int, match.groups())) < (5, 0, 0):
            with (ROOT / "constraints-dcu.txt").open("a") as constraints:
                constraints.write(f"transformers=={version}\n")

        self.begin("install_python_dependencies")
        self.run([str(python), "-m", "pip", "install", "-c", "constraints-dcu.txt",
                  "-r", "requirements-start.txt"], timeout=1800)
        self.begin("install_lerobot")
        self.run([str(python), "-m", "pip", "install", "--no-deps", "-e", str(source)])

        self.begin("verify_environment")
        after = self.probe(python, "environment-after.json")
        self.verify_vendor(baseline, after)
        assert after["packages"]["lerobot"] == "0.4.4", "LeRobot installation version is not 0.4.4"
        self.run([str(python), "-c", HARDWARE_CHECK])
        self.phase = "complete"
        self.status("complete")
        print(f"Setup complete. Log: {ROOT / 'logs/setup.log'}", flush=True)


def main() -> int:
    setup = Setup()
    lock = ROOT / ".setup.lock"
    # flock releases automatically after failure, avoiding stale PID lock files.
    import fcntl
    lock_handle = lock.open("a")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another setup process is already running for this START_ROOT", file=sys.stderr)
        setup.log.close()
        lock_handle.close()
        return 2
    try:
        setup.install()
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        setup.log.write(traceback.format_exc())
        setup.log.flush()
        message = str(exc) or type(exc).__name__
        setup.status("error", message)
        print(f"Setup stopped in {setup.phase}: {message}", file=sys.stderr, flush=True)
        return 1
    finally:
        setup.log.close()
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
