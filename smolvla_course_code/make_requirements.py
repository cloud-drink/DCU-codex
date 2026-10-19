"""Generate a minimal training/evaluation install without replacing vendor torch."""
import importlib.metadata as md
import re
import sys
import tomllib
from pathlib import Path

root = Path(sys.argv[1])
project = tomllib.loads((root / 'pyproject.toml').read_text())['project']
assert project['version'] == '0.4.4', '请先切换到 LeRobot v0.4.4'
try:
    import torchvision
except Exception as exc:
    raise SystemExit(f'先完成 torchvision 安装与导入/图像变换检查，再安装其余依赖：{exc}')
skip = {'torch', 'torchvision', 'torchcodec', 'rerun-sdk'}
reqs = []
for req in project['dependencies']:
    name = re.split(r'[<>=\[; @]', req, maxsplit=1)[0].lower().replace('_', '-')
    if name not in skip:
        reqs.append(req)
reqs += [
    'transformers>=4.57.1,<5', 'num2words>=0.5.14,<0.6.0',
    'safetensors>=0.4.3,<1.0.0', 'peft==0.18.0', 'hf-libero==0.1.3',
    'mujoco==3.3.2',
    'numpy>=1.26,<2', 'opencv-python<4.12', 'opencv-python-headless<4.12',
]
Path('requirements-start.txt').write_text('\n'.join(reqs) + '\n')
constraints = []
for name in ('torch', 'torchvision', 'torchaudio'):
    try:
        constraints.append(f'{name}=={md.version(name)}')
    except md.PackageNotFoundError:
        pass
assert any(x.startswith('torch==') for x in constraints)
Path('constraints-dcu.txt').write_text('\n'.join(constraints) + '\n')
print('已生成 requirements-start.txt 和 constraints-dcu.txt')
print('锁定镜像原有版本：', constraints)
print('入门路径不安装 torchcodec / rerun-sdk；视频使用 PyAV。')
