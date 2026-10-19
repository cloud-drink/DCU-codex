#!/usr/bin/env python3
"""Export an allowlisted server runtime; preserve its absolute path in the image.

Run with the prepared server .venv Python. No model training, downloads, Docker
build, or source deletion is performed. Experiment code is delivered separately.
"""
import argparse
import fnmatch
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

MODELS = ('lerobot/smolvla_base', 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct')
MODEL_FILES = ('config.json', 'generation_config.json', '*processor_config.json',
               'chat_template*', 'special_tokens_map.json', 'tokenizer*',
               'vocab.json', 'merges.txt', '*.model', 'model*.safetensors*',
               'pytorch_model*.bin', 'added_tokens.json', 'policy_*.json',
               'policy_*.safetensors')
SYSTEM_PACKAGES = ('git', 'ffmpeg', 'build-essential', 'pkg-config', 'libosmesa6',
                   'libosmesa6-dev', 'libgl1-mesa-dev', 'libegl1-mesa-dev',
                   'libglib2.0-0', 'libglfw3', 'libglew-dev')


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def copy_file(source, target):
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Dereference selected HF snapshot files: do not depend on excluded blobs.
    shutil.copy2(source, target)


def safe_relative(value):
    value = Path(value)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError('Unsafe relative path: ' + str(value))
    return value


def export(args):
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError('Build context and runtime must be separate, non-nested directories')
    if output.exists():
        raise FileExistsError('Choose a new build-context directory: ' + str(output))
    if not Path(sys.prefix).resolve() == (root / '.venv').resolve():
        raise ValueError('Use the prepared runtime .venv/bin/python to export')
    if not args.base_image or any(c.isspace() for c in args.base_image):
        raise ValueError('Use one real registry image reference, preferably with @sha256:digest')
    os.environ['START_ROOT'] = str(root)
    os.environ['HF_HOME'] = str(root / 'cache/huggingface')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from download_assets import matches, REVISION
    assert md.version('lerobot') == '0.4.4'
    assert torch.version.hip, 'Export from the vendor HIP Torch runtime'
    revision = (root / 'dataset_revision.txt').read_text().strip()
    episodes_path = Path(args.episodes_file)
    if not episodes_path.is_absolute():
        episodes_path = root / episodes_path
    episodes = json.loads(episodes_path.read_text())
    assert isinstance(episodes, list) and episodes and all(type(ep) is int and ep >= 0 for ep in episodes)
    assert len(set(episodes)) == len(episodes)
    dataset = LeRobotDataset('lerobot/libero', root=root / 'data/libero',
                             revision=revision, episodes=episodes, video_backend='pyav')
    assert {int(x) for x in dataset.hf_dataset.with_format(None)['task_index']} == {2}
    sample = dataset[0]  # verifies video loading before export
    assert tuple(sample['action'].shape) == (7,) and tuple(sample['observation.state'].shape) == (8,)
    required_data = set()
    for episode in episodes:
        required_data.add(safe_relative(dataset.meta.get_data_file_path(episode)))
        for camera in dataset.meta.video_keys:
            required_data.add(safe_relative(dataset.meta.get_video_file_path(episode, camera)))
    assets_manifest = json.loads((root / 'assets_manifest.json').read_text())
    assert assets_manifest['revision'] == REVISION and len(assets_manifest['files']) == 586
    for record in assets_manifest['files']:
        relative = safe_relative(record['path'])
        if not matches(root / 'data/libero-assets' / relative, record):
            raise ValueError('Missing/invalid asset: ' + str(relative))
    model_snapshots = {}
    for repo in MODELS:
        cache = root / 'cache/huggingface/hub' / ('models--' + repo.replace('/', '--'))
        revision_model = (cache / 'refs/main').read_text().strip()
        assert len(revision_model) == 40 and all(c in '0123456789abcdef' for c in revision_model)
        snapshot = cache / 'snapshots' / revision_model
        if not (snapshot / 'config.json').is_file() or not any(snapshot.glob('*.safetensors')):
            raise ValueError('Incomplete model snapshot: ' + repo)
        model_snapshots[repo] = {'revision': revision_model, 'cache': cache, 'snapshot': snapshot}
    query = subprocess.check_output(
        ['dpkg-query', '-W', '-f=${Package}\t${Version}\n', *SYSTEM_PACKAGES], text=True)
    system_versions = dict(line.split('\t') for line in query.splitlines())
    package_versions = {name: md.version(name) for name in
        ('torch', 'torchvision', 'lerobot', 'transformers', 'peft', 'hf-libero', 'robosuite', 'mujoco', 'numpy', 'av')}
    output.mkdir(parents=True)
    payload = output / 'payload'
    payload.mkdir()
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo')
    # Preserve symlinks in the venv: interpreter links refer to the identical base
    # image, and editable source/assets links retain the same absolute runtime path.
    shutil.copytree(root / '.venv', payload / '.venv', symlinks=True, ignore=ignore)
    framework_ignore = shutil.ignore_patterns('.git', '.github', 'tests', 'docs',
        'examples', 'benchmarks', '__pycache__', '*.pyc', '*.pyo')
    shutil.copytree(root / 'lerobot', payload / 'lerobot', symlinks=True, ignore=framework_ignore)
    for relative in ('env.sh', 'dataset_revision.txt', 'assets_manifest.json'):
        copy_file(root / relative, payload / relative)
    copy_file(root / 'cache/libero-config/config.yaml', payload / 'cache/libero-config/config.yaml')
    for file in sorted((root / 'data/libero/meta').rglob('*')):
        if file.is_file() and '.cache' not in file.relative_to(root / 'data/libero/meta').parts:
            copy_file(file, payload / file.relative_to(root))
    for relative in sorted(required_data):
        copy_file(root / 'data/libero' / relative, payload / 'data/libero' / relative)
    for record in assets_manifest['files']:
        relative = safe_relative(record['path'])
        copy_file(root / 'data/libero-assets' / relative, payload / 'data/libero-assets' / relative)
    for repo, details in model_snapshots.items():
        cache_target = payload / details['cache'].relative_to(root)
        copy_file(details['cache'] / 'refs/main', cache_target / 'refs/main')
        for file in sorted(details['snapshot'].rglob('*')):
            relative = str(file.relative_to(details['snapshot']))
            if file.is_file() and any(fnmatch.fnmatch(relative, pattern) for pattern in MODEL_FILES):
                copy_file(file, cache_target / 'snapshots' / details['revision'] / file.relative_to(details['snapshot']))
    # Provide only episode lists whose corresponding files are actually included.
    (payload / 'episodes_image.json').write_text(json.dumps(episodes) + '\n')
    (payload / 'episodes_smoke.json').write_text(json.dumps(sorted(episodes)[:2]) + '\n')
    all_file = root / 'episodes_all.json'
    if all_file.is_file() and set(json.loads(all_file.read_text())) == set(episodes):
        copy_file(all_file, payload / 'episodes_all.json')
    manifest = {
        'runtime_root': str(root), 'base_image': args.base_image,
        'base_image_is_digest_pinned': '@sha256:' in args.base_image,
        'python': sys.version, 'base_prefix': sys.base_prefix,
        'torch_hip': torch.version.hip, 'packages': package_versions,
        'system_packages': system_versions, 'dataset_revision': revision,
        'task_index': 2, 'episodes': episodes,
        'models': {repo: details['revision'] for repo, details in model_snapshots.items()},
        'assets_revision': REVISION,
        'excluded': ['experiment scripts (separate ZIP)', 'project tests', 'logs', 'runs',
                     'checkpoint pointers', 'verification JSON', 'pip caches', 'HF tokens and locks'],
    }
    # Do not ship links that only resolve through the original project volume.
    links = {}
    system_roots = (Path(sys.base_prefix), Path('/usr'), Path('/lib'), Path('/lib64'))
    for link in sorted(payload.rglob('*')):
        if not link.is_symlink():
            continue
        target = Path(os.readlink(link))
        if target.is_absolute():
            if target == root or root in target.parents:
                included = payload / target.relative_to(root)
                if not included.exists():
                    raise ValueError('Symlink refers to an excluded runtime file: ' + str(link))
            elif not any(target == parent or parent in target.parents for parent in system_roots):
                raise ValueError('Symlink requires an external project path: ' + str(link))
        else:
            resolved = (link.parent / target).resolve()
            in_payload = resolved == payload or payload in resolved.parents
            in_base = any(resolved == parent or parent in resolved.parents for parent in system_roots)
            if not resolved.exists() or not (in_payload or in_base):
                raise ValueError('Relative symlink leaves payload or is dangling: ' + str(link))
        links[str(link.relative_to(payload))] = str(target)
    manifest['symlinks'] = links
    # Store runtime provenance plus checksums of every regular payload file.
    inventory = {}
    for file in sorted(payload.rglob('*')):
        if file.is_file() and not file.is_symlink():
            inventory[str(file.relative_to(payload))] = {'bytes': file.stat().st_size, 'sha256': sha256(file)}
    manifest['files'] = inventory
    (payload / 'IMAGE_MANIFEST.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    # Verify selected vendor/Python versions during build without requiring a DCU.
    verification = """import importlib.metadata as md, json, os, sys
from pathlib import Path
import torch
p = Path(os.environ['START_ROOT'])
m = json.loads((p/'IMAGE_MANIFEST.json').read_text())
assert sys.version == m['python'], 'Base Python changed'
assert sys.base_prefix == m['base_prefix'], 'Base interpreter path changed'
assert torch.version.hip == m['torch_hip'], 'Vendor HIP Torch changed'
for name, expected in m['packages'].items():
    assert md.version(name) == expected, (name, expected, md.version(name))
print('IMAGE_SOFTWARE_VERSIONS_OK')
"""
    (output / 'verify_image_software.py').write_text(verification)
    packages = ' '.join(name + '=' + system_versions[name] for name in SYSTEM_PACKAGES)
    dockerfile = (
        'ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\nARG RUNTIME_USER=root\nUSER root\n'
        'RUN apt-get update && apt-get install -y --no-install-recommends ' + packages +
        ' && rm -rf /var/lib/apt/lists/*\n'
        'COPY ' + json.dumps(['payload/', str(root) + '/']) + '\n'
        'ENV START_ROOT=' + json.dumps(str(root)) + '\n'
        'ENV HF_HOME=' + json.dumps(str(root / 'cache/huggingface')) + '\n'
        'ENV LIBERO_CONFIG_PATH=' + json.dumps(str(root / 'cache/libero-config')) + '\n'
        'ENV MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa\n'
        'COPY verify_image_software.py /tmp/verify_image_software.py\n'
        'RUN "$START_ROOT/.venv/bin/python" /tmp/verify_image_software.py && rm /tmp/verify_image_software.py\n'
        'RUN chown -R "$RUNTIME_USER" "$START_ROOT"\n'
        'USER ${RUNTIME_USER}\n'
        '# Keep the platform base image JupyterLab entrypoint/CMD.\n')
    (output / 'Dockerfile').write_text(dockerfile)
    (output / '.dockerignore').write_text('*\n!Dockerfile\n!payload/\n!payload/**\n!verify_image_software.py\n')
    (output / 'export_summary.json').write_text(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, indent=2) + '\n')
    print('IMAGE_CONTEXT:', output)
    print('RUNTIME_ROOT (must not relocate/mount over):', root)
    print('EPISODES:', episodes)
    print('PAYLOAD_FILE_COUNT:', len(inventory))
    print('No Docker build performed. Build and validate on a compatible DCU platform.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=os.environ.get('START_ROOT'), required=not os.environ.get('START_ROOT'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--base-image', required=True, help='real vendor base registry reference, preferably digest-pinned')
    parser.add_argument('--episodes-file', default='episodes_smoke.json')
    export(parser.parse_args())


if __name__ == '__main__':
    main()
