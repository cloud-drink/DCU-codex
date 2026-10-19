"""Download fixed LIBERO assets with GET and verify Git/LFS content hashes."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import quote
import requests

REVISION = '0b3ea86be5fe169d0fd036ae63d1070ec09e90f6'


def matches(path, record):
    if not path.is_file() or path.stat().st_size != record['size']:
        return False
    data = path.read_bytes()
    if record.get('lfs'):
        return hashlib.sha256(data).hexdigest() == record['lfs']['oid']
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == record['oid']


def download_assets(destination):
    manifest = json.loads(Path(__file__).with_name('assets_manifest.json').read_text())
    assert manifest['revision'] == REVISION
    files = manifest['files']
    assert len(files) == 586
    endpoint = os.environ.get('LIBERO_ASSET_ENDPOINT', 'https://hf-mirror.com').rstrip('/')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    needed = [item for item in files if not matches(destination/item['path'], item)]
    print(f'Assets: {len(files)-len(needed)}/{len(files)} validated; downloading {len(needed)} from {endpoint}', flush=True)

    def get_file(item):
        relative = Path(item['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.part')
        url = f"{endpoint}/datasets/lerobot/libero-assets/resolve/{REVISION}/{quote(item['path'], safe='/')}"
        last = None
        attempts = 10
        for attempt in range(attempts):
            delay = min(60, 2 * (attempt + 1))
            try:
                with requests.get(url, stream=True, timeout=(15, 45)) as response:
                    if response.status_code == 429:
                        try:
                            retry_after = float(response.headers.get('Retry-After', '0'))
                        except ValueError:
                            retry_after = 0
                        delay = min(60, max(retry_after, 15 * (attempt + 1)))
                    response.raise_for_status()
                    with temporary.open('wb') as stream:
                        for chunk in response.iter_content(1024*1024):
                            stream.write(chunk)
                if not matches(temporary, item):
                    raise ValueError('content hash/size mismatch')
                temporary.replace(target)
                return item['path']
            except Exception as exc:
                last = type(exc).__name__
                print(f"Retry {attempt+1}/{attempts}: {item['path']} ({last}); wait {delay:g}s", flush=True)
                if attempt + 1 < attempts:
                    time.sleep(delay)
        raise RuntimeError(f"Asset download failed: {item['path']} ({last})")

    workers = max(1, min(4, int(os.environ.get('LIBERO_ASSET_WORKERS', '2'))))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(get_file, item) for item in needed]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 20 == 0 or count == len(needed):
                print(f'Assets validated: {len(files)-len(needed)+count}/{len(files)}', flush=True)
    assert all(matches(destination/item['path'], item) for item in files)
    print('All 586 LIBERO assets passed size and content hash validation', flush=True)


if __name__ == '__main__':
    download_assets(Path(os.environ['START_ROOT'])/'data/libero-assets')
