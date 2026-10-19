"""Download official weights and ONLY load episodes for dataset task_index=2."""
import json
import os
import re
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

root = Path(os.environ['START_ROOT'])
repo = 'lerobot/libero'
revision_file = root / 'dataset_revision.txt'
if revision_file.exists():
    revision = revision_file.read_text().strip()
else:
    revision = HfApi().dataset_info(repo, revision='main').sha
    revision_file.write_text(revision)
data_root = root / 'data/libero'
meta = LeRobotDatasetMetadata(repo, root=data_root, revision=revision)
print('Dataset commit:', revision)
print('Dataset version:', meta.info['codebase_version'])
assert meta.info['codebase_version'] == 'v3.0'
matches = meta.tasks[meta.tasks['task_index'] == 2]
assert len(matches) == 1, matches
language = str(matches.index[0])
print('task_index=2:', language)
assert all(x in language.lower() for x in ('yellow', 'white', 'mug', 'microwave', 'close')), (
    '任务含义与 PDF 不符，停止；不要按猜测的 episode 范围训练。'
)
complete_catalog = 'tasks' in meta.episodes.column_names
if complete_catalog:
    episodes = [int(ep['episode_index']) for ep in meta.episodes if list(ep['tasks']) == [language]]
else:
    # Some official v3 exports omit per-episode tasks. Read only numerical
    # parquet columns to build the mapping; videos are fetched for selected episodes later.
    import pandas as pd
    import pyarrow.parquet as pq
    def find_cached_episodes():
        paths = sorted((data_root / 'data').rglob('*.parquet'))
        if not paths:
            return []
        mapping = pd.concat([pq.read_table(p, columns=['episode_index', 'task_index']).to_pandas()
                             for p in paths], ignore_index=True)
        grouped = mapping.groupby('episode_index')['task_index'].agg(['min', 'max'])
        matched = grouped.index[(grouped['min'] == 2) & (grouped['max'] == 2)]
        return sorted(int(x) for x in matched)

    complete_catalog = os.environ.get('FULL_TASK_SCAN') == '1'
    if complete_catalog:
        snapshot_download(repo, repo_type='dataset', revision=revision, local_dir=data_root,
                          allow_patterns=['data/**'], max_workers=2)
    episodes = find_cached_episodes()
    if len(episodes) < 2:
        references = sorted({str(meta.get_data_file_path(int(ep['episode_index'])))
                             for ep in meta.episodes})
        for filename in references:
            if not (data_root / filename).is_file():
                hf_hub_download(repo, filename=filename, repo_type='dataset', revision=revision,
                                local_dir=data_root)
                episodes = find_cached_episodes()
            if len(episodes) >= 2:
                break
    print('Confirmed cached target episodes:', len(episodes), 'Complete catalog:', complete_catalog)
assert episodes, '未找到目标任务轨迹'
episodes.sort()
(root / 'episodes_available.json').write_text(json.dumps(episodes))
if complete_catalog:
    (root / 'episodes_all.json').write_text(json.dumps(episodes))
small = episodes[:2]
(root / 'episodes_smoke.json').write_text(json.dumps(small))
ds = LeRobotDataset(repo, root=data_root, revision=revision, episodes=small, video_backend='pyav')
indices = ds.hf_dataset.with_format(None)['task_index']
assert {int(x) for x in indices} == {2}, '加载数据中混入了其他任务'
sample = ds[0]
for key in ('observation.images.image', 'observation.images.image2', 'observation.state', 'action'):
    print(key, tuple(sample[key].shape))
print('Smoke episodes:', small, 'Confirmed target episodes available:', len(episodes))
for model in ('lerobot/smolvla_base', 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'):
    required = ['config.json', 'generation_config.json', '*processor_config.json', 'chat_template*',
                'special_tokens_map.json', 'tokenizer*', 'vocab.json', 'merges.txt', '*.model',
                'model*.safetensors*', 'pytorch_model*.bin', 'added_tokens.json',
                'policy_*.json', 'policy_*.safetensors']
    for attempt in range(3):
        try:
            path = snapshot_download(model, allow_patterns=required, max_workers=2)
            break
        except OSError as exc:
            match = re.search(r'Consistency check failed:.*\(([^()/]+)\)', str(exc))
            if not match or attempt == 2:
                raise
            filename = match.group(1)
            print('Re-fetching file that failed size validation:', model, filename)
            hf_hub_download(model, filename=filename, force_download=True)
    print('Model cached:', model, path)
print('数据读取和模型下载完成。训练仍需传 --dataset.episodes。')
