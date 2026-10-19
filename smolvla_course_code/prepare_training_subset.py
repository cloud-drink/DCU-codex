"""Fetch a selected episode subset and enforce dataset task_index=2."""
import argparse
import json
import os
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('episodes_file')
args=parser.parse_args()
root=Path(os.environ['START_ROOT'])
episodes=json.loads(Path(args.episodes_file).read_text())
assert isinstance(episodes,list) and episodes and all(type(ep) is int and ep>=0 for ep in episodes)
assert len(set(episodes))==len(episodes),'轨迹编号重复'
revision=(root/'dataset_revision.txt').read_text().strip()
ds=LeRobotDataset('lerobot/libero',root=root/'data/libero',revision=revision,episodes=episodes,video_backend='pyav')
assert {int(x) for x in ds.hf_dataset.with_format(None)['task_index']}=={2},'包含非目标任务'
sample=ds[0]
assert sample['action'].shape==(7,)
assert sample['observation.state'].shape==(8,)
print('TRAINING_SUBSET_OK episodes:',episodes,'frames:',len(ds),'task_index=2')
