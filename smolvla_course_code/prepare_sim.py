"""Initialize hf-libero paths and test one CPU-rendered simulator step."""
import importlib.util
import argparse
import json
import os
from pathlib import Path
import yaml
from download_assets import download_assets

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prepare-baseline', action='store_true', help='复用本地基础权重，准备 LIBERO 输入和 processor')
parser.add_argument('--check-policy', action='store_true', help='严格加载已准备基线，仅计算并执行一次动作')
args = parser.parse_args()

root = Path(os.environ['START_ROOT'])
assert os.environ.get('MUJOCO_GL') == 'osmesa', '先 source env.sh 设置 CPU 渲染'
package = Path(next(iter(importlib.util.find_spec('libero').submodule_search_locations))) / 'libero'
config_dir = Path(os.environ['LIBERO_CONFIG_PATH'])
config_dir.mkdir(parents=True, exist_ok=True)
assets = root / 'data/libero-assets'
download_assets(assets)
package_assets = package / 'assets'
required_assets = ('articulated_objects', 'stable_scanned_objects', 'turbosquid_objects', 'stable_hope_objects', 'scenes')
assert all((assets / name).is_dir() for name in required_assets), '场景资产下载不完整'
if package_assets.is_symlink() and not package_assets.exists():
    package_assets.unlink()
if not package_assets.exists():
    package_assets.symlink_to(assets, target_is_directory=True)
assert all((package_assets / name).is_dir() for name in required_assets), '包内已有 assets 路径不完整，请检查该目录后重试'
assert (package / 'bddl_files').is_dir() and (package / 'init_files').is_dir()
paths = {
    'benchmark_root': str(package), 'bddl_files': str(package / 'bddl_files'),
    'init_states': str(package / 'init_files'), 'datasets': str(root / 'data'),
    'assets': str(package_assets),
}
config = config_dir / 'config.yaml'
config.write_text(yaml.safe_dump(paths))
from libero.libero import benchmark
suite = benchmark.get_benchmark_dict()['libero_10']()
task = suite.get_task(9)
print('Simulator task:', task.name, task.language)
assert all(x in task.language.lower() for x in ('yellow', 'white', 'mug', 'microwave', 'close'))
states = suite.get_task_init_states(9)
print('Initial states:', len(states))
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
import numpy as np
env_config = LiberoEnv(task='libero_10', task_ids=[9], episode_length=500)
baseline = root / 'models/baseline_libero'
if args.prepare_baseline:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType
    from lerobot.envs.utils import env_to_policy_features
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla import processor_smolvla  # register processor

    base_root = Path(os.environ['SMOLVLA_BASE_ROOT']).resolve()
    base = base_root / 'models/smolvla_base'
    vlm = base_root / 'models/SmolVLM2-500M-Video-Instruct'
    assert (base / 'model.safetensors').is_file() and (vlm / 'tokenizer.json').is_file()
    if not baseline.exists():
        stats_json = json.loads((root / 'data/libero/meta/stats.json').read_text())
        stats = {key: {name: torch.tensor(value, dtype=torch.float32)
                       for name, value in stats_json[key].items()}
                 for key in ('observation.state', 'action')}
        assert tuple(stats['observation.state']['mean'].shape) == (8,)
        assert tuple(stats['action']['mean'].shape) == (7,)
        cfg = PreTrainedConfig.from_pretrained(str(base), local_files_only=True)
        features = env_to_policy_features(env_config)
        cfg.input_features = {k: v for k, v in features.items() if v.type is not FeatureType.ACTION}
        cfg.output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
        cfg.device = 'cuda'
        cfg.use_peft = False
        cfg.use_amp = False
        cfg.compile_model = False
        cfg.push_to_hub = False
        cfg.vlm_model_name = str(vlm)
        cfg.load_vlm_weights = False
        staging = baseline.with_name(baseline.name + '.preparing')
        assert not staging.exists(), f'存在未完成的准备目录：{staging}'
        staging.mkdir(parents=True)
        cfg.save_pretrained(staging)
        pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
        pre.save_pretrained(staging)
        post.save_pretrained(staging)
        (staging / 'model.safetensors').symlink_to(base / 'model.safetensors')
        (staging / 'baseline_manifest.json').write_text(json.dumps({
            'kind': 'unfinetuned_base_model', 'source': str(base),
            'dataset_stats': str(root / 'data/libero/meta/stats.json'),
            'suite': 'libero_10', 'task_id': 9, 'training_performed': False,
        }, indent=2) + '\n')
        staging.rename(baseline)
    assert (baseline / 'baseline_manifest.json').is_file(), '已有目录不是本工具准备的基线'
    (root / 'last_baseline_checkpoint.txt').write_text(str(baseline) + '\n')
    print('BASELINE_PREPARED:', baseline, flush=True)

env = make_env(env_config, n_envs=1)['libero_10'][9]
try:
    obs, info = env.reset(seed=[42])
    print('Main camera:', obs['pixels']['image'].shape)
    print('Wrist camera:', obs['pixels']['image2'].shape)
    action = np.zeros((1, 7), dtype=np.float32)
    if args.check_policy:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.utils import preprocess_observation, add_envs_task
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla import processor_smolvla

        torch.manual_seed(42)
        cfg = PreTrainedConfig.from_pretrained(str(baseline), local_files_only=True)
        policy = SmolVLAPolicy.from_pretrained(str(baseline), config=cfg, local_files_only=True, strict=True)
        policy.eval()
        pre, post = make_pre_post_processors(cfg, pretrained_path=str(baseline))
        env_pre, env_post = make_env_pre_post_processors(env_config, cfg)
        observation = env_pre(add_envs_task(env, preprocess_observation(obs)))
        assert observation['observation.state'].shape == (1, 8)
        batch = pre(observation)
        with torch.inference_mode():
            action = post(policy.select_action(batch))
            action = env_post({'action': action})['action'].cpu().numpy()
        assert action.shape == (1, 7) and np.isfinite(action).all()
        print('STRICT_BASELINE_ONE_ACTION_OK:', action.shape, flush=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print('LIBERO reset/render/step OK')
finally:
    env.close()
