"""Validate the newest evaluation and create a notebook for recorded playback."""
import json
import os
from pathlib import Path
import av

root = Path(os.environ['START_ROOT'])
pointer = root/'last_inference_result.txt'
if pointer.is_file():
    result = Path(pointer.read_text().strip())/'eval_info.json'
    assert result.is_file(), '本次推理结果不存在，请检查 last_inference_result.txt'
else:
    results = list((root/'runs').glob('eval_clean_*/eval_info.json'))
    assert results, '未找到推理结果，请先完成 bash eval_smoke.sh'
    result = max(results, key=lambda p: p.stat().st_mtime)
checkpoint_file = result.parent/'inference_checkpoint.txt'
checkpoint = checkpoint_file.read_text().strip() if checkpoint_file.is_file() else '旧结果未记录 checkpoint'
info = json.loads(result.read_text())
assert info['overall']['n_episodes'] == 1
tasks = info['per_task']
assert len(tasks) == 1 and tasks[0]['task_group'] == 'libero_10' and tasks[0]['task_id'] == 9
video = Path(info['overall']['video_paths'][0])
if not video.is_absolute():
    video = root/video
assert video.is_file() and video.stat().st_size > 0
with av.open(str(video)) as container:
    assert container.streams.video, '文件中没有视频流'
    stream = container.streams.video[0]
    video_info = {
        'codec_name': stream.codec_context.name,
        'width': stream.codec_context.width,
        'height': stream.codec_context.height,
        'duration': float(stream.duration * stream.time_base) if stream.duration is not None else None,
        'r_frame_rate': str(stream.average_rate),
    }
    video_info['decoded_frames'] = sum(1 for _ in container.decode(video=0))
    assert video_info['decoded_frames'] > 0, '视频未解码出任何帧'
print('推理 checkpoint:', checkpoint)
print('单次推理 JSON:', result)
print('完整 episode 数:', info['overall']['n_episodes'])
print('成功率:', info['overall']['pc_success'])
print('推理视频:', video)
print('视频字节数:', video.stat().st_size)
print('视频信息:', json.dumps(video_info, ensure_ascii=False))
print('视频完整解码通过')
source = ('from IPython.display import Video, display\n'
          'import json\nfrom pathlib import Path\n'
          f'info = json.loads(Path({str(result)!r}).read_text())\n'
          f'print("推理 checkpoint:", {checkpoint!r})\n'
          "print('episode:', info['overall']['n_episodes'], '成功率:', info['overall']['pc_success'])\n"
          f'display(Video({str(video)!r}, embed=True, width=480))\n')
nb = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'}},
    'cells': [
        {'id': 'intro', 'cell_type': 'markdown', 'metadata': {},
         'source': ['# SmolVLA 单卡推理结果\n', '运行下方单元格展示所选 checkpoint 的真实仿真视频。单次运行用于验证训练、重载和推理流程；视频编码时长与推理耗时不同。']},
        {'id': 'playback', 'cell_type': 'code', 'metadata': {}, 'execution_count': None,
         'outputs': [], 'source': source.splitlines(keepends=True)},
    ],
}
notebook = root/'results_view.ipynb'
notebook.write_text(json.dumps(nb, ensure_ascii=False, indent=2))
print('在 JupyterLab 打开并运行:', notebook)
