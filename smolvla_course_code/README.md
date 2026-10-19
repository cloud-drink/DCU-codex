# SmolVLA 课程最小代码包

当前范围：基础镜像配置、官方目标数据读取、新 LoRA 模型训练、保存、显式重载、单次 LIBERO 推理和视频播放。正式测评暂不操作。

2026-10-19 当前离线实例：工作目录为 `/public/home/xdzs2026_b0196/smolvla_recording_20261015_231050`。场景资产已校验，OSMesa 解压在项目 `.runtime/osmesa` 内，`env.sh` 复用现有平台环境路径。原始 SmolVLA 权重复用账号目录下 `SothisAI/smolvla/models`，不复制另一份大权重。

基础模型单次仿真（未微调，不代表优化效果）由用户在服务器 Terminal 执行：

```bash
export START_ROOT=/public/home/xdzs2026_b0196/smolvla_recording_20261015_231050
cd "$START_ROOT"
source env.sh
CHECKPOINT="$START_ROOT/models/baseline_libero" bash eval_smoke.sh
python show_result.py
```

首次准备或需要复查时，`python prepare_sim.py --prepare-baseline --check-policy` 仅检查环境 reset、相机渲染、严格权重加载和一个动作。它会基于已有官方数据统计适配 state=8/action=7，生成 processor 和基础模型配置，保留 `last_baseline_checkpoint.txt`；没有训练或完整 episode。`eval_smoke.sh` 仍可通过 CHECKPOINT 显式选择微调模型，不会把基线当作已训练模型。

上述小测试已在当前实例通过，两路相机为 `(1,256,256,3)`，动作输出为 `(1,7)`，环境单步成功。完整单次仿真仍由用户启动。结果展示使用现有 PyAV 完整解码视频，不再要求系统 PATH 中另有 ffmpeg/ffprobe。

先阅读 `RECORDING_GUIDE.md`，按 8 段录制。继续训练自己的模型用 `TRAINING_INFERENCE_GUIDE.md`；准备共享镜像用 `IMAGE_PACKAGING_GUIDE.md`；赛题对应核对见 `PDF_ALIGNMENT.md` 和 `参考资料/任务二-赛题介绍.pdf`。

这是代码包，不含 Python 环境、模型权重、机器人数据或场景资产。基础镜像起步时按操作稿下载并安装；在完整课程镜像中使用时只需上传本包、激活已有环境。

新模型必须明确指定后推理：

```bash
export START_ROOT=/你的实际实验目录
cd "$START_ROOT"
source env.sh
STEPS=100 WARMUP_STEPS=10 LR=0.0001 LORA_R=8 BATCH_SIZE=1 \
EXPERIMENT_NAME=my_model bash train_experiment.sh
CHECKPOINT="$(cat last_experiment_checkpoint.txt)" bash eval_smoke.sh
python show_result.py
```

导出工具 `export_image_context.py` 按必要清单生成 Docker 构建上下文，保留环境绝对路径，排除历史测试和实验输出；它不会自动构建镜像。使用前需在兼容服务器准备资源，构建后在新 DCU 实例离线验收。

本包不含项目 tests、正式测评代码、短验证 JSON/CSV、历史日志、旧 checkpoint、推理视频或代理凭据。文件 SHA-256 见 `PACKAGE_MANIFEST.json`。开发者完整包另行保留，不用作公共最小上传包。
