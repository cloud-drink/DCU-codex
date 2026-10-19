# 海光单卡 SmolVLA：从基础镜像到新模型推理的录屏操作稿

更新：2026-10-14。当前录屏范围是配置环境、上传代码、下载模型和数据、训练新模型、保存并重载、实际推理、播放结果。正式测评和评分暂不操作。

使用基础镜像 `jupyterlab-pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.11-devel`。PDF 要求海光 DCU-BW1000 单卡、100 小时内完成参数高效微调。平台资源页应核实 BW1000 卡型；原实测显示 BW/约 64 GiB，不能仅据此认定型号一致。

赛题 PDF 已保存在课程目录 `参考资料/任务二-赛题介绍.pdf`。训练限定 `lerobot/libero/task_index=2`，基础模型 `lerobot/smolvla_base`；动作、状态和语言标签不得修改。具体对应关系见 `PDF_ALIGNMENT.md`。

全部命令在服务器 JupyterLab Terminal 的 Bash 中执行，不能直接粘到 Python Notebook。每一步确认成功后再继续；推理失败时不执行结果展示，以免读取先前成功运行的产物。建议分别录制下面 8 段，再拼接成完整视频。

## 1. 展示实例并介绍任务

录制平台资源页：基础镜像名称、单张 DCU、CPU/RAM/磁盘配置。进入实例的 JupyterLab，选择 File → New → Terminal。

口述：“从海光 PyTorch 基础镜像开始，配置 SmolVLA 的训练和仿真环境，训练一份新模型并实际运行推理。目标任务是将黄白色马克杯放入微波炉并关门。”

## 2. 上传代码、建立目录、检查计算卡

在文件浏览器进入持久化用户目录，上传最小代码包 `smolvla_course_code.zip`。本账号目录如下，其他账号需要替换：

```bash
cd /public/home/xdzs2026_b0130
export START_ROOT="$PWD/smolvla_recording_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$START_ROOT"
python3 -m zipfile -e smolvla_course_code.zip "$START_ROOT"
cd "$START_ROOT"
pwd
python3 --version
df -h .
ls
python3 check_env.py
```

录制脚本列表、设备信息与 `DCU forward/backward OK`。第一次提示 torchvision 尚未安装可继续到安装阶段；如果 DCU 不可用，先修正平台卡分配/挂载。不要通过安装普通 Torch 替换海光 Torch。

## 3. 配置网络并安装运行环境

需要平台认证代理时，暂停录屏，执行：

```bash
python3 proxy_shell.py
```

输入平台当前地址、用户名、密码，之后在它打开的 Bash 中继续。原地址为 `10.16.1.51:3128`，新实例以平台实际值为准。恢复录屏，不展示 env/printenv 或代理变量值。网络不需代理时跳过该入口。

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
python3 setup_server.py
```

脚本继承海光 Torch 建立 `.venv`，补齐 FFmpeg、OSMesa/OpenGL、编译工具，安装 CPU torchvision、LeRobot 0.4.4、PEFT、hf-libero 等，并固定 MuJoCo 3.3.2。

下载来源：系统包来自 Ubuntu 源；普通 Python 包来自清华 PyPI；torchvision 0.24.0+cpu 来自 PyTorch 官方 CPU wheel 源；LeRobot 来自 GitHub 官方 v0.4.4 压缩包并校验 SHA-256；模型/数据来自 Hugging Face 官方仓库，HTTP 端点可选官方或已验证的镜像。

安装结束确认：

```bash
python3 - <<'PY'
import json
from pathlib import Path
s=json.loads(Path('setup_status.json').read_text())
assert s['status']=='complete',s
print('环境安装完成')
PY
source env.sh
export HF_ENDPOINT=https://hf-mirror.com
python check_env.py
python - <<'PY'
import importlib.metadata as m
for name in ['torch','torchvision','lerobot','transformers','peft','numpy','mujoco','hf-libero','av']:
    print(name,m.version(name))
assert m.version('lerobot')=='0.4.4'
assert m.version('mujoco')=='3.3.2'
PY
```

`env.sh` 保留已经选择的 HF_ENDPOINT，否则默认官方站；仅在明确提供 START_PROXY 时设置代理。完整安装日志为 `logs/setup.log`。安装等待可剪辑，保留开始、阶段进度和完成版本。

## 4. 下载模型和目标训练数据

```bash
printf '%s\n' a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 > dataset_revision.txt
python prepare_data.py 2>&1 | tee logs/prepare_data_recording.log
cat dataset_revision.txt
cat episodes_smoke.json
python prepare_training_subset.py episodes_smoke.json
```

下载 SmolVLA、其内部使用的 SmolVLM2-500M、必要配置/tokenizer/processor，以及目标轨迹对应的数值和视频文件。该 commit 是复现上次实验的选择，PDF 没有指定 commit。

录制任务描述 `put the yellow and white mug in the microwave and close it`，轨迹 `[2,3]`，以及：

```text
observation.images.image (3, 256, 256)
observation.images.image2 (3, 256, 256)
observation.state (8,)
action (7,)
```

`[2,3]` 是轨迹编号，`task_index=2` 是任务编号。训练脚本核验全部加载帧的 task_index 为 2；合并数据文件可能物理包含其他任务，不能因此把其他任务用于训练。最小下载不等于全任务数据；不要把旧缓存中发现 34 条轨迹当成这次完整扫描结果。

## 5. 真正训练一份新模型并保存

```bash
STEPS=100 WARMUP_STEPS=10 LR=0.0001 LORA_R=8 BATCH_SIZE=1 \
EPISODES_FILE=episodes_smoke.json EXPERIMENT_NAME=recording_new \
bash train_experiment.sh

export NEW_CHECKPOINT="$(cat last_experiment_checkpoint.txt)"
cat last_experiment_checkpoint.txt
ls -lh "$NEW_CHECKPOINT"
```

单进程、batch=1、LoRA r8、seed=42，关闭 AMP、compile 和图像增强。录制 step:10 至 step:100、loss 和保存输出。100 步是入门示例，不是 PDF 指定步数，不保证任务成功。

本入口从指定基础模型重新开始 LoRA，不接续旧 adapter。保存新 run，不覆盖基线 `last_checkpoint.txt`。新 checkpoint 指针为 `last_experiment_checkpoint.txt`，本次 run 内另有 `training_recipe.json`。

展示 adapter、config、policy processor 配置及状态文件。它是 LoRA checkpoint，推理还需要基础模型缓存。完整新训练与扩大目标数据方法见 `TRAINING_INFERENCE_GUIDE.md`。

口述：“这是一份本次实际训练的新模型，保存了完整 adapter 与处理器。接下来会明确指定它进行推理。”

## 6. 下载场景资产并准备仿真

```bash
python prepare_sim.py 2>&1 | tee logs/prepare_sim_recording.log
```

下载固定 commit 的 586 个场景文件，逐一验证尺寸/Git/LFS 内容哈希。hf-libero 包提供 BDDL 和官方初始状态。录制：

```text
All 586 LIBERO assets passed size and content hash validation
Main camera: (1, 256, 256, 3)
Wrist camera: (1, 256, 256, 3)
LIBERO reset/render/step OK
```

仿真为 `libero_10/task_id=9`，与训练 task_index=2 对应同一任务。模型在 DCU 计算，OSMesa 在 CPU 渲染，不需实体机器人。当前实测版本为 MuJoCo 3.3.2 + robosuite 1.4.0 + hf-libero 0.1.3；这些不是 PDF 明文指定版本。

## 7. 明确加载刚训练的新模型，执行实际推理

```bash
export NEW_CHECKPOINT="$(cat last_experiment_checkpoint.txt)"
CHECKPOINT="$NEW_CHECKPOINT" bash eval_smoke.sh
```

必须保留 CHECKPOINT 指定；不传时旧入口默认读取基线 `last_checkpoint.txt`。终端会打印 `INFERENCE_CHECKPOINT`，结果目录中的 `inference_checkpoint.txt` 也记录来源。

推理过程：离线加载基础模型 + 本次新 adapter + processor，输入两路图像、机器人状态和任务文字，DCU 输出 7 维动作，模拟器执行后返回新观测，继续循环。推理过程不更新参数。

保留模型加载、实际运行进度、正常结束输出。运行单个 episode，seed=42，最多 500 个控制步。脚本名称沿用 eval_smoke.sh，此处仅用于单次推理演示，正式状态遍历与评分另做。

口述：“现在重新加载刚刚保存的新模型，让它根据图像和状态持续生成动作，完成一次实际仿真运行。”

## 8. 核对结果来源并播放真实视频

```bash
python show_result.py
cat last_inference_result.txt
python -m pip freeze > environment-after.txt
```

脚本优先读取本次推理指针，核对单 episode、task_id，完整解码视频，并打印实际 checkpoint 与结果路径；随后生成 `results_view.ipynb`。

在 JupyterLab 刷新文件列表，打开 Notebook、运行单元格、点击视频播放。核对来源确实为新训练的 checkpoint。未成功可以说明流程连通，不能宣称调优有效。JSON 附带的单次成功信息暂不展开分析。

视频编码时长与推理耗时不同。原实验 500 帧/80 FPS 的视频为 6.25 秒，实际 episode 推理约 85 秒；本次数值以实际输出为准。

## 保存个人实验与后续镜像交付

个人实验保留训练参数、数据 revision/episode 清单、日志、完整 checkpoint、推理 JSON 和视频。下载等待可以加速，保留命令开始、进度与完成；代理凭据输入暂停录屏。

公共镜像仅收录必要运行环境、框架、模型 snapshot、选定目标数据、场景和配置。项目 tests、短验证模型、历史日志和结果不进入镜像。运行代码通过 `smolvla_course_code.zip` 单独上传。按 `IMAGE_PACKAGING_GUIDE.md` 导出干净构建上下文，在新 DCU 实例离线验收后再交付。

重新打开 Terminal：

```bash
export START_ROOT=/public/home/xdzs2026_b0130/你的实际录屏目录
cd "$START_ROOT"
source env.sh
```

联网仍需代理时先进入 `proxy_shell.py` 打开的 Bash。安装失败查 `setup_status.json`/`logs/setup.log`；训练失败查本次 logs；OSMesa 缺失需补系统库。不要因为换实例仅复制 `.venv` 就认定系统层也完整。不要删除状态文件或旧 run 来强行重训。
