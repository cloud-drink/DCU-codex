# 新模型训练与单次推理操作指南

更新：2026-10-14。当前范围为环境配置、训练新 SmolVLA、保存、重载和单次仿真推理。正式成功率测评和评分暂不操作。

## 一、先明确赛题边界

依据课程目录 `参考资料/任务二-赛题介绍.pdf` 第 2–6 页：

- 基础模型必须为 `lerobot/smolvla_base`；SmolVLM2 是它内部的视觉语言骨干。
- 训练仅使用 `lerobot/libero` 中 `task_index=2` 的轨迹。任务为“将黄白色马克杯放入微波炉并关闭微波炉”。episode 编号和 task_index 不同。
- 比赛算力要求为海光 DCU-BW1000 单卡，并在 100 小时内完成参数高效微调。平台资源页需核实具体卡型；原记录只显示 BW/约 64 GiB，不能据此断言已经核实 BW1000。100 小时是比赛限制，不是脚本默认时长；自行记录平台任务起止与累计运行时间。
- 允许基于官方视觉生成整路屏蔽、初始帧冻结；禁止其他机器人轨迹，也不得修改原始动作、机器人状态或语言标签。本入口默认关闭图像增强，直接进行 LoRA。
- 推理任务是 `libero_10/task_id=9`；policy seed=42，最多 500 个控制步。以 LIBERO 的成功判定为准：杯在微波炉内部且门关闭。

PDF 未指定 LeRobot、MuJoCo 版本、LoRA rank、学习率或训练步数。以下版本和参数来自实际跑通的课程实现。100/12/1000 步均不是 PDF 的硬性要求。

## 二、选择工作目录并进入环境

从基础镜像起步：先按 `RECORDING_GUIDE.md` 完成第 1–4 段及仿真准备。所有命令在服务器 JupyterLab Terminal 的 Bash 中运行。

继续现有环境：

```bash
export START_ROOT=/public/home/xdzs2026_b0130/smolvla_start
cd "$START_ROOT"
source env.sh
python check_env.py
```

其他实例应替换成自己的实际目录。代码默认不会写死个人账号；`env.sh` 未设置 START_ROOT 时取自身所在目录，不会自动套用旧平台代理。联网需要认证时先运行 `python3 proxy_shell.py`，然后在它打开的 Bash 中继续。

新启动的基础镜像只保留项目卷时，可能缺 OSMesa/FFmpeg：在有系统安装权限的实例中运行 `python3 restore_system_dependencies.py`。已封装这些库的完整镜像不需重装。

## 三、准备允许使用的训练数据

基础镜像初次准备：

```bash
printf '%s\n' a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 > dataset_revision.txt
python prepare_data.py
cat episodes_smoke.json
python prepare_training_subset.py episodes_smoke.json
```

该数据 revision 来自上次实际运行，是复现选择，不是 PDF 指定的 commit。脚本核验任务描述及所有训练帧 task_index 集合为 `{2}`。

固定 revision、干净目录的起步轨迹为 `[2,3]`。两路图像 `(3,256,256)`，状态 `(8,)`，动作 `(7,)`。两条轨迹用于流程演示，不代表已经使用全部允许数据。

镜像离线使用时，直接使用已包含的 episode 清单，不再下载：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python prepare_training_subset.py episodes_smoke.json
```

若需在联网准备阶段扩大数据范围：

```bash
FULL_TASK_SCAN=1 python prepare_data.py
python prepare_training_subset.py episodes_all.json
```

完整扫描和完整下载所选训练视频是两个步骤；第二条命令负责下载缺失文件并验证目标任务。镜像只能离线训练其中实际包含的轨迹。不要在准备了两条轨迹的镜像中直接指定其他 episode。

## 四、训练一个新的模型

录屏建议训练新 100 步/r8 模型：

```bash
STEPS=100 WARMUP_STEPS=10 LR=0.0001 LORA_R=8 BATCH_SIZE=1 \
EPISODES_FILE=episodes_smoke.json EXPERIMENT_NAME=recording_new \
bash train_experiment.sh

export NEW_CHECKPOINT="$(cat last_experiment_checkpoint.txt)"
cat last_experiment_checkpoint.txt
ls -lh "$NEW_CHECKPOINT"
cat "$(dirname "$(dirname "$(dirname "$NEW_CHECKPOINT")")")/training_recipe.json"
```

本入口从 `lerobot/smolvla_base` 开始新的 LoRA 微调，不接续旧 adapter。新 run 使用独立目录；保留原基线 `last_checkpoint.txt`，新路径写入 `last_experiment_checkpoint.txt`。`steps` 与 scheduler decay 同步，warmup 必须小于 steps。

checkpoint 是 LoRA adapter，至少保留：`adapter_model.safetensors`、`adapter_config.json`、`config.json`、两份 policy processor 配置以及它们引用的处理器状态文件。最稳妥的交付是整个 `pretrained_model` 目录，同时保留基础模型与骨干缓存。

入口可配置 `STEPS/LR/LORA_R/BATCH_SIZE/WARMUP_STEPS/DECAY_LR/EPISODES_FILE/EXPERIMENT_NAME`。例如在已准备好的目标轨迹上进一步训练：

```bash
STEPS=1000 WARMUP_STEPS=10 LR=0.00005 LORA_R=8 BATCH_SIZE=1 \
EPISODES_FILE=episodes_all.json EXPERIMENT_NAME=student_a \
bash train_experiment.sh
```

参数只是示例，不承诺提高效果。多次运行也需遵守单卡和比赛时间限制。

## 五、准备仿真并明确指定新模型推理

首次准备场景时执行：

```bash
python prepare_sim.py
```

资产为固定 commit 的 586 文件，经尺寸和内容哈希校验。OSMesa 在 CPU 上渲染，两路相机使用 256×256；模型计算使用 DCU。hf-libero 包内还必须有 BDDL 场景与官方初始状态。

推理必须显式传入刚训练的模型：

```bash
export NEW_CHECKPOINT="$(cat last_experiment_checkpoint.txt)"
CHECKPOINT="$NEW_CHECKPOINT" bash eval_smoke.sh
python show_result.py
```

不传 CHECKPOINT 时，旧入口默认读取基线 `last_checkpoint.txt`；它不会自动改用新训练的模型。本次更新添加了显式 checkpoint 选择、结果指针和来源记录。

该入口实际调用 LeRobot 的单 episode 仿真推理：

1. 离线加载 SmolVLA 基础模型、LoRA adapter 和 processor。
2. 读取主相机/腕部相机图像、机器人状态及固定任务指令。
3. DCU 模型输出 7 维动作；CPU 模拟器执行并返回下一次观测。
4. 持续推理至 episode 结束或最多 500 步，保存 JSON 和真实仿真视频。

`eval_smoke.sh` 名称沿用既有脚本，此处用于一次推理演示。它不是完整 60 次成绩评测；普通 LeRobot 包装器的状态推进与终止细节也不作为正式评测协议保证。

## 六、播放本次模型的结果

`show_result.py` 优先读取 `last_inference_result.txt`，核验单 episode 和 task_id、完整解码视频，并生成 `results_view.ipynb`。打开 Notebook、运行单元格、点击播放。

结果目录中的 `inference_checkpoint.txt` 记录实际模型来源，可与 `last_experiment_checkpoint.txt` 对照。保留本次 `training_recipe.json`、训练日志、完整 checkpoint、推理 JSON 和视频。它们是实验输出，不默认进入公共环境镜像。

单次结果未成功也可证明训练、保存、重载和推理链路运行；不能据此宣称优化有效。视频播放长度与实际推理耗时不同。

## 七、原聊天中新模型的真实验证情况

2026-10-14 已实测：`STEPS=12 WARMUP_STEPS=2 LR=0.00005 LORA_R=4 BATCH_SIZE=1`，使用 `[2,3]`、630 帧，从基础模型训练并保存新 adapter；第 10 步 loss=0.922。随后新旧两份 checkpoint 均做过五种组合各 20 步的短推理验证。

新验证模型位于服务器：

```text
/public/home/xdzs2026_b0130/smolvla_start/runs/experiment_entry_validation_20261014_105016/checkpoints/last/pretrained_model
```

12 步模型用于入口验证，不是精度提升成果。本次录屏应实际训练自己的新模型；历史测试 run、验证 JSON 和旧视频不加入公共镜像。此次对 CHECKPOINT 选择、来源指针和打包工具的更新做本地检查，仍需按 `IMAGE_PACKAGING_GUIDE.md` 在新镜像 DCU 实例验收。
