# 必要环境镜像与独立代码包交付

更新：2026-10-14。目标为其他同学在兼容的海光 DCU 实例中离线训练 SmolVLA、保存模型并运行 LIBERO 单次推理。正式测评暂不纳入最小交付。本文件提供可执行的导出工具与构建步骤；当前尚未实际构建、导入或验收新镜像。

## 一、交付拆成两件

| 交付件 | 内容 | 使用方式 |
|---|---|---|
| 环境镜像 | 原海光 Torch/DTK/Python、系统库、`.venv`、LeRobot 可编辑源码、两个模型仓库的选定 snapshot、所选目标训练数据、完整场景资产、LIBERO 场景/初始状态、环境入口、版本/哈希清单 | 平台导入镜像，分配单卡 BW1000，在兼容宿主驱动上启动 |
| `smolvla_course_code.zip` | 安装/准备/训练/单次推理/播放代码及操作说明 | 本地上传 JupyterLab，解压到镜像的 START_ROOT；可独立更新 |

公共环境镜像默认不带任何微调 checkpoint。学生可以从基础模型训练新的 adapter。确需交付一个指定模型时，把其整个 `pretrained_model` 目录作为独立模型包上传；不可只拿 adapter 权重而丢弃 processor。不要把历史训练目录整体加入镜像。

开发用 `smolvla_single_dcu_start.zip` 仍保留评测、测试、验证记录，供维护者复核。它和公共最小代码包用途不同，不能把开发 ZIP 的全部内容都塞进镜像。

## 二、进入镜像的必要文件

导出工具 `export_image_context.py` 使用明确清单，而不是复制整个项目目录。

| 路径/对象 | 收录规则 |
|---|---|
| 原基础镜像 | 使用平台实际 registry 引用，优先固定 digest；包含海光 Torch、DTK、系统 Python 及继承的包 |
| 系统库 | 从已准备服务器记录 apt 包版本，Dockerfile 补齐 FFmpeg、OSMesa/OpenGL 和相关安装依赖 |
| `.venv/` | 收录安装后的依赖与包元数据，排除 pycache；保留解释器及内部 symlink |
| `lerobot/` | 框架可编辑源码和安装元数据；排除仓库 `.git/.github/tests/docs/examples/benchmarks` |
| `cache/huggingface/hub/` | 仅 `lerobot/smolvla_base` 与 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 的 main 引用及其选定 snapshot；解引用 snapshot 文件，无需另外携带整个 blobs 缓存 |
| `data/libero/meta/` | 官方数据元信息，保持原始内容 |
| `data/libero/data/`、`videos/` | 仅 episode 清单实际引用的 parquet 和视频，导出前离线加载并检查 task_index=2 |
| `data/libero-assets/` | 固定资产清单中的 586 文件，经尺寸/Git/LFS 哈希校验；不凭猜测删除 task9 可能间接依赖的纹理、网格 |
| `.venv` 中 hf-libero | 保留 BDDL、官方初始状态和仿真代码 |
| `cache/libero-config/config.yaml` | 包路径与场景路径，导出时保留绝对路径 |
| `env.sh`、`dataset_revision.txt`、episode 清单、`assets_manifest.json`、`IMAGE_MANIFEST.json` | 进入环境、确定数据/模型/资产来源与实际收录范围 |

合并 parquet/video 文件可能物理包含其他任务，不能随意剪裁它们；训练加载范围必须通过所选 episode 限制并检查为 `{2}`。这与 PDF 的“不得使用其他机器人轨迹”要求一致。

不收录项目的 `tests/`、`logs/`、`runs/`、历史 checkpoint、`last_*` 指针、验证 JSON/CSV、`results_view.ipynb`、录屏视频、备份、安装临时目录、源码压缩包、pip 下载缓存、HF token/locks 和代理凭据。导出清单不会读取整个用户 home。依赖包内部文件保持完整，避免为删小文件破坏包的正常导入。

## 三、先确定镜像中的固定路径

`.venv/bin` 的入口、可编辑安装和 LIBERO 配置都可能包含绝对路径。本工具将源 START_ROOT 保留为镜像同一路径，**不迁移或重写已验证的环境**。

可以直接导出当前服务器的 `/public/home/xdzs2026_b0132/smolvla_start`，前提是新实例不会用另一个挂载覆盖这个路径。公共课程镜像更建议先在 `/opt/smolvla` 独立准备，再导出；这样不会受不同账号的 home 挂载影响。

在兼容实例中，由平台允许的系统管理员创建 `/opt/smolvla` 并赋予实验用户读写权限。然后将最小代码 ZIP 上传到此目录：

```bash
export START_ROOT=/opt/smolvla
cd "$START_ROOT"
python3 -m zipfile -e smolvla_course_code.zip .
python3 check_env.py
```

当前平台使用各账号自己的 `.ai_user_info/ai_proxy`，先加载代理，再继续（不要复制凭据到镜像）：

```bash
source /public/home/你的账号/.ai_user_info/ai_proxy
export START_PROXY="${http_proxy:-${HTTP_PROXY:-}}"
export HTTP_PROXY="$START_PROXY" HTTPS_PROXY="$START_PROXY"
export http_proxy="$START_PROXY" https_proxy="$START_PROXY"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
python3 setup_server.py
source env.sh
printf '%s\n' a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 > dataset_revision.txt
python prepare_data.py
python prepare_sim.py
```

不能直接把旧路径的 `.venv` 复制到 `/opt/smolvla`。应在该固定路径建立并验证环境。上述安装流程复用原基础镜像海光 Torch；生成新 `.venv`。实际包版本由导出清单记录。

为让同学可以选择全部允许轨迹，联网准备阶段建议补齐目标任务数据：

```bash
FULL_TASK_SCAN=1 python prepare_data.py
python prepare_training_subset.py episodes_all.json
```

若仅准备两条，则镜像可离线调整学习率、步数、LoRA 等，但不能声称已携带整个目标任务数据。运行前需确认磁盘足够容纳源资源、构建上下文和最终镜像。导出所选文件后会显示清单和实际大小，不能假定 ZIP 大小就是镜像大小。

## 四、导出干净构建上下文

工具随最小代码 ZIP 一并上传，但不进入环境镜像。无需 Docker 即可导出；它不删除源目录、不训练、不下载。

先从平台查到真实基础镜像的完整 registry 引用，不能仅凭 UI 展示名称猜仓库。优先取得 digest：

```bash
read -r -p '平台原始基础镜像完整引用（优先带 @sha256:digest）: ' BASE_IMAGE
export IMAGE_CONTEXT=/public/home/你的账号/smolvla_image_context_$(date +%Y%m%d_%H%M%S)

python export_image_context.py \
  --root="$START_ROOT" --output="$IMAGE_CONTEXT" \
  --base-image="$BASE_IMAGE" --episodes-file=episodes_all.json
```

`IMAGE_CONTEXT` 必须是源目录之外的新目录，替换账号路径。两条轨迹版本把最后一项改为 `--episodes-file=episodes_smoke.json`。

工具预先检查：当前解释器属于对应 `.venv`、海光 HIP Torch、LeRobot 0.4.4、目标轨迹离线可读取、模型 snapshot 完整基础项、586 场景哈希正确和系统库已安装。失败时停止，不能用缺失资源继续交付。

生成：

```text
smolvla_image_context_日期时间/
├── Dockerfile
├── .dockerignore
├── verify_image_software.py
├── export_summary.json
└── payload/                   镜像必需内容
    └── IMAGE_MANIFEST.json    版本、模型/数据 revision、文件 SHA-256
```

镜像内不含维护者运行脚本，因此不含测试或历史输出。snapshot 的文件解引用复制后无需依赖被排除的 HF blob symlink。版本清单也记录当前镜像是否使用 digest 固定；只有 tag 时不能宣称原基础镜像不可变。

## 五、构建与导入

在支持 Docker 构建的 Linux 主机或平台镜像构建服务中：

```bash
docker pull "$BASE_IMAGE"
RUNTIME_USER="$(docker image inspect --format '{{.Config.User}}' "$BASE_IMAGE")"
docker build --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg RUNTIME_USER="${RUNTIME_USER:-root}" \
  -t smolvla-course:20261014 "$IMAGE_CONTEXT"
docker save -o smolvla-course-20261014.tar smolvla-course:20261014
```

构建需要访问基础镜像 registry 和 Ubuntu 软件源；运行时模型/数据/资产可离线。Dockerfile 固定已记录的 apt 包版本，版本源失效时应修复软件源并重建、验证，不能悄悄替换版本。构建检查 Python/HIP Torch 和关键包版本，不需要在构建机上分配 DCU。

使用干净的官方基础镜像加 `payload` 构建，不对当前混有临时文件和凭据的实例直接 commit。持久化挂载不自动进入平台“保存镜像”；必须显式加入本工具生成的 payload。平台若只提供保存实例镜像，需要先新建干净构建实例，仅放入 payload、系统库，再保存，并检查平台是否把挂载内容真正收录。

本 Dockerfile 保留基础镜像 JupyterLab 的 ENTRYPOINT/CMD，构建时用 RUNTIME_USER 恢复基础镜像用户并给实验目录赋权。平台可能覆盖运行用户、端口与挂载，应按实际平台设置；实验用户必须能写 START_ROOT 中的源码、cache/libero-config、logs 和 runs，管理员应按平台 UID/GID 赋权。不要将空目录挂到整个 START_ROOT 上。

镜像还需兼容的 BW1000 与宿主驱动/设备挂载。宿主内核驱动和实体 DCU 无法封装进应用镜像；不要套用 NVIDIA 的设备参数。

## 六、其他同学如何使用

1. 从导入镜像启动一个新实例，分配一张符合 PDF 的 BW1000。
2. 进入 JupyterLab，找到固定 START_ROOT（见镜像 IMAGE_MANIFEST）。上传 `smolvla_course_code.zip` 并解压到这个目录。
3. 打开 Terminal，进入 Bash、激活环境，按镜像包含的数据范围训练新模型。

以 `/opt/smolvla` 镜像为例：

```bash
export START_ROOT=/opt/smolvla
cd "$START_ROOT"
python3 -m zipfile -e smolvla_course_code.zip .
source env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python check_env.py
python prepare_training_subset.py episodes_image.json

STEPS=100 WARMUP_STEPS=10 LR=0.0001 LORA_R=8 BATCH_SIZE=1 \
EPISODES_FILE=episodes_smoke.json EXPERIMENT_NAME=my_first_model \
bash train_experiment.sh
CHECKPOINT="$(cat last_experiment_checkpoint.txt)" bash eval_smoke.sh
python show_result.py
```

若上传独立模型，把整个目录放入 `models/my_model/pretrained_model`，然后：

```bash
CHECKPOINT="$START_ROOT/models/my_model/pretrained_model" bash eval_smoke.sh
python show_result.py
```

代码可以重新上传更新，环境镜像无需每次重建。修改依赖或 LeRobot 源码后应记录版本并重新验收，不能继续宣称与原环境相同。

## 七、发布前必须实际完成的镜像验收

不能把源服务器跑通过当成镜像验证完成。在一个未挂载旧实验目录的新 DCU 实例中，临时限制外网，实际执行上面的离线数据读取、新训练、保存、新 checkpoint 重载、单次 500 步上限推理和视频解码。

检查 `IMAGE_MANIFEST.json` 指定资源全部存在、Torch 未替换、只用所选 task_index=2 数据、推理记录确实指向新模型。确认项目 tests、验证 JSON、旧 runs/logs 和凭据未打入镜像。成功率不作为当前入门验收门槛。

保存新实例的验收记录在镜像外，并记录镜像 digest/导出 tar 的 SHA-256。上述验收通过后，才可对其他同学声明镜像能够离线复用。正式 60 次测评仍留到后续阶段。
