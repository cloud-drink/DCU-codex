# DCU 比赛 JupyterLab 控制脚本

脚本在本机运行，通过 Playwright 控制 Edge，在登录会话内使用 Jupyter 文件接口和终端 WebSocket。服务器不需要联网或安装额外组件。

当前远端主目录为 `/public/home/xdzs2026_b0196`，默认命令 cwd 已更新为该路径。基础模型和推理环境放在 `SothisAI/smolvla/`，已有课程代码及历史安装环境放在 `smolvla_recording_20261015_231050/`。`SothisAI/instance/` 与 `SothisAI/instance_service/` 为平台服务目录，项目操作不修改它们。

2026-10-19 已完成基础环境复制：9,333 个文件，19 个模型文件 SHA-256 校验通过，并修复副本中的旧环境入口路径；原目录保留。新路径 Python、torchvision、LeRobot 导入与 DCU 可见性检查通过。另已核实课程目录自己的 `.venv` 中包含 PEFT 0.18.0、hf-libero 0.1.3、MuJoCo 3.3.2、robosuite、bddl 等依赖，torchvision 和 PEFT 导入通过；课程任务优先复用该环境，基础权重复用 `SothisAI/smolvla/models/`。此次没有运行训练或推理主流程。

## 环境与首次登录

当前工作区已创建 Python 3.12 的 `.venv`，并安装 Playwright。使用本机已安装的 Microsoft Edge，无需再下载 Chromium。

依赖版本记录在 `requirements.lock.txt`，如需重装，在工作区虚拟环境中执行 `.venv/bin/python -m pip install -r requirements.lock.txt`。

```bash
.venv/bin/python dcu.py login
```

在新打开的**自动化 Edge 窗口**中登录比赛平台，进入 JupyterLab。脚本验证文件接口可用后会关闭这个窗口，登录状态保存在 `.dcu/browser-profile/`，后续命令默认在后台运行。日常 Edge 窗口中的登录状态不会自动出现在独立配置中。验证码、双因素验证由你手动完成。

默认读取 `config.example.json`。服务器重新分配后，将配置复制到 `config.json` 并修改 `lab_url`、`remote_cwd`：

```bash
cp config.example.json config.json
.venv/bin/python dcu.py --config config.json login
.venv/bin/python dcu.py --config config.json doctor
```

全局选项 `--config`、`--visible`、`--cdp` 放在子命令前。需要查看浏览器时加 `--visible`。

更换容器时，先更新实际使用的配置中的 `lab_url`，保持 `remote_cwd` 为 `/public/home/xdzs2026_b0196`，然后运行 `doctor`。本机浏览器配置会保留，但新容器是否接受原登录会话取决于平台；如果认证失效，再运行 `login`。更新配置后不会自动发现或选择新容器，需提供它的新 JupyterLab 地址。连接成功后还需确认项目目录和项目 Python 环境在新容器中可用。

## 文件与命令

```bash
# 检查登录、文件接口和终端接口（终端 WebSocket 由 exec 实际验证）
.venv/bin/python dcu.py doctor

# 目录查看和文本读取
.venv/bin/python dcu.py ls /public/home/xdzs2026_b0196
.venv/bin/python dcu.py read /public/home/xdzs2026_b0196/smolvla_recording_20261015_231050/README.md

# 创建独立的测试目录，父目录需要已存在
.venv/bin/python dcu.py mkdir /public/home/xdzs2026_b0196/dcu_bridge_test

# 上传后回读校验；覆盖已有文件需要明确添加 --overwrite
.venv/bin/python dcu.py upload examples/smoke_test.py /public/home/xdzs2026_b0196/dcu_bridge_test/smoke_test.py
.venv/bin/python dcu.py download /public/home/xdzs2026_b0196/dcu_bridge_test/smoke_test.py ./downloaded_smoke_test.py

# cwd 是服务器的绝对路径；不会继承你手动打开的终端环境
.venv/bin/python dcu.py exec 'pwd; command -v python; ls -la' --cwd /public/home/xdzs2026_b0196
.venv/bin/python dcu.py exec 'python smoke_test.py' --cwd /public/home/xdzs2026_b0196/dcu_bridge_test

# 执行本地 shell 脚本的内容；仅发送内容，不会自动上传脚本引用的其他文件
.venv/bin/python dcu.py exec --file examples/smoke_test.sh --timeout 60
```

`exec` 每次创建专用终端，实时获取合并后的 stdout/stderr，以随机完成标记判断退出码，不依赖提示符或固定等待。它会返回远端命令的退出码，并将输出和状态写到 `.dcu/runs/*.log`、`*.json`。命令结束、超时或本地 Ctrl+C 时清理此次创建的终端。输出保留 ANSI 控制序列，尾部最多约 100 个字符会暂存，等待确认完成标记。

超时会请求 Ctrl+C 并关闭专用终端，**不保证停止主动脱离终端的后台进程或已提交的调度任务**。断连意味着命令状态可能未知；提交作业等有副作用的命令不能直接重试。当前版本适合前台命令，尚未实现后台作业管理、断线续接和目录批量同步。

每次命令启动新的 bash 登录子 shell。若需要 Conda、module 或环境变量，须在命令中显式激活，例如 `source .../activate; python ...`。文件接口路径相对 Jupyter 文件浏览器根目录；当前截图根目录是 `/`，所以示例使用对应绝对路径。`remote_cwd` 不决定文件接口根目录。

## 可选：连接已有调试浏览器

如果浏览器**本来就开启了 CDP 调试端口**，可用 `--cdp http://127.0.0.1:9222` 复用其登录。普通 Edge 窗口不能直接这样连接。本工具不会替你退出现有 Edge、修改它的启动参数或复制日常浏览器的登录凭据。连接模式结束时仅断开控制连接，保留已有标签页。

## 本地验证

```bash
.venv/bin/python -m unittest discover -s tests -v
```

测试通过真实 Edge 连接本机模拟 Jupyter 服务，覆盖代理前缀、XSRF、二进制和中文路径、覆盖保护、过期登录、WebSocket 分帧、非零退出码和超时清理。测试不连接比赛服务器。

真实服务器联调已于 2026-10-19 通过：登录状态复用、独立测试目录创建、文件上传回读校验、下载校验、终端执行及退出码和日志回收均成功。测试主机是 `worker-0`，烟雾测试使用 `/usr/local/bin/python3`（Python 3.11.9），实际比赛任务仍需显式激活自己的环境。

需要重新验证真实服务器时执行：

```bash
.venv/bin/python verify_remote.py
# 自定义配置：.venv/bin/python verify_remote.py --config config.json
```

每次验证都会在 `remote_cwd` 下创建唯一的 `dcu_bridge_test_时间-随机值` 目录，并保留测试文件供查看。测试报告、下载文件和执行日志保存到本地 `.dcu/verification/`。

`.dcu/` 包含登录会话和运行日志，已加入 `.gitignore`。同一浏览器配置一次只能运行一个命令；不要并发启动脚本。TLS 校验默认开启。

协议参考：[Playwright 持久化会话](https://playwright.dev/python/docs/api/class-browsertype#browser-type-launch-persistent-context)、[Jupyter Server API](https://jupyter-server.readthedocs.io/en/latest/developers/rest-api.html)。
