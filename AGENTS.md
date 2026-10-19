# 工作约定

- 默认使用工作区自带的 Python 环境，除非用户指定其他解释器；本机控制脚本使用 `.venv/bin/python`。
- 后续远端主目录为 `/public/home/xdzs2026_b0196`。默认在该目录及其子目录中进行代码、文件和命令操作。迁移后的基础环境位于 `SothisAI/smolvla`，已有课程代码位于 `smolvla_recording_20261015_231050`。执行命令时明确设置 cwd，文件操作明确填写路径。原 `/public/home/smolvla` 作为迁移源保留。
- 课程代码优先复用 `smolvla_recording_20261015_231050/.venv/bin/python`，其中 PEFT、hf-libero、MuJoCo 等核心依赖已核实存在，torchvision 和 PEFT 导入通过。`SothisAI/smolvla/venv/bin/python` 是独立的基础推理环境，模型在 `SothisAI/smolvla/models/`。不要将旧最小环境的缺包结论套用于课程环境，也不要无必要地重建或混装两套环境。
- 课程目录下 `data/libero-assets` 的 586 个场景资产已校验；OSMesa 与配套 GLAPI 仅解压在 `.runtime/osmesa`，由 `env.sh` 配置，不替换系统库。未微调基线位于 `models/baseline_libero`，复用原权重并适配 state=8/action=7。`prepare_sim.py --prepare-baseline --check-policy` 的渲染、严格加载和单动作小测试已通过；完整 episode 仍交给用户执行。
- 尽可能复用现有代码。修改需求优先在原文件上就地修改，避免新建替代脚本、重复实现或多个版本；仅在确有必要时新增代码文件。
- 负责代码的上传和修改，可以运行必要的小测试。主流程由用户运行，不主动启动主流程、完整训练、全量推理或正式评测，也不提交相关作业。交付时提供修改文件的位置和可直接复制的运行命令，由用户执行；仅在用户后续明确授权时改变此分工。
- 更换容器后，先更新配置中的 `lab_url`，保持 `remote_cwd` 为 `/public/home/xdzs2026_b0196`，运行 `doctor` 验证连接。登录会话可能继续有效，不能默认要求重新登录；若认证失效，再运行 `login`。新容器仍需验证项目目录和项目 Python 环境是否可用。
