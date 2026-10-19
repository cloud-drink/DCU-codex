"""Run inside the user's DCU container, before and after dependency installation."""
import os
import sys
import torch

print('Python:', sys.version)
print('torch:', torch.__version__, torch.__file__)
try:
    import torchvision
    print('torchvision:', torchvision.__version__, torchvision.__file__)
except Exception as exc:
    print('torchvision 导入未通过:', type(exc).__name__, str(exc))
print('HIP:', torch.version.hip)
print('CUDA version field:', torch.version.cuda)
print('Visible accelerators:', torch.cuda.device_count())
print('CPU cores:', os.cpu_count())
assert torch.cuda.is_available(), 'DCU 不可见：先检查平台是否分配卡、设备挂载和镜像驱动。'
print('Device:', torch.cuda.get_device_name(0))
print('Device memory GiB:', torch.cuda.get_device_properties(0).total_memory / 2**30)
x = torch.randn(256, 256, device='cuda', requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
print('DCU forward/backward OK:', loss.item())
