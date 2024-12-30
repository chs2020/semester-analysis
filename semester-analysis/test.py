import torch

print("CUDA 可用性:", torch.cuda.is_available())
print("当前设备:", torch.cuda.current_device())
print("设备名称:", torch.cuda.get_device_name(torch.cuda.current_device()))