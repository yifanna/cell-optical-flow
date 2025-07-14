
import torch

# 加载模型参数
params_dict = torch.load("C:/Users/m1384/Desktop/100000_raft-cell.pth")

# 计算参数总量
total_params = sum(param_tensor.numel() for param_tensor in params_dict.values())

# 打印参数量
print(f"参数量约为：{total_params / 1_000_000:.2f}M（百万个参数）。")
