# import torch
 
# model_dict = torch.load("C:/Users/m1384/Desktop/100000_raft-sintel.pth")
# # print(model_dict.keys())
# params_dict=model_dict['state_dict']
 
# total_params = 0
# for param_tensor in params_dict.values():
#     # 将当前参数的元素数（即参数大小）加到总和中
#     total_params += param_tensor.numel()
 
# print(f"参数量约为：{total_params/1000000:.2f}M（百万个参数）。")
import torch

# 加载模型参数
params_dict = torch.load("C:/Users/m1384/Desktop/100000_raft-sintel.pth")

# 计算参数总量
total_params = sum(param_tensor.numel() for param_tensor in params_dict.values())

# 打印参数量
print(f"参数量约为：{total_params / 1_000_000:.2f}M（百万个参数）。")
