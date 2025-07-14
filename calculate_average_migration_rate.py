
# print("细胞的平均迁移速率：", average_migration_rate)
import cv2
import numpy as np
import os
import pandas as pd

def decode_optical_flow_color_image(flow_img):
    # 确保输入为 BGR 格式图像
    hsv_img = cv2.cvtColor(flow_img, cv2.COLOR_BGR2HSV)  # 转换为 HSV
    
    angle = hsv_img[..., 0].astype(np.float32) * (2 * np.pi / 180)  # 色调转换为弧度
    magnitude = hsv_img[..., 2].astype(np.float32) / 255.0  # 归一化亮度

    u = magnitude * np.cos(angle)  # 计算 u 分量
    v = magnitude * np.sin(angle)  # 计算 v 分量
    print("u 的最小值:", np.min(u), "u 的最大值:", np.max(u), "u 的平均值:", np.mean(u))
    print("v 的最小值:", np.min(v), "v 的最大值:", np.max(v), "v 的平均值:", np.mean(v))

    return u, v


def calculate_average_migration_rate(u, v, cell_mask):
    magnitude = np.sqrt(u**2 + v**2)
    cell_magnitude = magnitude[cell_mask]
    # print(f"cell_magnitude: {cell_magnitude}, mean: {np.mean(cell_magnitude) if cell_magnitude.size > 0 else 0}")
    
    average_migration_rate = np.mean(cell_magnitude) if cell_magnitude.size > 0 else 0
    return average_migration_rate

# 输入文件夹路径
optical_flow_folder = 'L:/fenge/true_flo/3'
# optical_flow_img = cv2.imread('L:/fenge/true_flo/2/frame_0029_part_2.jpg')
# print("光流图像的形状:", optical_flow_img.shape)
# print("光流图像的通道:", optical_flow_img[0, 0])  # 查看第一个像素的值

cell_mask_folder = 'L:/fenge/true_dsc/3'
output_csv_path = 'L:/fenge/true_dsc/average_migration_rates1.csv'

# 初始化结果列表
results = []

# 遍历光流文件夹中的每个图像文件
for image_file in os.listdir(optical_flow_folder):
    if image_file.endswith(('.jpg', '.png', '.jpeg')):
        # 读取光流图像
        optical_flow_img = cv2.imread(os.path.join(optical_flow_folder, image_file))
        if optical_flow_img is None:
            print(f"光流图像未找到：{image_file}")
            continue
        
        # 从颜色编码图像中提取 u 和 v 分量
        u, v = decode_optical_flow_color_image(optical_flow_img)

        # 读取细胞区域掩码图像
        mask_file = os.path.join(cell_mask_folder, image_file)
        cell_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if cell_mask is None:
            print(f"细胞掩码图像未找到：{mask_file}")
            continue
        cell_mask = cell_mask > 1

        # 计算细胞的平均迁移速率
        average_migration_rate = calculate_average_migration_rate(u, v, cell_mask)

        # 将结果添加到列表中
        results.append({
            'Image': image_file,
            'Average Migration Rate (pixels/frame)': average_migration_rate
        })

# 将结果保存为CSV文件
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_path, index=False)

print(f"平均迁移速率已保存到 {output_csv_path}")
