import cv2
import os

# 输入文件夹路径
input_folder = 'L:/fenge/dsc'
output_folder = 'L:/fenge/huafen-dsc'

# 获取文件夹中的所有图像文件
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 遍历每个图像文件
for image_file in image_files:
    # 读取图像
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    H, W = image.shape[:2]

    # 计算每个部分的尺寸
    part_height = H // 2
    part_width = W // 3

    # 切分图像并保存
    for i in range(2):  # 行
        for j in range(3):  # 列
            part = image[i * part_height:(i + 1) * part_height, j * part_width:(j + 1) * part_width]
            part_filename = f'{os.path.splitext(image_file)[0]}_part_{i * 3 + j + 1}.jpg'
            cv2.imwrite(os.path.join(output_folder, part_filename), part)
