import os
from PIL import Image

def resize_images(input_folder, output_folder, target_size):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中的所有图像文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        # 读取图像
        img = Image.open(input_path)

        # 调整图像大小
        img_resized = img.resize(target_size, 3)

        # 保存调整大小后的图像
        img_resized.save(output_path)

if __name__ == "__main__":
    # 设置输入文件夹、输出文件夹和目标尺寸
    path_to_folder1 = 'E:/data_4/flow/image_2'
    path_to_folder2 = 'E:/data_4/flow/image_3'
    output_folder1 = 'D:/cell/test/flow/clean/clean'
    output_folder2 = 'D:/cell/test/flow/final/final'

    target_size = (384, 521)

    # 调整第一个文件夹中的图像
    resize_images(path_to_folder1, output_folder1, target_size)

    # 调整第二个文件夹中的图像
    resize_images(path_to_folder2, output_folder2, target_size)
