from PIL import Image, ImageEnhance
import numpy as np
import cv2
import matplotlib.pyplot as plt

# def enhance_contrast(img):
#     # 将图像转换为PIL格式
#     pil_img = Image.fromarray(img)

#     # 检查并转换图像模式为适合增强的模式 ('L' 表示灰度图像)
#     if pil_img.mode not in ['RGB', 'L']:
#         pil_img = pil_img.convert('L')

#     enhancer = ImageEnhance.Contrast(pil_img)
#     enhanced_img = enhancer.enhance(10)  # 增强对比度，值越大对比度越高
#     return np.array(enhanced_img)

# # 读取.tif文件
# img = cv2.imread('man_seg114.tif', cv2.IMREAD_UNCHANGED)

# if img is not None:
#     # 增强对比度
#     enhanced_img = enhance_contrast(img)
    
#     # 将增强后的图像转换为PIL格式并保存为.png
#     enhanced_pil_img = Image.fromarray(enhanced_img)
#     enhanced_pil_img.save('enhanced_image.png')  # 保存为.png格式

#     # 显示图像
#     plt.imshow(enhanced_img, cmap='gray')
#     plt.title('Enhanced Segmentation Image')
#     plt.show()
# else:
#     print("Failed to load the image. Please check the file path.")

import os
from PIL import Image, ImageEnhance
import numpy as np
import cv2

def enhance_contrast(img):
    # 将图像转换为PIL格式
    pil_img = Image.fromarray(img)

    # 检查并转换图像模式为适合增强的模式 ('L' 表示灰度图像)
    if pil_img.mode not in ['RGB', 'L']:
        pil_img = pil_img.convert('L')

    enhancer = ImageEnhance.Contrast(pil_img)
    enhanced_img = enhancer.enhance(10)  # 增强对比度，值越大对比度越高
    return np.array(enhanced_img)

def process_images_in_folder(tif_folder, png_folder):
    # 确保保存文件夹存在
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)

    # 遍历tif文件夹中的所有.tif文件
    for filename in os.listdir(tif_folder):
        if filename.endswith('.tif'):
            # 读取.tif文件
            tif_path = os.path.join(tif_folder, filename)
            img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)

            if img is not None:
                # 增强对比度
                enhanced_img = enhance_contrast(img)

                # 检查增强后的图像是否为正确格式
                if enhanced_img is None:
                    print(f"Failed to enhance contrast for {filename}")
                    continue

                # 将增强后的图像转换为PIL格式并保存为.png
                png_filename = os.path.splitext(filename)[0] + '.png'
                png_path = os.path.join(png_folder, png_filename)

                try:
                    # 保存图片时确保是uint8格式，否则PIL可能无法处理
                    enhanced_pil_img = Image.fromarray(enhanced_img.astype(np.uint8))
                    enhanced_pil_img.save(png_path)
                    print(f"Processed {filename} and saved as {png_filename}")
                except Exception as e:
                    print(f"Error saving {png_filename}: {e}")
            else:
                print(f"Failed to load {filename}")

# 使用示例
tif_folder = 'C:/Users/m1384/Desktop/SEG'  # 替换为.tif文件夹路径
png_folder = 'C:/Users/m1384/Desktop/PNG'  # 替换为保存.png文件的文件夹路径

process_images_in_folder(tif_folder, png_folder)


# ######两图相减
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # 读取灰度图像
# img1 = cv2.imread('Figure_4.png', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('Figure_3.png', cv2.IMREAD_GRAYSCALE)

# # 检查两张图像的尺寸是否相同
# if img1.shape == img2.shape:
#     # 计算像素级的差异
#     img_diff = np.abs(img1 - img2)  # 计算绝对值差异，确保不会得到负值

#     # 显示结果图像
#     plt.imshow(img_diff, cmap='gray')
#     plt.title('Absolute Difference Image')
#     plt.show()
# else:
#     print("The images do not have the same dimensions.")
# ################只保存前景##########################
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def extract_foreground(a, b):
#     # 调整图像尺寸以匹配
#     if a.shape != b.shape[:2]:
#         print(f"Resizing image b from {b.shape[:2]} to {a.shape}")
#         b = cv2.resize(b, (a.shape[1], a.shape[0]))  # 调整 b 的尺寸为与 a 相同
    
#     # 将掩模图像转换为二值图像
#     _, mask = cv2.threshold(a, 127, 255, cv2.THRESH_BINARY)
    
#     # 计算前景图像
#     foreground = cv2.bitwise_and(b, b, mask=mask)
    
#     return foreground

# # 读取图像，a为分割图像(掩模)，b为原始图像
# a = cv2.imread('Figure_7.png', cv2.IMREAD_GRAYSCALE)  # 分割图像作为掩模
# b = cv2.imread('t113.png')  # 原始图像

# # 打印图像尺寸
# print(f"Image a size: {a.shape}")
# print(f"Image b size: {b.shape}")

# # 提取前景图像
# foreground_img = extract_foreground(a, b)

# # 显示前景图像
# plt.imshow(cv2.cvtColor(foreground_img, cv2.COLOR_BGR2RGB))
# plt.title('Foreground Image')
# plt.show()

# # 保存前景图像
# cv2.imwrite('foreground.png', foreground_img)


# # 提取前景图像
# foreground_img = extract_foreground(a, b)

# # 显示前景图像
# plt.imshow(cv2.cvtColor(foreground_img, cv2.COLOR_BGR2RGB))
# plt.title('Foreground Image')
# plt.show()

# # 保存前景图像
# cv2.imwrite('foreground.png', foreground_img)



