import cv2
import matplotlib.pyplot as plt

def otsu(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用 OpenCV 提供的 Otsu 二值化方法
    _, binary_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print(f"Optimal threshold by OpenCV: {binary_img.max()}")  # 打印阈值
    plt.imshow(binary_img, cmap='gray')
    plt.show()
    return binary_img

# 读取图像并调用Otsu算法
img_name0 = 'frame_0115.png'
img = cv2.imread(img_name0)
img_gray2 = otsu(img)
