import os
import cv2
import numpy as np

def generate_optical_flow(image):
    # Replace this function with your method for optical flow computation
    # For example, you can use cv2.calcOpticalFlowFarneback or any other method
    # to compute optical flow based on the input image.
    # Return a flow field (2D numpy array) representing the optical flow.
    flow = np.zeros((image.shape[0], image.shape[1], 2), dtype=np.float32)
    flow[:, :, 0] = 1.0  # Horizontal motion
    flow[:, :, 1] = 0.0  # Vertical motion
    return flow

def png_to_flo(png_path, flo_path):
    image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    flow = generate_optical_flow(image)

    with open(flo_path, 'wb') as f:
        f.write(b'PIEH')
        np.array([image.shape[1], image.shape[0]], dtype=np.int32).tofile(f)
        flow.tofile(f)

def batch_convert_png_to_flo(png_folder, flo_folder):
    if not os.path.exists(flo_folder):
        os.makedirs(flo_folder)

    for file_name in os.listdir(png_folder):
        if file_name.endswith(".png"):
            png_path = os.path.join(png_folder, file_name)
            flo_path = os.path.join(flo_folder, file_name.replace(".png", ".flo"))
            png_to_flo(png_path, flo_path)

if __name__ == "__main__":
    png_folder_path = "E:/data_4/flow/image_3"
    flo_folder_path = "E:/data_4/flow/image_3_flo"

    batch_convert_png_to_flo(png_folder_path, flo_folder_path)
