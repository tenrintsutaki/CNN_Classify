import os

import cv2
from rembg import remove
import time

def resize_image(input_path, output_path, size=(256, 256)):
    # 读取图像
    image = cv2.imread(input_path)

    # 检查是否成功读取图像
    if image is None:
        print("Error: Unable to read image")
        return

    # 调整图像大小
    resized_image = cv2.resize(image, size)

    # 保存调整后的图像
    cv2.imwrite(output_path, resized_image)

def remove_bg(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)


if __name__ == '__main__':
    input_directory = 'images/Origin/Silver/'
    resize_directory = 'images/Resize/Silver/'
    output_directory = 'images/Out/Silver/'
    process = 0

    for filename in os.listdir(input_directory): # 全都变成256 x 256
        resize_image(input_directory + filename, resize_directory + filename)
        print(f"Resizing image process {process/len(os.listdir(input_directory))}")
        process += 1

    process = 0
    for filename in os.listdir(resize_directory): # 全都变成256 x 256
        remove_bg(resize_directory + filename, output_directory + filename)
        print(f"Removing BG image process {process / len(os.listdir(resize_directory))}")
        process += 1
