import os

from PIL import Image

from pillow_heif import register_heif_opener

def heic_to_jpg(input_file, output_file):
    image = Image.open(input_file)
    image.convert("RGB").save(output_file, "JPEG")

def convert_images_in_directory(directory):
    for filename in os.listdir(directory):
        input_path = os.path.join(directory, filename)
        if filename.lower().endswith('.heic'):
            output_path = os.path.splitext(input_path)[0] + '.jpg'
            heic_to_jpg(input_path, output_path)
        else:
            print(f"Skipped non-livp and non-heic file: {filename}")

register_heif_opener()
# 示例调用
directory = 'images/Origin/Silver/'  # 当前目录
convert_images_in_directory(directory)

#
# # 调用方式
# heic_to_jpg("images/Origin/Silver/2024-07-27 144522.heic", "images/Out/Silver/test.jpg")