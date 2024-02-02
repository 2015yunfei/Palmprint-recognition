import os
from PIL import Image


def merge_images_from_folder(folder_path, output_path):
    # 获取文件夹中所有的 JPG 图片文件路径
    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(".jpg")]

    if not image_paths:
        print("文件夹中没有 JPG 图片。")
        return

    images = [Image.open(image_path) for image_path in image_paths]

    # 获取最大宽度和高度
    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)

    # 创建新的空白图片
    merged_image = Image.new("RGB", (max_width * len(images), max_height), (255, 255, 255))

    # 将图片逐个粘贴到新图片中，居中显示
    for i, image in enumerate(images):
        offset = ((max_width - image.width) // 2, (max_height - image.height) // 2)
        merged_image.paste(image, (i * max_width + offset[0], offset[1]))

    # 保存合并后的图片
    merged_image.save(output_path, format="BMP")


if __name__ == "__main__":
    # 请替换为您的文件夹路径
    folder_path = "./merge_texture.py"

    # 请替换为您希望保存合并后图片的路径
    output_path = "merged_image.jpg"

    merge_images_from_folder(folder_path, output_path)
