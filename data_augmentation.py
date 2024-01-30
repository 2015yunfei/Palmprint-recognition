import os
from PIL import Image, ImageEnhance, ImageOps
import random
import numpy as np
import shutil

my_noise_factor = 0.03
my_size = (100, 100)


# 因为旋转处理造成的黑边会对模型训练产生影响，所以删除这个函数
# def get_random_angle():
#     """
#     返回一个旋转角度
#     :return:得到一个随机的旋转角度
#     """
#     return random.uniform(0, 180)


def get_random_factor():
    random_factor = round(random.uniform(0.75, 1.30), 2)
    return random_factor


def random_flip0(image):
    """
    该函数一定会翻转图片
    :param image: 原图像
    :return: 返回一个进行水平翻转或者竖直翻转或者水平和竖直同时翻转的图像
    """
    if random.choice([True, False]):
        # 随机选择是否进行水平翻转
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif random.choice([True, False]):
        # 随机选择是否进行竖直翻转
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        # 既没有水平翻转也没有竖直翻转的话，就既水平也竖直的翻转一下
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image


def random_flip1(image):
    """
    随机返回进行下列变换
    1.进行水平翻转
    2.竖直翻转
    3.水平和竖直都反转
    4.不翻转
    的图像
    :param image: 原图像
    :return: 翻转处理
    """
    if random.choice([True, False]):
        # 随机选择是否进行水平翻转
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif random.choice([True, False]):
        # 随机选择是否进行竖直翻转
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    elif random.choice([True, False]):
        # 随机选择是否进行水平和竖直翻转
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        pass
    return image


def crop_center(image, size=(100, 100)):
    # 获取图像的宽度和高度
    width, height = image.size

    # 计算截取区域的左上角和右下角坐标
    left = (width - size[0]) // 2
    top = (height - size[1]) // 2
    right = left + size[0]
    bottom = top + size[1]

    # 截取图像中心的指定大小区域
    cropped_image = image.crop((left, top, right, bottom))

    return cropped_image


# 添加噪声函数
def add_noise(image, noise_factor=0.05):
    """
    在图像中添加随机噪声。
    :param image: 输入图像
    :param noise_factor: 噪声因子，控制噪声的强度
    :return: 添加噪声后的图像
    """
    img_array = np.array(image)
    noise = np.random.normal(scale=noise_factor, size=img_array.shape)
    noisy_image_array = img_array + noise
    noisy_image_array = np.clip(noisy_image_array, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image_array)


# 定义数据增强函数
def data_augmentation(input_path, output_path, num_augmented_images=5):
    # 创建输出路径
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 遍历输入路径下的所有BMP图像文件
    for filename in os.listdir(input_path):
        if filename.endswith(".bmp"):
            image_path = os.path.join(input_path, filename)

            # 打开图像
            img = Image.open(image_path)

            # 在新目录中保存原始图像
            img.save(os.path.join(output_path, f"{filename[:-4]}_old.bmp"))

            for i in range(0, 4):
                # 调整亮度
                enhancer = ImageEnhance.Brightness(img)
                bright_img = enhancer.enhance(get_random_factor())  # 亮度可以根据需求调整
                bright_img.save(os.path.join(output_path, f"{filename[:-4]}_bright_{1 + i}.bmp"))

                # 调整对比度
                enhancer = ImageEnhance.Contrast(img)
                contrast_img = enhancer.enhance(get_random_factor())  # 对比度可以根据需求调整
                contrast_img.save(os.path.join(output_path, f"{filename[:-4]}_contrast_{1 + i}.bmp"))

                # 调整对比度和亮度
                enhancer = ImageEnhance.Contrast(img)
                contrast_img = enhancer.enhance(get_random_factor())  # 对比度可以根据需求调整
                enhancer = ImageEnhance.Brightness(contrast_img)
                bright_img = enhancer.enhance(get_random_factor())  # 亮度可以根据需求调整
                bright_img.save(os.path.join(output_path, f"{filename[:-4]}_contrast_bright_{1 + i}.bmp"))

                # 在图像中心截取一个my_size大小的图片
                # small_img = crop_center(img, my_size)
                # small_img.save(os.path.join(output_path, f"{filename[:-4]}_small_{1 + i}.bmp"))


# 遍历主文件夹下的子文件夹
def process_subfolders(main_folder, output_folder, num_augmented_images=5):
    for root, dirs, files in os.walk(main_folder):
        for subfolder in dirs:
            subfolder_path = os.path.join(root, subfolder)
            output_subfolder_path = os.path.join(output_folder, subfolder)

            # 对每个子文件夹中的bmp图片应用数据增强
            data_augmentation(subfolder_path, output_subfolder_path, num_augmented_images)


def delete_directory(directory_path):
    if os.path.exists(directory_path):
        # 如果目录存在，递归删除目录及其内容
        shutil.rmtree(directory_path)
        print(f"目录 {directory_path} 已删除。")
    else:
        print(f"目录 {directory_path} 不存在。")


if __name__ == "__main__":
    main_folder = "./Palmprint/trainold"
    output_folder = "./Palmprint/train"
    trained_folder = "./models_resnet18_ep300"

    delete_directory(output_folder)
    delete_directory(trained_folder)

    # main_folder = "./testimg"
    # output_folder = "./testoutput"

    process_subfolders(main_folder, output_folder, num_augmented_images=2)
