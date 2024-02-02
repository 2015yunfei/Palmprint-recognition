import os
from PIL import Image, ImageEnhance, ImageOps
import random
import numpy as np
import shutil


def get_random_angle():
    """
    返回一个旋转角度
    :return:得到一个随机的旋转角度
    """
    return random.uniform(0, 180)


# def get_random_factor():
#     random_factor = round(random.uniform(0.75, 1.30), 2)
#     return random_factor


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


def data_augmentation(image_path, output_path):
    # 创建输出路径
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 打开图像
    original_img = Image.open(image_path)

    # 调整亮度
    enhancer = ImageEnhance.Brightness(original_img)
    bright_img = enhancer.enhance(1.12)  # 亮度可以根据需求调整

    # 调整对比度
    enhancer = ImageEnhance.Contrast(original_img)
    contrast_img = enhancer.enhance(1.12)  # 对比度可以根据需求调整

    # 随机噪声
    noisy_img = add_noise(original_img, noise_factor=my_noise_factor)

    # 图像翻转
    flipped_img = original_img.transpose(Image.FLIP_LEFT_RIGHT)

    # 图像旋转
    rotated_img = original_img.rotate(get_random_angle())

    # 在图像中心截取一个my_size大小的图片
    small_img = crop_center(original_img, my_size)

    # 输出每个增强图像及原图像的拼接图像
    bright_img_with_original = Image.new('RGB', (original_img.width * 2, original_img.height))
    bright_img_with_original.paste(original_img, (0, 0))
    bright_img_with_original.paste(bright_img, (original_img.width, 0))
    bright_img_with_original.save(os.path.join(output_path, f"bright_with_original.bmp"))

    contrast_img_with_original = Image.new('RGB', (original_img.width * 2, original_img.height))
    contrast_img_with_original.paste(original_img, (0, 0))
    contrast_img_with_original.paste(contrast_img, (original_img.width, 0))
    contrast_img_with_original.save(os.path.join(output_path, f"contrast_with_original.bmp"))

    noisy_img_with_original = Image.new('RGB', (original_img.width * 2, original_img.height))
    noisy_img_with_original.paste(original_img, (0, 0))
    noisy_img_with_original.paste(noisy_img, (original_img.width, 0))
    noisy_img_with_original.save(os.path.join(output_path, f"noisy_with_original.bmp"))

    flipped_img_with_original = Image.new('RGB', (original_img.width * 2, original_img.height))
    flipped_img_with_original.paste(original_img, (0, 0))
    flipped_img_with_original.paste(flipped_img, (original_img.width, 0))
    flipped_img_with_original.save(os.path.join(output_path, f"flipped_with_original.bmp"))

    rotated_img_with_original = Image.new('RGB', (original_img.width * 2, original_img.height))
    rotated_img_with_original.paste(original_img, (0, 0))
    rotated_img_with_original.paste(rotated_img, (original_img.width, 0))
    rotated_img_with_original.save(os.path.join(output_path, f"rotated_with_original.bmp"))

    small_img_with_original = Image.new('RGB', (original_img.width * 2, original_img.height))
    small_img_with_original.paste(original_img, (0, 0))
    small_img_with_original.paste(small_img, (original_img.width, 0))
    small_img_with_original.save(os.path.join(output_path, f"small_with_original.bmp"))


if __name__ == "__main__":
    output_path = "./samples"
    image_path = "008-3.bmp"

    my_noise_factor = 0.03
    my_size = (100, 100)

    data_augmentation(image_path, output_path)
