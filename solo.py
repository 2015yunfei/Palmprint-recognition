from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt

original_image = '001-1.bmp'

saturation_factor = 1.5  # 增加饱和度的因子
enhancer = ImageEnhance.Color(original_image)
saturated_image = enhancer.enhance(saturation_factor)

# 显示饱和度调整后的图像
plt.subplot(4, 4, 8)
plt.title('Saturated Image')
plt.imshow(saturated_image)
plt.axis('off')
