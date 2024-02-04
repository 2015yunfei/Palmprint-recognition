import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from PIL import Image, ImageEnhance, ImageOps
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import os


# 训练和验证
def train_and_valid(model, loss_function, optimizer, epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_function(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()

            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = loss_function(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()

        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss:"
            " {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start
            ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
        model_path = 'models_resnet18_ep' + str(num_epochs)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(model, model_path + '/' + dataset + '_model_' + str(epoch + 1) + '.pt')
    return model, history, best_acc, best_epoch


def get_random_factor():
    random_factor = round(random.uniform(0.75, 1.30), 2)
    return random_factor


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

            for i in range(0, num_augmented_images):
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


# 定义超参
batch_size = 64
num_epochs = 300  # 因为要让他过拟合，epoch数要多一些

image_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# 迁移学习
resnet18 = models.resnet18(pretrained=True)
for param in resnet18.parameters():
    param.requires_grad = False
fc_inputs = resnet18.fc.in_features
resnet18.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 99),
    nn.LogSoftmax(dim=1)
)
# 用GPU进行训练
resnet18 = resnet18.to('cuda:0')
# 定义损失函数和优化器
loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet18.parameters())

# 加载数据
dataset = 'Palmprint'
train_directory = os.path.join(dataset, 'trainold')
valid_directory = os.path.join(dataset, 'valid')
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
}

train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)

print(train_data_size, valid_data_size)

main_folder = "./Palmprint/trainold"
output_folder = "./Palmprint/train"
trained_folder = "./models_resnet18_ep300"

# 先删除可能存在之前的训练数据
delete_directory(output_folder)
delete_directory(trained_folder)

trained_model, history, best_acc_old, best_epoch = train_and_valid(resnet18, loss_func, optimizer, num_epochs)

history = np.array(history)
# 绘制第一组数据
plt.plot(history[:, 3], label='Original Accuracy')

my_noise_factor = 0.03
my_size = (100, 100)
total_augmented_images = 6

# 删除第一次训练的数据
delete_directory(output_folder)
delete_directory(trained_folder)

process_subfolders(main_folder, output_folder, num_augmented_images=total_augmented_images)

# 加载数据
dataset = 'Palmprint'
train_directory = os.path.join(dataset, 'train')
valid_directory = os.path.join(dataset, 'valid')
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
}

train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)

print(train_data_size, valid_data_size)

trained_model, history, best_acc, best_epoch = train_and_valid(resnet18, loss_func, optimizer, num_epochs)

model_path = 'models_resnet18_ep' + str(num_epochs)

history = np.array(history)
plt.plot(history[:, 3], label='Enhanced Accuracy')
plt.legend()
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.title('Best original accuracy:' + str(best_acc_old)[0:6] + '\nBest enhanced accuracy:' + str(best_acc)[0:6])
plt.savefig(dataset + model_path + '_accuracy_curve.png')
plt.close()

# 删除第二次训练的数据
delete_directory(output_folder)
delete_directory(trained_folder)
