import sys

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class CNN(nn.Module):
    """
    卷积层1：输入为 3×256×256，输出通道数为16，卷积核大小为3，填充为1。
    池化层1：池化核大小为2，步幅为2。
    卷积层2：输入为 16×128×128，输出通道数为32，卷积核大小为3，填充为1。
    池化层2：池化核大小为2，步幅为2。
    卷积层3：输入为 32×64×64，输出通道数为64，卷积核大小为3，填充为1。
    池化层3：池化核大小为2，步幅为2。
    全连接层1：输入为 64×32×32，输出为512个神经元。
    全连接层2：输入为512个神经元，输出为2个神经元（类别数）
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 加载模型
if __name__ == '__main__':
    arg_path = sys.argv[1]
    model = CNN().cuda()
    model.load_state_dict(torch.load('cnn_classifier.pth'))
    model.eval()  # 将模型设置为评估模式

    # 定义数据预处理
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 读取和预处理图片
    image_path = arg_path
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(e)

    image = data_transforms(image)
    image = image.unsqueeze(0)  # 添加批次维度

    # 将图片移到GPU上
    image = image.cuda()

    # 推理
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    # 输出结果
    classes = ['Gold', 'Silver']
    predicted_class = classes[predicted.item()]

    print(f'The image is predicted to be: {predicted_class}')
