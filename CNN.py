import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

# 设置超参数
# 输入尺寸 32 * 3 * 256 * 256
batch_size = 32
learning_rate = 0.001
num_epochs = 10
k_folds = 5

# 数据预处理和数据集
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(), # 变换到 [0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # 将数据变换到[-1,1]之间
])

data_dir = 'images/Out'
dataset = datasets.ImageFolder(data_dir, data_transforms)


# 定义卷积神经网络模型
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



kf = KFold(n_splits=k_folds, shuffle=True) # K折交叉验证，用于评测模型综合性能

results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)): # 对于每一折
    print(f'Fold {fold + 1}/{k_folds}')
    print(len(train_idx), len(val_idx))
    print('-' * 10)

    train_subsampler = Subset(dataset, train_idx) # 训练集采样器
    val_subsampler = Subset(dataset, val_idx) # 验证集采样器

    train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True) # 把挑选出来的数据拼成batch size为32的张量
    val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

    model = CNN().cuda() # 网络
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(num_epochs):# 训练模型
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader: # Batch size = 32 的 loader
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0) # 本batch损失 * batch的size大小 = 累计loss
            running_corrects += torch.sum(preds == labels.data) # 累计正确

        epoch_loss = running_loss / len(train_subsampler) # avg loss 累计
        epoch_acc = running_corrects.double() / len(train_subsampler) # acc 累计
        print(f'Epoch: {epoch} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


    model.eval() # 验证模型
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(val_subsampler)
    epoch_acc = running_corrects.double() / len(val_subsampler)
    print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    results.append({'fold': fold + 1, 'val_loss': epoch_loss, 'val_acc': epoch_acc})


for result in results: # 打印每折的结果
    print(f"Fold {result['fold']} - Val Loss: {result['val_loss']:.4f}, Val Acc: {result['val_acc']:.4f}")


avg_loss = sum(result['val_loss'] for result in results) / k_folds
avg_acc = sum(result['val_acc'] for result in results) / k_folds
print(f'Average Val Loss: {avg_loss:.4f}, Average Val Acc: {avg_acc:.4f}') # 打印平均结果


torch.save(model.state_dict(), 'cnn_classifier.pth') # 保存最终模型
