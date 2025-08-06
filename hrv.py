#-------------------------------------------------------------------------------------------------------------------data
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
#load
X=np.load(r'./X1500.npy')
y=np.load(r'./y1500.npy')
#counter
counter=Counter(y)
print(f'Original y Distribution:\n{counter}')
#encoder
encoder=LabelEncoder()
y=encoder.fit_transform(y)
print(f'Encoder y Distribution:\n{Counter(y)}')
#split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train,X_eval,y_train,y_eval=train_test_split(X_train,y_train,test_size=0.2)
#scaler
scaler=StandardScaler()
X_train,X_eval,X_test=scaler.fit_transform(X_train),scaler.transform(X_eval),scaler.transform(X_test)

#--------------------------------------------------------------------------model
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
#module
class Net(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(Net,self).__init__()
        self.fc1=nn.Linear(input_dim,hidden_dim)#-------------------
        self.fc2=nn.Linear(hidden_dim,output_dim)
        self.dropout=nn.Dropout(0.2)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=self.dropout(x)
        x=self.fc2(x)
        return x
#parameter
input_dim=X_train.shape[1]
output_dim=4
hidden_dim=16#------------------------
learn_rate=0.1#-----------------------
num_epochs=460#-----------------------
#device
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#to GPU
model=Net(input_dim,hidden_dim,output_dim).to(device)
X_train,X_eval=torch.FloatTensor(X_train).to(device),torch.FloatTensor(X_eval).to(device)
y_train,y_eval=torch.LongTensor(y_train).to(device),torch.LongTensor(y_eval).to(device)
criterion=nn.CrossEntropyLoss()
#optimizer-Adam
optimizer=optim.Adam(model.parameters(),lr=learn_rate)

#---------------------------------------------------------------------fit
best_eval_acc = 0.0  # 记录最佳验证集准确率
for epoch in range(num_epochs):
    # 训练阶段
    model.train()  # 设置为训练模式
    optimizer.zero_grad()  # 清空梯度
    output = model(X_train)  # 前向传播
    loss = criterion(output, y_train)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    # 每 10 个 epoch 在验证集上评估
    if epoch % 10 == 9:
        with torch.no_grad():  # 禁用梯度计算
            model.eval()  # 设置为评估模式
            outputs = model(X_eval)  # 预测验证集
            _, predicted = torch.max(outputs, 1)  # 取概率最大的类别
            acc = (predicted == y_eval).float().mean().item()  # 计算准确率
            # 如果当前模型更好，则保存
            if acc > best_eval_acc:
                best_eval_acc = acc
                torch.save(model.state_dict(), 'best_model.pth')  # 保存最佳模型
            print(f'Epoch {epoch}: 训练损失={loss:.4f}\t验证集准确率={acc * 100:.2f}%')
# 测试阶段（加载最佳模型）
with torch.no_grad():
    model.load_state_dict(torch.load('best_model.pth',weights_only=True))  # 加载最佳模型
    model.eval()  # 设置为评估模式
    # 确保测试数据在正确的设备上（CPU/GPU）
    device = next(model.parameters()).device  # 获取模型所在的设备
    X_test = torch.FloatTensor(X_test).to(device)  # 转换并移至设备
    y_test = torch.LongTensor(y_test).to(device)  # 转换并移至设备
    outputs = model(X_test)  # 预测测试集
    _, predicted = torch.max(outputs, 1)  # 取概率最大的类别
    acc = (predicted == y_test).float().mean().item()  # 计算测试准确率
    print(f'测试集准确率: {acc * 100:.2f}%')