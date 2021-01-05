# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 22:14:32 2021

@author: Richie Bao-caDesign设计(cadesign.cn)
"""
import torch
from runx.logx import logx
import torch.utils.data as Data
from torch import nn
import numpy as np
from runx.logx import logx

#初始化logx
logx.initialize(logdir="./logs/",     #指定日志文件保持路径（如果不指定，则自动创建）
                coolname=True,        #是否在logdir下生成一个独有的目录
                hparams=None,         #配合runx使用，保存超参数
                tensorboard=True,     #是否自动写入tensorboard文件
                no_timestamp=False,   #是否不启用时间戳命名
                global_rank=0,        
                eager_flush=True,     
               ) 

#A-生成数据集
num_inputs=2 #包含的特征数
num_examples=5000
true_w=[3,-2.3]
true_b=5.9
features=torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels=true_w[0]*features[:, 0]+true_w[1]*features[:, 1]+true_b #由回归方程：y=w1*x1+-w2*x2+b，计算输出值
labels+=torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float) #变化输出值
logx.msg(r"the expression generating labels:w1*x1+-w2*x2+b") #可以将需要查看的对象以文本的形式保存在logging.log文件，此处保存了生成类标的线性方程

#B-读取数据（小批量-随机读取指定数量/batch_size的样本）
batch_size=100
dataset=Data.TensorDataset(features,labels) #建立PyTorch格式的数据集(组合输入与输出值)
train_size, val_size=[4000,1000]
train_dataset, val_dataset=torch.utils.data.random_split(dataset, [train_size, val_size])
train_data_iter=Data.DataLoader(train_dataset,batch_size,shuffle=True) #随机读取小批量
val_data_iter=Data.DataLoader(val_dataset,batch_size,shuffle=True)


#C-定义模型
#方法-01-继承nn.Module类构造模型
class MLP(nn.Module): #继承nn.Module父类
    #声明带有模型参数的层
    def __init__(self,n_feature):
        super(MLP,self).__init__()
        self.hidden=nn.Linear(n_feature,2) #隐藏层,torch.nn.Linear(in_features: int, out_features: int, bias: bool = True)，参数指定特征数（输入），以及类标数（输出）
        self.activation=nn.ReLU() #激活层
        self.output=nn.Linear(2,1) #输出层
        
    #定义前向传播，根据输入x计算返回所需要的模型输出
    def forward(self,x):
        y=self.activation(self.hidden(x))
        return self.output(y)
net=MLP(num_inputs)
print("nn.Module构造模型：")
logx.msg("net:{}".format(net)) #此处保存了神经网络结构
print("_"*50)

#方法-02-使用nn.Sequential类构造模型。
net_sequential=nn.Sequential(
    nn.Linear(num_inputs,2),
    nn.ReLU(),
    nn.Linear(2,1)
    )
print("nn.Sequential构造模型：",net_sequential)

#nn.Sequential的.add_module方式
net_sequential_=nn.Sequential() 
net_sequential_.add_module('hidden',nn.Linear(num_inputs,2))
net_sequential_.add_module('activation',nn.ReLU())
net_sequential_.add_module("output",nn.Linear(2,1))
print("nn.Sequential()-->.add_module方式：",net_sequential_)

#nn.Sequential的OrderedDict方式
from collections import OrderedDict
net_sequential_orderDict=nn.Sequential(
    OrderedDict([
        ('hidden',nn.Linear(num_inputs,2)),
        ('activation',nn.ReLU()),
        ("output",nn.Linear(2,1))
    ])
    )
print("nn.Sequential-->OrderedDict方式：",net_sequential_orderDict)
print("_"*50)

#方法-03-使用nn-ModuleList类构造模型。注意nn.ModuleList仅仅是一个存储各类模块的列表，这些模块之间没有联系，也没有顺序，没有实现forward功能，但是加入到nn.ModuleList中模块的参数会被自动添加到整个网络。
net_moduleList=nn.ModuleList([nn.Linear(num_inputs,2),nn.ReLU()])
net_moduleList.append(nn.Linear(2,1)) #可以像列表已有追加层
print("nn.ModuleList构造模型：",net_moduleList)
print("_"*50)

#方法-04-使用nn.ModuleDict类构造模型。注意nn.ModuleDict与nn.ModuleList类似，模块间没有关联，没有实现forward功能，但是参数会被自动添加到整个网络中。
net_moduleDict=nn.ModuleDict({
    'hidden':nn.Linear(num_inputs,2),
    'activation':nn.ReLU()    
    })
net_moduleDict['output']=nn.Linear(2,1) #象字典一样添加模块
print("nn.ModuleDict构造模型：",net_moduleDict)
print("_"*50)

#D-查看参数
for param in net.parameters():
    print(param)
print("_"*50)

#E-初始化模型参数
from torch.nn import init
init.normal_(net.hidden.weight,mean=0,std=0.01) #权值初始化为均值为0，标准差为0.01的正态分布。如果是用net_sequential,则可以使用net[0].weight指定权值，如果层自定义了名称，则可以用.layerName来指定
init.constant_(net.hidden.bias,val=0) #偏置初始化为0

print("初始化参数后：")
for param in net.parameters():
    print(param)
print("_"*50)

#F-定义损失函数
loss=nn.MSELoss()

#J-定义优化算法
import torch.optim as optim
optimizer=optim.SGD(net.parameters(),lr=0.03)
print("optimizer:",optimizer)

#配置不同的学习率-方法-01
optimizer_=optim.SGD([                
                {'params': net.hidden.parameters(),'lr':0.03},  
                {'params': net.output.parameters(), 'lr':0.01}
            ], lr=0.02) # 如果对某个参数不指定学习率，就使用最外层的默认学习率
print("配置不同的学习率-optimizer_:",optimizer_)

#配置不同的学习率-方法-02-新建优化器
for param_group in optimizer_.param_groups:
    param_group['lr'] *= 0.1 # 学习率为之前的0.1倍
print("新建优化器调整学习率-optimizer_:",optimizer_)
print("_"*50)

#H-训练模型
num_epochs=1000
best_loss=np.inf
for epoch in range(1,num_epochs+1):
    loss_acc_val=0
    for X,y in train_data_iter:
        output=net(X)
        l=loss(output,y.view(-1,1))
        optimizer.zero_grad() #梯度清零，
        l.backward()
        optimizer.step()      
    
    #验证数据集，计算预测值与真实值之间的均方误差累加和（用MSELoss损失函数）
    with torch.no_grad():
        for X_,y_ in val_data_iter:
            y_pred=net(X_)
            valid_loss=loss(y_pred,y_.view(-1,1))
            loss_acc_val+=valid_loss 
        
    #print('epoch %d, loss: %f' % (epoch, l.item()))  
    if epoch%100==0:
        logx.msg('epoch %d, loss: %f' % (epoch, l.item())) #logx.msg也会打印待保存的结果，因此注释掉上行的print
        print("验证数据集-精度-loss：{}".format(loss_acc_val))
    
    metrics={'loss':l.item() ,
            'lr':optimizer.param_groups[-1]['lr']}
    curr_iter=epoch*len(train_data_iter)
    #print("+"*50)
    #print(metrics,curr_iter)
    logx.metric(phase="train",metrics=metrics,epoch=curr_iter) #对传入字典中的数据进行保存记录，参数phase可以选择'train'和'val'；参数metrics为传入的字典；参数epoch表示全局轮次。若开启了tensorboard则自动增加。
    
    if loss_acc_val<best_loss:
        best_loss=loss_acc_val
    save_dict={"state_dict":net.state_dict(),
              'epoch':epoch+1,
              'optimizer':optimizer.state_dict()
              }
    ''' logx.save_model在JupyterLab环境下运行目前会出错，可以在Spyder下运行
    logx.save_model(
        save_dict=save_dict,     #checkpoint字典形式保存，包括epoch，state_dict等信息
        metric=best_loss,        #保存评估指标
        epoch=epoch,             #当前轮次
        higher_better=False,     #是否更高更好，例如准确率
        delete_old=True          #是否删除旧模型
        )
    '''