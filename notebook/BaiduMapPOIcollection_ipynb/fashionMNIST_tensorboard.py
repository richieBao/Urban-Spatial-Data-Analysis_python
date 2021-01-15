import torchvision
import torchvision.transforms as transforms

#读取（下载）FashionMNIST数据。如果已经下载，则直接读取
transform=transforms.Compose(
                                [transforms.ToTensor(), #转换PIL图像或numpy.ndarray为tensor张量
                                transforms.Normalize((0.5,), (0.5,))] #torchvision.transforms.Normalize(mean, std, inplace=False)，用均值和标准差，标准化张量图像
                            )
mnist_train=torchvision.datasets.FashionMNIST(root='./datasets/FashionMNIST_norm', train=True, download=True, transform=transform) 
mnist_test=torchvision.datasets.FashionMNIST(root='./datasets/FashionMNIST_norm', train=False, download=True, transform=transform)

#DataLoade-读取小批量
import torch.utils.data as data_utils
batch_size=4
num_workers=0
trainloader=data_utils.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader=data_utils.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def fashionMNIST_show_v2(img,figsize=(5,5),one_channel=False):
    import matplotlib.pyplot as plt
    import numpy as np
    '''
    function - 显示单张图像
    '''
    if one_channel:
        img=img.mean(dim=0)
    img=img/2+0.5 #逆标准化, unnormalize
    npimg=img.numpy()
    print("image size={}".format(npimg.shape))
    plt.figure(figsize=figsize)
    if one_channel:
        plt.imshow(npimg,cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg,(1,2,0)))

fashionMNIST_show_v2(mnist_test[4][0])


#-----------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F

class net_fashionMNIST(nn.Module):
    def __init__(self):
        super(net_fashionMNIST,self).__init__()
        self.conv1=nn.Conv2d(1,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*4*4,120) #torch.nn.Linear(in_features: int, out_features: int, bias: bool = True)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
    
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*4*4)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
net_fashionMNIST=net_fashionMNIST()
print(net_fashionMNIST)

#----------------------------------------------------------------------------
import torch.optim as optim

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net_fashionMNIST.parameters(), lr=0.001, momentum=0.9)

#----------------------------------------------------------------------------
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter(r'./runs/fashion_mnist_experiment_1')

#提取图像（数量为batch_size） get some random training images
dataiter=iter(trainloader)
images,labels=dataiter.next()
#建立图像格网 create grid of images
img_grid = torchvision.utils.make_grid(images)
#显示图像 show images
fashionMNIST_show_v2(img_grid, one_channel=True,figsize=(10,10))
# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)

#----------------------------------------------------------------------------
# writer.add_graph(net_fashionMNIST,images)
# #writer.flush
# writer.close

#----------------------------------------------------------------------------
import torch

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# helper function
def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# select random images and their target indices
images, labels = select_n_random(mnist_train.data, mnist_train.targets)

# get the class labels for each image
class_labels = [classes[lab] for lab in labels]

# log embeddings
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))
writer.close()