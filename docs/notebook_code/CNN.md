> Created on Tue Jan 12 13\48\15 2021 @author: Richie Bao-caDesign设计(cadesign.cn)

## 1. 卷积神经网络，可视化卷积层/卷积核,tensorboard,torchvision.models与VGG网络
### 1.1 卷积神经网络(Convolutional neural network, CNN)—— 卷积原理与卷积神经网络
在阅读卷积神经网络(CNNs)之前，如果已经阅读‘卷积’，‘计算机视觉，特征提取和尺度空间’等部分章节，可以更好的理解CNNs。其中‘尺度空间’的概念，通过降采样的不同空间分辨率影像（类似池化，pooLing）和不同$\sigma$值的高斯核卷积(即卷积计算或称为互相关运算)，来提取图像的概貌特征，这与CNNs的多层卷积网络如出一辙，只是CNNs除了卷积层，还可以自由加入其它的数据处理层，提升图像特征捕捉的几率。

一幅图像的意义来自于邻近的一组像素，而不是单个像素自身，因为单个像素并不包含关于整个图像的信息。例如下图(引自*PyTorch Artificial Intelligence Fundamentals*)很好的说明了这两者的差异。一个全连接的神经网络（密集层，dense layer），一层的每个节点都连接到下一层的每个节点，并不能很好的反映节点之间的关系，也具有较大的计算量；但是卷积网络利用像素之间的空间结构减少了层之间的连接数量，显著提高了训练的速度，减少了模型的参数，更重要的是反映了图像特征像素之间的关系，即图像内容中各个对象是由自身与邻近像素关系决定的空间结构（即特征）所表述。

<a href=""><img src="./imgs/22_01.jpg" height='auto' width='700' title="caDesign"></a>

二维卷积层的卷积计算，就是表征图像的数组shape(c,h,w)（如果为灰度图像则只有一个颜色通道，如果是彩色图像(RGB)则有三个图像通道），每个通道(h,w)二维数组中每一像素，与卷积核(filter/kernal)的加权和计算。这个过程存在一些可变动的因素：一个是步幅(stride)，卷积窗口（卷积核）从输入数组（二维图像数组）的最左上方开始，按从左到右，从上往下的顺序，依次在输入数组上滑动，每次滑动的行数和列数称为步幅。步幅越小代表扑捉不同内容对象的精度越高。下述图表（参考[*Convolutional Neural Networks*](https://cs231n.github.io/convolutional-networks/)）卷积核的滑动步幅为2，即行列均跨2步后计算；二是填充(padding)，如果不设置填充，并且从左上角滑动，会有一部分卷积核对应空值（没有图像/数据）。同时要保持四周填充的行列数相同，通常使用奇数高宽的卷积核；三是，如果是对图像数据实现卷积，通常包括1个单通道，或3个多通道的情况。对于多通道的计算是可以配置不同的卷积核对应不同的通道，各自通道分别卷积后，计算和（累加）作为结果输出。同时可以增加并行的新的卷积计算，获取多个输出，例如下图的Filter W0，和Filter W1的($3 \times 3$)卷积核，W0和W1各自包含3个卷积核对应3个通道输入，并各自输出。

卷积运算，步幅、填充配置，以及多通道卷积都没有改变图像的空间尺寸（空间分辨率），即尺度空间概念下降采样的表述（可以反映不同对象的尺度大小，或理解为只有在不同的尺度下才可以捕捉到对象的特征）。池化层(pooling)正是降采样在卷积神经网络中的表述，可以降低输入的空间维数，保留输入的深度。在图像卷积过程中，识别出比实际像素信息更多的概念意义，识别保留输入的关键信息，丢弃冗余部分。池化层不仅捕捉尺度空间下的对象特征，同时可以减少训练所需时间，减小模型的参数数量，降低模型复杂度，更好的泛化等。池化层可以用`nn.MaxPool2d`，取最大值；`nn.AvgPool2d`，取均值等方法。

<a href=""><img src="./imgs/22_05.jpg" height='auto' width='auto' title="caDesign"></a>

卷积层和池化层都有一个`dilation`参数，可以翻译为膨胀。通过dilation配置，可以调整卷积核与图像的作用域，即感受野（receptive filed）。当$3 \times 3$的卷积核dilation=1时，即没有膨胀效应；当dilation=2时，感受野扩展至$7 \times 7$；当dilation=24时，感受野扩展至$15 \times 15$。可以确定当dilation线性增加时，其感受野时呈指数增加。

<a href=""><img src="./imgs/22_11.jpg" height='auto' width='1000' title="caDesign"></a>


>  参考文献：
1. Jibin Mathew.PyTorch Artificial Intelligence Fundamentals: A recipe-based approach to design, build and deploy your own AI models with PyTorch 1.x[m].UK:Packt Publishing (February 28, 2020)
2. Aston Zhang,Zack C. Lipton,Mu Li,etc.Dive into Deep Learning[M].；中文版-阿斯顿.张,李沐,扎卡里.C. 立顿,亚历山大.J. 斯莫拉.动手深度学习[M].人民邮电出版社,北京,2019-06-01
3. [Convolutional Neural Networks (CNNs / ConvNets)](https://cs231n.github.io/convolutional-networks/)
4. [Understanding 2D Dilated Convolution Operation with Examples in Numpy and Tensorflow with Interactive Code](https://towardsdatascience.com/understanding-2d-dilated-convolution-operation-with-examples-in-numpy-and-tensorflow-with-d376b3972b25)

建立图表中的输入数据`t_input`，张量形状为(batchsize, nChannels, Height, Width)=(1,3,7,7)，即只有一幅图像，通道数为3，高度为7，宽度为7。通常查看时，只要确定最后一维内的数据为图像每一行的像素值（由上至下）。PyTorch提供的`nn.Conv2d`卷积方法不需自定义卷积核，其参数为`torch.nn.Conv2d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T, T]], stride: Union[T, Tuple[T, T]] = 1, padding: Union[T, Tuple[T, T]] = 0, dilation: Union[T, Tuple[T, T]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros')`。如果需要自定义卷积核，则调用`torch.nn.functional.conv2d`方法，其中`weight`参数即为卷积核，其输入参数为`torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) → Tensor`。


```python
import torch
import torch.nn as nn

#batchsize, nChannels, Height, Width
t_input=torch.tensor([[[[0,0,0,0,0,0,0],[0,2,2,2,0,0,0],[0,1,0,2,0,0,0],[0,0,0,1,0,1,0],[0,1,1,1,1,0,0],[0,2,2,0,0,0,0],[0,0,0,0,0,0,0]],
                      [[0,0,0,0,0,0,0],[0,1,2,1,1,1,0],[0,2,1,0,1,1,0],[0,0,0,0,0,1,0],[0,2,0,2,1,1,0],[0,0,2,0,1,2,0],[0,0,0,0,0,0,0]],
                      [[0,0,0,0,0,0,0],[0,1,0,0,1,1,0],[0,1,0,2,0,1,0],[0,2,0,1,2,1,0],[0,1,2,1,2,2,0],[0,1,1,2,0,0,0],[0,0,0,0,0,0,0]],
                     ]],dtype=torch.float)
print("t_input.shape={}".format(t_input.shape))
conv_2d_3c=nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3,stride=2,padding=0,bias=1)
conv_2d_3c(t_input)
```

    t_input.shape=torch.Size([1, 3, 7, 7])
    




    tensor([[[[ 0.1740, -0.2426,  0.2305],
              [ 0.2810,  0.3236,  0.4322],
              [ 0.0234, -0.0917, -0.8094]],
    
             [[-0.1936, -0.0077,  0.0944],
              [ 0.1876, -0.2772, -0.5696],
              [-0.1700,  0.2354, -0.5344]]]], grad_fn=<ThnnConv2DBackward>)




```python
import torch.nn.functional as F
w_0=torch.tensor([[[[-1, 1, 0],
                    [ 0,-1, 0],
                    [-1,-1, 1]],

                   [[0,-1,-1],
                    [-1,1,1],
                    [-1,-1,1]],

                   [[-1,1,0],
                    [0,0,1],
                    [0,0,-1]]]],dtype=torch.float)

w_1=torch.tensor([[[[0, -1, -1],
                    [ 1,-1, 1],
                    [1,1, 1]],

                   [[0,0,1],
                    [0,1,1],
                    [0,1,0]],

                   [[0,0,0],
                    [0,0,1],
                    [0,1,-1]]]],dtype=torch.float)

b_0=torch.tensor([1])
b_1=torch.tensor([0])
output_0=F.conv2d(input=t_input,weight=w_0,stride=2,padding=0,bias=b_0)
output_1=F.conv2d(input=t_input,weight=w_1,stride=2,padding=0,bias=b_1)
print(output_0)
print(output_1)
```

    tensor([[[[ 0., -2., -1.],
              [-4.,  1., -2.],
              [ 2., -4.,  0.]]]])
    tensor([[[[7., 7., 3.],
              [3., 4., 4.],
              [1., 2., 2.]]]])
    

最大池化，即设定池化（卷积核）大小，返回覆盖范围内最大值，其参数为`torch.nn.MaxPool2d(kernel_size: Union[T, Tuple[T, ...]], stride: Optional[Union[T, Tuple[T, ...]]] = None, padding: Union[T, Tuple[T, ...]] = 0, dilation: Union[T, Tuple[T, ...]] = 1, return_indices: bool = False, ceil_mode: bool = False)`。


```python
pooling_input=torch.tensor([[[0,-2,-1],[-4,1,-2],[2,-4,0]]],dtype=torch.float)
print("输入数据形状={}".format(pooling_input.shape))

maxPool_2d=nn.MaxPool2d(kernel_size=2,stride=1)
maxPool_2d(pooling_input)
```

    输入数据形状=torch.Size([1, 3, 3])
    




    tensor([[[1., 1.],
             [2., 1.]]])



### 1.2 卷积_特征提取器--->_分类器，可视化卷积层/卷积核，及tensorboard

#### 1.2.1 卷积层，池化层输出尺寸（形状）计算，及根据输入输出尺寸反推填充pad

构建卷积神经网络，很重要的一步是确定图像经过一次或多次卷积后，输出的尺寸，用于分类器（线性模型，全连接层）的输入。在PyTorch的['torch.nn.Conv2d'](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)类方法说明中都会给出卷积输出的尺寸计算公式，但是手工的计算方式容易出错，并且耗时耗力，因此通常将其编写为代码程序。池化层的输出尺寸计算实际上与卷积的输出相同，但是为了区分，仍然在`conv2d_output_size_calculation`类中，增加了` pooling_output_shape`方法（直接调用`conv2d_output_shape`）。卷积的方式除了给定h_w/图像高宽，kernel_size/卷积核（过滤器，filter）,stride/步幅，pad/填充，以及dilation/膨胀参数，卷积核的初始位置可以分为：以卷积核左上角对位图像左上角第一个像素值开始由左-->右，由上-->下卷积计算，即`conv2d_output_shape`函数； 以卷积核右下角对位图像左上角第一个像素值开始由左-->右，由上-->下卷积计算，即`convtransp2d_output_shape`函数。

同时，也可以根据卷积的输入和输出的尺寸，反推填充的大小，对应卷积核初始位置的不同分别为`conv2d_get_padding`和`convtransp2d_get_padding`函数。

对于卷积神经网络'net_fashionMNIST'的网络结构如下：

<a href=""><img src="./imgs/22_06.jpg" height='auto' width='1000' title="caDesign"></a>

输入图像的大小为$28 \times 28$，经过卷积层conv1-->池化层pool-->卷积层conv2-->池化层pool后，图像尺寸的大小变为$4 \times 4$，该值（卷积层的输出值）由自定义`conv2d_output_size_calculation`计算。注意两次池化，其参数值相同，因此卷积神经网络结构定义时，可以仅定义一个池化方法，用于生成不同位置的池化层。计算应用卷积提取特征部分的图像输出大小后，因为conv2+pool之后，其输出的'out_channels'输出通道数配置为16，因此到分类器部分（线性函数/全连接层/展平层）的输入大小为$16 \times 4 \times 4 = 256$。


```python
class conv2d_output_size_calculation:
    '''
    class - PyTorch 卷积层，池化层输出尺寸(shape)计算，及根据输入，输出尺寸(shape)反推pad填充大小
    
    @author:sytelus Shital Shah
    Updated on Tue Jan 12 19:17:22 2021 @author: Richie Bao-caDesign设计(cadesign.cn)
    '''   
    
    def num2tuple(self,num):
        '''
        function - 如果num=2，则返回(2,2)；如果num=(2,2)，则返回(2,2).
        '''
        return num if isinstance(num, tuple) else (num, num)
    
    def conv2d_output_shape(self,h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        import math
        '''
        funciton - 计算PyTorch的nn.Conv2d卷积方法的输出尺寸。以卷积核左上角对位图像左上角第一个像素值开始由左-->右，由上-->下卷积计算
        '''
        
        h_w, kernel_size, stride, pad, dilation = self.num2tuple(h_w), \
            self.num2tuple(kernel_size), self.num2tuple(stride), self.num2tuple(pad), self.num2tuple(dilation)
        pad = self.num2tuple(pad[0]), self.num2tuple(pad[1])
        
        h = math.floor((h_w[0] + sum(pad[0]) - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
        w = math.floor((h_w[1] + sum(pad[1]) - dilation[1]*(kernel_size[1]-1) - 1) / stride[1] + 1)
        
        return h, w
    
    def convtransp2d_output_shape(self,h_w, kernel_size=1, stride=1, pad=0, dilation=1, out_pad=0):
        '''
        function - 以卷积核右下角对位图像左上角第一个像素值开始由左-->右，由上-->下卷积计算
        '''
        h_w, kernel_size, stride, pad, dilation, out_pad = self.num2tuple(h_w), \
            self.num2tuple(kernel_size), self.num2tuple(stride), self.num2tuple(pad), self.num2tuple(dilation), self.num2tuple(out_pad)
        pad = self.num2tuple(pad[0]), self.num2tuple(pad[1])
        
        h = (h_w[0] - 1)*stride[0] - sum(pad[0]) + dilation[0]*(kernel_size[0]-1) + out_pad[0] + 1
        w = (h_w[1] - 1)*stride[1] - sum(pad[1]) + dilation[1]*(kernel_size[1]-1) + out_pad[1] + 1
        
        return h, w
    
    def conv2d_get_padding(self,h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1):
        import math
        '''
        function - conv2d_output_shape 方法的逆，求填充pad
        '''
        h_w_in, h_w_out, kernel_size, stride, dilation = self.num2tuple(h_w_in), self.num2tuple(h_w_out), \
            self.num2tuple(kernel_size), self.num2tuple(stride), self.num2tuple(dilation)
        
        p_h = ((h_w_out[0] - 1)*stride[0] - h_w_in[0] + dilation[0]*(kernel_size[0]-1) + 1)
        p_w = ((h_w_out[1] - 1)*stride[1] - h_w_in[1] + dilation[1]*(kernel_size[1]-1) + 1)
        
        return (math.floor(p_h/2), math.ceil(p_h/2)), (math.floor(p_w/2), math.ceil(p_w/2))  #((pad_up, pad_bottom)， (pad_left, pad_right))
    
    def convtransp2d_get_padding(self,h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1, out_pad=0):
        import math
        '''
        function - convtransp2d_output_shape 方法的逆，求填充pad
        '''
        h_w_in, h_w_out, kernel_size, stride, dilation, out_pad = self.num2tuple(h_w_in), self.num2tuple(h_w_out), \
            self.num2tuple(kernel_size), self.num2tuple(stride), self.num2tuple(dilation), self.num2tuple(out_pad)
            
        p_h = -(h_w_out[0] - 1 - out_pad[0] - dilation[0]*(kernel_size[0]-1) - (h_w_in[0] - 1)*stride[0]) / 2
        p_w = -(h_w_out[1] - 1 - out_pad[1] - dilation[1]*(kernel_size[1]-1) - (h_w_in[1] - 1)*stride[1]) / 2
        
        return (math.floor(p_h/2), math.ceil(p_h/2)), (math.floor(p_w/2), math.ceil(p_w/2))
    
    def pooling_output_shape(self,h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        '''
        function - pooling池化层输出尺寸，同conv2d_output_shape
        '''
        return self.conv2d_output_shape(h_w, kernel_size=kernel_size, stride=stride, pad=pad, dilation=dilation)
```


```python
conv2dSize_cal=conv2d_output_size_calculation()
size_conv1=conv2dSize_cal.conv2d_output_shape(28, kernel_size=5, stride=1, pad=0, dilation=1)
size_pool1=conv2dSize_cal.pooling_output_shape(24, kernel_size=2, stride=2, pad=0, dilation=1)
size_conv2=conv2dSize_cal.conv2d_output_shape(12, kernel_size=5, stride=1, pad=0, dilation=1)
size_pool2=conv2dSize_cal.pooling_output_shape(8, kernel_size=2, stride=2, pad=0, dilation=1)

print("conv1_size={}\nsize_pool1={}\nsize_conv2={}\nsize_pool2={}".format(size_conv1,size_pool1,size_conv2,size_pool2))
```

    conv1_size=(24, 24)
    size_pool1=(12, 12)
    size_conv2=(8, 8)
    size_pool2=(4, 4)
    

在设计卷积层，或者包含很多卷积层时，上述的方法可以进一步改进，定义一个函数可以一次性计算所有的卷积层。在输入参数设置时，使用了列表和元组的形式，固定输入参数'input'，'conv'和'pool'，值的参数依次为[h_w, kernel_size, stride, pad, dilation]。


```python
convs_params=[
            ('input',(28,28)),
            ('conv',[5,1,0,1]),  #h_w, kernel_size, stride, pad, dilation
            ('pool',[2,2,0,1]),
            ('conv',[5,1,0,1]),
            ('pool',[2,2,0,1]),    
            ]

def conv2d_outputSize_A_oneTime(convs_params):
    '''
    fucntion - 一次性计算卷积输出尺寸
    '''
    conv2dSize_cal=conv2d_output_size_calculation()
    for v in convs_params:
        if v[0]=='input':
            h_w=v[1]
        elif v[0]=='conv' or v[0]=='pool':            
            kernel_size, stride, pad, dilation=v[1]
            h_w, kernel_size, stride, pad, dilation=conv2dSize_cal.num2tuple(h_w),conv2dSize_cal.num2tuple(kernel_size), conv2dSize_cal.num2tuple(stride),conv2dSize_cal.num2tuple(pad),conv2dSize_cal.num2tuple(dilation)
            h_w=conv2dSize_cal.conv2d_output_shape(h_w, kernel_size, stride, pad, dilation)        
    return h_w
 
output_h_w=conv2d_outputSize_A_oneTime(convs_params)    
print("卷积层输出尺寸={}".format(output_h_w))
```

    卷积层输出尺寸=(4, 4)
    

#### 1.2.2 构建简单的卷积神经网络识别fashionMNIST数据集，与tensorboard

此处构建上图给出的神经网络结构，input-->conv1(relu)-->pool-->conv2(relu)-->pool--->fc1(relu)-->fc2(relu)-->fc3-->output，同时应用tensorboard可以写入并自动的根据写入的内容图示卷积神经网络结构，损失曲线，预测结果，样本信息，以及自定义的内容等。详细内容可以查看[torch.utils.tensorboard](https://pytorch.org/docs/stable/tensorboard.html)。

> 参考：[Visualizing Models, Data, and Training with TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html?highlight=fashion%20mnist)


```python
#01-下载/读取fashinMNIST数据，以及构建训练、测试可迭代对象。（如果已经下载，则直接读取）

def load_fashionMNIST(root,batchsize=4,num_workers=2,resize=None,n_mean=0.5,n_std=0.5):
    import torchvision
    import torchvision.transforms as transforms
    '''
    function - 下载读取fashionMNIST数据集，并建立训练、测试可迭代数据集
    '''
    trans= [transforms.ToTensor(), #转换PIL图像或numpy.ndarray为tensor张量
            transforms.Normalize((0.5,), (0.5,))] #torchvision.transforms.Normalize(mean, std, inplace=False)，用均值和标准差，标准化张量图像       
    if resize:
        trans.append(transforms.Resize(size=resize))  
    transform=transforms.Compose(trans)
    mnist_train=torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform) 
    mnist_test=torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    #DataLoade-读取小批量
    import torch.utils.data as data_utils
    batch_size=batchsize
    num_workers=num_workers
    trainloader=data_utils.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader=data_utils.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return trainloader,testloader

trainloader,testloader=load_fashionMNIST(root='./datasets/FashionMNIST_norm')
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
```


```python
#02-定义网络结构，特征提取层+分类器

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
        return x
net_fashionMNIST_=net_fashionMNIST()
print(net_fashionMNIST_)

#03-定义损失函数核优化算法

import torch.optim as optim
criterion=nn.CrossEntropyLoss() #定义损失函数
optimizer=optim.SGD(net_fashionMNIST_.parameters(), lr=0.001, momentum=0.9) #定义优化算法
```

    net_fashionMNIST(
      (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
      (fc1): Linear(in_features=256, out_features=120, bias=True)
      (fc2): Linear(in_features=120, out_features=84, bias=True)
      (fc3): Linear(in_features=84, out_features=10, bias=True)
    )
    


```python
#04-定义显示图像函数（一个batch批次）

import torchvision
# helper functions
def matplotlib_imshow(img, one_channel=False):
    import matplotlib.pyplot as plt
    import numpy as np    
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

#05-调入初始化tensorboard.SummaryWriter（指定数据写入的保存路径），并写入训练图像与模型
        
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer=SummaryWriter(r'./runs/fashion_mnist_experiment_1')

#提取图像（数量为batch_size） get some random training images
dataiter=iter(trainloader)
images,labels=dataiter.next()
#建立图像格网 create grid of images
img_grid = torchvision.utils.make_grid(images)
#显示图像 show images
matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)

writer.add_graph(net_fashionMNIST_, images)
writer.close()
```


    
<a href=""><img src="./imgs/22_01.png" height='auto' width='700' title="caDesign"></a>
    


在终端，切换到tensorboard数据保存文件夹runs所在的目录，执行`tensorboard --logdir=runs`后，通常提示在浏览器中打开`http://localhost:6006/`地址，可以查看`writer.add_image('four_fashion_mnist_images', img_grid)`写入的训练图像信息内容(其语法为：`add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')`)，如下：

<a href=""><img src="./imgs/22_07.png" height='auto' width='1000' title="caDesign"></a>

`writer.add_graph(net_fashionMNIST_, images)`（其语法为：`add_graph(model, input_to_model=None, verbose=False)`）,写入的信息内容下，可以查看网络结构及运算的流程，这对模型的构建与调整有所帮助。

<a href=""><img src="./imgs/22_08.png" height='auto' width='1000' title="caDesign"></a>

下述在训练过程中，`riter.add_scalar('training loss',running_loss / 1000,epoch * len(trainloader) + i)`(其语法为：`add_scalar(tag, scalar_value, global_step=None, walltime=None)`)，写入损失数据图表，如下：

<a href=""><img src="./imgs/22_10.png" height='auto' width='1000' title="caDesign"></a>

writer.add_figure('predictions vs. actuals',plot_classes_preds(net_fashionMNIST_, inputs, labels),global_step=epoch * len(trainloader) + i)(其语法为：`add_figure(tag, figure, global_step=None, close=True, walltime=None)`)，写入`plot_classes_preds`返回图表，包括图像，预测值及其概率，如下：

<a href=""><img src="./imgs/22_09.png" height='auto' width='1000' title="caDesign"></a>


```python
#06-定义将图像应用训练的网络（模型）预测及其概率待写入tensorboard文件的函数

def images_to_probs(net, images):
    '''
    function - 用训练的网络预测给定的一组图像，并计算相应的概率 Generates predictions and corresponding probabilities from a trainednetwork and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(net, images, labels):
    import matplotlib.pyplot as plt
    
    '''
    function - 用训练的网络预测给定的一组图像，并计算概率后，显示图像、预测值以及概率，和实际的标签 Generates matplotlib Figure using a trained network, along with images and labels from a batch, that shows the network's top prediction along with its probability, alongside the actual label, coloring this information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig
```


```python
#07-训练模型，同时向tensorboard文件写入相关信息

from tqdm.auto import tqdm
import torch
import numpy as np
running_loss = 0.0
epochs=1
for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net_fashionMNIST_(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()                  
        if i % 1000==999:    # every 1000 mini-batches...
            # ...log the running loss
            writer.add_scalar('training loss',running_loss / 1000,epoch * len(trainloader) + i) #写入损失数值

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',plot_classes_preds(net_fashionMNIST_, inputs, labels),global_step=epoch * len(trainloader) + i)  #写入自定义函数  plot_classes_preds返回的图表，包括图像，预测值及其概率
            loss_temp=running_loss
            running_loss=0.0
    
    print("epoch={},running_loss={}".format(epoch,loss_temp))     
    loss_temp=0
    
print('Finished Training')
```


    HBox(children=(HTML(value=''), FloatProgress(value=0.0, max=1.0), HTML(value='')))


    epoch=0,running_loss=433.0562955379719
    
    Finished Training
    

#### 1.2.3 可视化卷积层/卷积核

通常一个卷积层的输出通道（output channel）有6，16，32，64，256，等不确定的输出个数及更多的输出个数，即一个卷积层包含输出通道数目的filter/kernal过滤器/卷积核；而卷积核通过$3 \times 3$，$5 \times 5$，$7 \times 7$，等不确定尺寸（通常为奇数）或者更大尺寸来提取图像的特征，不同的卷积核提取的图像特征不同，或者表述为不同的卷积核关注不同的特征提取，这类似于图像的关键点描述子（位于不同的尺度空间下），不过因为卷积核的多样性，卷积提取的特征更加丰富多样。通过一个卷积层的多个卷积核（尺度空间的水平向，表述各个像素值与各周边像素值的差异程度），及多个卷积层（尺度空间的纵深向，表述图像特征所在的（对应的）空间分辨率，例如遥感影像看清建筑轮廓的空间分辨率约为5-15m，看清行人的空间分辨率约为0.3-1m，而若要看清人的五官，空间分辨率率则约为0.01-0.05m，这与对象（特征）的尺寸有关）提取了大量的图像特征，将这些图像特征flatten展平，就构成了该图像的特征集合(feature maps)。

下述定义的`conv_retriever`类用于取回卷积神经网络中所有卷积层及其权重值（卷积核），取回的卷积层可以直接输入图像数据计算该卷积。例如上述网络取回的卷积层有两个。函数`visualize_convFilte`则可以打印显示卷积核。函数`visualize_convLaye`则能打印显示指定数目的所有卷积图像结果。

> 参考 [Visualizing Filters and Feature Maps in Convolutional Neural Networks using PyTorch](https://debuggercafe.com/visualizing-filters-and-feature-maps-in-convolutional-neural-networks-using-pytorch/)


```python
class conv_retriever:
    '''
    class - 取回卷积神经网络中所有卷积层及其权重
    '''
    def __init__(self,net):
        self.net=net
        self.model_weights=[] # we will save the conv layer weights in this list
        self.conv_layers=[] # we will save the 49 conv layers in this list
        self.model_children=list(net.children()) # get all the model children as list
        self.counter=0
        
    def retriever(self):
        import torch.nn as nn
        for i in range(len(self.model_children)):
            if type(self.model_children[i])==nn.Conv2d:                
                self.counter+=1
                self.model_weights.append(self.model_children[i].weight)
                self.conv_layers.append(self.model_children[i])
            elif type(self.model_children[i])==nn.Sequential:
                for j in range(len(self.model_children[i])):
                    for child in self.model_children[i][j].children():
                        if type(child)==nn.Conv2d:
                            self.counter+=1
                            self.model_weights.append(child.weight)
                            self.conv_layers.append(child)       
         
        
conv_retriever_=conv_retriever(net_fashionMNIST_)
conv_retriever_.retriever()
#print(conv_retriever_.conv_layers) #conv_retriever_.model_weights
for weight, conv in zip(conv_retriever_.model_weights, conv_retriever_.conv_layers):
    print(f"CONV: {conv} ====> SHAPE: {weight.shape}")
```

    CONV: Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1)) ====> SHAPE: torch.Size([6, 1, 5, 5])
    CONV: Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1)) ====> SHAPE: torch.Size([16, 6, 5, 5])
    

卷积核显示中，像素越黑，值越小，趋于0；而像素越亮，值越大，趋于255.0。因此越白的像素，在卷积过程中对应位置图像的像素的权重越大，对特征影响越重。


```python
def visualize_convFilter(conv_layer,model_weight,output_name,figsize=(10,10)):
    import matplotlib.pyplot as plt
    '''
    function - 可视化卷积核 visualize the conv layer filters
    '''
    plt.figure(figsize=figsize)
    kernel_size=conv_layer.kernel_size
    for i,filter in enumerate(model_weight):
        plt.subplot(kernel_size[0]+1,kernel_size[1]+1,i+1)
        plt.imshow(filter[0,:,:].detach(),cmap='gray')
        plt.axis('off')
        plt.savefig(r'./results/%s'%output_name)
    plt.show()        
        
visualize_convFilter(conv_retriever_.conv_layers[0],conv_retriever_.model_weights[0],output_name='fashion_MNIST_filter.png')
```


    
<a href=""><img src="./imgs/22_02.png" height='auto' width='700' title="caDesign"></a>
    


卷积（层）结果的显示，往往可以观察到不同的特征，例如明显的黑色区域轮廓为商标标识，对象的轮廓也能够通过代表不同颜色的值区分开来等。这为观察不同的卷积核提取了图像哪些特征，为相关研究或者网络调试提供参照。


```python
def visualize_convLayer(imgs_batch,conv_layers,model_weights,num_show=6,figsize=(10,10)):
    import matplotlib.pyplot as plt
    '''
    function - 可视化所有卷积层（卷积结果）
    '''
    results=[conv_layers[0](imgs_batch)]
    for i in range(1,len(conv_layers)):
        results.append(conv_layers[i](results[-1]))
    outputs=results
    
    for num_layer in range(len(outputs)):
        plt.figure(figsize=figsize)
        layer_viz=outputs[num_layer][0,:,:,:]
        layer_viz=layer_viz.data
        print("num_layer:{},layer.size={}".format(num_layer,layer_viz.size()))
        
        print()
        kernel_size=conv_layers[num_layer].kernel_size
        for i,filter in enumerate(layer_viz):
            if i==num_show:
                break
            
            plt.subplot(kernel_size[0]+1,kernel_size[1]+1,i+1)
            plt.imshow(filter,cmap='gray')
            plt.axis('off')
        print(f"Saving layer {num_layer} feature maps...")
        plt.savefig(f'./results/layer_{num_layer}.png')
        plt.show()
        #plt.close() #如果只保存，不需要显示打印，则可以开启plt.close()，并注释掉plt.show()
            
    
    
visualize_convLayer(images,conv_retriever_.conv_layers,conv_retriever_.model_weights,num_show=6)    
```

    num_layer:0,layer.size=torch.Size([6, 24, 24])
    
    Saving layer 0 feature maps...
    


    
<a href=""><img src="./imgs/22_03.png" height='auto' width='700' title="caDesign"></a>
    


    num_layer:1,layer.size=torch.Size([16, 20, 20])
    
    Saving layer 1 feature maps...
    


    
<a href=""><img src="./imgs/22_04.png" height='auto' width='700' title="caDesign"></a>
    


### 1.3 [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html)

`torchvision.models`库包含有解决不同任务的预先模型定义(通过配置参数pretrained=True，可以下载已经训练模型的参数)，包括：image classification图像分类、pixelwise sementic segmentation像素语义分割、object detection对象检测、instance segmentation实例分割、person keypoint detection 人关键点检测和video classification视频分类等。目前包括的模型如下：

| 用途      | 模型/网络 |
| ----------- | ----------- |
| Classification      | AlexNet, VGG,ResNet, SqueezeNet, DenseNet, Inception v3, GoogLeNet,ShuffleNet v2,MobileNet v2,ResNeXt, Wide ResNet, MNASNet|
| Semantic Segmentation   | FCN ResNet50, ResNet101; DeepLabV3 ResNet50, ResNet101       |
| Object Detection, Instance Segmentation and Person Keypoint Detection|Faster R-CNN ResNet-50 FPN ,Mask R-CNN ResNet-50 FPN|
| Video classification|ResNet 3D,  ResNet Mixed Convolution,ResNet (2+1)D|

#### 1.3.1 自定义VGG网络
VGG作者研究了在大规模图像识别设置中，卷积深度网络对其精度的影响。其贡献是使用非常小的卷积核($3 \times 3$)滤波器的架构对深度增加的网络进行全面的评估，表面深度推进到16-19个权重层，可以实现对现有配置的显著改进。同时该模型可以很好的泛化到其它数据集。VGG的网络结构可以看作不断重复的模块，因此在定义模型时可以先定义规律性的模块，然后给定配置的卷积层数(num_convs)，输入通道数(in_channels)和输出通道数(out_channels)，调用模块实现完了的架构，避免代码冗长。其中包括5个卷积模块（block），前2块为单层卷积，后3层为双层卷积。该网络总共8个卷积层和3个全连接层，所以也称为VGG-11。

> 参考文献：
```
1. @misc{simonyan2015deep,
      title={Very Deep Convolutional Networks for Large-Scale Image Recognition}, 
      author={Karen Simonyan and Andrew Zisserman},
      year={2015},
      eprint={1409.1556},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

> 注： * 收集元组中所有位置参数 ** 收集字典中的所有关键字参数


```python
def func_asterisk(*s_position,**d_keywords):
    print("*s_position={}\n**d_keywords={}".format(s_position,d_keywords))

func_asterisk('a','b','c','d','e',param_a=1,param_b=2,param_c=3)
```

    *s_position=('a', 'b', 'c', 'd', 'e')
    **d_keywords={'param_a': 1, 'param_b': 2, 'param_c': 3}
    


```python
'''
@author:《动手学深度学习》/'Dive into PyTorch'
Updated on Thu Jan 14 17:45:17 2021 @author: Richie Bao-caDesign设计(cadesign.cn)
'''

import torch
from torch  import nn,optim
import util

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vgg_block(num_convs,in_channels,out_channels):
    blk=[]
    for i in range(num_convs):
        if i==0:
            blk.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        else:
            blk.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*blk)  

conv_arch=((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
fc_features=512*7*7
fc_hidden_units=4096

def vgg(conv_arch,fc_features,fc_hidden_units=4096):
    net=nn.Sequential()
    for i,(num_convs,in_channels,out_channels) in enumerate(conv_arch):
        net.add_module('vgg_block_'+str(i+1),vgg_block(num_convs,in_channels,out_channels))
    net.add_module('fc',nn.Sequential(util.flattenLayer(),
                                      nn.Linear(fc_features,fc_hidden_units),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(fc_hidden_units,fc_hidden_units),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(fc_hidden_units,10)        
                                     ))
    return net

VGG_net=vgg(conv_arch,fc_features,fc_hidden_units)
print("VGG网络结构：\n",VGG_net)
```

    VGG网络结构：
     Sequential(
      (vgg_block_1): Sequential(
        (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (vgg_block_2): Sequential(
        (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (vgg_block_3): Sequential(
        (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (vgg_block_4): Sequential(
        (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (vgg_block_5): Sequential(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (fc): Sequential(
        (0): flattenLayer()
        (1): Linear(in_features=25088, out_features=4096, bias=True)
        (2): ReLU()
        (3): Dropout(p=0.5, inplace=False)
        (4): Linear(in_features=4096, out_features=4096, bias=True)
        (5): ReLU()
        (6): Dropout(p=0.5, inplace=False)
        (7): Linear(in_features=4096, out_features=10, bias=True)
      )
    )
    

配置完卷积层（特征提取层），可以通过上述定义的`conv2d_outputSize_A_oneTime`函数，计算全连接层的输入尺寸。计算结果为(7,7)，将其与卷积层最后一层的输出通道数相乘就为全连接层的输入尺寸$512 \times 7 \times 7=25088$，这与VGG的输入值相同。


```python
VGG_convs_params=[
            ('input',(224,224)),
            ('conv',[3,1,1,1]),  #kernel_size, stride, pad, dilation
            ('pool',[2,2,0,1]),
    
            ('conv',[3,1,1,1]),
            ('pool',[2,2,0,1]),
    
            ('conv',[3,1,1,1]),
            ('conv',[3,1,1,1]),
            ('pool',[2,2,0,1]),  
    
            ('conv',[3,1,1,1]),
            ('conv',[3,1,1,1]),
            ('pool',[2,2,0,1]),  
    
            ('conv',[3,1,1,1]),
            ('conv',[3,1,1,1]),
            ('pool',[2,2,0,1])     
            ]

VGG_output_h_w=conv2d_outputSize_A_oneTime(VGG_convs_params)    
print("卷积层输出尺寸={}".format(VGG_output_h_w))
```

    卷积层输出尺寸=(7, 7)
    

可以先生成随机的样本数据(batchsize, nChannels, Height, Width)，逐个循环每一模块，即Sequential对象，获取每一模块计算后数据的形状，从而观察、验证并用于辅助调整模型参数。


```python
X=torch.rand(1, 1, 224, 224)
for name, blk in VGG_net.named_children(): 
    X = blk(X)
    print(name, 'output shape: ', X.shape)
```

    vgg_block_1 output shape:  torch.Size([1, 64, 112, 112])
    vgg_block_2 output shape:  torch.Size([1, 128, 56, 56])
    vgg_block_3 output shape:  torch.Size([1, 256, 28, 28])
    vgg_block_4 output shape:  torch.Size([1, 512, 14, 14])
    vgg_block_5 output shape:  torch.Size([1, 512, 7, 7])
    fc output shape:  torch.Size([1, 10])
    

VGG的原始输入图像尺寸为224,目前实验的数据为fashionMNIST数据集，一幅图像的尺寸为$28 \times 28$，因此使用`torchvision.transforms.Resize`的方法调整图像大小，该方法已经包含于自定义`load_fashionMNIST`中，因此只需要配置输入参数`resize=224`就可以修改图像尺寸，满足VGG输入数据尺寸的要求。

因为卷积后的图像输出尺寸只与图像的输入尺寸，卷积层、池化层的卷积核大小，填充、步幅等有关，因此可以自由的修改卷积层的输入、输出通道，以及全连接层隐含层的数量。在修改时只需要配置一个缩放比例参数`ratio`，将所有的相关值除以该值完成新的配置。


```python
ratio = 8
small_conv_arch = [(1, 1, 64//ratio), (1, 64//ratio, 128//ratio), (2, 128//ratio, 256//ratio), (2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio)]
VGG_net_= vgg(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
print("VGG网络结构_减少通道数：\n",VGG_net_)
```

    VGG网络结构_减少通道数：
     Sequential(
      (vgg_block_1): Sequential(
        (0): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (vgg_block_2): Sequential(
        (0): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (vgg_block_3): Sequential(
        (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (vgg_block_4): Sequential(
        (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (vgg_block_5): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (fc): Sequential(
        (0): flattenLayer()
        (1): Linear(in_features=3136, out_features=512, bias=True)
        (2): ReLU()
        (3): Dropout(p=0.5, inplace=False)
        (4): Linear(in_features=512, out_features=512, bias=True)
        (5): ReLU()
        (6): Dropout(p=0.5, inplace=False)
        (7): Linear(in_features=512, out_features=10, bias=True)
      )
    )
    


```python
batch_size=64
trainloader,testloader=load_fashionMNIST(root='./datasets/FashionMNIST_norm',batchsize=batch_size,num_workers=2,resize=224,n_mean=0.5,n_std=0.5)

def dataiter_view(dataiter):
    '''
    function - 查看可迭代数据形状
    '''
    dataiter_=iter(dataiter)
    images_,labels_=dataiter_.next()
    print('数据形状：',images_.shape)
    return images_,labels_
images_,labels_=dataiter_view(testloader)     
```

    数据形状： torch.Size([64, 1, 224, 224])
    


```python
lr,num_epochs=0.001,5
optimizer=torch.optim.Adam(VGG_net_.parameters(),lr=lr)
```


```python
def evaluate_accuracy_V2(data_iter, net, device=None):
    '''
    function - 模型精度计算
    '''    
    if device is None and isinstance(net,torch.nn.Module):
        device=list(net.parameters())[0].device #如果没指定device就使用net的device
    acc_sum,n=0.0,0
    with torch.no_grad():
        for X,y in data_iter:
            if isinstance(net,torch.nn.Module):
                net.eval() #评估模式，会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() #改回训练模式
            n+=y.shape[0]
    return acc_sum/n
    
def train_v2(net, train_iter, test_iter,optimizer, device, num_epochs):
    from tqdm.auto import tqdm
    import time
    '''
    function - 训练模型，v2版
    '''
    net = net.to(device)
    print("training on-",device)
    loss=torch.nn.CrossEntropyLoss()
    for epoch in tqdm(range(num_epochs)):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X,y in train_iter:
            X=X.to(device)
            y=y.to(device)
            y_pred=net(X)
            l=loss(y_pred,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc=evaluate_accuracy_V2(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'% (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))           
```


```python
train_v2(net=VGG_net_, train_iter=trainloader, test_iter=testloader,optimizer=optimizer, device=device, num_epochs=num_epochs) 
```

    training on- cuda
    


    HBox(children=(HTML(value=''), FloatProgress(value=0.0, max=5.0), HTML(value='')))


    epoch 1, loss 0.5758, train acc 0.786, test acc 0.879, time 46.5 sec
    epoch 2, loss 0.3177, train acc 0.885, test acc 0.895, time 45.0 sec
    epoch 3, loss 0.2715, train acc 0.901, test acc 0.908, time 44.8 sec
    epoch 4, loss 0.2381, train acc 0.912, test acc 0.906, time 45.1 sec
    epoch 5, loss 0.2131, train acc 0.922, test acc 0.917, time 45.2 sec
    
    

#### 1.3.2 torchvision.models实现VGG网络
VGG网络包含于`torchvision.models`库中， 因此无需自行配置网络，直接下载使用。通常模型也包含预先训练的模型参数，可以配置`pretrained=True`下载，下载到本地的位置为`C:\Users\<your name>\.cache\torch\hub\checkpoints`，后缀名为.pth。有时直接下载的网络并不能直接应用到其它不同的数据集，例如fashionMNIST数据集，该数据集的图像为灰色，即只有一个通道；同时，只有10个标签。因此需要对应层修改输入、输出大小。在[Finetuning Torchvision Models](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)中，包含了模型参数微调的方法，对于VGG而言，修改卷积层，可以通过`.features[idx]`的方式读取；修改全连接层，可以通过`.classifier[idx]`的方式修改。需要修改的层为features[0]和classifier[6]这两个层。其它层不需修改。

注意在训练模型之前，需要调整优化函数`optimizer=torch.optim.Adam(VGG_model.parameters(),lr=lr)`的输入模型参数为当前的网络模型。


```python
import torchvision 
VGG_model=torchvision.models.vgg11(pretrained=False) #如果为真，返回在ImageNet上预先训练的模型
```


```python
print(VGG_model)
```

    VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): ReLU(inplace=True)
        (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): ReLU(inplace=True)
        (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (9): ReLU(inplace=True)
        (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (11): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (12): ReLU(inplace=True)
        (13): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (14): ReLU(inplace=True)
        (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (16): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (17): ReLU(inplace=True)
        (18): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (19): ReLU(inplace=True)
        (20): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
      (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace=True)
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace=True)
        (5): Dropout(p=0.5, inplace=False)
        (6): Linear(in_features=4096, out_features=1000, bias=True)
      )
    )
    


```python
VGG_model.features[0]=nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
VGG_model.classifier[6]=nn.Linear(in_features=4096, out_features=10, bias=True)
print(VGG_model)
```

    VGG(
      (features): Sequential(
        (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): ReLU(inplace=True)
        (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): ReLU(inplace=True)
        (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (9): ReLU(inplace=True)
        (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (11): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (12): ReLU(inplace=True)
        (13): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (14): ReLU(inplace=True)
        (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (16): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (17): ReLU(inplace=True)
        (18): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (19): ReLU(inplace=True)
        (20): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
      (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace=True)
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace=True)
        (5): Dropout(p=0.5, inplace=False)
        (6): Linear(in_features=4096, out_features=10, bias=True)
      )
    )
    


```python
lr,num_epochs=0.001,5
optimizer=torch.optim.Adam(VGG_model.parameters(),lr=lr)
train_v2(net=VGG_model, train_iter=trainloader, test_iter=testloader,optimizer=optimizer, device=device, num_epochs=num_epochs)  
```

    training on- cuda
    


    HBox(children=(HTML(value=''), FloatProgress(value=0.0, max=5.0), HTML(value='')))


    epoch 1, loss 0.5578, train acc 0.817, test acc 0.886, time 418.3 sec
    epoch 2, loss 0.2851, train acc 0.895, test acc 0.894, time 423.2 sec
    epoch 3, loss 0.2467, train acc 0.909, test acc 0.908, time 422.6 sec
    epoch 4, loss 0.2115, train acc 0.923, test acc 0.917, time 422.1 sec
    epoch 5, loss 0.1929, train acc 0.931, test acc 0.924, time 421.0 sec
    
    

### 1.5 要点
#### 1.5.1 数据处理技术

* 卷积层，池化层输出尺寸（形状）计算

* tensorboard图表化深度学习参数，辅助调参

* 可视化卷积层/卷积核

* \* 收集元组中所有位置参数 ** 收集字典中的所有关键字参数

#### 1.5.2 新建立的函数

* class - PyTorch 卷积层，池化层输出尺寸(shape)计算，及根据输入，输出尺寸(shape)反推pad填充大小. `conv2d_output_size_calculation`

包括：

* function - 如果num=2，则返回(2,2)；如果num=(2,2)，则返回(2,2). `num2tuple(self,num)`

* funciton - 计算PyTorch的nn.Conv2d卷积方法的输出尺寸。以卷积核左上角对位图像左上角第一个像素值开始由左-->右，由上-->下卷积计算. `conv2d_output_shape(self,h_w, kernel_size=1, stride=1, pad=0, dilation=1)`

* function - 以卷积核右下角对位图像左上角第一个像素值开始由左-->右，由上-->下卷积计算. `convtransp2d_output_shape(self,h_w, kernel_size=1, stride=1, pad=0, dilation=1, out_pad=0)`

* function - conv2d_output_shape 方法的逆，求填充pad. `conv2d_get_padding(self,h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1)`

* function - convtransp2d_output_shape 方法的逆，求填充pad. `convtransp2d_get_padding(self,h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1, out_pad=0)`

* function - pooling池化层输出尺寸，同conv2d_output_shape. ` pooling_output_shape(self,h_w, kernel_size=1, stride=1, pad=0, dilation=1)`

---

* fucntion - 一次性计算卷积输出尺寸. `conv2d_outputSize_A_oneTime(convs_params)`

* function - 下载读取fashionMNIST数据集，并建立训练、测试可迭代数据集. `load_fashionMNIST(root,batchsize=4,num_workers=2,resize=None,n_mean=0.5,n_std=0.5)`

* 自定义net_fashionMNIST(nn.Module)网络. `net_fashionMNIST(nn.Module)`

* 定义显示图像函数（一个batch批次）. `matplotlib_imshow(img, one_channel=False)`

* function - 用训练的网络预测给定的一组图像，并计算相应的概率. `images_to_probs(net, images)`

* function - 用训练的网络预测给定的一组图像，并计算概率后，显示图像、预测值以及概率，和实际的标签. `plot_classes_preds(net, images, labels)`

* class - 取回卷积神经网络中所有卷积层及其权重. `class conv_retriever`

* function - 可视化卷积核 visualize the conv layer filters. `visualize_convFilter(conv_layer,model_weight,output_name,figsize=(10,10))`

* function - 可视化所有卷积层（卷积结果）. `visualize_convLayer(imgs_batch,conv_layers,model_weights,num_show=6,figsize=(10,10))`

* function - 自定义VGG. `vgg_block(num_convs,in_channels,out_channels)`;`vgg(conv_arch,fc_features,fc_hidden_units=4096)`

* function - 查看可迭代数据形状. `dataiter_view(dataiter)`

* function - 模型精度计算. `evaluate_accuracy_V2(data_iter, net, device=None)`

* function - 训练模型，v2版. `train_v2(net, train_iter, test_iter,optimizer, device, num_epochs)`

#### 1.5.3 所调用的库


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import time
```

#### 1.5.4 参考文献
1. Jibin Mathew.PyTorch Artificial Intelligence Fundamentals: A recipe-based approach to design, build and deploy your own AI models with PyTorch 1.x[m].UK:Packt Publishing (February 28, 2020)
2. Aston Zhang,Zack C. Lipton,Mu Li,etc.Dive into Deep Learning[M].；中文版-阿斯顿.张,李沐,扎卡里.C. 立顿,亚历山大.J. 斯莫拉.动手深度学习[M].人民邮电出版社,北京,2019-06-01
3. [Convolutional Neural Networks (CNNs / ConvNets)](https://cs231n.github.io/convolutional-networks/)
4. [Understanding 2D Dilated Convolution Operation with Examples in Numpy and Tensorflow with Interactive Code](https://towardsdatascience.com/understanding-2d-dilated-convolution-operation-with-examples-in-numpy-and-tensorflow-with-d376b3972b25)
5. [Visualizing Models, Data, and Training with TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html?highlight=fashion%20mnist)
6. [Visualizing Filters and Feature Maps in Convolutional Neural Networks using PyTorch](https://debuggercafe.com/visualizing-filters-and-feature-maps-in-convolutional-neural-networks-using-pytorch/)
