> Created on Tue Dec 29 10\12\57 2020 @author: Richie Bao-caDesign设计(cadesign.cn)

## 1. 从解析解(analytical solution)-->到数值解(numerical solution) | 从机器学习[scikit-learn](https://scikit-learn.org/stable/)-->到深度学习[pytorch](https://PyTorch.org/)
解析解(analytical solution)，又称闭式解，是可以用解析表达式来表达的解（有时也称为公式解）。在数学上，如果一个方程或者方程组存在的某些解，是由有限次常见运算的组合给出的形式，则称该方程存在解析解。二次方程的根就是一个解析解的典型例子。当解析解不存在时，比如五次以及更高次的代数方程，则该方程只能用数值分析的方法求解近似值（有限元的方法、数值逼近，差值的方法，大多数深度学习通过优化算法有限次迭代模型参数来尽可能降低损失函数的值），则是数值解(numerical solution) 。例如大多数偏微分方程，尤其是非线性偏微分方程。

在'简单回归，多元回归'部分使用解析解的方法（对真实值与预测值之差的平方和，即残差平方和求微分或偏微分）解得回归系数；在'梯度下降法'部分，则应用了数值解的优化算法梯度下降法，通过配置学习率、精度、和随机初始化系数，定义模型，定义损失函数（残差平方和为一种损失函数，又称代价函数），定义梯度下降函数（对损失函数求梯度，即对模型变量微分），迭代训练模型直至损失函数计算的结果小于预设值，解得模型权重值。

机器学习(machine learning)是人工智能的一个分支，机器学习理论主要是设计和分析一些让计算机可以自动'学习'的算法。机器学习算法是一类从数据中自动分析获得规律，并利用规律对未知数据进行预测的算法。常用的机器学习库是scikit-learn，包括数据预处理、聚类、回归和分类三大方向，其中提供了SGDRegressor随机梯度下降方法。深度学习(deep learning)是机器学习的一个分支，是一种以人工神经网络为结构，对资料数据进行表征学习的算法。深度学习库推荐使用[pytorch](https://pytorch.org/)，是以python优先的深度学习框架，其设计符合人类的思维方式（而[TensorFlow](https://www.tensorflow.org/)因为不符合人们的思维习惯，很难应用，但是后来结合到keras框架，确立为Tensorflow高阶API，使得应用TensorFlow的环境得以改善）。基于pytorch的深度学习资源推荐[*Dive into Deep Learning*](https://d2l.ai/)，中文版[*动手深度学习*](https://tangshusen.me/Dive-into-DL-PyTorch/#/)。下述的深度学习方法基础解析亦是以*Dive into Deep Learning*为主要参考。

### 1.1 张量(tensor)

最常用的数据格式是使用numpy库提供的数组(array)形式（也是机器学习库Sklearn数据集，及数据处理的格式），以及pandas库提供的DataFrame（常配合地理信息系统中的table/表，以及数据库使用）、Series数据格式。在深度学习（[PyTorch](https://pytorch.org/)库，或者[TensorFlow](https://www.tensorflow.org/)）中，引入了张量(tensor)，可以用于GPU计算，并有自动求梯度等更多功能。张量和数组类似，0维（一个数，rank=0,rank为维度数）张量即为标量（Scalar），1维张量（rank=1）即为向量/矢量(Vector)，2维张量(rank=2)即为矩阵(Matrix)，3维张量(rank=3)即为矩阵数组。深度学习中，张量可以看作一个多维数组(multidimentional array)。

> 参考文献：
> 1. Aston Zhang,Zack C. Lipton,Mu Li,etc.[Dive into Deep Learning](https://d2l.ai/)[M].；中文版-阿斯顿.张,李沐,扎卡里.C. 立顿,亚历山大.J. 斯莫拉.[动手深度学习](https://tangshusen.me/Dive-into-DL-PyTorch/#/)[M].人民邮电出版社,北京,2019-06-01


```python
import torch
print("_"*50,"A-tensor创建")
#01-未初始化，shape=（5，3）的tensor
t_a=torch.empty(5,3)
print("01-创建未初始化，shape=（5，3）的tensor:\nt_a={},\nshape={}".format(t_a,t_a.shape))
#02-随机初始化shape=（5，3）的tensor
t_b=torch.rand(5,3)
print("02-随机初始化shape=（5，3）的tensor:\nt_b={}".format(t_b))
#03-数据类型为long，全0的tensor
t_c=torch.zeros(5,3,dtype=torch.long)
print("03-数据类型为long，全0的tensor:\nt_c={}".format(t_c))
#04-由给定数据直接创建
t_d=torch.tensor([[3.1415,0],[9,2.71828]])
print("04-由给定数据直接创建:\nt_d={}".format(t_d))
#05-tensor.new_ones()方法重用已有tensor，保持数据类型，及torch.device(CPU或GPU)
t_e=t_c.new_ones(4,3)
print("05-tensor.new_ones()方法重用已有tensor，保持数据类型，及torch.device(CPU或GPU):\nt_e={}\ndtype={},device={}".format(t_e,t_e.dtype,t_e.device))
#06-torch.randn_like()方法重用已有tensor,并可重定义数据类型
t_f=torch.randn_like(t_c,dtype=torch.float)
print("#06-torch.randn_like()方法重用已有tensor,并可重定义数据类型:\nt_f={},\ndtype={}".format(t_f,t_f.dtype))
#07-tensor.size()和tensor.shape方法查看tensor形状
print("07-tensor.size()和tensor.shape方法查看tensor形状:\nt_f.size()={},t_f.shape={}".format(t_f.size(),t_f.shape))
```

    __________________________________________________ A-tensor创建
    01-创建未初始化，shape=（5，3）的tensor:
    t_a=tensor([[-0.6849, -0.0610,  2.3693],
            [ 1.0276, -0.7919,  2.6917],
            [ 1.2662, -0.9841,  0.4744],
            [ 0.6238,  2.5234, -0.4556],
            [ 0.9540,  1.8516,  0.1717]]),
    shape=torch.Size([5, 3])
    02-随机初始化shape=（5，3）的tensor:
    t_b=tensor([[0.5356, 0.7655, 0.6168],
            [0.2533, 0.6416, 0.1587],
            [0.9901, 0.6007, 0.1130],
            [0.2830, 0.9496, 0.3424],
            [0.5047, 0.4998, 0.1619]])
    03-数据类型为long，全0的tensor:
    t_c=tensor([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]])
    04-由给定数据直接创建:
    t_d=tensor([[3.1415, 0.0000],
            [9.0000, 2.7183]])
    05-tensor.new_ones()方法重用已有tensor，保持数据类型，及torch.device(CPU或GPU):
    t_e=tensor([[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]])
    dtype=torch.int64,device=cpu
    #06-torch.randn_like()方法重用已有tensor,并可重定义数据类型:
    t_f=tensor([[-0.7853,  1.5429,  1.4266],
            [-0.0986,  0.2011, -2.0210],
            [ 1.6421, -1.0019, -1.2939],
            [-0.3767,  2.2305,  0.6250],
            [ 2.0690, -0.4641, -1.1207]]),
    dtype=torch.float32
    07-tensor.size()和tensor.shape方法查看tensor形状:
    t_f.size()=torch.Size([5, 3]),t_f.shape=torch.Size([5, 3])
    


```python
print("_"*50,"B-tensor操作")
#08-加法形式-1-'+'
print("08-加法形式-1-'+':\nt_a+t_b={}".format(t_a+t_b))
#09-加法形式-2-torch.add()
print("09-加法形式-2-torch.add():\ntorch.add(t_a,t_b)={}".format(torch.add(t_a,t_b)))
#10-加法形式-3-原地结果替换inplace/add_(PyTorch原地操作inplace都有后缀_)
print("10-加法形式-3-原地结果替换inplace/add_:\nt_a.add_(t_b)={}，\nt_a={}".format(t_a.add_(t_b),t_a))
#11-索引，共享存储地址
t_g=t_a[0,:]
print("11-索引，共享存储地址:\nt_a[0,:]={}".format(t_g))
t_g+=1
print("t_g+=1后,tg={},t_a[0,:]={}".format(t_g,t_a[0,:]))
#12-view()方法改变tensor形状（shape），但为同一存储地址
print("12-view()方法改变tensor形状（shape），但为同一存储地址:\nt_a.shape={},t_a.view(15).shape={},t_a.view(-1,5).shape={}".format(t_a.shape,t_a.view(15).shape,t_a.view(-1,5).shape))
#13-clone()
print("13-clone():\nid(t_a)==id(t_a.clone())={}".format(id(t_a)==id(t_a.clone())))

print("_"*50,"C-tensor广播机制")
#14-广播机制
t_h=torch.arange(1,3).view(1,2)
t_i=torch.arange(1,4).view(3,1)
print("14-广播机制:\nt_h={}\nt_i={}\nt_h.shape={},t_i.shape={}\nt_h+t_i={}".format(t_h,t_i,t_h.shape,t_i.shape,t_h+t_i))

print("_"*50,"D-inplace原地操作，节约内存")
#15-inplace——out参数
print("15-inplace——out参数:\nid(t_a)==id(torch.add(t_a,t_b,out=t_a))={}".format(id(t_a)==id(torch.add(t_a,t_b,out=t_a))))
#16-inplace——+=(add_())
t_j=t_a
t_a+=1
print("16-inplace——+=(add_()):\nid(t_a)==id(t_j)={}".format(id(t_a)==id(t_j)))
#17-inplace——[:]
t_k=t_a
t_a[:]=t_a+t_b
print("17-inplace——[:]:\nid(t_a)==id(t_k)={}".format(id(t_a)==id(t_k)))

print("_"*50,"E-tensor<-->array(NumPy)")
#18-tensor-->array
print("18-tensor-->array:\ntype(t_a)={},type(t_a.numpy())={}".format(type(t_a),type(t_a.numpy())))
#19-array-->tenssor
import numpy as np
array=np.arange(15).reshape(3,5)
print("19-array-->tenssor:\ntype(array)={},type(torch.from_numpy(array))={}".format(type(array),type(torch.from_numpy(array))))
```

    __________________________________________________ B-tensor操作
    08-加法形式-1-'+':
    t_a+t_b=tensor([[53.4192, 64.3870, 60.1233],
            [21.4257, 37.0823, 18.8311],
            [54.8192, 35.0464, 14.5611],
            [22.3603, 54.2553, 23.9502],
            [32.6671, 33.3424, 16.4583]])
    09-加法形式-2-torch.add():
    torch.add(t_a,t_b)=tensor([[53.4192, 64.3870, 60.1233],
            [21.4257, 37.0823, 18.8311],
            [54.8192, 35.0464, 14.5611],
            [22.3603, 54.2553, 23.9502],
            [32.6671, 33.3424, 16.4583]])
    10-加法形式-3-原地结果替换inplace/add_:
    t_a.add_(t_b)=tensor([[53.4192, 64.3870, 60.1233],
            [21.4257, 37.0823, 18.8311],
            [54.8192, 35.0464, 14.5611],
            [22.3603, 54.2553, 23.9502],
            [32.6671, 33.3424, 16.4583]])，
    t_a=tensor([[53.4192, 64.3870, 60.1233],
            [21.4257, 37.0823, 18.8311],
            [54.8192, 35.0464, 14.5611],
            [22.3603, 54.2553, 23.9502],
            [32.6671, 33.3424, 16.4583]])
    11-索引，共享存储地址:
    t_a[0,:]=tensor([53.4192, 64.3870, 60.1233])
    t_g+=1后,tg=tensor([54.4192, 65.3870, 61.1233]),t_a[0,:]=tensor([54.4192, 65.3870, 61.1233])
    12-view()方法改变tensor形状（shape），但为同一存储地址:
    t_a.shape=torch.Size([5, 3]),t_a.view(15).shape=torch.Size([15]),t_a.view(-1,5).shape=torch.Size([3, 5])
    13-clone():
    id(t_a)==id(t_a.clone())=False
    __________________________________________________ C-tensor广播机制
    14-广播机制:
    t_h=tensor([[1, 2]])
    t_i=tensor([[1],
            [2],
            [3]])
    t_h.shape=torch.Size([1, 2]),t_i.shape=torch.Size([3, 1])
    t_h+t_i=tensor([[2, 3],
            [3, 4],
            [4, 5]])
    __________________________________________________ D-inplace原地操作，节约内存
    15-inplace——out参数:
    id(t_a)==id(torch.add(t_a,t_b,out=t_a))=True
    16-inplace——+=(add_()):
    id(t_a)==id(t_j)=True
    17-inplace——[:]:
    id(t_a)==id(t_k)=True
    __________________________________________________ E-tensor<-->array(NumPy)
    18-tensor-->array:
    type(t_a)=<class 'torch.Tensor'>,type(t_a.numpy())=<class 'numpy.ndarray'>
    19-array-->tenssor:
    type(array)=<class 'numpy.ndarray'>,type(torch.from_numpy(array))=<class 'torch.Tensor'>
    

* tensor在GPU和CPU上互相转换，及在GPU上运算


```python
if torch.cuda.is_available():
    device=torch.device("cuda")
    print("device={}".format(device))
    x=torch.tensor([[3.1415,0],[9,2.71828]],device=device)
    print("x={}".format(x))
    y=torch.rand(2,2)
    print("默认CPU,y={}".format(y))
    y=y.to(device)
    print(".to(device)，y={}".format(y))    
    print("GPU运算,x+y={}".format(x+y))
    print(".to('cpu'),(x+y).to('cpu',torch.double)={}".format((x+y).to("cpu",torch.double)))
```

    device=cuda
    x=tensor([[3.1415, 0.0000],
            [9.0000, 2.7183]], device='cuda:0')
    默认CPU,y=tensor([[0.9658, 0.6832],
            [0.6438, 0.1294]])
    .to(device)，y=tensor([[0.9658, 0.6832],
            [0.6438, 0.1294]], device='cuda:0')
    GPU运算,x+y=tensor([[4.1073, 0.6832],
            [9.6438, 2.8477]], device='cuda:0')
    .to('cpu'),(x+y).to('cpu',torch.double)=tensor([[4.1073, 0.6832],
            [9.6438, 2.8477]], dtype=torch.float64)
    

### 1.2 微积分-链式法则(chain rule)
在'微积分基础的代码表述'部分，求导的基本公式中给出了复合函数的微分公式为：$\{g(f(x))\}' = g'(f(x)) f' (x) $ ，也表示为：$\frac{dy}{dx}= \frac{dy}{du}   .  \frac{du}{dx} $。链式法则表述为，两个函数组合起来的复合函数，导数等于内层函数代入外层函数的导函数，乘以内层函数的导函数。下述验证代码中，定义了复合函数为$5 \times  sin( x^{3}+5 )$，将其分解为外层函数：$5 \times  sin( x)$和内层函数:$ x^{3}+5 $。使用'sympy'库diff方法求导，分别对各个分解函数求导，求导后，外层导函数带入内层函数，最后求积即为应用链式法则求复合函数的结果。其结果与直接对复合函数求导的结果保持一致。

链式法则应用于深度学习神经网络的反向传播计算。


```python
import sympy
from sympy import diff,pprint
import matplotlib.pyplot as plt
import numpy as np

x,x_i,x_o=sympy.symbols('x x_i,x_o')
composite_func=5*sympy.sin(x**3+5)
composite_func_=sympy.lambdify(x,composite_func,"numpy")
print("复合函数：")
pprint(composite_func)

inner_func=x_i**3+5
inner_func_=sympy.lambdify(x_i,inner_func,"numpy")
print("内层函数：")
pprint(inner_func)

outer_func=5*sympy.sin(x_o)
outer_func_=sympy.lambdify(x_o,outer_func,"numpy")
print("外层函数：")
pprint(outer_func)

#求导
d_composite=diff(composite_func,x)
d_composite_=sympy.lambdify(x,d_composite,"numpy")
print("复合函数-求导：")
pprint(d_composite)

d_chain_rule_form_1=diff(outer_func,x_o).subs(x_o,inner_func)*diff(inner_func,x_i)
print("链式法则-求导:")
pprint(d_chain_rule_form_1)

t=np.arange(-np.pi,np.pi,0.01)
y_composite=composite_func_(t)
y_inner=inner_func_(t)
y_outer=outer_func_(t)

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(t,y_composite,label="composite_func")
ax.plot(t,y_inner,label="inner_func")
ax.plot(t,y_outer,label="outer_func")
ax.plot(t,d_composite_(t)/10,label="derivative/5",dashes=[6, 2],color='gray',alpha=0.3)
ax.axhline(y=0,color='r',linestyle='-.',alpha=0.5)

ax.legend(loc='upper left')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()
```

    复合函数：
         ⎛ 3    ⎞
    5⋅sin⎝x  + 5⎠
    内层函数：
      3    
    xᵢ  + 5
    外层函数：
    5⋅sin(xₒ)
    复合函数-求导：
        2    ⎛ 3    ⎞
    15⋅x ⋅cos⎝x  + 5⎠
    链式法则-求导:
         2    ⎛  3    ⎞
    15⋅xᵢ ⋅cos⎝xᵢ  + 5⎠
    

<a href=""><img src="./imgs/20_02.png" height='auto' width='auto' title="caDesign"></a>    


### 1.3 激活函数(activation function)
在多层神经网络中（多层感知机，multilayer perception,MLP），如果不使用激活函数，则每一层节点（神经元）的输入都是上一层输出的线性函数，那么无论神经网络有多少层，输出都是输入的线性组合，即为原始的感知机(单层感知机，perception)，其网络的逼近能力有限。因此引入了非线性函数作为激活函数，使神经网络可以逼近任意函数。常用的激活函数有ReLU(rectified linear unit)，sigmoid，和tanh函数。

给定元素$x$，ReLU函数的定义为:$ReLU(x)=max(x,0)$，可知ReLU函数只保留正数元素，并将负数元素清零，为两段线性函数。ReLU函数的导数对应在负数时，为0，正数时为1.值为0时，ReLU函数不可导，在此处将其导数配置为0.

sigmoid函数可以将元素的值变换到0-1之间，其公式为：$sigmoid(x)= \frac{1}{1+exp(-x)} $。sigmoid函数在早期的神经网络中较为普遍，目前逐渐被更为简单的ReLU函数取代。当输入接近0时，sigmoid函数接近线性变换。sigmoid函数的导数，在输入为0时，导数达到最大值0.25；当输入偏离0时，导数趋近于0.

tanh（双曲正切）函数可以将元素的值变换到-1到1之间。其公式为：$tanh(x)= \frac{1-exp(-2x)}{1+exp(-2x)} $，当输入接近0时，tanh函数接近线性变换。同时，tanh函数在坐标系原点上对称。依据链式法则，tanh函数的导数为：$ tanh' (x)=1- tanh^{2}(x) $。当输入为0时，tanh的导数达到最大值1；当输入偏离0时，tanh函数的导数趋近于0.


```python
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
fig, axs=plt.subplots(1,3,figsize=(28,5))

#A-ReLU
x=torch.arange(-8.0,8.0,0.1,requires_grad=True)
y_relu=x.relu()
axs[0].plot(x.detach().numpy(), y_relu.detach().numpy(),label="ReLU")

#ReLU函数的导数
y_relu.sum().backward()
axs[0].plot(x.detach().numpy(), x.grad.detach().numpy(),label="grad of ReLU",linestyle='-.')

#B-sigmoid
y_sigmoid=x.sigmoid()
axs[1].plot(x.detach().numpy(), y_sigmoid.detach().numpy(),label="sigmoid")

#sigmoid函数的导函数
x.grad.zero_() #参数梯度置零
y_sigmoid.sum().backward()
axs[1].plot(x.detach().numpy(), x.grad.detach().numpy(),label="grad of ReLU",linestyle='-.')

#C-tanh
y_tanh=x.tanh()
axs[2].plot(x.detach().numpy(), y_tanh.detach().numpy(),label="tanh")

#tanh函数的导函数
x.grad.zero_()
y_tanh.sum().backward()
axs[2].plot(x.detach().numpy(), x.grad.detach().numpy(),label="grad of tanh",linestyle='-.')

axs[0].legend(loc='upper left', frameon=False)
axs[1].legend(loc='upper left', frameon=False)
axs[2].legend(loc='upper left', frameon=False)
plt.show()
```


    
<a href=""><img src="./imgs/20_03.png" height='auto' width='auto' title="caDesign"></a> 
    


### 1.4 前向(正向)传播(forward propagation)与后向（反向）传播(back propagation)
构建一个典型的三层神经网络，包括输入层(input layer)包含两个神经元(neuron)$i_{1},i_{2}$，和一个偏置（偏差/截距，bias）项$b_{1}$；隐含层(hidden layter)包含两个神经元$h_{1},h_{2}$，和一个偏置$b_{2}$；及输出层(output layer)$o_{1},o_{2}$。假设数据集仅包含一个特征向量(feature)并含两个值，即输入数据：$i_{1} =0.05, i_{2} =0.10$；对应类标(label)，即输出数据：$o_{1}=0.01,o_{2}=0.99$；同时随机初始化权重值：$w_{1} =0.15,w_{2} =0.20,w_{3} =0.25,w_{4} =0.30,w_{5} =0.40,w_{6} =0.45,w_{7} =0.50,w_{8} =0.55$;偏置值为$b_{1} =0.35,b_{2} =0.60$

通过神经网络，输入值经过与权重及偏置值的计算，使得其结果与类标接近，从而最终确定权重值。

<a href=""><img src="./imgs/20_01.jpg" height='auto' width='700' title="caDesign"></a>

> 参考：https://www.cnblogs.com/charlotte77/p/5629865.html

**Step-0：初始化** 

初始化特征值，类标，以及权重值和偏置


```python
i_1,i_2=0.05,0.10
w_1,w_2,w_3,w_4,w_5,w_6,w_7,w_8=0.15,0.20,0.25,0.30,0.40,0.45,0.50,0.55
b_1,b_2=0.35,0.60
o_1,o_2=0.01,0.99
```

**Step-1：前向传播** 
* 1. 输入层--->隐含层

计算神经元$h_{1}$的输入加权和


```python
net_h_1=w_1*i_1+w_2*i_2+b_1*1
print("net_h_1={}".format(net_h_1))
```

    net_h_1=0.3775
    

对h1应用激活函数-sigmoid


```python
def sigmoid(x):
    import math
    '''
    function - sigmoid函数
    '''
    return 1/(1+math.exp(-x))
out_h_1=sigmoid(net_h_1)
print("out_h_1={}".format(out_h_1))
```

    out_h_1=0.5932699921071872
    

同理，计算神经元$h_{2}$的输入加权和，并应用sigmoid函数


```python
net_h_2=w_3*i_1+w_4*i_2+b_1*1
out_h_2=sigmoid(net_h_2)
print("out_h_2={}".format(out_h_2))
```

    out_h_2=0.596884378259767
    

* 2. 隐含层--->输出层

计算神经元$o_{1}, o_{2}$的值


```python
net_o_1=w_5*out_h_1+w_6*out_h_2+b_2*1
out_o_1=sigmoid(net_o_1)
print("out_o_1={}".format(out_o_1))
net_o_2=w_7*out_h_1+w_8*out_h_2+b_2*1
out_o_2=sigmoid(net_o_2)
print("out_o_2={}".format(out_o_2))
```

    out_o_1=0.7513650695523157
    out_o_2=0.7729284653214625
    

随机初始化权重值，逐层计算输入加权和(Summation and Bias):$\sum_{i=1}^m( w_{i} x_{i}  )+bias$，并应用激活函数（sigmoid），获得输出值'out_o_1'和'out_o_2'，与实际值$o_{1}=0.01,o_{2}=0.99$相差还很远，现在对误差进行反向传播，更新权重/权值，重新计算输出。

**Step-2：反向传播** 

* 1. 计算总误差（残差平方误差，Residual square error）

分别计算各个输出（此时有2个输出）的误差，再求和。


```python
E_o_1=1/2*(out_o_1-o_1)**2
E_o_2=1/2*(out_o_2-o_2)**2
E_total=E_o_1+E_o_2
print("E_total={}".format(E_total))
```

    E_total=0.2983711087600027
    

* 2. 隐含层--->输出层的权值更新

以w_5权重值参数为例，计算w_5对整体误差的影响，用整体误差对w_5求偏导，应用微分链式法则，其公式为：$\frac{ \partial  E_{total} }{ \partial  w_{5} } =\frac{ \partial  E_{total} }{ \partial  out_{o1} } \times \frac{ \partial  out_{o1} }{ \partial  net_{o1} } \times \frac{ \partial  net_{o1} }{ \partial  w_{5} }$，分别是对总误差计算公式、激活函数和加权和函数求导。只是各个输入值分别对应w_5计算路径下的各个对应值，可以很容易的从以上神经网络结构图中确定，即函数的输入端值。总误差计算公式导函数对应$ out_{o1}$即`out_o_1`，激活函数导函数对应$net_{o1} $即`net_o_1`，加权和函数导函数对应$out_{h1} $即`out_h_1`。


```python
import sympy
from sympy import diff,pprint

#w_5-链式法则-01-总误差函数（公式）对out_o_1的偏导数
out_o_1_,o_1_,out_o_2_,o_2_=sympy.symbols('out_o_1_ o_1_,out_o_2_ o_2_')
d_E_total_2_out_o_1=diff(1/2*(out_o_1_-o_1_)**2+1/2*(out_o_2_-o_2_)**2,out_o_1_)
d_E_total_2_out_o_1_value=d_E_total_2_out_o_1.subs([(o_1_,o_1),(out_o_1_,out_o_1)])
print("d_E_total_2_out_o_1_value={}".format(d_E_total_2_out_o_1_value))

#w_5-链式法则-02-激活函数对net_o_1的偏导数
from sympy.functions import exp
x_sigmoid=sympy.symbols('x_sigmoid')
d_activation=diff(1/(1+exp(-x_sigmoid)),x_sigmoid)
d_out_o_1_2_net_o_1_value=d_activation.subs([(x_sigmoid,net_o_1)])
print("d_out_o_1_2_activation_value={}".format(d_out_o_1_2_net_o_1_value))

#w_5-链式法则-03-隐藏层h1，加权和函数（公式）对out_h_1的偏导数
w_5_,out_h_1_,w_6_,out_h_2_,b_2_=sympy.symbols('w_5,out_h_1,w_6,out_h_2,b_2')
d_net_o_1_2_w_5=diff(w_5_*out_h_1_+w_6_*out_h_2_+b_2_*1,w_5_)
d_net_o_1_2_w_5_value=d_net_o_1_2_w_5.subs([(out_h_1_,out_h_1)])
print("d_net_o_1_2_w_5_value={}".format(d_net_o_1_2_w_5_value))

#w_5-链式法则-各部分相乘
d_E_total_2_w_5=d_E_total_2_out_o_1_value*d_out_o_1_2_net_o_1_value*d_net_o_1_2_w_5_value
print("d_E_total_2_w_5={}".format(d_E_total_2_w_5))

#w_5-权重值更新
lr=0.5 #学习速率
w_5_update=w_5-lr*d_E_total_2_w_5
print("w_5_update={}".format(w_5_update))
```

    d_E_total_2_out_o_1_value=0.741365069552316
    d_out_o_1_2_activation_value=0.186815601808960
    d_net_o_1_2_w_5_value=0.593269992107187
    d_E_total_2_w_5=0.0821670405642308
    w_5_update=0.358916479717885
    

所有权值的偏导求法基本相同，为了减少重复的代码，可以定义公用的导函数，只是不同的权值根据其在神经网络结构下的路径，调整链式法则中的各个偏导函数，并对应替换值。


```python
#定义总误差偏导
def partialD_E_total_prediction(true_value_,predicted_value_):
    import sympy
    from sympy import diff
    '''
    function - 定义总误差偏导
    '''
    true_value,predicted_value=sympy.symbols('true_value predicted_value')
    partialD_E_total_prediction=diff(1/2*(predicted_value-true_value)**2,predicted_value)
    return partialD_E_total_prediction.subs([(predicted_value,predicted_value_),(true_value,true_value_)])

#定义激活函数偏导
def partialD_activation(x):
    import sympy
    from sympy import diff
    from sympy.functions import exp
    '''
    function -定义激活函数偏导
    '''
    x_sigmoid=sympy.symbols('x_sigmoid')
    partialD_activation=diff(1/(1+exp(-x_sigmoid)),x_sigmoid)
    return partialD_activation.subs([(x_sigmoid,x)])

#定义加权和偏导
def partialD_weightedSUM(w_):
    import sympy
    from sympy import diff
    '''
    fucntion - 定义加权和偏导
    '''
    w,x_w=sympy.symbols('w x_w')
    partialD_weightedSUM=diff(w*x_w,w)
    return partialD_weightedSUM.subs([(x_w,w_)])
```


```python
w_5_update=w_5-lr*partialD_E_total_prediction(o_1,out_o_1)*partialD_activation(net_o_1)*partialD_weightedSUM(out_h_1)
w_6_update=w_6-lr*partialD_E_total_prediction(o_1,out_o_1)*partialD_activation(net_o_1)*partialD_weightedSUM(out_h_2)
w_7_update=w_7-lr*partialD_E_total_prediction(o_2,out_o_2)*partialD_activation(net_o_2)*partialD_weightedSUM(out_h_1)
w_8_update=w_8-lr*partialD_E_total_prediction(o_2,out_o_2)*partialD_activation(net_o_2)*partialD_weightedSUM(out_h_2)

print("w_5_update={}\nw_6_update={}\nw_7_update={}\nw_8_update={}".format(w_5_update,w_6_update,w_7_update,w_8_update))
```

    w_5_update=0.358916479717885
    w_6_update=0.408666186076233
    w_7_update=0.511301270238738
    w_8_update=0.561370121107989
    

* 3. 输入层--->隐含层的权值更新

因为隐含层$ out_{h1}$会受到$ E_{o1},E_{o2}$两个地方传来的误差，因此分别计算并求和，再与其它路径下的导数求积。$ out_{h2}$与之同。


```python
w_1_update=w_1-lr*(partialD_E_total_prediction(o_1,out_o_1)*partialD_activation(net_o_1)*partialD_weightedSUM(w_5)+partialD_E_total_prediction(o_2,out_o_2)*partialD_activation(net_o_2)*partialD_weightedSUM(w_7))*partialD_activation(net_h_1)*partialD_weightedSUM(i_1)
w_2_update=w_2-lr*(partialD_E_total_prediction(o_1,out_o_1)*partialD_activation(net_o_1)*partialD_weightedSUM(w_5)+partialD_E_total_prediction(o_2,out_o_2)*partialD_activation(net_o_2)*partialD_weightedSUM(w_7))*partialD_activation(net_h_1)*partialD_weightedSUM(i_2)
w_3_update=w_3-lr*(partialD_E_total_prediction(o_1,out_o_1)*partialD_activation(net_o_1)*partialD_weightedSUM(w_6)+partialD_E_total_prediction(o_2,out_o_2)*partialD_activation(net_o_2)*partialD_weightedSUM(w_8))*partialD_activation(net_h_2)*partialD_weightedSUM(i_1)
w_4_update=w_4-lr*(partialD_E_total_prediction(o_1,out_o_1)*partialD_activation(net_o_1)*partialD_weightedSUM(w_6)+partialD_E_total_prediction(o_2,out_o_2)*partialD_activation(net_o_2)*partialD_weightedSUM(w_8))*partialD_activation(net_h_2)*partialD_weightedSUM(i_2)

print("w_1_update={}\nw_2_update={}\nw_3_update={}\nw_4_update={}".format(w_1_update,w_2_update,w_3_update,w_4_update))
```

    w_1_update=0.149780716132763
    w_2_update=0.199561432265526
    w_3_update=0.249751143632370
    w_4_update=0.299502287264739
    

至此误差的反向传播全部计算完，并更新权重值，重复前向传播。为了方法计算将前向传播定义为一个函数，计算结果显示其总误差为0.29102777369359933，较之前次0.2983711087600027有所下降。


```python
def forward_func(input_list,weight_list,bias_list,output_list):
    '''
    function - 三层神经网络，各层2个神经元的示例，前向传播函数
    '''
    def sigmoid(x):
        import math
        '''
        function - sigmoid函数
        '''
        return 1/(1+math.exp(-x))
    i_1,i_2=input_list
    w_1,w_2,w_3,w_4,w_5,w_6,w_7,w_8=weight_list
    b_1,b_2,b_3,b_4=bias_list
    o_1,o_2=output_list
    
    net_h_1=w_1*i_1+w_2*i_2+b_1*1
    out_h_1=sigmoid(net_h_1)
    
    net_h_2=w_3*i_1+w_4*i_2+b_2*1
    out_h_2=sigmoid(net_h_2)    
    
    net_o_1=w_5*out_h_1+w_6*out_h_2+b_3*1
    out_o_1=sigmoid(net_o_1)

    net_o_2=w_7*out_h_1+w_8*out_h_2+b_4*1
    out_o_2=sigmoid(net_o_2)

    E_total=1/2*(out_o_1-o_1)**2+1/2*(out_o_2-o_2)**2
    print("out_o_1={}\nout_o_2={}\nE_total={}".format(out_o_1,out_o_2,E_total))
    return out_o_1,out_o_2,E_total
out_o_1,out_o_2,E_total=forward_func(input_list=[0.05,0.10],weight_list=[w_1_update,w_2_update,w_3_update,w_4_update,w_5_update,w_6_update,w_7_update,w_8_update],bias_list=[0.35,0.35,0.60,0.60],output_list=[0.01,0.99])
```

    out_o_1=0.7420881111907824
    out_o_2=0.7752849682944595
    E_total=0.29102777369359933
    

**多层感知机-代码整合**

将上述的分解过程整合，实现迭代计算求取权重值。代码的结构包括三个类，NEURON类定义神经元前向传播和反向传播的函数；NEURAL_LAYER类是基于NEURON的层级别神经元汇总；NEURAL_NETWORK类构建神经网络结构，初始化权值并更新权值。此处的输入值保持不变为[0.05,0.1],输出值修改为[0.01,0.09]，能够更好的观察收敛的过程，如果仍然保持输出值为[0.01,0.99]，因为偏置保持不变，造成损失函数下降的趋势不明显。


```python
class NEURON:
    '''
    class - 每一神经元的输入加权和、激活函数，以及输出计算
    '''
    def __init__(self,bias):
        '''
        function - 初始化权重和偏置
        '''
        self.bias=bias
        self.weights=[]
        
    def activation(self,x_sigmoid):
        import math
        '''
        function - 定义sigmoid激活函数
        '''        
        return 1/(1+math.exp(-x_sigmoid))
        
    def net_input(self):
        import numpy as np
        '''
        function - 每一神经元的输入加权和，inputs，及weights数组形状为(-1,1)
        '''
        y=np.array(self.inputs).reshape(-1,1)*np.array(self.weights).reshape(-1,1)
        return y.sum()+self.bias
            
    def net_output(self,inputs):
        '''
        function - 每一神经元，先计算输入加权和，然后应用激活函数获得输出
        '''
        self.inputs=inputs
        self.output=self.activation(self.net_input())
        return self.output
    
    def neuron_error(self,predicted_output):
        '''
        function - 每一神经元的误差（平方差）
        '''
        return 0.5*(predicted_output-self.output)**2
    
    def pd_activation(self):
        import sympy
        from sympy import diff
        from sympy.functions import exp
        '''
        function -定义神经元激活函数偏导
        '''
        x_sigmoid=sympy.symbols('x_sigmoid')
        partialD_activation=diff(1/(1+exp(-x_sigmoid)),x_sigmoid)
        return partialD_activation.subs([(x_sigmoid,self.output)]) #由sympy的diff方法获取sigmoid导函数
        #return self.output * (1 - self.output) #推到公式
    
    def pd_E_prediction(self,predicted_output):
        import sympy
        from sympy import diff
        '''
        function - 定义神经元误差偏导
        '''
        true_value,predicted_value=sympy.symbols('true_value predicted_value')
        partialD_E_total_prediction=diff(0.5*(predicted_value-true_value)**2,true_value)
        return partialD_E_total_prediction.subs([(predicted_value,predicted_output),(true_value,self.output)])

    
    def pd_net_output(self,predicted_output):
        '''
        function - 每一神经元误差偏导，由链式法则计算，包括神经元激活函数偏导，和神经元总误差偏导
        '''
        return self.pd_activation()*self.pd_E_prediction(predicted_output)
    
    def pd_weightedSUM(self,index):
        import sympy
        from sympy import diff
        '''
        fucntion - 定义神经元加权和偏导
        '''
        w,x_w=sympy.symbols('w x_w')
        partialD_weightedSUM=diff(w*x_w,w)
        return partialD_weightedSUM.subs([(x_w,self.inputs[index])])
    
class NEURAL_LAYER:
    '''
    class - 神经网络每一层的定义
    '''
    def __init__(self,num_neurons,bias):
        '''
        fucntion - 初始化每层偏置，及神经元
        '''
        self.bias=bias if bias else random.random() #同一层各个神经元共享同一个偏置        
        self.neurons=[NEURON(self.bias) for i in range(num_neurons)]
        
    def layer_info(self):
        '''
        function - 查看每层的神经元信息
        '''
        print('neurons:{}'.format(len(self.neurons)))
        for i in range(len(self.neurons)):
            print('neuron-{}'.format(i))
            print('weights:{}'.format([self.neurons[i].weights[j] for j in range(len(self.neurons[i].weights))]))
            print('bias:{}'.format(self.bias))
            
    def layer_output(self,inputs):
        outputs=[neuron.net_output(inputs) for neuron in self.neurons]
        return outputs
    
class NEURAL_NETWORK:
    '''
    class - 构建多层感知机（神经网络）
    '''
    def __init__(self,learning_rate,num_inputs,num_hidden,num_outputs,hiddenLayer_weights=None,hiddenLayer_bias=None,outputLayer_weights=None,outputLayer_bias=None):
        self.lr=learning_rate
        self.num_inputs=num_inputs
        self.hidden_layer=NEURAL_LAYER(num_hidden,hiddenLayer_bias)
        self.output_layer=NEURAL_LAYER(num_inputs,outputLayer_bias)
        self.weights_init(hiddenLayer_weights,self.hidden_layer,self.num_inputs)
        self.weights_init(outputLayer_weights,self.output_layer,len(self.hidden_layer.neurons))       
        
    def weights_init(self,weights,neu_layer,num_previous):
        '''
        fucntion - 初始化权值
        '''
        weights_num=0
        for i in range(len(neu_layer.neurons)):
            for j in range(num_previous):
                if not weights:
                    neu_layer.neurons[i].weights.append(random.random())
                else:
                    neu_layer.neurons[i].weights.append(weights[weights_num])
                weights_num+=1
                
    def neural_info(self):
        '''
        function - 查看神经网络结构
        '''
        print("_"*50)
        print("inputs number:{}".format(self.num_inputs))
        print("_"*50)
        print("hidden layer:{}".format(self.hidden_layer.layer_info()))
        print("_"*50)
        print("output layer:{}".format(self.output_layer.layer_info()))
        print("_"*50)
        
    def forward_propagation(self,inputs):
        '''
        fucntion - 前向传播
        '''
        hiddenLayer_outputs=self.hidden_layer.layer_output(inputs)
        return self.output_layer.layer_output(hiddenLayer_outputs)
    
    def train(self,training_inputs,training_outputs):
        '''
        function - 训练，迭代更新权值
        '''
        self.forward_propagation(training_inputs)        
       
        #1. 输出层神经元的值
        pd_output_values=[0]*len(self.output_layer.neurons)
        for i in range(len(self.output_layer.neurons)):
            # ∂E/∂zⱼ
            pd_output_values[i]=self.output_layer.neurons[i].pd_net_output(training_outputs[i])
       # print(pd_output_values)
        
        
        #2. 隐含层神经元的值
        hidden_values=[0]*len(self.hidden_layer.neurons)
        for i in range(len(self.hidden_layer.neurons)):
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_errors=0
            for j in range(len(self.output_layer.neurons)):
                d_errors+=pd_output_values[j]*self.output_layer.neurons[j].weights[i]
            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            hidden_values[i]=d_errors*self.hidden_layer.neurons[i].pd_activation()
        #print(hidden_values)
        
        #3. 更新输出层权重系数
        for i in range(len(self.output_layer.neurons)):
            for j in range(len(self.output_layer.neurons[i].weights)):
                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                weights_update=pd_output_values[i]*self.output_layer.neurons[i].pd_weightedSUM(j)
                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[i].weights[j]-=self.lr*weights_update
                
        #4. 更新隐含层的权重系数
        for i in range(len(self.hidden_layer.neurons)):
            for j in range(len(self.hidden_layer.neurons[i].weights)):
                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                weights_update=hidden_values[i]*self.hidden_layer.neurons[i].pd_weightedSUM(j)
                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[i].weights[j]-=self.lr*weights_update
                
    def total_error(self,training_sets):
        total_error=0
        for i in range(len(training_sets)):            
            training_inputs, training_outputs=training_sets[i]
            self.forward_propagation(training_inputs)
            for j in range(len(training_outputs)):
                total_error+=self.output_layer.neurons[j].neuron_error(training_outputs[j])
        return total_error

        
learning_rate=0.5        
nn=NEURAL_NETWORK(learning_rate,2, 2, 2, hiddenLayer_weights=[0.15, 0.2, 0.25, 0.3], hiddenLayer_bias=0.35, outputLayer_weights=[0.4, 0.45, 0.5, 0.55], outputLayer_bias=0.6)    
nn.neural_info()

import numpy as np
from tqdm import tqdm
for i in tqdm(range(10000)):
    nn.train([0.05,0.1],[0.01,0.09])
    if i%1000==0:
        print(i,round(nn.total_error([[[0.05, 0.1], [0.01, 0.09]]]),9))
hidenLayer_weights_=[nn.hidden_layer.neurons[i].weights for i in range(len(nn.hidden_layer.neurons))]
output_layer_weights_=[nn.output_layer.neurons[i].weights for i in range(len(nn.output_layer.neurons))]
print("hidenLayer_weights_={}\noutput_layer_weights_={}".format(hidenLayer_weights_,output_layer_weights_))
```

      0%|          | 9/10000 [00:00<01:55, 86.70it/s]

    __________________________________________________
    inputs number:2
    __________________________________________________
    neurons:2
    neuron-0
    weights:[0.15, 0.2]
    bias:0.35
    neuron-1
    weights:[0.25, 0.3]
    bias:0.35
    hidden layer:None
    __________________________________________________
    neurons:2
    neuron-0
    weights:[0.4, 0.45]
    bias:0.6
    neuron-1
    weights:[0.5, 0.55]
    bias:0.6
    output layer:None
    __________________________________________________
    0 0.493712428
    

     10%|█         | 1018/10000 [00:10<01:38, 90.97it/s]

    1000 2.4963e-05
    

     20%|██        | 2017/10000 [00:21<01:20, 98.74it/s]

    2000 1.878e-06
    

     30%|███       | 3015/10000 [00:31<01:14, 94.07it/s]

    3000 2.28e-07
    

     40%|████      | 4020/10000 [00:42<01:04, 92.57it/s]

    4000 3.2e-08
    

     50%|█████     | 5016/10000 [00:52<00:50, 98.09it/s]

    5000 5e-09
    

     60%|██████    | 6017/10000 [01:03<00:40, 99.24it/s] 

    6000 1e-09
    

     70%|███████   | 7017/10000 [01:13<00:33, 89.81it/s] 

    7000 0.0
    

     80%|████████  | 8013/10000 [01:23<00:20, 96.50it/s] 

    8000 0.0
    

     90%|█████████ | 9017/10000 [01:34<00:09, 99.85it/s] 

    9000 0.0
    

    100%|██████████| 10000/10000 [01:44<00:00, 95.40it/s]

    hidenLayer_weights_=[[0.374907950905178, 0.649815901810356], [0.469098467292651, 0.738196934585302]]
    output_layer_weights_=[[-4.28132748279580, -4.25790301529841], [-2.41120108789009, -2.37807889000506]]
    

    
    

### 1.5 自动求梯度(gradient)
在上述前向和反向传播中，反向传播的梯度计算是应用sympy函数的diff方法求得导函数计算。因为在深度学习中经常要对函数求梯度，因此pytorch提供了`autograd`，可以根据输入和前向传播过程自动构建计算图，并执行反向传播。在定义张量时，如果配置参数`requires_grad=True`，将追踪其上的所有操作，因此可以应用链式法则进行梯度传播计算。完成前向传播计算后，可以调用`.backward()`来完成梯度计算。梯度将累积到.grad属性中。如果不想被追踪，可以执行`.detach()`将其从追踪记录中剥离开来，或者使用`with torch.no_grad()`将不想追踪的操作代码包裹起来（例如在评估模型时）。

注意：grad在反向传播过程中是累加的(accumulated)，一般在反向传播之前把应用`x.grad.data.zero_()`梯度清零。


> 梯度,gradient，一种关于多元导数的概况。一元（单变量）函数的导数是标量值函数，而多元函数的梯度是向量值函数。多元可微函数$f$在点$P$上的梯度，是以$f$在$P$上的偏导数为分量的向量。一元函数的导数表示这个函数图形切线的斜率，如果多元函数在点$P$上的梯度不是零向量，则它的方向是这个函数在$P$上最大增长的方向，它的量是在这个方向上的增长率。


```python
import torch
#01-定义包含梯度追踪的张量
x = torch.tensor([3.0, 2.0, 1.0, 4.0], requires_grad=True) #x.requires_grad_(True)的方式可以将未指定requires_grad = False的张量转换为追踪梯度传播
print("01-定义包含梯度追踪的张量:")
print("x={}\nx.grad_fn={}\nx.requires_grad={}".format(x,x.grad_fn,x.requires_grad))
print("_"*50)

#02-追踪运算的梯度
print("02-追踪运算的梯度:")
y=(x+1)**3+2
z=y.view(2,2)
print("z={}\nz.grad_fn={}".format(z,z.grad_fn))
print("x.is_leaf={},y.is_leaf={},z.is_leaf={}".format(x.is_leaf,y.is_leaf,z.is_leaf))
print("_"*50)

#03-z(张量)关于x的梯度
print("03-z(张量)关于x的梯度:")
v=torch.tensor([[1.0, 0.1], [0.01, 0.001]],dtype=torch.float)
z.backward(v) #因为z不是标量，在调用backward时需要传入一个和z同形状的权值向量（权值）进行加权求和得到一个标量
print("dz/dx={}".format(x.grad)) #x.grad是和x同形的张量
print("_"*50)

#04-中断梯度追踪
print("04-中断梯度追踪:")
x_1=torch.tensor(1.0,requires_grad=True)
y_1=x_1**2
with torch.no_grad():
    y_2=x_1**3
y_3=y_1+y_2
print("x.requires_grad={}\ny_1={};y_1.requires_grad={}\ny_2={};y_2.requires_grad={}\ny_3={};y_3.requires_grad={}\n".format(x.requires_grad,y_1,y_1.requires_grad,y_2,y_2.requires_grad,y_3,y_3.requires_grad))
y_3.backward()
print("dy_3/dx={}".format(x_1.grad)) #y_2梯度并未被追踪，与y_2有关的梯度并不会回传
print("_"*50)

#05-不影响反向传播下修改张量值，tensor.data
print("05-不影响反向传播下修改张量值，tensor.data:")
x_2=torch.ones(1,requires_grad=True)
print("x_2.data={}\nx_2.data.requires_grad={}".format(x_2,x_2.data.requires_grad))
y_4=2*x_2
x_2.data*=100 #只改变了值，不会记录在计算图中（autograd记录），不影响梯度传播
y_4.backward()
print("x_2={}\nx_2.grad={}".format(x_2,x_2.grad))
```

    01-定义包含梯度追踪的张量:
    x=tensor([3., 2., 1., 4.], requires_grad=True)
    x.grad_fn=None
    x.requires_grad=True
    __________________________________________________
    02-追踪运算的梯度:
    z=tensor([[ 66.,  29.],
            [ 10., 127.]], grad_fn=<ViewBackward>)
    z.grad_fn=<ViewBackward object at 0x0000026720AE9340>
    x.is_leaf=True,y.is_leaf=False,z.is_leaf=False
    __________________________________________________
    03-z(张量)关于x的梯度:
    dz/dx=tensor([48.0000,  2.7000,  0.1200,  0.0750])
    __________________________________________________
    04-中断梯度追踪:
    x.requires_grad=True
    y_1=1.0;y_1.requires_grad=True
    y_2=1.0;y_2.requires_grad=False
    y_3=2.0;y_3.requires_grad=True
    
    dy_3/dx=2.0
    __________________________________________________
    05-不影响反向传播下修改张量值，tensor.data:
    x_2.data=tensor([1.], requires_grad=True)
    x_2.data.requires_grad=False
    x_2=tensor([100.], requires_grad=True)
    x_2.grad=tensor([2.])
    

### 1.6 用PyTorch构建多层感知机（多层神经网络）
#### 1.6.1-自定义激活函数、模型、损失函数及梯度下降法
PyTorch自动求梯度的方式，让神经网络模型的构建变得轻松，可以将更多的注意力放在模型的构建上，而不是梯度的计算上。上述实现了典型三层神经网络的逐步计算、以及代码整合，对应输入输出值，权值和偏置、激活函数、损失函数、模型结构以及权值更新（优化算法：梯度下降法）保持不变，应用PyTorch分别重新定义。从迭代计算结果来看，因为偏置值也得以更新，损失函数下降的比较快，迅速的收敛。


```python
import torch
import numpy as np

#A-训练数据
X=torch.tensor([0.05,0.1])
y=torch.tensor([0.01,0.9])

#B-定义模型参数
num_inputs,num_outputs,num_hiddens=2,2,2

W1=torch.tensor([[0.15, 0.2], [0.25, 0.3]]) #hiddenLayer_weights
b1=torch.tensor([0.35,0.35]) #hiddenLayer_bias
W2=torch.tensor([[0.4, 0.45], [0.5, 0.55]]) #outputLayer_weights
b2=torch.tensor([0.6,0.6]) #outputLayer_bias
params=[W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)

#C-定义激活函数
def sigmoid(X):
    return 1/(1+torch.exp(-X))

#D-定义模型
def net(X):
    X=X.view((-1,num_inputs))
    H=sigmoid(torch.matmul(X,W1)+b1)
    return torch.matmul(H,W2)+b2

#F-定义损失函数
def loss(y_hat,y):
    return 0.5*(y_hat-y)**2

#G-优化算法
def sgd(params,lr):
    for param in params:
        param.data-=lr*param.grad

#H-训练模型
def train(net,X,y,loss,num_epochs,params=None,lr=None,optimizer=None):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for epoch in range(num_epochs):
        y_hat=net(X)
        l=loss(y_hat,y).sum()
        #梯度清零
        if params is not None and params[0].grad is not None:
            for param in params:
                param.grad.data.zero_()        
        l.backward()
        sgd(params,lr)
        print('epoch %d, loss %.9f'%(epoch+1,l))
    return net,params
        
num_epochs, lr=5, 0.5
net_,params_=train(net,X,y,loss,num_epochs,params,lr)
print("应用训练好的模型验证预测值：\nnet_(x)={}".format(net_(X)))
print("参数(权值+偏置)更新结果：\nparams_={}".format(params_))
```

    epoch 1, loss 0.677512288
    epoch 2, loss 0.013031594
    epoch 3, loss 0.000361601
    epoch 4, loss 0.000010227
    epoch 5, loss 0.000000290
    应用训练好的模型验证预测值：
    net_(x)=tensor([[0.0101, 0.9000]], grad_fn=<AddBackward0>)
    参数(权值+偏置)更新结果：
    params_=[tensor([[0.1463, 0.1954],
            [0.2427, 0.2907]], requires_grad=True), tensor([0.2770, 0.2573], requires_grad=True), tensor([[0.0102, 0.3532],
            [0.1094, 0.4529]], requires_grad=True), tensor([-0.0585,  0.4366], requires_grad=True)]
    

#### 1.6.2-直接使用PyTorch提供的函数
PyTorch已经内置了多种激活函数、损失函数、优化算法以及各类模型算法，可以直接调用参与计算。因为延续了以上的典型三层神经网络，假设的输入输出值比较简单，因此损失函数仍然采取自定义的平方误差形式。


```python
import torch
from torch import nn

#A-训练数据
X=torch.tensor([0.05,0.1])
y=torch.tensor([0.01,0.9])

#B-定义模型参数:权值和偏置此次随机初始化
num_inputs,num_outputs,num_hiddens=2,2,2
    
#C-定义模型
net=nn.Sequential(
    nn.Linear(num_inputs,num_hiddens),
    nn.Sigmoid()
    )

#D-初始化参数：可以省略，pytorch定义模型时，已经初始化
#for params in net.parameters():
    #nn.init.normal_(params,mean=0,std=0.01)

#E-定义损失函数
def loss(y_hat,y):
    return 0.5*(y_hat-y)**2

#F-优化算法
optimizer=torch.optim.SGD(net.parameters(),lr=0.5)

#G-训练模型
def train(net,X,y,loss,num_epochs,params=None,lr=None,optimizer=None):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for epoch in range(num_epochs):
        y_hat=net(X)
        l=loss(y_hat,y).sum()
        #梯度清零
        if optimizer is not None:
            optimizer.zero_grad()
        elif params is not None and params[0].grad is not None:
            for param in params:
                param.grad.data.zero_()        
        #自动求梯度
        l.backward()
        optimizer.step()
        
        if epoch%1000==0:
            print('epoch %d, loss %.9f'%(epoch+1,l))
    return net,params


num_epochs=10000
net,params=train(net,X,y,loss,num_epochs,params,lr,optimizer)
print("应用训练好的模型验证预测值：\nnet_(x)={}".format(net_(X)))
print("参数(权值+偏置)更新结果：\nparams_={}".format(params_))
```

    epoch 1, loss 0.106545240
    epoch 1001, loss 0.000367269
    epoch 2001, loss 0.000144827
    epoch 3001, loss 0.000080614
    epoch 4001, loss 0.000051799
    epoch 5001, loss 0.000036051
    epoch 6001, loss 0.000026406
    epoch 7001, loss 0.000020044
    epoch 8001, loss 0.000015620
    epoch 9001, loss 0.000012420
    应用训练好的模型验证预测值：
    net_(x)=tensor([[0.0101, 0.9000]], grad_fn=<AddBackward0>)
    参数(权值+偏置)更新结果：
    params_=[tensor([[0.1463, 0.1954],
            [0.2427, 0.2907]], requires_grad=True), tensor([0.2770, 0.2573], requires_grad=True), tensor([[0.0102, 0.3532],
            [0.1094, 0.4529]], requires_grad=True), tensor([-0.0585,  0.4366], requires_grad=True)]



### 1.7 [runx.logx](https://github.com/NVIDIA/runx#introduction--a-simple-example)-深度学习实验管理/Deep Learning Experiment Management+模型构建
runx是NVIDIA开源的一款深度学习实验管理工具。可以帮助深度学习研究者自动执行一些常见的任务。例如参数扫描，输出日志记录，保存训练模型等。该库包括3个模块，runx,logx以及sumx。其中最为常用的是logx模块，可以应用`logx.metric()`保存metrics(字典格式保存的指定评估参数等，例如损失函数值，学习率等)；应用`logx.msg()`保存message（任意指定的信息）；应用`logx.save_model()`保存模型checkpoint，并可以指定一个模型精度评估指标，保存精度最好的模型； 以及初始化时配置`tensorboard=True`，写入tensorboard文件，可以应用tensorboard打开训练时的一些信息图表，例如损失曲线，学习率，输入输出图片(`logx.add_image()`)，卷积核的参数分布等。这些信息能够帮助监督网络的训练过程，为参数优化提供帮助。logx保存文件位于指定的文件夹下，包括'best_checkpoint_ep1000.pth','last_checkpoint_ep1000.pth‘网络模型，logging.log日志，metrics.csv评估参数，以及'events.out.tfevents.1609820463.LAPTOP-GH6EM1TC'文件。

下述代码也涵盖了模型构建的几种方法，包括继承nn.Module类构造模型，使用nn.Sequential类构造模型（又包括.add_module方式和OrderedDict方式），使用nn-ModuleList类构造模型，以及使用nn.ModuleDict类构造模型。其中优先使用nn.Sequential类构造模型方法，如果需要增加模型构造的灵活性，可以使用继承nn.Module类构造模型。

在生成数据集时，是使用了一个二元一次函数根据随机生成的特征值（特征数为2，即输入值）生成对应的类标（即输出值）。但是在神经网络模型构建时，并未使用单纯的一个线性模型，而是构建了线性模型-->激活函数-->线性模型的神经网络结构，用以说明模型构建的方法。如果希望验证一个层线性回归的参数是否与生成数据集类标的方程权值和偏置趋于一致，可以仅保留第一个线性模型，而移除激活函数以及第2个线性模型。

将数据集划分了训练数据集和验证数据集，在训练过程中的每一epoch增加了验证数据集根据已训练的网络预测输出值，并应用损失函数-`nn.MSELoss()`均方误差的累加和用于网络评估。


```python
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
```

    the expression generating labels:w1*x1+-w2*x2+b
    nn.Module构造模型：
    net:MLP(
      (hidden): Linear(in_features=2, out_features=2, bias=True)
      (activation): ReLU()
      (output): Linear(in_features=2, out_features=1, bias=True)
    )
    __________________________________________________
    nn.Sequential构造模型： Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): ReLU()
      (2): Linear(in_features=2, out_features=1, bias=True)
    )
    nn.Sequential()-->.add_module方式： Sequential(
      (hidden): Linear(in_features=2, out_features=2, bias=True)
      (activation): ReLU()
      (output): Linear(in_features=2, out_features=1, bias=True)
    )
    nn.Sequential-->OrderedDict方式： Sequential(
      (hidden): Linear(in_features=2, out_features=2, bias=True)
      (activation): ReLU()
      (output): Linear(in_features=2, out_features=1, bias=True)
    )
    __________________________________________________
    nn.ModuleList构造模型： ModuleList(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): ReLU()
      (2): Linear(in_features=2, out_features=1, bias=True)
    )
    __________________________________________________
    nn.ModuleDict构造模型： ModuleDict(
      (hidden): Linear(in_features=2, out_features=2, bias=True)
      (activation): ReLU()
      (output): Linear(in_features=2, out_features=1, bias=True)
    )
    __________________________________________________
    Parameter containing:
    tensor([[-0.3995, -0.0165],
            [ 0.3065, -0.0651]], requires_grad=True)
    Parameter containing:
    tensor([0.3005, 0.4148], requires_grad=True)
    Parameter containing:
    tensor([[-0.7020, -0.0650]], requires_grad=True)
    Parameter containing:
    tensor([0.4247], requires_grad=True)
    __________________________________________________
    初始化参数后：
    Parameter containing:
    tensor([[-0.0142, -0.0086],
            [-0.0022,  0.0051]], requires_grad=True)
    Parameter containing:
    tensor([0., 0.], requires_grad=True)
    Parameter containing:
    tensor([[-0.7020, -0.0650]], requires_grad=True)
    Parameter containing:
    tensor([0.4247], requires_grad=True)
    __________________________________________________
    optimizer: SGD (
    Parameter Group 0
        dampening: 0
        lr: 0.03
        momentum: 0
        nesterov: False
        weight_decay: 0
    )
    配置不同的学习率-optimizer_: SGD (
    Parameter Group 0
        dampening: 0
        lr: 0.03
        momentum: 0
        nesterov: False
        weight_decay: 0
    
    Parameter Group 1
        dampening: 0
        lr: 0.01
        momentum: 0
        nesterov: False
        weight_decay: 0
    )
    新建优化器调整学习率-optimizer_: SGD (
    Parameter Group 0
        dampening: 0
        lr: 0.003
        momentum: 0
        nesterov: False
        weight_decay: 0
    
    Parameter Group 1
        dampening: 0
        lr: 0.001
        momentum: 0
        nesterov: False
        weight_decay: 0
    )
    __________________________________________________
    epoch 100, loss: 0.000117
    验证数据集-精度-loss：0.07787404209375381
    epoch 200, loss: 0.000105
    验证数据集-精度-loss：0.03207176923751831
    epoch 300, loss: 0.000074
    验证数据集-精度-loss：0.016309896484017372
    epoch 400, loss: 0.000103
    验证数据集-精度-loss：0.009164282120764256
    epoch 500, loss: 0.000116
    验证数据集-精度-loss：0.0055949147790670395
    epoch 600, loss: 0.000108
    验证数据集-精度-loss：0.003492299932986498
    epoch 700, loss: 0.000089
    验证数据集-精度-loss：0.0021637314930558205
    epoch 800, loss: 0.000128
    验证数据集-精度-loss：0.0014685962814837694
    epoch 900, loss: 0.000128
    验证数据集-精度-loss：0.0011613850947469473
    epoch 1000, loss: 0.000111
    验证数据集-精度-loss：0.0012093255063518882
    


```python
print("params_hidden:{}\nbias:{}\nparams_output:{}\nbias:{}".format(net.hidden.weight,net.hidden.bias,net.output.weight,net.output.bias))
```

    params_hidden:Parameter containing:
    tensor([[ 0.9073, -0.6878],
            [-0.5067,  0.4085]], requires_grad=True)
    bias:Parameter containing:
    tensor([2.4939, 0.8909], requires_grad=True)
    params_output:Parameter containing:
    tensor([[ 2.7124, -1.0936]], requires_grad=True)
    bias:Parameter containing:
    tensor([0.1000], requires_grad=True)
    

在cmd命令行中输入`tensorboard --logdir=logs`可以根据提示在`http://localhost:6006/`页面下打开tensorboard，并显示以下损失曲线和学习率的图表。

<a href=""><img src="./imgs/20_04.png" height='auto' width='1200' title="caDesign"></a>  


> 更多Pytorch深度学习的内容推荐其官方的[教程和案例](https://pytorch.org/tutorials/)，以及[动手深度学习（Dive into Deep Learning)](https://tangshusen.me/Dive-into-DL-PyTorch/)。

本章是从机器学习(以[scikit-learn](https://scikit-learn.org/stable/)库为主)各类分析数据规律的算法（聚类、回归和分类），到深度学习(以[pytorch](https://PyTorch.org/)库为主)人工神经网络(多层感知机)的解读，从典型三层神经网络的逐步构建，代码整合中可以深度的理解前向传播和反向传播，以及梯度计算是如何实现权值（参数）更新，并应用梯度下降法（一种优化算法）实现迭代，降低损失函数（代价函数，例如平方误差），最终确定权值的过程。进一步应用同一神经网络结构，用PyTorch库实现，包括自定义函数实现，和直接使用PyTorch的内置算法实现。通过同一神经网络结构的逐步构建、代码整合和PyTorch实现，能够清晰的理解深度学习的核心思想，为自行自由的构建神经网络结构打下很好的基础。

### 1.5 要点
#### 1.5.1 数据处理技术

* Pytorch库深度学习库-张量与自动求梯度

* 链式法则

* 激活函数

* 前向和反向传播

* PyTorch构建神经网络模型

* runx.logx-深度学习实验管理

#### 1.5.2 新建立的函数

* function - sigmoid函数, `sigmoid(x)`

* function - 定义总误差偏导, `partialD_E_total_prediction(true_value_,predicted_value_)`

* function -定义激活函数偏导, `partialD_activation(x)`

* fucntion - 定义加权和偏导, `partialD_weightedSUM(w_)`

* function - 三层神经网络，各层2个神经元的示例，前向传播函数, `forward_func(input_list,weight_list,bias_list,output_list)`

* class - 每一神经元的输入加权和、激活函数，以及输出计算, `NEURON`

* class - 神经网络每一层的定义, `NEURAL_LAYER`

* class - 构建多层感知机（神经网络）, ` NEURAL_NETWORK`

* PyTorch自定义激活函数、模型、损失函数及梯度下降法, `sigmoid(X), net(X), loss(y_hat,y)`

#### 1.5.3 所调用的库


```python
import torch
import torch.nn as nn

import sympy
from sympy import diff,pprint
from sympy.functions import exp

import matplotlib.pyplot as plt
import numpy as np
import math
```

#### 1.5.4 参考文献
1. Aston Zhang,Zack C. Lipton,Mu Li,etc.Dive into Deep Learning[M].；中文版-阿斯顿.张,李沐,扎卡里.C. 立顿,亚历山大.J. 斯莫拉.动手深度学习[M].人民邮电出版社,北京,2019-06-01
