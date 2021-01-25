> Created on Fri Jan 22 11\45\33 2021 @author: Richie Bao-caDesign设计(cadesign.cn)

## 1. 开放神经网络交换-ONNX(Open Neural Network,Exchage)，netron，Cityscapes数据集，DUC(Dense Upsampling Convolution)图像分割与城市空间要素组成，时空量度，绿视率

### 1.1 [开放神经网络交换-ONNX(Open Neural Network,Exchage)](https://onnx.ai/supported-tools.html)
[开放神经网络交换-ONNX](https://onnx.ai/supported-tools.html)是针对深度学习所设计的开放式文件格式，用于存储训练好的模型。ONNX相信人工智能社区需要更强的互操作性，不会被局限于一种人工智能框架，使得不同类型的人工智能框架（例如PyTorch，MXNet)可以采用相同格式存储模型数据并交互，共享模型。我们推荐使用PyTorch的人工智能框架（深度学习），并利用[torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html)（预训练）模型库，实现城市空间下行人的对象检测来估算人流量变化；应用对象分割（Instance Segmentation），统计城市空间对象内容，建立关联网络结构分析对象之间的关系。因此预训练好的模型可以帮助其它领域的研究者迅速应用已有的研究模型，避免重新构建模型网络，以及不菲的训练时间成本（即使有多GPU加速，有些海量数据集或高度复杂的网络，训练时间也要多达数周）。非计算机科学领域的研究者，往往也不具有理想的硬件条件，因此预训练好的模型共享，显得尤为重要，这也是将人工智能从研究带到现实的有效途径。


[ONNX Model Zoo](https://github.com/onnx/models)汇集了已有的大量模型，包括的种类有，

1. Vision 视觉类：Image Classification图像分类，Object Detection & Image Segmentation对象检测和图像分割，Body, Face & Gesture Analysis人体，人脸和姿势分析，Image Manipulation图像处理。

2. Language语言类：Machine Comprehension机器理解，Machine Translation机器翻译，Language Modelling语言模型。

3. 及其它类：Visual Question Answering & Dialog视觉问答&对话，Speech & Audio Processing语音和音频处理，和其它有趣的模型。

### 1.2 [netron](https://github.com/lutzroeder/Netron) 网络可视化工具
tensorboard可以在深度学习模型建立过程中帮助分析网络结构，显示训练图像和预测精度，以及损失曲线等。但是如果想更加方便的查看模型网络结构，[netron](https://github.com/lutzroeder/Netron)可以直接读取开放神经网络交换格式(.onnx)。`torch.onnx.export`方法可以将PyTorch模型导出为.onnx交换格式，默认`export_params=True`，保存预训练模型的参数。直接用netron工具打开，可以查看到下述模型的网络结构。


<a href=""><img src="./imgs/24_01.png" height='auto' width='800' title="caDesign"></a>


```python
import torch
import torchvision

dummy_input = torch.randn(10, 3, 224, 224, device='cuda') #输入张量结构（batchsize, nChannels, Height, Width）
model = torchvision.models.alexnet(pretrained=True).cuda()
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model,# model being run 待输出的模型
                  dummy_input, #model input (or a tuple for multiple inputs) 因为torch.onnx.export需要运行所输入的模型（待输出），所以需要提供一个输入张量(x)。张量只需类型和大小正确，其中的值可以为随机数。
                  "alexnet.onnx", #where to save the model (can be a file or file-like object) 模型保存路径
                  export_params=True, # store the trained parameter weights inside the model file 将训练过的参数权重存储在模型文件中，默认为True
                  verbose=True, #causes the exporter to print out a human-readable representation of the network 打印输出一个可读的网络模型陈述
                  input_names=input_names, #the model's input names 模型的输入名称，用于设置模型图中值的显示名称。这些设置不会改变图的语义，知识为了可读性
                  output_names=output_names) #the model's output names 模型的输出名称，解释同输入
```

    Downloading: "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth" to C:\Users\richi/.cache\torch\hub\checkpoints\alexnet-owt-4df8aa71.pth
    22.9%IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    100.0%
    

    graph(%actual_input_1 : Float(10:150528, 3:50176, 224:224, 224:1, requires_grad=0, device=cuda:0),
          %learned_0 : Float(64:363, 3:121, 11:11, 11:1, requires_grad=1, device=cuda:0),
          %learned_1 : Float(64:1, requires_grad=1, device=cuda:0),
          %learned_2 : Float(192:1600, 64:25, 5:5, 5:1, requires_grad=1, device=cuda:0),
          %learned_3 : Float(192:1, requires_grad=1, device=cuda:0),
          %learned_4 : Float(384:1728, 192:9, 3:3, 3:1, requires_grad=1, device=cuda:0),
          %learned_5 : Float(384:1, requires_grad=1, device=cuda:0),
          %learned_6 : Float(256:3456, 384:9, 3:3, 3:1, requires_grad=1, device=cuda:0),
          %learned_7 : Float(256:1, requires_grad=1, device=cuda:0),
          %learned_8 : Float(256:2304, 256:9, 3:3, 3:1, requires_grad=1, device=cuda:0),
          %learned_9 : Float(256:1, requires_grad=1, device=cuda:0),
          %learned_10 : Float(4096:9216, 9216:1, requires_grad=1, device=cuda:0),
          %learned_11 : Float(4096:1, requires_grad=1, device=cuda:0),
          %learned_12 : Float(4096:4096, 4096:1, requires_grad=1, device=cuda:0),
          %learned_13 : Float(4096:1, requires_grad=1, device=cuda:0),
          %learned_14 : Float(1000:4096, 4096:1, requires_grad=1, device=cuda:0),
          %learned_15 : Float(1000:1, requires_grad=1, device=cuda:0)):
      %17 : Float(10:193600, 64:3025, 55:55, 55:1, requires_grad=1, device=cuda:0) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[11, 11], pads=[2, 2, 2, 2], strides=[4, 4]](%actual_input_1, %learned_0, %learned_1) # C:\Users\richi\anaconda3\envs\mxnet\lib\site-packages\torch\nn\modules\conv.py:420:0
      %18 : Float(10:193600, 64:3025, 55:55, 55:1, requires_grad=1, device=cuda:0) = onnx::Relu(%17) # C:\Users\richi\anaconda3\envs\mxnet\lib\site-packages\torch\nn\functional.py:1134:0
      %19 : Float(10:46656, 64:729, 27:27, 27:1, requires_grad=1, device=cuda:0) = onnx::MaxPool[kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[2, 2]](%18) # C:\Users\richi\anaconda3\envs\mxnet\lib\site-packages\torch\nn\functional.py:586:0
      %20 : Float(10:139968, 192:729, 27:27, 27:1, requires_grad=1, device=cuda:0) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[5, 5], pads=[2, 2, 2, 2], strides=[1, 1]](%19, %learned_2, %learned_3) # C:\Users\richi\anaconda3\envs\mxnet\lib\site-packages\torch\nn\modules\conv.py:420:0
      %21 : Float(10:139968, 192:729, 27:27, 27:1, requires_grad=1, device=cuda:0) = onnx::Relu(%20) # C:\Users\richi\anaconda3\envs\mxnet\lib\site-packages\torch\nn\functional.py:1134:0
      %22 : Float(10:32448, 192:169, 13:13, 13:1, requires_grad=1, device=cuda:0) = onnx::MaxPool[kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[2, 2]](%21) # C:\Users\richi\anaconda3\envs\mxnet\lib\site-packages\torch\nn\functional.py:586:0
      %23 : Float(10:64896, 384:169, 13:13, 13:1, requires_grad=1, device=cuda:0) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%22, %learned_4, %learned_5) # C:\Users\richi\anaconda3\envs\mxnet\lib\site-packages\torch\nn\modules\conv.py:420:0
      %24 : Float(10:64896, 384:169, 13:13, 13:1, requires_grad=1, device=cuda:0) = onnx::Relu(%23) # C:\Users\richi\anaconda3\envs\mxnet\lib\site-packages\torch\nn\functional.py:1134:0
      %25 : Float(10:43264, 256:169, 13:13, 13:1, requires_grad=1, device=cuda:0) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%24, %learned_6, %learned_7) # C:\Users\richi\anaconda3\envs\mxnet\lib\site-packages\torch\nn\modules\conv.py:420:0
      %26 : Float(10:43264, 256:169, 13:13, 13:1, requires_grad=1, device=cuda:0) = onnx::Relu(%25) # C:\Users\richi\anaconda3\envs\mxnet\lib\site-packages\torch\nn\functional.py:1134:0
      %27 : Float(10:43264, 256:169, 13:13, 13:1, requires_grad=1, device=cuda:0) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%26, %learned_8, %learned_9) # C:\Users\richi\anaconda3\envs\mxnet\lib\site-packages\torch\nn\modules\conv.py:420:0
      %28 : Float(10:43264, 256:169, 13:13, 13:1, requires_grad=1, device=cuda:0) = onnx::Relu(%27) # C:\Users\richi\anaconda3\envs\mxnet\lib\site-packages\torch\nn\functional.py:1134:0
      %29 : Float(10:9216, 256:36, 6:6, 6:1, requires_grad=1, device=cuda:0) = onnx::MaxPool[kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[2, 2]](%28) # C:\Users\richi\anaconda3\envs\mxnet\lib\site-packages\torch\nn\functional.py:586:0
      %30 : Float(10:9216, 256:36, 6:6, 6:1, requires_grad=1, device=cuda:0) = onnx::AveragePool[kernel_shape=[1, 1], strides=[1, 1]](%29) # C:\Users\richi\anaconda3\envs\mxnet\lib\site-packages\torch\nn\functional.py:936:0
      %31 : Float(10:9216, 9216:1, requires_grad=1, device=cuda:0) = onnx::Flatten[axis=1](%30) # C:\Users\richi\anaconda3\envs\mxnet\lib\site-packages\torch\nn\functional.py:983:0
      %32 : Float(10:4096, 4096:1, requires_grad=1, device=cuda:0) = onnx::Gemm[alpha=1., beta=1., transB=1](%31, %learned_10, %learned_11) # C:\Users\richi\anaconda3\envs\mxnet\lib\site-packages\torch\nn\functional.py:1690:0
      %33 : Float(10:4096, 4096:1, requires_grad=1, device=cuda:0) = onnx::Relu(%32) # C:\Users\richi\anaconda3\envs\mxnet\lib\site-packages\torch\nn\functional.py:983:0
      %34 : Float(10:4096, 4096:1, requires_grad=1, device=cuda:0) = onnx::Gemm[alpha=1., beta=1., transB=1](%33, %learned_12, %learned_13) # C:\Users\richi\anaconda3\envs\mxnet\lib\site-packages\torch\nn\functional.py:1690:0
      %35 : Float(10:4096, 4096:1, requires_grad=1, device=cuda:0) = onnx::Relu(%34) # C:\Users\richi\anaconda3\envs\mxnet\lib\site-packages\torch\nn\functional.py:1134:0
      %output1 : Float(10:1000, 1000:1, requires_grad=1, device=cuda:0) = onnx::Gemm[alpha=1., beta=1., transB=1](%35, %learned_14, %learned_15) # C:\Users\richi\anaconda3\envs\mxnet\lib\site-packages\torch\nn\functional.py:1690:0
      return (%output1)
    
    


```python
import onnx
model = onnx.load("alexnet.onnx") # Load the ONNX model 加载.onnx模型
onnx.checker.check_model(model) ## Check that the IR is well formed
onnx.helper.printable_graph(model.graph) # Print a human readable representation of the graph 打具有印可读性的图结构陈述
```




    'graph torch-jit-export (\n  %actual_input_1[FLOAT, 10x3x224x224]\n) initializers (\n  %learned_0[FLOAT, 64x3x11x11]\n  %learned_1[FLOAT, 64]\n  %learned_10[FLOAT, 4096x9216]\n  %learned_11[FLOAT, 4096]\n  %learned_12[FLOAT, 4096x4096]\n  %learned_13[FLOAT, 4096]\n  %learned_14[FLOAT, 1000x4096]\n  %learned_15[FLOAT, 1000]\n  %learned_2[FLOAT, 192x64x5x5]\n  %learned_3[FLOAT, 192]\n  %learned_4[FLOAT, 384x192x3x3]\n  %learned_5[FLOAT, 384]\n  %learned_6[FLOAT, 256x384x3x3]\n  %learned_7[FLOAT, 256]\n  %learned_8[FLOAT, 256x256x3x3]\n  %learned_9[FLOAT, 256]\n) {\n  %17 = Conv[dilations = [1, 1], group = 1, kernel_shape = [11, 11], pads = [2, 2, 2, 2], strides = [4, 4]](%actual_input_1, %learned_0, %learned_1)\n  %18 = Relu(%17)\n  %19 = MaxPool[kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [2, 2]](%18)\n  %20 = Conv[dilations = [1, 1], group = 1, kernel_shape = [5, 5], pads = [2, 2, 2, 2], strides = [1, 1]](%19, %learned_2, %learned_3)\n  %21 = Relu(%20)\n  %22 = MaxPool[kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [2, 2]](%21)\n  %23 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%22, %learned_4, %learned_5)\n  %24 = Relu(%23)\n  %25 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%24, %learned_6, %learned_7)\n  %26 = Relu(%25)\n  %27 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%26, %learned_8, %learned_9)\n  %28 = Relu(%27)\n  %29 = MaxPool[kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [2, 2]](%28)\n  %30 = AveragePool[kernel_shape = [1, 1], strides = [1, 1]](%29)\n  %31 = Flatten[axis = 1](%30)\n  %32 = Gemm[alpha = 1, beta = 1, transB = 1](%31, %learned_10, %learned_11)\n  %33 = Relu(%32)\n  %34 = Gemm[alpha = 1, beta = 1, transB = 1](%33, %learned_12, %learned_13)\n  %35 = Relu(%34)\n  %output1 = Gemm[alpha = 1, beta = 1, transB = 1](%35, %learned_14, %learned_15)\n  return %output1\n}'



### 1.3 [Cityscapes数据集](https://www.cityscapes-dataset.com/)
Cityscapes数据集集中于城市街道场景的语义解释（semantic understanding），这非常适用于对城市空间内容的分析。这个数据集的主要特点有（引自官网说明）：

**Polygonal annotations 多边形标注**

* Dense semantic segmentation 密集的语义分割
* Instance segmentation for vehicle and people 车辆和行人的实例（对象）分割

**Complexity 复杂性**

* 30 classes 30个分类
* Class Definitions for a list of all classes  分类定义如下：


| Group        | Classes                                                                           |
|--------------|-----------------------------------------------------------------------------------|
| flat         | road · sidewalk · parking+ · rail track+                                          |
| human        | person* · rider*                                                                  |
| vehicle      | car* · truck* · bus* · on rails* · motorcycle* · bicycle* · caravan*+ · trailer*+ |
| construction | building · wall · fence · guard rail+ · bridge+ · tunnel+                         |
| object       | pole · pole group+ · traffic sign · traffic light                                 |
| nature       | vegetation · terrain                                                              |
| sky          | sky                                                                               |
| void         | ground+ · dynamic+ · static+   

\* 包含单个实例对象注释。但是，如果多个对象（实例）之间界限不清楚，整个组(crowd/group)，包含多个对象被标记在一起，并标注为组，例如车辆组。

\+ 该标签不包括在任何评估中，并视为无效。（这是关于模型评估的说明）


**Diversity 多样性**

* 50 cities 包含有50座城市
* Several months (spring, summer, fall) 春夏秋等多季节变化
* Daytime 白天
* Good/medium weather conditions 好/中天气状况
* Manually selected frames 手动选择的帧
* Large number of dynamic objects 大量的动态对象
* Varying scene layout 多样的场景布局
* Varying background 多样的背景

**Volume 量**

* 5,000 annotated images with fine annotations 5,000张精标注
* 20,000 annotated images with coarse annotations 20,000张粗标注

**metadata 元数据**

* Preceding and trailing video frames. Each annotated image is the 20th image from a 30 frame video snippets (1.8s) 前后视频帧。每一幅注释图像都是30帧视频片段(1.8s)中的第20幅图像。
* Corresponding right stereo views 对应右侧立体试图
* GPS coordinates GPS坐标
* Ego-motion data from vehicle odometry 汽车里程测量的运动数据
* Outside temperature from vehicle sensor 来自车辆传感器的外部温度

**Extensions by other researchers 其他研究者的扩展**

* Bounding box annotations of people 行人锚框标注
* Images augmented with fog and rain 雨雾图像增广

**Benchmark suite and evaluation server 基准测试套件和评估服务器**

* Pixel-level semantic labeling 像素级语义标签
* Instance-level semantic labeling 实例（对象）级语义标签
* Panoptic semantic labeling 展示全景的语义标签

图像分割训练的数据集在[Cityscapes下载页](https://www.cityscapes-dataset.com/login/) 注册后获取。主要包括两个文件，一个是11GB大小的`leftImg8bit_trainvaltest.zip`，为训练/验证/测试的图像；二是241ＭＢ大小的`gtFine_trainvaltest.zip`，为训练/验证图像数据集对应的精细标注。

### 1.4 [Hierarchical Multi-Scale Attention for Semantic Segmentation](https://github.com/NVIDIA/semantic-segmentation)解析
下述代码解析 [Hierarchical Multi-Scale Attention for Semantic Segmentation](https://github.com/NVIDIA/semantic-segmentation)前半部分，主要目的除了读取Cityscapes数据集，理解该数据集的文件结构和数据结构外，就是能够从该作者的代码中学习到一些比较重要的编程思维，以及相关使用库。但是并不用该作者的模型实现图像分割，而会直接用ONNX中的[DUC](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/duc)预训练模型。

**知识点-01：**`super().__init__()`——继承父类的`init`方法

通过下述实例理解继承父类的方法，对于子类`Child_robin`虽然继承了父类，可以调用父类的方法（函数），但是因为子类自身的__init__初始化，覆盖了父类的属性，因此无法调用父类属性。对于子类` Child_sparrow`，增加了`super().__init__()`方法，从而可以调用父类属性。


```python
class Parent:
    def __init__(self,name="bird"):
        self.name=name
        
class Child_robin(Parent):
    def __init__(self,species="robin"):
        self.species=species
        
class Child_sparrow(Parent):
    def __init__(self,species="sparrow"):
        self.species=species
        super(Child_sparrow,self).__init__()
        
p=Parent()
print("获取Parent类属性：name=%s"%p.name)
print("_"*50)
c_r=Child_robin()
print("获取子类Child_robin的属性：species=%s"%c_r.species)
try:
    print("获取父类Parent属性：name=%s"%c_r.name)
except AttributeError as error:
    print(error)
print("_"*50)
c_s=Child_sparrow()
print("获取子类Child_sparrow的属性：species=%s"%c_s.species)
print("获取父类Parent属性：name=%s"%c_s.name)
```

    获取Parent类属性：name=bird
    __________________________________________________
    获取子类Child_robin的属性：species=robin
    'Child_robin' object has no attribute 'name'
    __________________________________________________
    获取子类Child_sparrow的属性：species=sparrow
    获取父类Parent属性：name=bird
    

**知识点-02：**`__getattr__`,`__setattr__`,`__delattr__`

类Class_A的实例(instance)C_A，通过`C_A.attri_a`访问实例属性attri_a（对象变量），并返回属性对应值；通过实例的`C_A.__dict__`可以查看所有实例的属性（即实例的属性存储在__dict__中）。如果预提取实例中不存在的属性，则会调用`__getattri__`。如果类的变量（属性）定义在初始化函数外部，例如attri_c(类变量)，则实例的`C_A.__dict__`并不包含该属性，但是在类自身的`Class_A.__dict__`对象中包含该属性。

在实例初始化、重新赋值，以及增加新的属性时均会自动调用`__setattr__`方法，并用`self.__dict__[name]=value`语句，把属性键值对保存在__dict__对象中。其中` __setattr__(self,name,value)`的参数'name'和'value'为固定参数，代表属性的键值对。

如果要删除实例的属性键值对，则可以执行`del C_A.attri_d`，调用`__delattr__`，用`del self.__dict__[name]`方法删除属性键值对。


```python
class Class_A:
    attri_c="attri_C"
    def __init__(self,attri_a,attri_b):
        self.attri_a=attri_a
        self.attri_b=attri_b
        
    def __getattr__(self,attri):
        return('invoke __getattr__',attri)
    
    def __setattr__(self,name,value):
        print("invoke __setattr__",)
        self.__dict__[name]=value
        
    def __delattr__(self,name):
        print("invoke __delattr__",)
        print("deleting `{}`".format(str(name)))
        try:
            del self.__dict__[name]
            print ("`{}` deleted".format(str(name)))
        except KeyError as k:
            return None
                       
    
C_A=Class_A("attri_A","attri_B")
print("实例C_A包含的属性及其值：",C_A.__dict__)
print("实例C_A已有属性attri_a，则直接返回该属性对应值，不会调用__getattr__，attri_a=",C_A.attri_a)
print("实例C_A没有属性attri_none,则调用__getattr__：",C_A.attri_none)
print(Class_A.__dict__)
print("用类的实例C_A提取属性attri_c=%s"%C_A.attri_c,";" "用类自身Class_A直接提取属性attri_c=%s"%Class_A.attri_c)
print("_"*50)
C_A.attri_b="attri_B_assignment"
print("对已有属性重新赋值：",C_A.__dict__)
C_A.attri_d="attri_D"
print("增加新的属性，并赋值：",C_A.__dict__)
```

    invoke __setattr__
    invoke __setattr__
    实例C_A包含的属性及其值： {'attri_a': 'attri_A', 'attri_b': 'attri_B'}
    实例C_A已有属性attri_a，则直接返回该属性对应值，不会调用__getattr__，attri_a= attri_A
    实例C_A没有属性attri_none,则调用__getattr__： ('invoke __getattr__', 'attri_none')
    {'__module__': '__main__', 'attri_c': 'attri_C', '__init__': <function Class_A.__init__ at 0x0000015A60FF90D0>, '__getattr__': <function Class_A.__getattr__ at 0x0000015A60FF9158>, '__setattr__': <function Class_A.__setattr__ at 0x0000015A60FF91E0>, '__delattr__': <function Class_A.__delattr__ at 0x0000015A60FF9268>, '__dict__': <attribute '__dict__' of 'Class_A' objects>, '__weakref__': <attribute '__weakref__' of 'Class_A' objects>, '__doc__': None}
    用类的实例C_A提取属性attri_c=attri_C ;用类自身Class_A直接提取属性attri_c=attri_C
    __________________________________________________
    invoke __setattr__
    对已有属性重新赋值： {'attri_a': 'attri_A', 'attri_b': 'attri_B_assignment'}
    invoke __setattr__
    增加新的属性，并赋值： {'attri_a': 'attri_A', 'attri_b': 'attri_B_assignment', 'attri_d': 'attri_D'}
    


```python
print("_"*50)
del C_A.attri_d
print("删除属性attri_d：",C_A.__dict__)
```

    __________________________________________________
    invoke __delattr__
    deleting `attri_d`
    `attri_d` deleted
    删除属性attri_d： {'attri_a': 'attri_A', 'attri_b': 'attri_B_assignment'}
    

**知识点-03：**mutable(可变)与immutable(不可变)

python的数据类型分为mutable(可变)与immutable(不可变)，mutable就是创建后可以修改，而immutable是创建后不可修改。

对于mutable，下述代码定义了变量a，并将变量b指向了变量a，因此a和b指向同一对象；但是当变量b执行运算后，则变量b指向新的对象（地址）。同样定义列表lst_a，并将列表lst_b指向列表lst_a，则lst_a和lst_b指向同一对象。即使二者分别追加新的值，仍然指向同一对象。但是，重新定义变量lst_b为新的列表，则lst_b指向新的对象。


```python
a=0
b=a
print("a,b 是否指向同一个对象：id_a={};id_b={}".format(id(a),id(b)),id(a)==id(b))

b+=1
print("b执行运算后，a,b 是否指向同一个对象：",id(a)==id(b))
```

    a,b 是否指向同一个对象：id_a=1713352864;id_b=1713352864 True
    b执行运算后，a,b 是否指向同一个对象： False
    


```python
lst_a=[0]
lst_b=lst_a
print("列表lst_a和lst_b是否指向同一个对象：",id(lst_a)==id(lst_b))
lst_b.append(99)
print("lit_b追加值后,列表lst_a和lst_b是否指向同一个对象：",id(lst_a)==id(lst_b))
lst_a.append(79)
print("lit_a追加值后,列表lst_a和lst_b是否指向同一个对象：",id(lst_a)==id(lst_b))
lst_b=[0]
print("lit_b定义新的列表,列表lst_a和lst_b是否指向同一个对象：",id(lst_a)==id(lst_b))
```

    列表lst_a和lst_b是否指向同一个对象： True
    lit_b追加值后,列表lst_a和lst_b是否指向同一个对象： True
    lit_a追加值后,列表lst_a和lst_b是否指向同一个对象： True
    lit_b定义新的列表,列表lst_a和lst_b是否指向同一个对象： False
    

对于immutable，因为自定义的python类型一般都是mutable，如果实现immutable数据类型，通常需要重写对象(object)的`__setattr_`_和`__delattr__`方法。例如下述重新定义了`__setattr__`，并不会将待增加或修改的属性写入`__dict__`中，而是直接引起TypeError异常。为保证不能删除类实例对象，令`__delattr__ = __setattr__`。因此待类immutable实例化为cls对象，修改删除和增加属性值都会引发异常。


```python
class immutable:
    def __setattr__(self, *args): 
        print("invoke __setattr__")
        raise TypeError("cannot modify the value of immutable instance")
    __delattr__ = __setattr__
    def __init__(self,name,value):
        super(immutable,self).__setattr__(name,value)  
cls=immutable("attri_e","attri_E")
print("实例初始化属性值，并读取 attri_e=%s"%cls.attri_e)
cls.attri_e="attri_new"
```

    实例初始化属性值，并读取 attri_e=attri_E
    invoke __setattr__
    


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-12-d05cf92e1c5b> in <module>
          8 cls=immutable("attri_e","attri_E")
          9 print("实例初始化属性值，并读取 attri_e=%s"%cls.attri_e)
    ---> 10 cls.attri_e="attri_new"
    

    <ipython-input-12-d05cf92e1c5b> in __setattr__(self, *args)
          2     def __setattr__(self, *args):
          3         print("invoke __setattr__")
    ----> 4         raise TypeError("cannot modify the value of immutable instance")
          5     __delattr__ = __setattr__
          6     def __init__(self,name,value):
    

    TypeError: cannot modify the value of immutable instance



```python
del cls.attri_e
```

    invoke __setattr__
    


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-13-7e95f192cfc0> in <module>
    ----> 1 del cls.attri_e
    

    <ipython-input-12-d05cf92e1c5b> in __setattr__(self, *args)
          2     def __setattr__(self, *args):
          3         print("invoke __setattr__")
    ----> 4         raise TypeError("cannot modify the value of immutable instance")
          5     __delattr__ = __setattr__
          6     def __init__(self,name,value):
    

    TypeError: cannot modify the value of immutable instance



```python
cls.attri_f="attri_F"
```

    invoke __setattr__
    


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-14-df574e27048a> in <module>
    ----> 1 cls.attri_f="attri_F"
    

    <ipython-input-12-d05cf92e1c5b> in __setattr__(self, *args)
          2     def __setattr__(self, *args):
          3         print("invoke __setattr__")
    ----> 4         raise TypeError("cannot modify the value of immutable instance")
          5     __delattr__ = __setattr__
          6     def __init__(self,name,value):
    

    TypeError: cannot modify the value of immutable instance


**知识点-04：** `_variable`,`__variable`,及`__variable__`

python中成员函数和变量都是公开的public，在python中没有public和private方法修饰成员函数或变量。虽然没有支持私有化priviate，但是可以应用下划线的方法限制成员函数和成员变量的访问权限（尽力避免定义以下划线开头的变量）。`_variable`单下划线开始的成员变量叫做包含变量，只有类的实例和子类的实例能访问这些变量，并需要通过类的接口访问，不能用`from module imort *`的方法导入。`__variable`双下划线开始的成员变量为私有成员，只有类对象自己能访问，子类对象不能访问。`__variable__`前后双下划线，为python特殊方法专用的标识，例如__init__()类的构造函数。


```python
class private:
    def __init__(self):
        self.attri='attri public'
        self._attri='attri_singleUnderscore'
        self.__attri='attri__doubleUnderscore'
    
    def func(self):
        return self.attri+' func'
    def _func(self):
        return self._attri+' _func'
    def __func(self):
        return self.__attri+' __func'
    def invoke__func(self):
        return self.__func()
    
class private_Child(private):
    def __init__(self):
        self.attri_Child='attri child'
        super(private_Child,self).__init__()
        
p=private()
print("类属性——公有成员：attri=%s"%p.attri)
print("类属性——包含变量(单下划线):_attri=%s"%p._attri)
try:
    print("类属性——私有变量（双下划线）:__attri=%s"%p.__attri)
except AttributeError as error:
    print(error)
print("_"*50)    
print("类方法——公有成员：func=",p.func())
print("类方法——单下划线：_func=",p._func())
try:
    print("类方法——双下划线：__func=",p.__func())
except:
    print("没有类方法：__func")
print("_"*50)
p_child=private_Child()
print("子类调用父类单下划线方法：",p_child._func())
try:
    print("子类调用父类双下划线方法：",p_child.__func())
except:
    print("子类没有父类方法：__func()")
```

    类属性——公有成员：attri=attri public
    类属性——包含变量(单下划线):_attri=attri_singleUnderscore
    'private' object has no attribute '__attri'
    __________________________________________________
    类方法——公有成员：func= attri public func
    类方法——单下划线：_func= attri_singleUnderscore _func
    没有类方法：__func
    __________________________________________________
    子类调用父类单下划线方法： attri_singleUnderscore _func
    子类没有父类方法：__func()
    

---

#### 1.4.1  参数管理
应用(App)涉及到的参数很多，为了便于管理，这里使用了[argparse](https://docs.python.org/3/library/argparse.html)命令行参数解析包，可以将参数和代码分离开来，方便读取命令行参数，尤其适合于参数的频繁修改。同时，在程序执行过程中，为了避免参数变化导致难以调试或难以理解代码，在参数配置完之后，需要将mutable参数转变为immutable参数，迁移代码类`class AttrDict(dict)`实现。AttrDict类可以定义类的属性，并通过该类的`immutable`方法，实现类属性的批量类型转换（mutable到immutable，或反之）。

下述仅是提取了*Hierarchical Multi-Scale Attention for Semantic Segmentation(HMA4SS)*的部分代码，用于说明批量参数管理的方法。代码集中于文件'semantic-segmentation-main/train.py'，'semantic-segmentation-main/config.py'，'semantic-segmentation-main/utils/attr_dict.py.py'三个文件中。


```python
#semantic-segmentation-main\train.py
import argparse #
# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')

#...
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--global_rank', default=0, type=int,help='parameter used by apex library')
parser.add_argument('--test_mode', action='store_true', default=False,help=('Minimum testing to verify nothing failed, ''Runs code for 1 epoch of train and val'))
parser.add_argument('--init_decoder', default=False, action='store_true',help='initialize decoder with kaiming normal')
parser.add_argument('--syncbn', action='store_true', default=False,help='Use Synchronized BN')
parser.add_argument('--extra_scales', type=str, default='0.5,2.0')
parser.add_argument('--set_cityscapes_root', type=str, default=None,help='override cityscapes default root dir')
parser.add_argument('--dataset', type=str, default='cityscapes',help='cityscapes, mapillary, camvid, kitti')
parser.add_argument('--result_dir', type=str, default='./logs_sementic',help='where to write log output')
parser.add_argument('--cv', type=int, default=0,help=('Cross-validation split id to use. Default # of splits set' ' to 3 in config'))

parser.add_argument('--crop_size', type=str, default='448',help='dynamically scale training images down to this size') #896
parser.add_argument('--scale_min', type=float, default=0.5, help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,help='dynamically scale training images up to this size')
parser.add_argument('--full_crop_training', action='store_true', default=False,help='Full Crop Training')
parser.add_argument('--pre_size', type=int, default=None,help=('resize long edge of images to this before'' augmentation'))
parser.add_argument('--color_aug', type=float,default=0.25, help='level of color augmentation')
parser.add_argument('--jointwtborder', action='store_true', default=False,help='Enable boundary label relaxation')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,help='Dump Augmentated Images for sanity check')

parser.add_argument('--rmi_loss', action='store_true', default=False,help='use RMI loss')
parser.add_argument('--img_wt_loss', action='store_true', default=False,help='per-image class-weighted loss')
parser.add_argument('--arch', type=str, default='deepv3.DeepV3Plus',help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50)and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--trunk', type=str, default='resnet101',help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--apex', action='store_true', default=False,help='Use Nvidia Apex Distributed Data Parallel')
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--lr_schedule', type=str, default='poly',help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,help='polynomial LR exponent')

#...

args = parser.parse_args([]) #JupyterLab需要在args = parser.parse_args()中传入空的[]，否则引发异常
print("--syncbn=%f"%args.syncbn)

args.world_size = 1
print("增加新的参数-world_size=%d"%args.world_size)
print("args参数解析:",args)
```

    --syncbn=0.000000
    增加新的参数-world_size=1
    args参数解析: Namespace(apex=False, arch='deepv3.DeepV3Plus', color_aug=0.25, crop_size='448', cv=0, dataset='cityscapes', dump_augmentation_images=False, extra_scales='0.5,2.0', full_crop_training=False, global_rank=0, img_wt_loss=False, init_decoder=False, jointwtborder=False, lr=0.002, lr_schedule='poly', max_epoch=180, momentum=0.9, optimizer='sgd', poly_exp=1.0, pre_size=None, result_dir='./logs_sementic', rmi_loss=False, scale_max=2.0, scale_min=0.5, set_cityscapes_root=None, start_epoch=0, syncbn=False, test_mode=False, trunk='resnet101', weight_decay=0.0001, world_size=1)
    


```python
#semantic-segmentation-main\utils\attr_dict.py
"""
# Code adapted from:
# https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/collections.py

Source License
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""

class AttrDict(dict):

    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not self.__dict__[AttrDict.IMMUTABLE]:
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                self[name] = value
        else:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.
                format(name, value)
            )

    def immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested AttrDicts.
        """
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]
```

实例化类`AttrDict`，使用类的属性存储参数，并配置参数immutable与mutable互相转换的方法。


```python
#semantic-segmentation-main\config.py
import os

__C = AttrDict() #非嵌套字典
cfg=__C
print("cfg=",cfg)

#...
__C.GLOBAL_RANK = 0
__C.EPOCH = 0

# Absolute path to a location to keep some large files, not in this dir.
__C.ASSETS_PATH = r'D:\Cityscapes_assets'

print("参数——非嵌套字典:",cfg)

#Attribute Dictionary for Options
__C.OPTIONS = AttrDict() #嵌套字典
__C.OPTIONS.TEST_MODE = False
__C.OPTIONS.INIT_DECODER = False

#Attribute Dictionary for Model
__C.MODEL = AttrDict()
__C.MODEL.EXTRA_SCALES = '0.5,1.5'
__C.MODEL.BNFUNC = None

WEIGHTS_PATH = os.path.join(__C.ASSETS_PATH, 'seg_weights')
__C.MODEL.WRN38_CHECKPOINT =os.path.join(WEIGHTS_PATH, 'wider_resnet38.pth.tar')

#Attribute Dictionary for Dataset
__C.DATASET = AttrDict()
__C.DATASET.CITYSCAPES_DIR =os.path.join(__C.ASSETS_PATH, 'data/Cityscapes')
__C.DATASET.IGNORE_LABEL = 255
__C.DATASET.NUM_CLASSES = 0
__C.DATASET.CV = 1  #cv_split - 0,1,2,3
__C.DATASET.CUSTOM_COARSE_PROB = None
__C.DATASET.CLASS_UNIFORM_PCT = 0.5
__C.DATASET.NAME = ''
__C.DATASET.CLASS_UNIFORM_TILE = 1024
__C.DATASET.CENTROID_ROOT =os.path.join(__C.ASSETS_PATH, 'uniform_centroids')
__C.DATASET.CITYSCAPES_CUSTOMCOARSE=os.path.join(__C.ASSETS_PATH, 'data/Cityscapes/autolabelled')

__C.DATASET.MEAN = [0.485, 0.456, 0.406]
__C.DATASET.STD = [0.229, 0.224, 0.225]
__C.DATASET.DUMP_IMAGES = False

# This enables there to always be translation augmentation during random crop
__C.DATASET.TRANSLATE_AUG_FIX = False
#...

print("参数——含嵌套字典:",cfg)
```

    cfg= {}
    参数——非嵌套字典: {'GLOBAL_RANK': 0, 'EPOCH': 0, 'ASSETS_PATH': 'D:\\Cityscapes_assets'}
    参数——含嵌套字典: {'GLOBAL_RANK': 0, 'EPOCH': 0, 'ASSETS_PATH': 'D:\\Cityscapes_assets', 'OPTIONS': {'TEST_MODE': False, 'INIT_DECODER': False}, 'MODEL': {'EXTRA_SCALES': '0.5,1.5', 'BNFUNC': None, 'WRN38_CHECKPOINT': 'D:\\Cityscapes_assets\\seg_weights\\wider_resnet38.pth.tar'}, 'DATASET': {'CITYSCAPES_DIR': 'D:\\Cityscapes_assets\\data/Cityscapes', 'IGNORE_LABEL': 255, 'NUM_CLASSES': 0, 'CV': 1, 'CUSTOM_COARSE_PROB': None, 'CLASS_UNIFORM_PCT': 0.5, 'NAME': '', 'CLASS_UNIFORM_TILE': 1024, 'CENTROID_ROOT': 'D:\\Cityscapes_assets\\uniform_centroids', 'CITYSCAPES_CUSTOMCOARSE': 'D:\\Cityscapes_assets\\data/Cityscapes/autolabelled', 'MEAN': [0.485, 0.456, 0.406], 'STD': [0.229, 0.224, 0.225], 'DUMP_IMAGES': False, 'TRANSLATE_AUG_FIX': False}}
    

`assert_and_infer_cfg`函数可以将`argparse`定义的参数args，有选择性的存储到类AttrDict的实例`cfg(__C)`中，并配置为immutable类型。


```python
#semantic-segmentation-main\config.py
import torch
def assert_and_infer_cfg(args, make_immutable=True, train_mode=True):
    """Call this function in your script after you have finished setting all cfg values that are necessary (e.g., merging a config from a file, merging command line config options, etc.). By default, this function will also mark the global cfg as immutable to prevent changing the global cfg
    settings during script execution (which can lead to hard to debug errors or code that's harder to understand than is necessary).
    """
    #...
    cfg.DATASET.CV_SPLITS = 3
    cfg.DATASET.CLASS_UNIFORM_BIAS = None
    cfg.DATASET.DUMP_IMAGES = args.dump_augmentation_images
    
    if hasattr(args, 'syncbn') and args.syncbn:
        if args.apex:
            import apex
            __C.MODEL.BN = 'apex-syncnorm'
            __C.MODEL.BNFUNC = apex.parallel.SyncBatchNorm
        else:
            raise Exception('No Support for SyncBN without Apex')
    else:
        __C.MODEL.BNFUNC = torch.nn.BatchNorm2d
        print('Using regular batch norm')  
    
    if not train_mode:
        cfg.immutable(True)
        return

    def str2list(s):
        alist = s.split(',')
        alist = [float(x) for x in alist]
        return alist
    
    if args.extra_scales:
        cfg.MODEL.EXTRA_SCALES = str2list(args.extra_scales)    
        
    if args.set_cityscapes_root is not None:
        # '/data/cs_imgs_cv0'
        # '/data/cs_imgs_cv2'
        __C.DATASET.CITYSCAPES_DIR = args.set_cityscapes_root            
    
    if make_immutable:
        cfg.immutable(True)
        
    
    #...

print("mutable——cfg.MODEL.EXTRA_SCALES=",cfg.MODEL.EXTRA_SCALES)
cfg.MODEL.EXTRA_SCALES='0.7,1.7'
print("mutable——updated_cfg.MODEL.EXTRA_SCALES=",cfg.MODEL.EXTRA_SCALES)

assert_and_infer_cfg(args)
try:
    cfg.MODEL.EXTRA_SCALES='0.9,1.9'
except:
    print("immutable——cfg.MODEL.EXTRA_SCALES, cannot be updated!")
```

    mutable——cfg.MODEL.EXTRA_SCALES= 0.5,1.5
    mutable——updated_cfg.MODEL.EXTRA_SCALES= 0.7,1.7
    Using regular batch norm
    immutable——cfg.MODEL.EXTRA_SCALES, cannot be updated!
    

通过`AttrDict`的`immutable`方法可以转换参数的mutable和immutable的类型，从而修改参数。


```python
cfg.immutable(False)
cfg.MODEL.EXTRA_SCALES='0.9,1.9'
print("mutable——updated_cfg.MODEL.EXTRA_SCALES=",cfg.MODEL.EXTRA_SCALES)
cfg.immutable(True)    
```

    mutable——updated_cfg.MODEL.EXTRA_SCALES= 0.9,1.9
    

#### 1.4.2 [cityscapes](https://www.cityscapes-dataset.com/)数据读取

**知识点-05：**[collections](https://docs.python.org/3/library/collections.html).namedtuple

`collections.namedtuple(typename, field_names, *, verbose=False, rename=False, module=None)`，其中参数`typename`为创建的一个元组子类类名，用于实例化各种元组对象；`field_names`类似于字典的键(key)，通过键提取对应的值(value)；`rename`默认为False，如果为True，则不能包含有‘非Python标识符，Python中的关键字以及重复的name’，如果有，则会默认重命名。


```python
from collections import namedtuple
#01-实例化nametuple对象
Point = namedtuple('Point', ['x', 'y'])
#02-使用关键字参数或位置参数初始化nametuple
p = Point(11, y=22)
print("02-使用关键字参数或位置参数初始化nametuple:p={}".format(p))
#03-使用键提取元组元素
print("03-使用键提取元组元素:p[0]={},p[1]={}".format(p[0],p[1]))
#04-拆包
x,y=p
print("04-拆包：x,y=p——>x={},y={}".format(x,y))
#05-instance.key的方式提取值
print("05-instance.key的方式提取值:p.x={},p.y={}".format(p.x,p.y))
#06-用已有序列或可迭代对象实例化一个nametuple
lst=[99,77]
print("06-用已有序列或可迭代对象实例化一个nametuple:Point._make(lst)={}".format(Point._make(lst)))
#07-将nametuple对象转换为有序字典OrderDict
print("07-将nametuple对象转换为有序字典OrderDict:p._asdict()={}".format(p._asdict()))
#08-有序字典转换为nametuple对象
dic={'x': 11, 'y': 22}
print("08-有序字典转换为nametuple对象:Point(**dic)={}".format(Point(**dic)))
#09-替换值
print("09-替换值:p._replace(x=33)——>{}".format(p._replace(x=33)))
#10-获取所有nametuple对象字段名
print("10-获取所有nametuple对象字段名:p._fields={}".format(p._fields))
```

    02-使用关键字参数或位置参数初始化nametuple:p=Point(x=11, y=22)
    03-使用键提取元组元素:p[0]=11,p[1]=22
    04-拆包：x,y=p——>x=11,y=22
    05-instance.key的方式提取值:p.x=11,p.y=22
    06-用已有序列或可迭代对象实例化一个nametuple:Point._make(lst)=Point(x=99, y=77)
    07-将nametuple对象转换为有序字典OrderDict:p._asdict()=OrderedDict([('x', 11), ('y', 22)])
    08-有序字典转换为nametuple对象:Point(**dic)=Point(x=11, y=22)
    09-替换值:p._replace(x=33)——>Point(x=33, y=22)
    10-获取所有nametuple对象字段名:p._fields=('x', 'y')
    

---

* cityscapes的标签数据处理

Cityscapes数据集的处理工具，可以从[cityscapesScripts](https://github.com/mcordts/cityscapesScripts)中查找。下属代码namedtuple对象labels,创建了不同类别之间(列之间)的映射关系，并通过`name2label`,`id2label `,`trainId2label `,`label2trainid`,`trainId2name`,`trainId2color`和`category2labels`定义了不同列转换的映射关系。方便后续cityscapes标签的变换。


```python
#semantic-segmentation-main\datasets\cityscapes_labels.py
"""
# File taken from https://github.com/mcordts/cityscapesScripts/
# License File Available at:
# https://github.com/mcordts/cityscapesScripts/blob/master/license.txt

# ----------------------
# The Cityscapes Dataset
# ----------------------
#
#
# License agreement
# -----------------
#
# This dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree:
#
# 1. That the dataset comes "AS IS", without express or implied warranty. Although every effort has been made to ensure accuracy, we (Daimler AG, MPI Informatics, TU Darmstadt) do not accept any responsibility for errors or omissions.
# 2. That you include a reference to the Cityscapes Dataset in any work that makes use of the dataset. For research papers, cite our preferred publication as listed on our website; for other media cite our preferred publication as listed on our website or link to the Cityscapes website.
# 3. That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data) and do not allow to recover the dataset or something similar in character.
# 4. That you may not use the dataset or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
# 5. That all rights not expressly granted to you are reserved by us (Daimler AG, MPI Informatics, TU Darmstadt).
#
#
# Contact
# -------
#
# Marius Cordts, Mohamed Omran
# www.cityscapes-dataset.net

"""
from collections import namedtuple
#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... . 标签的标识符，例如'car', 'person'等
                    # We use them to uniquely name a class 使用'name'命名唯一的类

    'id'          , # An integer ID that is associated with this label.  与标签关联的整形ID
                    # The IDs are used to represent the label in ground truth images ID被用于表示真实图像的标签
                    # An ID of -1 means that this label does not have an ID and thus is ignored when creating ground truth images (e.g. license plate). 为-1值的ID意为这个标签没有ID（被忽略），例如车牌（涉及到公共安全），在创建真实图像分类时就会标识为-1
                    # Do not modify these IDs, since exactly these IDs are expected by the evaluation server. 不要修改这些IDs，因为这些IDs是真实服务器所期望的的值

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create ground truth images with train IDs, using the tools provided in the  'preparation' folder. However, make sure to validate or submit results to our evaluation server using the regular IDs above!
                    #这列IDs可以随意修改，以满足不同的训练目的。在创建自己的真实图像分类时，可以在cityscapesScripts GitHub仓库中的preparation文件夹下寻找创建工具。但是，在验证模型，以及向评估服务器提交结果时，还是需要使用上述同一的ID
                    # For trainIds, multiple labels might have the same ID. Then, these labels are mapped to the same class in the ground truth images. For the inverse mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training, might make sense for some approaches. Max value is 255!
                    #对于'trainId'，多个标签可能具有相同的ID。然后这些标签映射到真实图像中同一类。例如,对于某些方法,将所有void类型的类映射到训练中的同一个ID可能是有意义的，值为255

    'category'    , # The name of the category that this label belongs to 此标签所属类别的名称

    'categoryId'  , # The ID of this category. Used to create ground truth images on category level. 这个类别的ID,用于在类别水平上创建真实图像分类

    'hasInstances', # Whether this label distinguishes between single instances or not 这个标签用于区分是否有单个实例(对象)

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored during evaluations or not 在评估中,像素有作为真实类标的分类被忽略,或者未被忽略

    'color'       , # The color of this label 类标对应的颜色
    ] )

#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego_vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification_border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out_of_roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail_track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard_rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic_light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic_sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license_plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# label2trainid
label2trainid   = { label.id      : label.trainId for label in labels   }
# trainId to label object
trainId2name   = { label.trainId : label.name for label in labels   }
trainId2color  = { label.trainId : label.color for label in labels      }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]
        
#--------------------------------------------------------------------------------
# Assure single instance name
#--------------------------------------------------------------------------------

# returns the label name that describes a single instance (if possible)
# e.g.     input     |   output
#        ----------------------
#          car       |   car
#          cargroup  |   car
#          foo       |   None
#          foogroup  |   None
#          skygroup  |   None
def assureSingleInstanceName( name ):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name
    
print(assureSingleInstanceName('ego_vehicle' ))  
```

    ego_vehicle
    

**知识点-06：** [importlib](https://docs.python.org/3/library/importlib.html) 与getattr

python标准库importlib，可以导入自定义的对象（.py文件/模块），并支持传入字符串导入模块。首先定义了'importlib_func_A.py'文件，将其置于'datasets'文件夹（包）下，使用`importlib`库，读取模块，并应用模块的基本操作，读取模块中的变量值、类的属性和方法。同时可以应用`importlib.util.find_spec`查看是否存在模块等。读取模块的属性使用`getattr`方法。


```python
#datasets/importlib_func_A.py
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 16:24:53 2020

Richie Bao-caDesign设计(cadesign.cn)
"""
attri_f="attri_F"

def func_A():
    print("importlib_func_A/func_A")

class cls_A:
    attri_g="attri_G"
    
    def func_B():
        print("importlib_func_A/cls_A/func_B")
        
if __name__=="__main__":
    func_A()
```

    importlib_func_A/func_A
    


```python
args.impoftlib_module="importlib_func_A"

def dynamic_import(package,module):
    import importlib
    '''
    function - 应用importlib调入自定义模块
    '''    
    return importlib.import_module('{}.{}'.format(package,module))

module=dynamic_import("datasets",args.impoftlib_module)
print("imported module:",module)
print("调入模块中的变量值：",getattr(module,"attri_f"))
cls_A=getattr(module,'cls_A')
print("调入模块中的类：",cls_A)
print("调入模块中类的属性：",cls_A.attri_g)
print("_"*50)
print("调入模块中类的方法：")
cls_A.func_B()
```

    imported module: <module 'datasets.importlib_func_A' from 'C:\\Users\\richi\\omen-richiebao\\omen_github\\Urban-Spatial-Data-Analysis_python\\notebook\\BaiduMapPOIcollection_ipynb\\datasets\\importlib_func_A.py'>
    调入模块中的变量值： attri_F
    调入模块中的类： <class 'datasets.importlib_func_A.cls_A'>
    调入模块中类的属性： attri_G
    __________________________________________________
    调入模块中类的方法：
    importlib_func_A/cls_A/func_B
    


```python
def check_module(package,module):
    import importlib
    '''
    function - 应用importlib查看是否存在模块
    '''
    module_spec=importlib.util.find_spec('{}.{}'.format(package,module))
    if module_spec is None:
        print("Module:{} not found.".format('{}.{}'.format(package,module)))
        return None
    else:
        print("Module:{} can be imported.".format('{}.{}'.format(package,module)))
    return module_spec
check_module('datasets',args.impoftlib_module)
```

    Module:datasets.importlib_func_A can be imported.
    




    ModuleSpec(name='datasets.importlib_func_A', loader=<_frozen_importlib_external.SourceFileLoader object at 0x0000015A5F93BDA0>, origin='C:\\Users\\richi\\omen-richiebao\\omen_github\\Urban-Spatial-Data-Analysis_python\\notebook\\BaiduMapPOIcollection_ipynb\\datasets\\importlib_func_A.py')



---


```python
#semantic-segmentation-main\datasets\cityscapes.py
root = cfg.DATASET.CITYSCAPES_DIR
id_to_trainid =label2trainid
print("id_to_trainid:",id_to_trainid)
print("_"*50)
trainid_to_name = trainId2name
print("trainid_to_name:",trainid_to_name)
```

    id_to_trainid: {0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: -1}
    __________________________________________________
    trainid_to_name: {255: 'trailer', 0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car', 14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle', -1: 'license plate'}
    


```python
#semantic-segmentation-main\datasets\cityscapes.py
def fill_colormap():
    palette = [128, 64, 128,
               244, 35, 232,
               70, 70, 70,
               102, 102, 156,
               190, 153, 153,
               153, 153, 153,
               250, 170, 30,
               220, 220, 0,
               107, 142, 35,
               152, 251, 152,
               70, 130, 180,
               220, 20, 60,
               255, 0, 0,
               0, 0, 142,
               0, 0, 70,
               0, 60, 100,
               0, 80, 100,
               0, 0, 230,
               119, 11, 32]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    return palette

color_mapping=fill_colormap()
```


```python
#semantic-segmentation-main\datasets\cityscapes.py
import os
import os.path as path
img_root = path.join(root, 'leftImg8bit_trainvaltest/leftImg8bit')
mask_root = path.join(root, 'gtFine_trainvaltest/gtFine')

import util
imgs_fn=util.filePath_extraction(img_root,["png"]) 
imgs_root=list(imgs_fn.keys())[0]
imgsFn_lst=imgs_fn[imgs_root]
imgsFn_lst.sort()
imgsFn_lst_=imgsFn_lst[:2]

columns=2
scale=1

util.imgs_layoutShow(imgs_root,imgsFn_lst_,columns,scale,figsize=(10,3))
print(imgsFn_lst_)
```


    
<a href=""><img src="./imgs/44_2.png" height="auto" width="auto" title="caDesign"></a>
    


    ['jena_000000_000019_leftImg8bit.png', 'jena_000001_000019_leftImg8bit.png']
    


```python
mask_fn=util.filePath_extraction(mask_root,["png"]) 
mask_root_=list(mask_fn.keys())[0]
maskFn_lst=mask_fn[mask_root_]
maskFn_lst.sort()
maskFn_lst_=maskFn_lst[:6]
columns=6
scale=1

util.imgs_layoutShow(mask_root_,maskFn_lst_,columns,scale,figsize=(30,3))
print(maskFn_lst_)
```


    
<a href=""><img src="./imgs/44_03.png" height="auto" width="auto" title="caDesign"></a>
    


    ['jena_000000_000019_gtFine_color.png', 'jena_000000_000019_gtFine_instanceIds.png', 'jena_000000_000019_gtFine_labelIds.png', 'jena_000001_000019_gtFine_color.png', 'jena_000001_000019_gtFine_instanceIds.png', 'jena_000001_000019_gtFine_labelIds.png']
    

### 1.5 [DUC(Dense Upsampling Convolution)图像分割](object_detection_segmentation)
目前[torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html)图像分割部分的预训练模型主要是针对COCO数据集子集Pascal VOC，包括有20个分类，并且分类中包括了部分室内物品，这不能够满足室外城市街道环境的语义分割。在[ONNX Model Zoo](https://github.com/onnx/models)汇集的大量模型中，语义分割部分可获取的模型只有[DUC](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/duc)，具体实现的方法在[Inference demo for DUC models](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/duc/dependencies/duc-inference.ipynb) notebook中给出了阐述，其使用的深度学习框架是[MXNet](https://mxnet.apache.org/versions/1.7.0/)，MXNet在PyTorch和TensorFlow双重夹击下发展缓慢，其架构从安装到应用并不友好，因此不建议使用。但是因为图像分割预训练的模型很少，即使有，例如上述[Hierarchical Multi-Scale Attention for Semantic Segmentation](https://github.com/NVIDIA/semantic-segmentation)解析案例，因为代码复杂，操作困难，并缺少维护，提供预训练模型参数因为代码更新与实际代码并不匹配，造成使用上的困难。所以，下述仍然使用MXNet框架，其具体安装以及相关依赖库在notebook文件中均有说明。

DUC所使用的数据集为针对建筑外街道环境的[Cityscapes数据集](https://www.cityscapes-dataset.com/)。

当读者察看该部分时，最好先在[torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html)中查看是否有类似针对[Cityscapes数据集](https://www.cityscapes-dataset.com/)的图像分割预训练模型，或者在[ONNX Model Zoo](https://github.com/onnx/models)中查看，是否有新的更新，如果有则使用最新的预训练模型，进行图像的语义分割。

> 需要注意，因为MXNet发展缓慢，可能DUC代码运行出行错误。可以尝试先安装notebook文件中给出的` pip install mxnet-cu90mkl --pre -U `，支持的python版本为3.6，因此可能需要在Anaconda中建立新环境。再安装一次MXNet，用`pip install mxnet`。这样操作可能是弥补MXNet升级所带来的某些错误。

* A-调入依赖库


```python
import mxnet as mx
import cv2 as cv
import numpy as np
import os
from PIL import Image
import math
from collections import namedtuple
from mxnet.contrib.onnx import import_model
import cityscapes_labels
```

* B-图像预处理，其颜色通道RGB均减去RGB的均值，并转换为MXNet的ndarray数据格式。


```python
def preprocess(im):
    # Convert to float32
    test_img = im.astype(np.float32)
    # Extrapolate image with a small border in order obtain an accurate reshaped image after DUC layer
    test_shape = [im.shape[0],im.shape[1]]
    cell_shapes = [math.ceil(l / 8)*8 for l in test_shape]
    test_img = cv.copyMakeBorder(test_img, 0, max(0, int(cell_shapes[0]) - im.shape[0]), 0, max(0, int(cell_shapes[1]) - im.shape[1]), cv.BORDER_CONSTANT, value=rgb_mean)
    test_img = np.transpose(test_img, (2, 0, 1))
    # subtract rbg mean
    for i in range(3):
        test_img[i] -= rgb_mean[i]
    test_img = np.expand_dims(test_img, axis=0)
    # convert to ndarray
    test_img = mx.ndarray.array(test_img)
    return test_img
```

* C-`get_palette()`：返回用于生成输出分割图的预定义调色板； `colorize()`:使用由模型生成的输出类标和`get_palette()`建立的分割图调色板构建分割映射；`predict()`:向模型传入预处理图像，执行前向传播，并将预测输出数据重新调整为输入图像的形状，使用`colorize()`生成彩色分割的分类掩码图。


```python

def get_palette():
    # get train id to color mappings from file
    trainId2colors = {label.trainId: label.color for label in cityscapes_labels.labels}
    # prepare and return palette
    palette = [0] * 256 * 3
    for trainId in trainId2colors:
        colors = trainId2colors[trainId]
        if trainId == 255:
            colors = (0, 0, 0)
        for i in range(3):
            palette[trainId * 3 + i] = colors[i]
    return palette

def colorize(labels):
    # generate colorized image from output labels and color palette
    result_img = Image.fromarray(labels).convert('P')
    result_img.putpalette(get_palette())
    return np.array(result_img.convert('RGB'))

def predict(imgs):
    # get input and output dimensions
    result_height, result_width = result_shape
    _, _, img_height, img_width = imgs.shape
    # set downsampling rate
    ds_rate = 8
    # set cell width
    cell_width = 2
    # number of output label classes
    label_num = 19
    
    # Perform forward pass
    batch = namedtuple('Batch', ['data'])
    mod.forward(batch([imgs]),is_train=False)
    labels = mod.get_outputs()[0].asnumpy().squeeze()

    # re-arrange output
    test_width = int((int(img_width) / ds_rate) * ds_rate)
    test_height = int((int(img_height) / ds_rate) * ds_rate)
    feat_width = int(test_width / ds_rate)
    feat_height = int(test_height / ds_rate)
    labels = labels.reshape((label_num, 4, 4, feat_height, feat_width))
    labels = np.transpose(labels, (0, 3, 1, 4, 2))
    labels = labels.reshape((label_num, int(test_height / cell_width), int(test_width / cell_width)))

    labels = labels[:, :int(img_height / cell_width),:int(img_width / cell_width)]
    labels = np.transpose(labels, [1, 2, 0])
    labels = cv.resize(labels, (result_width, result_height), interpolation=cv.INTER_LINEAR)
    labels = np.transpose(labels, [2, 0, 1])
    
    # get softmax output
    softmax = labels
    
    # get classification labels
    results = np.argmax(labels, axis=0).astype(np.uint8)
    raw_labels = results

    # comput confidence score
    confidence = float(np.max(softmax, axis=0).mean())

    # generate segmented image
    result_img = Image.fromarray(colorize(raw_labels)).resize(result_shape[::-1])
    
    # generate blended image
    blended_img = Image.fromarray(cv.addWeighted(im[:, :, ::-1], 0.5, np.array(result_img), 0.5, 0))

    return confidence, result_img, blended_img, raw_labels
```

* D-加载预训练的模型。导入ONNX预训练模型到MXＮｅｔ中，使用符号文件(symbol file)定义模型，使用参数文件(params file)绑定参数。


```python
def get_model(ctx, model_path):
    # import ONNX model into MXNet symbols and params
    sym,arg,aux = import_model(model_path)
    # define network module
    mod = mx.mod.Module(symbol=sym, data_names=['data'], context=ctx, label_names=None)
    # bind parameters to the network
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, im.shape[0], im.shape[1]))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params=arg, aux_params=aux,allow_missing=True, allow_extra=True)
    return mod
```

* E-给出一张[KITTI](http://www.cvlibs.net/datasets/kitti/index.php)数据中的影像并显示


```python
im = cv.imread('./data/0000000389.png')[:, :, ::-1]
# set output shape (same as input shape)
result_shape = [im.shape[0],im.shape[1]]
# set rgb mean of input image (used in mean subtraction)
rgb_mean = cv.mean(im)
```


```python
# display input image
Image.fromarray(im)
```




    
<a href=""><img src="./imgs/44_04.png" height="auto" width="auto" title="caDesign"></a>
    



* F-可以手动从'https://s3.amazonaws.com/onnx-model-zoo/duc/ResNet101_DUC_HDC.onnx' 下载预训练模型。并设置使用GPU，还是CPU。


```python
# Download ONNX model
#mx.test_utils.download('https://s3.amazonaws.com/onnx-model-zoo/duc/ResNet101_DUC_HDC.onnx')
# Determine and set context
if len(mx.test_utils.list_gpus())==0:
    ctx = mx.cpu()
else:
    ctx = mx.gpu(0)
```


```python
# Load ONNX model
mod = get_model(ctx, r'ResNet101_DUC_HDC.onnx')
print("The model is loaded...")
```

    The model is loaded...
    

* G-处理输入图像，并执行预测，查看预测结果


```python
pre = preprocess(im)
conf,result_img,blended_img,raw = predict(pre)
result_img
```




    
<a href=""><img src="./imgs/44_05.png" height="auto" width="auto" title="caDesign"></a>
    



* H-混合输出。分割图与真实图叠合，方便观察预测结果。


```python
blended_img
```




    
<a href=""><img src="./imgs/44_06.png" height="auto" width="auto" title="caDesign"></a>
    



* I-查看精度(confidence score)。为SoftMax回归分类模型输出的联合概率分布最大值。数值位于[0,1]，数值越大，像素属于某一分类可能性越大。


```python
print('Confidence = %f' %(conf))
```

    Confidence = 0.929088
    


```python

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego_vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification_border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out_of_roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail_track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard_rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic_light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic_sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license_plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]
```

### 1.6 城市空间要素组成，时空量度，绿视率，均衡度
cityscapes数据集，其标签/分类包括主要的城市街道场景内容，这为城市空间的分析提供了基础的数据支持，例如对于固定行进流线，视野方向和宽度下，通过标签'vegetation'可以计算绿视率(Good Looking Ratio)，当绿视率达到一定水平，会让行人在街道空间中觉得舒适；通过'sky'可以获知视野下所见天空的比例，这与天空视域因子(Sky View Factor, SVF)可以比较研究；对于其它项，例如'car' ，'truck'，'bus'  可以初步判断某一时刻街道的交通情况，'person' ，'rider'则可以初步判断行人情况。根据待分析的内容可以有意识的选择对应的要素进行分析，也可以综合考虑所有因素，计算每一位置的信息熵和均衡度，比较不同位置的混杂程度，通常混杂比较高的位置可能感觉会比较热闹，而低的区域则相对简单和冷清。

因为将DUC预训练模型用于KITTI数据集，无人驾驶项目拍摄的连续图像，是固定行进流线，视野方向和宽度的，这可以保证图像具有统一的属性，避免因为拍摄上下角度变化的问题，使得图像之间不具有比较性。预测计算实际较长，为了避免数据丢失，DUC预测的结果，conf 精度/概率；result_img 语义分割的掩码图像（颜色区分）；blended_img 叠合掩码和实际的图像，可以方便查看分割与实际之间的差异；raw trainId数字索引，分别保存为图像格式、以及存储在列表下，用pickle保存。


```python
import glob
drive_29_0071_img_fp_list=glob.glob("D:/dataset/KITTI/2011_09_29_drive_0071_sync/image_03/data/*.png")
```


```python
def sementicSeg_DUC_pred(DUC_output_root,img_fps,preprocess,predict):
    from tqdm.auto import tqdm
    import os,pickle
    import cv2 as cv
    from tqdm.auto import tqdm
    '''
    function - DUC图像分割，及预测图像保存
    '''

    DUC_output_root=DUC_output_root
    conf_list=[]
    raw_list=[]
    for i,img in enumerate(tqdm(img_fps)):
        im=cv.imread(img)[:, :, ::-1]
        # set output shape (same as input shape)
        result_shape = [im.shape[0],im.shape[1]]
        # set rgb mean of input image (used in mean subtraction)
        rgb_mean = cv.mean(im)
        pre=preprocess(im)
        conf,result_img,blended_img,raw=predict(pre)
        conf_list.append(conf)
        raw_list.append(raw)

        if not os.path.exists(os.path.join(DUC_output_root,"result_img")):
            os.makedirs(os.path.join(DUC_output_root,"result_img"))
        if not os.path.exists(os.path.join(DUC_output_root,"blended_img")):
            os.makedirs(os.path.join(DUC_output_root,"blended_img"))        
        result_img.save(os.path.join(DUC_output_root,"result_img/result_img_{}.png".format(i)))
        blended_img.save(os.path.join(DUC_output_root,"blended_img/blended_img_{}.png".format(i)))

    with open(os.path.join(DUC_output_root,'KITTI_DUC_confi.pkl'),'wb') as f1:    
        pickle.dump(conf_list,f1)    
    with open(os.path.join(DUC_output_root,'KITTI_DUC_raw.pkl'),'wb') as f2:    
        pickle.dump(raw_list,f2)   
        
DUC_output_root=r'D:\dataset\KITTI_DUC'
sementicSeg_DUC_pred(DUC_output_root,img_fps=drive_29_0071_img_fp_list,preprocess=preprocess,predict=predict)
```


      0%|          | 0/1059 [00:00<?, ?it/s]



```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from IPython.display import HTML
import os,glob

DUC_output_root=r'D:\dataset\KITTI_DUC'
result_img_fp_list=glob.glob(DUC_output_root+r'/result_img/*.png')

fig=plt.figure(figsize=(20,10))
ims=[[plt.imshow(mpimg.imread(f),animated=True)] for f in result_img_fp_list[:]]
print("finished reading the imgs.")

ani=animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=1000) #conda install -c conda-forge ffmpeg
ani.save(os.path.join(DUC_output_root,r'DUC_result_imgs.mp4'))
print(".mp4 saved.")
HTML(ani.to_html5_video())
```

    finished reading the imgs.
    




<video width='auto' height='auto' controls><source src="./imgs/DUC_result_imgs.mp4" height='auto' width='auto' title="caDesign" type='video/mp4'></video>
    


对每一位置计算所有对象的频数，可以得知各对象在图像所代表的视野下所占的比例。   


```python
def DUC_pred_frequency_moment(KITTI_DUC_raw_fp,KITTI_DUC_confi_fp):
    import pickle
    import numpy as np
    import pandas as pd
    from tqdm.auto import tqdm
    '''
    function - 读取DUC语义分割结果保存的'KITTI_DUC_confi.pkl'和'KITTI_DUC_raw.pkl'文件，用于位置图像对象/语义分割类别频数统计
    '''
    with open(KITTI_DUC_raw_fp,'rb') as f:
        KITTI_DUC_raw=pickle.load(f)
    with open(KITTI_DUC_confi_fp,'rb') as f:
        KITTI_DUC_confi=pickle.load(f) 

    unique_id_all=np.unique(np.stack(KITTI_DUC_raw))
    print("所有出现的id:{}".format(unique_id_all)) #对应trainId
    id_info_df=pd.DataFrame(columns=unique_id_all)
    
    for seg in tqdm(KITTI_DUC_raw):
        unique_id, counts_id=np.unique(seg, return_counts=True)
        id_fre_dic=dict(list(zip(unique_id, counts_id)))
        for i in unique_id_all:
            if i not in unique_id.tolist():
                id_fre_dic.setdefault(i,0)
        id_info_df=id_info_df.append(id_fre_dic,ignore_index=True)    
     
    id_info_df["confidence"]=KITTI_DUC_confi
    return id_info_df,unique_id_all
    
KITTI_DUC_raw_fp=r'D:\dataset\KITTI_DUC\KITTI_DUC_raw.pkl'   
KITTI_DUC_confi_fp=r'D:\dataset\KITTI_DUC\KITTI_DUC_confi.pkl'
id_info_df,unique_id_all=DUC_pred_frequency_moment(KITTI_DUC_raw_fp,KITTI_DUC_confi_fp)
```

    所有出现的id:[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18]
    


      0%|          | 0/1059 [00:00<?, ?it/s]


读取KITTI数据集的经纬度位置信息和时间戳。和频数的DataFrame格式数据合并在一个DataFrame之下，方便后续数据处理。


```python
import util
KITTI_info_fp=r'D:\dataset\KITTI\2011_09_29_drive_0071_sync\oxts\data'
timestamps_fp=r'D:\dataset\KITTI\2011_09_29_drive_0071_sync\image_03\timestamps.txt'
drive_29_0071_info=util.KITTI_info(KITTI_info_fp,timestamps_fp)
drive_29_0071_info_coordi=drive_29_0071_info[['lat','lon','timestamps_']]
```


```python
trainID_label_mapping={id_:trainId2label[id_].name for id_ in unique_id_all} #建立trainID到类别的映射字典，trainId2label方法向上，在cityscapes的标签数据处理部分
id_info_df=id_info_df.rename(columns=trainID_label_mapping)
id_info_df['trainID']=id_info_df.index
id_info_df=id_info_df.join(drive_29_0071_info_coordi)
id_info_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>road</th>
      <th>sidewalk</th>
      <th>building</th>
      <th>wall</th>
      <th>fence</th>
      <th>pole</th>
      <th>traffic_light</th>
      <th>traffic_sign</th>
      <th>vegetation</th>
      <th>terrain</th>
      <th>...</th>
      <th>truck</th>
      <th>bus</th>
      <th>train</th>
      <th>motorcycle</th>
      <th>bicycle</th>
      <th>confidence</th>
      <th>trainID</th>
      <th>lat</th>
      <th>lon</th>
      <th>timestamps_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>88520</td>
      <td>9010</td>
      <td>317546</td>
      <td>2399</td>
      <td>135</td>
      <td>1818</td>
      <td>0</td>
      <td>236</td>
      <td>1640</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>70</td>
      <td>0.954470</td>
      <td>0</td>
      <td>49.008645</td>
      <td>8.398104</td>
      <td>2011-09-29 13:54:59.990872576</td>
    </tr>
    <tr>
      <th>1</th>
      <td>88991</td>
      <td>9251</td>
      <td>321157</td>
      <td>2638</td>
      <td>637</td>
      <td>1673</td>
      <td>0</td>
      <td>118</td>
      <td>1118</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>645</td>
      <td>0.955859</td>
      <td>1</td>
      <td>49.008645</td>
      <td>8.398103</td>
      <td>2011-09-29 13:55:00.094612992</td>
    </tr>
    <tr>
      <th>2</th>
      <td>89471</td>
      <td>11162</td>
      <td>324130</td>
      <td>2735</td>
      <td>75</td>
      <td>1046</td>
      <td>0</td>
      <td>200</td>
      <td>837</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>458</td>
      <td>0.953789</td>
      <td>2</td>
      <td>49.008646</td>
      <td>8.398102</td>
      <td>2011-09-29 13:55:00.198486528</td>
    </tr>
    <tr>
      <th>3</th>
      <td>91988</td>
      <td>9927</td>
      <td>314206</td>
      <td>2806</td>
      <td>12524</td>
      <td>1131</td>
      <td>0</td>
      <td>136</td>
      <td>969</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>593</td>
      <td>0.948840</td>
      <td>3</td>
      <td>49.008646</td>
      <td>8.398101</td>
      <td>2011-09-29 13:55:00.302340864</td>
    </tr>
    <tr>
      <th>4</th>
      <td>94986</td>
      <td>13557</td>
      <td>310776</td>
      <td>1036</td>
      <td>16818</td>
      <td>1668</td>
      <td>0</td>
      <td>103</td>
      <td>887</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>818</td>
      <td>0.945306</td>
      <td>4</td>
      <td>49.008646</td>
      <td>8.398100</td>
      <td>2011-09-29 13:55:00.406079232</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1054</th>
      <td>121743</td>
      <td>15678</td>
      <td>252994</td>
      <td>1281</td>
      <td>52565</td>
      <td>1562</td>
      <td>58</td>
      <td>935</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>255</td>
      <td>1562</td>
      <td>0</td>
      <td>0</td>
      <td>0.944332</td>
      <td>1054</td>
      <td>49.009498</td>
      <td>8.395251</td>
      <td>2011-09-29 13:56:49.458599424</td>
    </tr>
    <tr>
      <th>1055</th>
      <td>125205</td>
      <td>12000</td>
      <td>247483</td>
      <td>4974</td>
      <td>52823</td>
      <td>1994</td>
      <td>90</td>
      <td>971</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>118</td>
      <td>2218</td>
      <td>0</td>
      <td>0</td>
      <td>0.939795</td>
      <td>1055</td>
      <td>49.009498</td>
      <td>8.395250</td>
      <td>2011-09-29 13:56:49.562463744</td>
    </tr>
    <tr>
      <th>1056</th>
      <td>120627</td>
      <td>16324</td>
      <td>247972</td>
      <td>4548</td>
      <td>53390</td>
      <td>2144</td>
      <td>134</td>
      <td>972</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>642</td>
      <td>557</td>
      <td>0</td>
      <td>0</td>
      <td>0.939858</td>
      <td>1056</td>
      <td>49.009497</td>
      <td>8.395250</td>
      <td>2011-09-29 13:56:49.666327808</td>
    </tr>
    <tr>
      <th>1057</th>
      <td>120940</td>
      <td>15959</td>
      <td>247862</td>
      <td>4554</td>
      <td>53147</td>
      <td>2140</td>
      <td>99</td>
      <td>897</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1119</td>
      <td>813</td>
      <td>0</td>
      <td>0</td>
      <td>0.937643</td>
      <td>1057</td>
      <td>49.009498</td>
      <td>8.395249</td>
      <td>2011-09-29 13:56:49.770316544</td>
    </tr>
    <tr>
      <th>1058</th>
      <td>117905</td>
      <td>18114</td>
      <td>247659</td>
      <td>6028</td>
      <td>52716</td>
      <td>2294</td>
      <td>180</td>
      <td>839</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1524</td>
      <td>219</td>
      <td>0</td>
      <td>0</td>
      <td>0.939307</td>
      <td>1058</td>
      <td>49.009498</td>
      <td>8.395248</td>
      <td>2011-09-29 13:56:49.874179584</td>
    </tr>
  </tbody>
</table>
<p>1059 rows × 24 columns</p>
</div>



打印所有位置时刻，每一图像包含对象的面积频数，即各对象像素占整体图像像素的比例。如果要查看单个对象，可以点击图例对应项。因为'2011_09_29_drive_0071_sync'部分数据位于城市的街巷内，因此可以明显的观察到'building'对象所占的数量较大，次之则为'road'，其它对象相对较小。'vegetation'在部分区域有较高的比例。


```python
import plotly.express as px
labels=list(trainID_label_mapping.values())
fig = px.line(id_info_df, x="trainID", y=labels,
              hover_data=['confidence','lat','lon','trainID'],
              title='id_info_df'
             )
fig.show()
```

<a href=""><img src="./imgs/44_08.png" height="auto" width="auto" title="caDesign"></a>

可以打印感兴趣的对象，观察其在实际空间地理位置上的分布情况。例如'sky'和'vegetation'的分布，通过量化的方式能够明确变化方式的具体位置。

```python
fig = px.density_mapbox(id_info_df, lat='lat', lon='lon', z='sky', radius=10,
                        center=dict(lat=49.008645, lon=8.398104), zoom=18,
                        mapbox_style="stamen-terrain",
                        hover_data=['confidence','lat','lon','trainID'],
                        title=r'sky Kernel Density')
fig.show()
```

<a href=""><img src="./imgs/44_07.png" height="auto" width="auto" title="caDesign"></a>

```python
import plotly.express as px
fig = px.density_mapbox(id_info_df, lat='lat', lon='lon', z='vegetation', radius=10,
                        center=dict(lat=49.008645, lon=8.398104), zoom=18,
                        mapbox_style="stamen-terrain",
                        hover_data=['confidence','lat','lon','trainID'],
                        title=r'vegetation Kernel Density')
fig.show()
```

<a href=""><img src="./imgs/44_09.png" height="auto" width="auto" title="caDesign"></a>


虽然对象的像素数量可以比较不同对象所占的比例，但是计算各自对象所占的百分比，则更容易得知对象在整个图像视野中比例的变化情况。


```python
sum_syntax='pixels='+''.join("%s+"%''.join(map(str,x)) for x in labels)[:-1]   
id_info_df=id_info_df.eval(sum_syntax)
id_info_df['vegetation_percent']=id_info_df.apply(lambda row:row.vegetation/row.pixels*100,axis=1)
id_info_df['sky_percent']=id_info_df.apply(lambda row:row.sky/row.pixels*100,axis=1)
id_info_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>road</th>
      <th>sidewalk</th>
      <th>building</th>
      <th>wall</th>
      <th>fence</th>
      <th>pole</th>
      <th>traffic_light</th>
      <th>traffic_sign</th>
      <th>vegetation</th>
      <th>terrain</th>
      <th>...</th>
      <th>motorcycle</th>
      <th>bicycle</th>
      <th>confidence</th>
      <th>trainID</th>
      <th>lat</th>
      <th>lon</th>
      <th>timestamps_</th>
      <th>pixels</th>
      <th>vegetation_percent</th>
      <th>sky_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>88520</td>
      <td>9010</td>
      <td>317546</td>
      <td>2399</td>
      <td>135</td>
      <td>1818</td>
      <td>0</td>
      <td>236</td>
      <td>1640</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>70</td>
      <td>0.954470</td>
      <td>0</td>
      <td>49.008645</td>
      <td>8.398104</td>
      <td>2011-09-29 13:54:59.990872576</td>
      <td>463012</td>
      <td>0.354202</td>
      <td>1.901463</td>
    </tr>
    <tr>
      <th>1</th>
      <td>88991</td>
      <td>9251</td>
      <td>321157</td>
      <td>2638</td>
      <td>637</td>
      <td>1673</td>
      <td>0</td>
      <td>118</td>
      <td>1118</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>645</td>
      <td>0.955859</td>
      <td>1</td>
      <td>49.008645</td>
      <td>8.398103</td>
      <td>2011-09-29 13:55:00.094612992</td>
      <td>463012</td>
      <td>0.241462</td>
      <td>1.868418</td>
    </tr>
    <tr>
      <th>2</th>
      <td>89471</td>
      <td>11162</td>
      <td>324130</td>
      <td>2735</td>
      <td>75</td>
      <td>1046</td>
      <td>0</td>
      <td>200</td>
      <td>837</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>458</td>
      <td>0.953789</td>
      <td>2</td>
      <td>49.008646</td>
      <td>8.398102</td>
      <td>2011-09-29 13:55:00.198486528</td>
      <td>463012</td>
      <td>0.180773</td>
      <td>1.835158</td>
    </tr>
    <tr>
      <th>3</th>
      <td>91988</td>
      <td>9927</td>
      <td>314206</td>
      <td>2806</td>
      <td>12524</td>
      <td>1131</td>
      <td>0</td>
      <td>136</td>
      <td>969</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>593</td>
      <td>0.948840</td>
      <td>3</td>
      <td>49.008646</td>
      <td>8.398101</td>
      <td>2011-09-29 13:55:00.302340864</td>
      <td>463012</td>
      <td>0.209282</td>
      <td>1.807729</td>
    </tr>
    <tr>
      <th>4</th>
      <td>94986</td>
      <td>13557</td>
      <td>310776</td>
      <td>1036</td>
      <td>16818</td>
      <td>1668</td>
      <td>0</td>
      <td>103</td>
      <td>887</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>818</td>
      <td>0.945306</td>
      <td>4</td>
      <td>49.008646</td>
      <td>8.398100</td>
      <td>2011-09-29 13:55:00.406079232</td>
      <td>463012</td>
      <td>0.191572</td>
      <td>1.777708</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1054</th>
      <td>121743</td>
      <td>15678</td>
      <td>252994</td>
      <td>1281</td>
      <td>52565</td>
      <td>1562</td>
      <td>58</td>
      <td>935</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0.944332</td>
      <td>1054</td>
      <td>49.009498</td>
      <td>8.395251</td>
      <td>2011-09-29 13:56:49.458599424</td>
      <td>463012</td>
      <td>0.000216</td>
      <td>0.048811</td>
    </tr>
    <tr>
      <th>1055</th>
      <td>125205</td>
      <td>12000</td>
      <td>247483</td>
      <td>4974</td>
      <td>52823</td>
      <td>1994</td>
      <td>90</td>
      <td>971</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0.939795</td>
      <td>1055</td>
      <td>49.009498</td>
      <td>8.395250</td>
      <td>2011-09-29 13:56:49.562463744</td>
      <td>463012</td>
      <td>0.000000</td>
      <td>0.053562</td>
    </tr>
    <tr>
      <th>1056</th>
      <td>120627</td>
      <td>16324</td>
      <td>247972</td>
      <td>4548</td>
      <td>53390</td>
      <td>2144</td>
      <td>134</td>
      <td>972</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0.939858</td>
      <td>1056</td>
      <td>49.009497</td>
      <td>8.395250</td>
      <td>2011-09-29 13:56:49.666327808</td>
      <td>463012</td>
      <td>0.000000</td>
      <td>0.030885</td>
    </tr>
    <tr>
      <th>1057</th>
      <td>120940</td>
      <td>15959</td>
      <td>247862</td>
      <td>4554</td>
      <td>53147</td>
      <td>2140</td>
      <td>99</td>
      <td>897</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0.937643</td>
      <td>1057</td>
      <td>49.009498</td>
      <td>8.395249</td>
      <td>2011-09-29 13:56:49.770316544</td>
      <td>463012</td>
      <td>0.000000</td>
      <td>0.026781</td>
    </tr>
    <tr>
      <th>1058</th>
      <td>117905</td>
      <td>18114</td>
      <td>247659</td>
      <td>6028</td>
      <td>52716</td>
      <td>2294</td>
      <td>180</td>
      <td>839</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0.939307</td>
      <td>1058</td>
      <td>49.009498</td>
      <td>8.395248</td>
      <td>2011-09-29 13:56:49.874179584</td>
      <td>463012</td>
      <td>0.000000</td>
      <td>0.017926</td>
    </tr>
  </tbody>
</table>
<p>1059 rows × 27 columns</p>
</div>



打印植被百分比的分布，可以得知该街道植被在前半部分开始逐渐增加，在中心广场部分则有相对较多的树木栽植，但是到了后半段，则迅速减少。同时也加入了天空的百分比。


```python
import plotly.express as px
labels=list(trainID_label_mapping.values())
extracted_labels=['vegetation_percent','sky_percent']
fig = px.line(id_info_df, x="trainID", y=extracted_labels,
              hover_data=['confidence','lat','lon','trainID'],
              title='vegetation_percent'
             )
fig.show()
```

<a href=""><img src="./imgs/44_10.png" height="auto" width="auto" title="caDesign"></a>


将'KITTI，动态街景视觉感知'部分的代码加入到'util.py'文件中，在此处调用计算视觉感知消失的距离，获得开始点和消失点的索引，计算在每一位置下与对应感知距离消失的位置间均衡度的变化。这一信息的比较可以粗略的得知当前位置到视觉感知消失位置城市空间场景混杂程度的变化。 当差值绝对值较大时，说明场景在混杂丰富和简单冷清间互相变换，即视觉消失的位置场景与当前场景有很大不同；如果差值绝对值较小，则说明城市空对象的混合程度基本保持不变，即视觉消失的位置场景与当前场景类似。

对于需要花时间计算的内容，通常都将其保存到硬盘中，需要时直接调用，避免重复计算。


```python
import util
import glob

drive_29_0071_img_fp_list=glob.glob("D:/dataset/KITTI/2011_09_29_drive_0071_sync/image_03/data/*.png")
dsv_vp=util.dynamicStreetView_visualPerception(drive_29_0071_img_fp_list[:]) #[:200]  #pip install opencv-python and pip install opencv-contrib-python 
matches_num=dsv_vp.sequence_statistics()
```

      0%|          | 3/1059 [00:00<00:50, 20.75it/s]

    计算关键点和描述子...
    

    100%|██████████| 1059/1059 [01:07<00:00, 15.69it/s]
      0%|          | 0/1058 [00:00<?, ?it/s]

    计算序列图像匹配数...
    

    100%|██████████| 1058/1058 [23:55<00:00,  1.36s/it]
    


```python
import util

KITTI_info_fp=r'D:\dataset\KITTI\2011_09_29_drive_0071_sync\oxts\data'
timestamps_fp=r'D:\dataset\KITTI\2011_09_29_drive_0071_sync\image_03\timestamps.txt'
drive_29_0071_info=util.KITTI_info(KITTI_info_fp,timestamps_fp)
drive_29_0071_info_coordi=drive_29_0071_info[['lat','lon','timestamps_']]

coordi_df=drive_29_0071_info_coordi
vanishing_gpd=util.vanishing_position_length(matches_num,coordi_df,epsg="EPSG:3857",threshold=0)
vanishing_gpd.to_pickle('./results/drive_29_0071_vanishing_gpd.pkl')
```

    C:\Users\richi\anaconda3\envs\PyTorch\lib\site-packages\pyproj\crs\crs.py:53: FutureWarning: '+init=<authority>:<code>' syntax is deprecated. '<authority>:<code>' is the preferred initialization method. When making the change, be mindful of axis order changes: https://pyproj4.github.io/pyproj/stable/gotchas.html#axis-order-changes-in-proj-6
      return _prepare_from_string(" ".join(pjargs))
    


```python
import pandas as pd
vanishing_gpd_=pd.read_pickle('./results/drive_29_0071_vanishing_gpd.pkl')
vanishing_gpd_
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>start_idx</th>
      <th>end_idx</th>
      <th>geometry</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>91</td>
      <td>LINESTRING (934872.661 6276328.368, 934872.550...</td>
      <td>26.590485</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>81</td>
      <td>LINESTRING (934872.550 6276328.441, 934872.449...</td>
      <td>25.073412</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>81</td>
      <td>LINESTRING (934872.449 6276328.505, 934872.346...</td>
      <td>24.953489</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>81</td>
      <td>LINESTRING (934872.346 6276328.570, 934872.232...</td>
      <td>24.832104</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>82</td>
      <td>LINESTRING (934872.232 6276328.643, 934872.132...</td>
      <td>24.815814</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1053</th>
      <td>1053</td>
      <td>1053</td>
      <td>GEOMETRYCOLLECTION EMPTY</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1054</th>
      <td>1054</td>
      <td>1054</td>
      <td>GEOMETRYCOLLECTION EMPTY</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1055</th>
      <td>1055</td>
      <td>1055</td>
      <td>GEOMETRYCOLLECTION EMPTY</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1056</th>
      <td>1056</td>
      <td>1056</td>
      <td>GEOMETRYCOLLECTION EMPTY</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1057</th>
      <td>1057</td>
      <td>1057</td>
      <td>GEOMETRYCOLLECTION EMPTY</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>1058 rows × 4 columns</p>
</div>




```python
id_info_df=id_info_df.join(vanishing_gpd_)
id_info_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>road</th>
      <th>sidewalk</th>
      <th>building</th>
      <th>wall</th>
      <th>fence</th>
      <th>pole</th>
      <th>traffic_light</th>
      <th>traffic_sign</th>
      <th>vegetation</th>
      <th>terrain</th>
      <th>...</th>
      <th>trainID</th>
      <th>lat</th>
      <th>lon</th>
      <th>timestamps_</th>
      <th>pixels</th>
      <th>vegetation_percent</th>
      <th>start_idx</th>
      <th>end_idx</th>
      <th>geometry</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>88520</td>
      <td>9010</td>
      <td>317546</td>
      <td>2399</td>
      <td>135</td>
      <td>1818</td>
      <td>0</td>
      <td>236</td>
      <td>1640</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>49.008645</td>
      <td>8.398104</td>
      <td>2011-09-29 13:54:59.990872576</td>
      <td>463012</td>
      <td>0.354202</td>
      <td>0.0</td>
      <td>91.0</td>
      <td>LINESTRING (934872.661 6276328.368, 934872.550...</td>
      <td>26.590485</td>
    </tr>
    <tr>
      <th>1</th>
      <td>88991</td>
      <td>9251</td>
      <td>321157</td>
      <td>2638</td>
      <td>637</td>
      <td>1673</td>
      <td>0</td>
      <td>118</td>
      <td>1118</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>49.008645</td>
      <td>8.398103</td>
      <td>2011-09-29 13:55:00.094612992</td>
      <td>463012</td>
      <td>0.241462</td>
      <td>1.0</td>
      <td>81.0</td>
      <td>LINESTRING (934872.550 6276328.441, 934872.449...</td>
      <td>25.073412</td>
    </tr>
    <tr>
      <th>2</th>
      <td>89471</td>
      <td>11162</td>
      <td>324130</td>
      <td>2735</td>
      <td>75</td>
      <td>1046</td>
      <td>0</td>
      <td>200</td>
      <td>837</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>49.008646</td>
      <td>8.398102</td>
      <td>2011-09-29 13:55:00.198486528</td>
      <td>463012</td>
      <td>0.180773</td>
      <td>2.0</td>
      <td>81.0</td>
      <td>LINESTRING (934872.449 6276328.505, 934872.346...</td>
      <td>24.953489</td>
    </tr>
    <tr>
      <th>3</th>
      <td>91988</td>
      <td>9927</td>
      <td>314206</td>
      <td>2806</td>
      <td>12524</td>
      <td>1131</td>
      <td>0</td>
      <td>136</td>
      <td>969</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>49.008646</td>
      <td>8.398101</td>
      <td>2011-09-29 13:55:00.302340864</td>
      <td>463012</td>
      <td>0.209282</td>
      <td>3.0</td>
      <td>81.0</td>
      <td>LINESTRING (934872.346 6276328.570, 934872.232...</td>
      <td>24.832104</td>
    </tr>
    <tr>
      <th>4</th>
      <td>94986</td>
      <td>13557</td>
      <td>310776</td>
      <td>1036</td>
      <td>16818</td>
      <td>1668</td>
      <td>0</td>
      <td>103</td>
      <td>887</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>49.008646</td>
      <td>8.398100</td>
      <td>2011-09-29 13:55:00.406079232</td>
      <td>463012</td>
      <td>0.191572</td>
      <td>4.0</td>
      <td>82.0</td>
      <td>LINESTRING (934872.232 6276328.643, 934872.132...</td>
      <td>24.815814</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1054</th>
      <td>121743</td>
      <td>15678</td>
      <td>252994</td>
      <td>1281</td>
      <td>52565</td>
      <td>1562</td>
      <td>58</td>
      <td>935</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1054</td>
      <td>49.009498</td>
      <td>8.395251</td>
      <td>2011-09-29 13:56:49.458599424</td>
      <td>463012</td>
      <td>0.000216</td>
      <td>1054.0</td>
      <td>1054.0</td>
      <td>GEOMETRYCOLLECTION EMPTY</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1055</th>
      <td>125205</td>
      <td>12000</td>
      <td>247483</td>
      <td>4974</td>
      <td>52823</td>
      <td>1994</td>
      <td>90</td>
      <td>971</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1055</td>
      <td>49.009498</td>
      <td>8.395250</td>
      <td>2011-09-29 13:56:49.562463744</td>
      <td>463012</td>
      <td>0.000000</td>
      <td>1055.0</td>
      <td>1055.0</td>
      <td>GEOMETRYCOLLECTION EMPTY</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1056</th>
      <td>120627</td>
      <td>16324</td>
      <td>247972</td>
      <td>4548</td>
      <td>53390</td>
      <td>2144</td>
      <td>134</td>
      <td>972</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1056</td>
      <td>49.009497</td>
      <td>8.395250</td>
      <td>2011-09-29 13:56:49.666327808</td>
      <td>463012</td>
      <td>0.000000</td>
      <td>1056.0</td>
      <td>1056.0</td>
      <td>GEOMETRYCOLLECTION EMPTY</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1057</th>
      <td>120940</td>
      <td>15959</td>
      <td>247862</td>
      <td>4554</td>
      <td>53147</td>
      <td>2140</td>
      <td>99</td>
      <td>897</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1057</td>
      <td>49.009498</td>
      <td>8.395249</td>
      <td>2011-09-29 13:56:49.770316544</td>
      <td>463012</td>
      <td>0.000000</td>
      <td>1057.0</td>
      <td>1057.0</td>
      <td>GEOMETRYCOLLECTION EMPTY</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1058</th>
      <td>117905</td>
      <td>18114</td>
      <td>247659</td>
      <td>6028</td>
      <td>52716</td>
      <td>2294</td>
      <td>180</td>
      <td>839</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1058</td>
      <td>49.009498</td>
      <td>8.395248</td>
      <td>2011-09-29 13:56:49.874179584</td>
      <td>463012</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1059 rows × 30 columns</p>
</div>




```python
id_info_df.to_pickle('./results/DUC_info_drive_29_0071_vanishing_gpd.pkl')
```


```python
id_info_df_=pd.read_pickle('./results/DUC_info_drive_29_0071_vanishing_gpd.pkl')
```

计算信息熵和均衡度，对二者的解释可以参看对应章节。计算的内容是图像中各个对象的频数或百分比。


```python
def entroy_df_row(row):
    import numpy as np
    import math
    '''
    function - 计算DataFrame每行的信息熵，用于df.apply(lambda)
    '''
    labels_percent=row[labels].to_numpy()*1.000/aa.iloc[[0]][["pixels"]].to_numpy()
    labels_percent=labels_percent[labels_percent!=0]
    entropy=-np.sum(labels_percent*np.log(labels_percent.astype(np.float)))
    max_entropy=math.log(len(labels))
    frank_e=entropy/max_entropy
    
    return frank_e
id_info_df_['equilibrium']=id_info_df_.apply(lambda row :entroy_df_row(row), axis=1 )    
id_info_df_
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>road</th>
      <th>sidewalk</th>
      <th>building</th>
      <th>wall</th>
      <th>fence</th>
      <th>pole</th>
      <th>traffic_light</th>
      <th>traffic_sign</th>
      <th>vegetation</th>
      <th>terrain</th>
      <th>...</th>
      <th>lat</th>
      <th>lon</th>
      <th>timestamps_</th>
      <th>pixels</th>
      <th>vegetation_percent</th>
      <th>start_idx</th>
      <th>end_idx</th>
      <th>geometry</th>
      <th>length</th>
      <th>equilibrium</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>88520</td>
      <td>9010</td>
      <td>317546</td>
      <td>2399</td>
      <td>135</td>
      <td>1818</td>
      <td>0</td>
      <td>236</td>
      <td>1640</td>
      <td>0</td>
      <td>...</td>
      <td>49.008645</td>
      <td>8.398104</td>
      <td>2011-09-29 13:54:59.990872576</td>
      <td>463012</td>
      <td>0.354202</td>
      <td>0.0</td>
      <td>91.0</td>
      <td>LINESTRING (934872.661 6276328.368, 934872.550...</td>
      <td>26.590485</td>
      <td>0.350934</td>
    </tr>
    <tr>
      <th>1</th>
      <td>88991</td>
      <td>9251</td>
      <td>321157</td>
      <td>2638</td>
      <td>637</td>
      <td>1673</td>
      <td>0</td>
      <td>118</td>
      <td>1118</td>
      <td>0</td>
      <td>...</td>
      <td>49.008645</td>
      <td>8.398103</td>
      <td>2011-09-29 13:55:00.094612992</td>
      <td>463012</td>
      <td>0.241462</td>
      <td>1.0</td>
      <td>81.0</td>
      <td>LINESTRING (934872.550 6276328.441, 934872.449...</td>
      <td>25.073412</td>
      <td>0.346034</td>
    </tr>
    <tr>
      <th>2</th>
      <td>89471</td>
      <td>11162</td>
      <td>324130</td>
      <td>2735</td>
      <td>75</td>
      <td>1046</td>
      <td>0</td>
      <td>200</td>
      <td>837</td>
      <td>0</td>
      <td>...</td>
      <td>49.008646</td>
      <td>8.398102</td>
      <td>2011-09-29 13:55:00.198486528</td>
      <td>463012</td>
      <td>0.180773</td>
      <td>2.0</td>
      <td>81.0</td>
      <td>LINESTRING (934872.449 6276328.505, 934872.346...</td>
      <td>24.953489</td>
      <td>0.336999</td>
    </tr>
    <tr>
      <th>3</th>
      <td>91988</td>
      <td>9927</td>
      <td>314206</td>
      <td>2806</td>
      <td>12524</td>
      <td>1131</td>
      <td>0</td>
      <td>136</td>
      <td>969</td>
      <td>0</td>
      <td>...</td>
      <td>49.008646</td>
      <td>8.398101</td>
      <td>2011-09-29 13:55:00.302340864</td>
      <td>463012</td>
      <td>0.209282</td>
      <td>3.0</td>
      <td>81.0</td>
      <td>LINESTRING (934872.346 6276328.570, 934872.232...</td>
      <td>24.832104</td>
      <td>0.367334</td>
    </tr>
    <tr>
      <th>4</th>
      <td>94986</td>
      <td>13557</td>
      <td>310776</td>
      <td>1036</td>
      <td>16818</td>
      <td>1668</td>
      <td>0</td>
      <td>103</td>
      <td>887</td>
      <td>0</td>
      <td>...</td>
      <td>49.008646</td>
      <td>8.398100</td>
      <td>2011-09-29 13:55:00.406079232</td>
      <td>463012</td>
      <td>0.191572</td>
      <td>4.0</td>
      <td>82.0</td>
      <td>LINESTRING (934872.232 6276328.643, 934872.132...</td>
      <td>24.815814</td>
      <td>0.368164</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1054</th>
      <td>121743</td>
      <td>15678</td>
      <td>252994</td>
      <td>1281</td>
      <td>52565</td>
      <td>1562</td>
      <td>58</td>
      <td>935</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>49.009498</td>
      <td>8.395251</td>
      <td>2011-09-29 13:56:49.458599424</td>
      <td>463012</td>
      <td>0.000216</td>
      <td>1054.0</td>
      <td>1054.0</td>
      <td>GEOMETRYCOLLECTION EMPTY</td>
      <td>0.000000</td>
      <td>0.419064</td>
    </tr>
    <tr>
      <th>1055</th>
      <td>125205</td>
      <td>12000</td>
      <td>247483</td>
      <td>4974</td>
      <td>52823</td>
      <td>1994</td>
      <td>90</td>
      <td>971</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>49.009498</td>
      <td>8.395250</td>
      <td>2011-09-29 13:56:49.562463744</td>
      <td>463012</td>
      <td>0.000000</td>
      <td>1055.0</td>
      <td>1055.0</td>
      <td>GEOMETRYCOLLECTION EMPTY</td>
      <td>0.000000</td>
      <td>0.430710</td>
    </tr>
    <tr>
      <th>1056</th>
      <td>120627</td>
      <td>16324</td>
      <td>247972</td>
      <td>4548</td>
      <td>53390</td>
      <td>2144</td>
      <td>134</td>
      <td>972</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>49.009497</td>
      <td>8.395250</td>
      <td>2011-09-29 13:56:49.666327808</td>
      <td>463012</td>
      <td>0.000000</td>
      <td>1056.0</td>
      <td>1056.0</td>
      <td>GEOMETRYCOLLECTION EMPTY</td>
      <td>0.000000</td>
      <td>0.435620</td>
    </tr>
    <tr>
      <th>1057</th>
      <td>120940</td>
      <td>15959</td>
      <td>247862</td>
      <td>4554</td>
      <td>53147</td>
      <td>2140</td>
      <td>99</td>
      <td>897</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>49.009498</td>
      <td>8.395249</td>
      <td>2011-09-29 13:56:49.770316544</td>
      <td>463012</td>
      <td>0.000000</td>
      <td>1057.0</td>
      <td>1057.0</td>
      <td>GEOMETRYCOLLECTION EMPTY</td>
      <td>0.000000</td>
      <td>0.436820</td>
    </tr>
    <tr>
      <th>1058</th>
      <td>117905</td>
      <td>18114</td>
      <td>247659</td>
      <td>6028</td>
      <td>52716</td>
      <td>2294</td>
      <td>180</td>
      <td>839</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>49.009498</td>
      <td>8.395248</td>
      <td>2011-09-29 13:56:49.874179584</td>
      <td>463012</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>NaN</td>
      <td>0.442050</td>
    </tr>
  </tbody>
</table>
<p>1059 rows × 31 columns</p>
</div>



打印均衡度的曲线分布。


```python
import plotly.express as px
labels=list(trainID_label_mapping.values())
fig = px.line(id_info_df_, x="trainID", y='equilibrium',
              hover_data=['confidence','lat','lon','trainID'],
              title='equilibrium'
             )
fig.show()
```

<a href=""><img src="./imgs/44_11.png" height="auto" width="auto" title="caDesign"></a>


打印均衡度的空间核密度。


```python
import plotly.express as px
fig = px.density_mapbox(id_info_df_, lat='lat', lon='lon', z='equilibrium', radius=10,
                        center=dict(lat=49.008645, lon=8.398104), zoom=18,
                        mapbox_style="stamen-terrain",
                        hover_data=['confidence','lat','lon','trainID'],
                        title=r'equilibrium Kernel Density')
fig.show()
```

<a href=""><img src="./imgs/44_12.png" height="auto" width="auto" title="caDesign"></a>

计算视觉感知消失距离对应图像之间的均衡度变化。


```python
id_info_df_.dropna(inplace=True)
id_info_df_["VP_equilibrium"]=id_info_df_.apply(lambda row: id_info_df_.iloc[[int(row.end_idx)]].equilibrium.values[0]-row.equilibrium,axis=1)

import plotly.express as px
fig = px.density_mapbox(id_info_df_, lat='lat', lon='lon', z='VP_equilibrium', radius=10,
                        center=dict(lat=49.008645, lon=8.398104), zoom=18,
                        mapbox_style="stamen-terrain",
                        hover_data=['confidence','lat','lon','trainID'],
                        title=r'VP_equilibrium Kernel Density')
fig.show()
```

<a href=""><img src="./imgs/44_13.png" height="auto" width="auto" title="caDesign"></a>


### 1.4 要点
#### 1.4.1 数据处理技术

* 开放神经网络交换-ONNX

* netron 网络可视化工具

* Cityscapes数据集

* 知识点-01：super().__init__()——继承父类的init方法

* 知识点-02：__getattr__,__setattr__,__delattr__

* 知识点-03：mutable(可变)与immutable(不可变)

* 知识点-04： _variable,__variable,及__variable__

* 参数管理

* 知识点-05：collections.namedtuple

* cityscapes的标签数据处理

* 知识点-06： importlib 与getattr

#### 1.4.2 新建立的函数

* class - 使用类的属性存储参数，并配置参数immutable与mutable互相转换的方法，迁移'https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/collections.py'.  `AttrDict(dict)`

* function - DUC图像分割，及预测图像保存. `sementicSeg_DUC_pred(DUC_output_root,img_fps,preprocess,predict)`

* function - 读取DUC语义分割结果保存的'KITTI_DUC_confi.pkl'和'KITTI_DUC_raw.pkl'文件，用于位置图像对象/语义分割类别频数统计. `DUC_pred_frequency_moment(KITTI_DUC_raw_fp,KITTI_DUC_confi_fp)`

* function - 计算DataFrame每行的信息熵，用于df.apply(lambda). `entroy_df_row(row)`

#### 1.4.3 所调用的库


```python
import torch
import torchvision
import onnx
import argparse
import pickle
import os.path as path
from collections import namedtuple

import mxnet as mx
import cv2 as cv
import numpy as np
import os
from PIL import Image
import math
from collections import namedtuple
from mxnet.contrib.onnx import import_model
import cityscapes_labels

import glob
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from IPython.display import HTML

import plotly.express as px
import pandas as pd
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-182-ad4f6b8f0ce8> in <module>
          1 import torch
          2 import torchvision
    ----> 3 import onnx
          4 import argparse
          5 import pickle
    

    ModuleNotFoundError: No module named 'onnx'


#### 1.4.4 参考文献
为文中超链接
