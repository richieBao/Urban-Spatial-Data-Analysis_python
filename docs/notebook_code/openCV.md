> Created on Wed Feb 7 02\35\31 2018 @author: Richie Bao-caDesign设计(cadesign.cn)__+updated on Fri Dec 11 16\11\14 2020 by Richie Bao

## 1. openCV-计算机视觉，特征提取，尺度空间（scale space），动态街景视觉感知
设计行业是对城市和自然的规划与设计，是在平衡可见事物（建筑、道路、植被、车辆...）与不可见事物(能量、感知、关系、结构...)。规划设计就是理解不可见来规划可见事物，而可见事物则影射着城市的结构、肌理、物质信息流、生活方式、城市感知、自然自身运行的发展等。而图像（照片，影像）是对可见事物的捕捉，是否可以从图像的信息中扑捉到不可见事物呢？目前计算机视觉的发展，可以解构图像信息，这为规划设计行业注入了又一新的技术手段。为方便计算机视觉的理解，可以粗略分为两个层级，一个是基于openCV库提供的各类算法对图像的解析，计算图像处理，图像特征提取，对象检测等；二是应用深度学习实现图像的语义分割、图像特征迁移、对抗生成等。这些方法对规划行业都有着重要的影响，百度、Google的街景图像，以及无人驾驶项目带来的大量序列图像（连接有GPS，惯性测量单元IMU,Inertial Measurement Unit等传感器信息数据），以及社交网络的图像，图像也通常包括GPS信息，都推动着计算机视觉在规划领域潜在的应用前景。

[OpenCV](https://opencv.org/)是计算机视觉开源库，包括有数百种计算机视觉算法。对应的OpenCV-Python包包括其核心功能、图像处理、视频分析、相机标定与三维重建、对象检测和特征提取等。

### 1.1 OpenCV读取图像与图像处理示例
OpenCV的具体内容可以参考其官方网址，其包括针对[python的应用部分](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)。图像处理部分，下述代码仅是示例了图像的读取和图像模糊、边缘检测、角点检测以及建立图像金字塔显示等内容，粗略了解图像处理的一些基本内容。对于该部分也可以结合到卷积部分的知识阐述。其中建立图像金字塔显示对于应用GIS等平台处理地理信息图像时并不陌生，大数据量的遥感影像显示并不是一次加载所有单元数据，而是根据所需要的显示效果，建立一系列不同分辨率的图像，这些图像叠合在一起，最高分辨率的位于底层，最低的则位于顶部，类似金字塔，因此图像的集合被称为图像金字塔，从而提示了显示的速度。

角点检测时，为了能够清晰显示主要的角点内容，通过放大菱角标记和定义显示阈值实现。注意角点检测在官方的归类中是将其纳入到特征提取的部分。


```python
import cv2 as cv
import sys

img_1_fp=r'./data/cv_IMG_0405.JPG'
img_2_fp=r'./data/cv_IMG_0407.JPG'

img_1=cv.imread(cv.samples.findFile(img_1_fp))
img_2=cv.imread(cv.samples.findFile(img_2_fp))

if img_1 is None:
    sys.exit("could not read the image")
cv.imshow("display window", img_1) #图像将在一个新打开的窗口中显示
k=cv.waitKey(0) 
```

使用`cv.imshow`会打开一个新的窗口显示图像内容。因为`cv.imread`读取图像为数组格式，因此可以使用matplotlib库打印，只是需要注意，OpenCV读取的图像为BGR，需要将其转换为RGB，否则显示的颜色会不正确。


```python
import matplotlib.pyplot as plt
import numpy as np
import copy
fig, axs=plt.subplots(ncols=2,nrows=3,figsize =(30, 15))

#01-原始图像
img_2_RGB=cv.cvtColor(img_2,cv.COLOR_BGR2RGB) #注意OpenCV读取图像，其格式为BGR，需要将其调整为RGB再用matplotlib库打印
axs[0][0].imshow(img_2_RGB)
axs[0][0].set_title("original image-shape=(684, 1920, 3)")

#02-averaging 均值模糊
axs[0][1].imshow(cv.blur(img_2_RGB,(15,15)))
axs[0][1].set_title("averaging blure")

#03-Canny边缘检测
axs[1][0].imshow(cv.Canny(img_2_RGB,200,300))
axs[1][0].set_title("Canny edge detection")

#04-Harris角点检测
img_2_gray=np.float32(cv.cvtColor(img_2,cv.COLOR_BGR2GRAY)) #将图像转换为灰度，为每一像素位置1个值，可理解为图像的强度(颜色，易受光照影响，难以提供关键信息，故将图像进行灰度化，同时也可以加快特征提取的速度。)，并强制转换为浮点值，用于棱角检测。
img_2_harris=cv.cornerHarris(img_2_gray,7,5,0.04) #哈里斯角检测器
img_2_harris_dilate=cv.dilate(img_2_harris,np.ones((1,1))) #放大棱角标记
img_2_copy=copy.deepcopy(img_2_RGB)
img_2_copy[img_2_harris_dilate>0.01*img_2_harris_dilate.max()]=[40,75,236]#定义阈值，显示重要的棱
axs[1][1].imshow(img_2_copy)
axs[1][1].set_title("Harris corner detector")

#05-建立图像显示金字塔-pyrDown-1次
axs[2][0].imshow(cv.pyrDown(img_2_RGB))
axs[2][0].set_title("image pyramids-shape=(342, 960, 3)")

#06-建立图像显示金字塔-pyrDown-3次-pyrUp-1次
axs[2][1].imshow(cv.pyrUp(cv.pyrDown(cv.pyrDown(cv.pyrDown(img_2_RGB))))) 
axs[2][1].set_title("image pyramids-shape=(172, 480, 3)")

plt.show()
```


    
<a href=""><img src="./imgs/17_05.png" height="auto" width="auto" title="caDesign"></a>
    


### 1.2 图像特征提取-尺度不变特征转换SIFT (Scale-Invariant Feature Transform)，Star特征检测器，图像匹配

#### 1.2.1 SIFT算法关键点阐述

* A 高斯分布与高斯模糊（高斯卷积核）

高斯分布即正态分布，可以查看正态分布与概率密度函数部分，其概率密度函数为：$f(x)= \frac{1}{ \sigma  \times  \sqrt{2 \pi } }  \times  e^{- \frac{1}{2}  [ \frac{x- \mu }{  \sigma  } ]^{2} } $，$\sigma$为标准差；$\mu$为平均值；$e$为自然对数的底。因为使用二维高斯分布作为卷积核，中心点对应着原点，因此$\mu=0$，公式可以简化为：$f(x)= \frac{1}{ \sigma \sqrt{2 \pi } } e^{ \frac{- x^{2} }{2  \sigma ^{2} } } $，对应的2维高斯函数为：$G(x,y)= \frac{1}{2 \pi    \sigma ^{2}  } e^{ \frac{- (x^{2} + y^{2} )}{2  \sigma ^{2} } } $。代码中的计算直接调用acipy库，分布使用norm和 multivariate_normal方法实现1维和2维高斯函数的计算。从打印的结果查看2维的高斯分布，以中心点为最大值，向四周逐渐降低，其分布的幅度由$\sigma$控制，$\sigma$较小时，中心点最高，曲线(曲面)较陡；反之，$\sigma$值较大时，中心点较低。曲线（曲面）趋于缓和。


高斯模糊（Gaussian blur）是以二维高斯分布为卷积核（权重值）对图像（或二维数组）进行卷积操作，即卷积计算为每个像素值与周围相邻像素值的加权平均。原始像素值对应着卷积核的中心，具有最大的权重值，相邻像素随距离中心点的距离增加而降低。高斯卷积核可以在模糊的同时，较好的保留图像的边缘效果。高斯卷积核做归一化处理，确保在[0,1]区间。例如下述代码中假设了4种典型的矩阵（数组）类型，均匀型、凸性、凹型和偏斜型，以及假设高斯权重卷积核，通过计算卷积（权重加权平均）得到结果为，均匀型值为100、凸性值为138.46、凹型值为61.54，偏斜型值为107.69。均匀性为衡量的标准，可见凹凸型都较大的偏离均匀型值，而偏斜型则靠近均匀型，因此通过这个典型的案例，可以更为清楚的理解到高斯卷积核可以计算目标像素与周边像素值之间的差异程度。


> 参考：
>  D.Lowe.Distinctive Image Features from Scale-Invariant Keypoints[J], 2004. SIFT提出者


```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(20,9))
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122,projection='3d')

#A-一维高斯分布（即正态分布）
x=np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
for scale in np.arange(1,5,0.3):
    ax1.plot(x, norm.pdf(x,loc=0,scale=scale),'-', lw=3, alpha=0.6, label='Sigma=%.2f'%scale) #位置loc（平均值）和比例scale（标准差）参数
    
#B-二维高斯分布
from scipy.stats import multivariate_normal
from matplotlib import cm
x_, y=np.mgrid[-1.0:1.0:30j, -1.0:1.0:30j] #x,y=np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
xy=np.column_stack([x_.flat, y.flat])

for scale in np.arange(1,5,0.3):
    mu=np.array([0.0, 0.0])
    sigma=np.array([scale, scale])
    covariance=np.diag(sigma**2)
    z=multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    z=z.reshape(x_.shape)
    ax2.plot_surface(x_,y,z,cmap=cm.coolwarm,alpha=0.4)

ax1.legend(loc='upper left', frameon=False)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.view_init(20,50)
plt.show()
```


    
<a href=""><img src="./imgs/17_06.png" height="auto" width="auto" title="caDesign"></a>
    



```python
matrix_even=np.array([[100,100,100],[100,100,100],[100,100,100]]) #均匀型
matrix_convex=np.array([[100,100,100],[100,200,100],[100,100,100]]) #凸型
matrix_concave=np.array([[100,100,100],[100,0,100],[100,100,100]]) #凹型
matrix_skew=np.array([[200,100,100],[100,100,100],[100,100,100]]) #偏斜型

kernal=np.array([[0.1,0.1,0.1],[0.1,0.5,0.1],[0.1,0.1,0.1]])
even_weightedAverage=np.average(matrix_even,weights=kernal)
convex_weightedAverage=np.average(matrix_convex,weights=kernal)
concave_weightedAverage=np.average(matrix_concave,weights=kernal)
skew_weightedAverage=np.average(matrix_skew,weights=kernal)

print("even_WA=%.2f,convex_WA=%.2f,concave_WA=%.2f,skew_WA=%.2f"%(even_weightedAverage,convex_weightedAverage,concave_weightedAverage,skew_weightedAverage))
```

    even_WA=100.00,convex_WA=138.46,concave_WA=61.54,skew_WA=107.69
    

OpenCV可以增加滑动条，实时调整数据，与显示更新后的图像，方便观察参数变化对计算结果的影响。下述实现了给定卷积核大小和$\sigma$两个参数，应用OpenCV库提供的`GaussianBlur`函数实现图像高斯模糊的过程。

<a href=""><img src="./imgs/17_01.png" height='auto' width=1000 title="caDesign"></a>

```python
def Gaussion_blur(img_fp):
    import cv2 as cv    
    '''
    function - 应用OpenCV计算高斯模糊，并给定滑动条调节参数
    '''
    #回调函数
    def Gaussian_blur_size(GBlur_size): #高斯核(卷积核大小)，值越大，图像越模糊
        global KSIZE 
        KSIZE = GBlur_size * 2 +3
        print("changes in kernel size:",KSIZE, SIGMA)
        dst = cv.GaussianBlur(img, (KSIZE,KSIZE), SIGMA, KSIZE) 
        cv.imshow(window_name,dst)

    def Gaussian_blur_Sigma(GBlur_sigma): #σ(sigma)设置，值越大，图像越模糊
        global SIGMA
        SIGMA = GBlur_sigma/10.0
        print ("changes in sigma:",KSIZE, SIGMA)
        dst = cv.GaussianBlur(img, (KSIZE,KSIZE), SIGMA, KSIZE) 
        cv.imshow(window_name,dst)

    #全局变量
    GBlur_size = 1
    GBlur_sigma = 15
    
    global KSIZE 
    global SIGMA
    KSIZE = 1
    SIGMA = 15
    max_value = 300
    max_type = 6
    window_name = "Gaussian Blur"
    trackbar_size = "Size*2+3"
    trackbar_sigema = "Sigma/10"

    #读入图片，模式为灰度图，创建窗口
    img= cv.imread(img_fp,0)
    cv.namedWindow(window_name)
   
    #创建滑动条
    cv.createTrackbar( trackbar_size, window_name,GBlur_size, max_type,Gaussian_blur_size)
    cv.createTrackbar( trackbar_sigema, window_name,GBlur_sigma, max_value, Gaussian_blur_Sigma)    
    
    #初始化
    Gaussian_blur_size(GBlur_size)
    Gaussian_blur_Sigma(GBlur_sigma)        

    if cv.waitKey(0) == 27:  
        cv.destroyAllWindows()   

img_2_fp=r'./data/cv_IMG_0407.JPG'
Gaussion_blur(img_2_fp)
```

    changes in kernel size: 5 15
    changes in sigma: 5 1.5
    changes in kernel size: 7 1.5
    changes in sigma: 7 7.5
    changes in sigma: 7 1.5
    changes in sigma: 7 7.5
    changes in sigma: 7 13.5
    

* B 尺度空间（scale space）与高斯差分金字塔

从高斯分布（函数）到高斯模糊可知，在固定卷积核大小时（可参看卷积部分），不同的$\sigma$值，因为产生权重值的分布陡峭程度不同，图像的模糊程度会对应变化。更好理解为基于变化权重值的图像拉伸，从图像整体上来讲就是各个像素值与各周边像素值的差异程度。而$\sigma$值的变化带来差异程度分布即加权均值分布的差异，这个差异变化是固定卷积核大小下尺度空间的纵深向空间。当$\sigma$设置值较大时，差异程度分布趋向于均值，反映图像的概貌特征；设置较小时，差异程度分布明显，强化了目标像素与周边像素的差异性，对应于图像的细节特征。如果对图像连续降采样，每一个降采样都是前一图像的1/4，即长宽分别缩小一倍，这一连续降采样获得的多个图像就是尺度空间的水平向空间。降采样类似于遥感影响的空间分辨率，如果分辨率为1m，即遥感影像的每一单元(cell)大小为1m，可以分辨出建筑、道路、车辆，甚至行人的信息，但是如果降采样到空间分辨率为30m时，则较大的建筑，农田，林地可以分辨出，但是小于30m的建筑，车辆和行人等信息是不能分辨的。降采样的水平向空间实际上反映了不同大小地物存在的尺度，或理解为不同的尺度反映不同大小的地物信息，结合到图像上，即为图像的不同内容存在不同的水平向尺度。将上述阐述对应到Lowe的论文明确其相应的定义，可以确定降采样的水平向空间为八度（octave）的定义，每一个八度即为连续降采样的图像之一，可以称之为组。而对每一降采样（octave）图像进行高斯卷积核计算，可以设置连续的多个$\sigma$值，这些不同参数下的卷积结果即为每一八度的层。组（降采样-八度，octave)反映图像内容（地物）的水平向尺度（空间分辨率），与层（不同$\sigma$值高斯核卷积）反映差异程度的纵深空间，构成了高斯金字塔的尺度空间，这样）。为方便理解，引用*Distinctive Image Features from Scale-Invariant Keypoints*中的图，具体也可以搜索查看该论文：

<a href=""><img src="./imgs/17_02.png" height='auto' width=500 title="caDesign"></a>
    
D.Lowe将图像的尺度空间定义为$L(x,y, \sigma )=G(x,y, \sigma )*I(x,y)$，其中$G(x,y, \sigma )$为变化尺度的高斯函数(2维)，$I(x,y)$为原图像（像素位置），$*$表示卷积运算。
    
为在连续的尺度空间下检测到特征点，建立高斯差分金字塔(difference-of-Gaussian ,DOG)，其高斯差分算子为$D(x,y, \sigma )=(G(x,y,k \sigma )-G(x,y, \sigma ))*I(x,y)=L(x,y,k \sigma )-L(x,y, \sigma )$，即在左图每一组（octave）下每相邻的两层图像相减得到右侧的图像。层之间的差值可以理解为相邻$\sigma$值的变化反映目标像素与周围像素差异程度表现在层之间的变化程度。而层间较大的程度变化代表着尺度空间纵深向目标像素点与周边像素点差异程度表现最为突出所在的层，即该层反映了目标像素的特征。如果同时考虑到水平向空间，假设5m的空间分辨率为车辆的水平向尺度空间的界定边界（一个组），即如果大于5m，车辆信息被淹没在更大尺度的地物中，例如车辆被分配到绿地空间里。要想在5m的水平向尺度下，将车辆提取出来，必须比较车辆所在像素与周边像素的差异，如果差异程度较高则可以提取，但是必须寻找到体现差异程度最大的纵深向尺度空间的层。
    
* C 空间关键点（极值点）检测
    
已获得DOG，为寻找极值点，需要比较中间的检测点与其同纵深向尺度空间的8个相邻点和上下相邻纵深向尺度层的18个点，共26个点进行比较。每组中的纵深向尺度空间，设置连续的$\sigma$值，可以产生无数连续的层，对于一个目标像素而言，就是需要不断比较连续的DOG中的相邻层，从而获得无数个离散的局部的极值点，离散的空间极值点并不是真正的极值点，因此可以对尺度空间DOG函数通过泰勒展开式（具体可参看微积分部分的泰勒展开式）拟合函数（二次方程）为：$D(X)=D+ \frac{ \partial  D^{ T } }{ \partial X}X+ \frac{1}{2}   X^{ T }  \frac{  \partial ^{2}D }{ \partial  X^{2} } X$，其中$X= (x,y, \sigma )^{ T } $。对$X$求导，并令方程等于0，求得极值点(关键点)的位置$\widehat{X} = \frac{  \partial^{2}  D^{-1}  }{\partial X^{2} }  \frac{\partial D}{\partial X} $。
    
<a href=""><img src="./imgs/17_03.png" height='auto' width=250 title="caDesign"></a>
    
* D 关键点方向(用一组向量描述关键点<关键点及其邻域像素点>，特征向量)

计算像素点的幅度（梯度大小，gradient magnitude）$m(x,y)$和方向(orientation)$\theta (x,y)$，使用像素的位置差异预先计算，例如像素的位置标识：$\begin{bmatrix}(-1,1) & (0,1)&(1,1) \\(-1,0) & (0,0)&(1,0) \\(-1,-1)&(0,-1)&(1,-1))\end{bmatrix} $，以关键点为中心，以$3 \times 1.5 \sigma $为半径的同一纵向尺度空间的邻域内计算，$m(x,y)= \sqrt{ (L(x+1,y)-L(x-1,y)^{2} +(L(x,y+1)-L(x,y-1)^{2}} $，$\theta (x,y)= tan^{-1} \frac{L(x,y+1)-L(x,y-1)}{L(x+1,y)-L(x-1,y)}  $，$L$为关键点所在的尺度空间值，即对应的高斯模糊图像。完成关键点的梯度大小和方向计算后，统计频数，将360度的方向划分为36个等分(bin)，频数最大值代表关键点的主方向。

<a href=""><img src="./imgs/17_04.png" height='auto' width=500 title="caDesign"></a>
    
在关键点位置周围区域内（指定半径的圆形区域，或关键点周围$16 \times 16$个像素点），首先计算每个图像样本点的梯度大小和方向（高斯卷积核加权），创建关键点描述子（符）。然后这些样本累积成$4 \times 4$次区域内方向直方图（每个关键点有16个子块/sub-block，即有16个直方图，每个块为$4 \times 4$的像素点，上图仅以$8 \times 8$邻域为例），每个箭头的长度对应区域内该方向附近的梯度大小之和，将所有方向特征相邻连接起来，够成$16 \times 8=128$维的特征向量。至此，每个关键点有三个信息$(x,y,\sigma,\theta,magnitude,region)$，即位置，纵深向空间尺度、方向和大小，及邻域。
    
使用OpenCV的`SIFT_create()`方法实现尺度不变特征转换，所返回的关键点信息对应位置点坐标、邻域（直径），水平向空间尺度（即octave）、梯度大小（响应程度）和方向，对图像分类（-1为未设置）。OpenCV的SIFT计算配置邻域大小同样为$16 \times 16$，划分为$4 \times 4$大小的16个子块，总共128份(bin)。


```python
def SIFT_detection(img_fp,save=False):
    import cv2 as cv
    import numpy as np
    import matplotlib.pyplot as plt
    
    '''
    function - SIFT(scale invariant feature transform) 尺度不变特征变换)特征点检测
    
    Paras:
    img_fp - 图像文件路径    
    '''
    img=cv.imread(img_fp)
    img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift=cv.SIFT_create() #SIFT特征实例化 cv.xfeatures2d.SIFT_create()
    key_points=sift.detect(img_gray,None)  #提取SIFT特征关键点detector
    
    #示例打印关键点数据
    for k in key_points[:5]:
        print("关键点点坐标:%s,直径:%.3f,金字塔层:%d,响应程度:%.3f,分类:%d,方向:%d"%(k.pt,k.size,k.octave,k.response,k.class_id,k.angle))
        """
        关键点信息包含：
        k.pt关键点点的坐标(图像像素位置)
        k.size该点范围的大小（直径）
        k.octave从高斯金字塔的哪一层提取得到的数据
        k.response响应程度，代表该点强壮大小，即角点的程度。角点：极值点，某方面属性特别突出的点(最大或最小)。
        k.class_id对图像进行分类时，可以用class_id对每个特征点进行区分，未设置时为-1
        k.angle角度，关键点的方向。SIFT算法通过对邻域做梯度运算，求得该方向。-1为初始值        
        """
    print("_"*50) 
    descriptor=sift.compute(img_gray,key_points) #提取SIFT调整描述子-descriptor，返回的列表长度为2，第1组数据为关键点，第2组数据为描述子(关键点周围对其有贡献的像素点)
    print("key_points数据类型:%s,descriptor数据类型:%s"%(type(key_points),type(descriptor)))
    print("关键点：")
    print(descriptor[0][:1]) #关键点
    print("描述子：")
    print(descriptor[1][:1]) #描述子
    print("描述子 shape:",descriptor[1].shape)      
    
    cv.drawKeypoints(img,key_points,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
    
    if save:
        cv.imshow('sift features',img)
        cv.imwrite('./data/sift_features.jpg',img) #保存图像
        cv.waitKey()
    else:        
        fig, ax=plt.subplots(figsize=(30,15))
        ax.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB) )
        plt.show()   

SIFT_detection(img_fp=img_2_fp)    
```

    关键点点坐标:(2.318413734436035, 498.3979187011719),直径:1.818,金字塔层:852479,响应程度:0.019,分类:-1,方向:300
    关键点点坐标:(2.7633209228515625, 405.3725891113281),直径:2.134,金字塔层:12452351,响应程度:0.015,分类:-1,方向:2
    关键点点坐标:(2.7633209228515625, 405.3725891113281),直径:2.134,金字塔层:12452351,响应程度:0.015,分类:-1,方向:155
    关键点点坐标:(2.925368547439575, 662.67041015625),直径:1.981,金字塔层:7078399,响应程度:0.017,分类:-1,方向:257
    关键点点坐标:(3.029824733734131, 606.0113525390625),直径:2.157,金字塔层:13238783,响应程度:0.042,分类:-1,方向:61
    __________________________________________________
    key_points数据类型:<class 'list'>,descriptor数据类型:<class 'tuple'>
    关键点：
    [<KeyPoint 0000021CED051FC0>]
    描述子：
    [[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   4.
       14.   3.   8.   0.   0.   0.   0.   9.  43.  48.  30.   2.   0.   0.
        0.  14.  32.  24.  12.  11.   0.   0.   8.  23.   2.   1. 129.  39.
        0.   0.   3.  21.  81.  43.  24.   0.   1.   2.  66.  65. 129.  74.
        8.   5.  23.   7.  27.  20.  33.  23.  42.  30.  15.  13.  10.  55.
       37.  23. 129.  18.   1.   6.   7.  26.  55. 129.  56.   6.   4. 121.
      129.  28.   8.  14.   1.   6.  93. 106.  60.   2.   0.   1. 129.  91.
       20.  15.   2.   1.  12.  53.  73.   4.   1.   6.  28.  24.  47. 103.
        4.   1.   1. 111. 129.  38.   3.   4.   3.   2.  17.  80.  55.   4.
        0.   1.]]
    描述子 shape: (19810, 128)
    


    
<a href=""><img src="./imgs/17_07.png" height='auto' width='auto' title="caDesign"></a>
    


OpenCV提供有多种类型的特征检测和描述算法，如果SIFT计算速度不过快，则可以尝试使用SURF(Speeded-Up Robust Features)，或者BRIEF(Binary Robust Independent Elementary Feature)算法，如果需要实时性，例如机器人领域里的SLAM(simultaneous localization and mapping)配合使用，则可以应用FAST算法。下述为Star算法实现，每种算法都有各自的优缺点，需要针对分析目标进行比较确定。


```python
def STAR_detection(img_fp,save=False):
    import cv2 as cv
    import numpy as np
    import copy
    import matplotlib.pyplot as plt
    '''
    function - 使用Star特征检测器提取图像特征
    '''
    img=cv.imread(img_fp)
    star=cv.xfeatures2d.StarDetector_create() 
    key_point=star.detect(img)
    cv.drawKeypoints(img,key_point,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if save:
        cv.imshow('star features',img_copy)
        cv.imwrite('./data/star_features.jpg',img) #保存图像
        cv.waitKey()
    else:        
        fig, ax=plt.subplots(figsize=(30,15))
        ax.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB) )
        plt.show()        
    
STAR_detection(img_fp=img_2_fp)
```


    
<a href=""><img src="./imgs/17_08.png" height='auto' width='auto' title="caDesign"></a>
    


#### 1.2.2 特征匹配
通过特征检测器可以获取图像关键点的描述子，那么包含同一事物（具有同样的特征）的两幅图像就可以通过描述子匹配。[OpenCV提供了三种方式](https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html)，一种是基于ORB(Oriented FAST and Rotated BRIEF)描述子的蛮力匹配（Brute-Force Matching with ORB Descriptors），再者为使用SIFT描述子和比率检测蛮力匹配(Brute-Force Matching with SIFT Descriptors and Ratio Test)，以及基于FLANN的匹配器(FLANN based Matcher)。

> 注意有些算法是已申请专利的


```python
def feature_matching(img_1_fp,img_2_fp,index_params=None,method='FLANN'):
    import numpy as np
    import cv2 as cv
    import matplotlib.pyplot as plt    
    '''
    function - OpenCV 根据图像特征匹配图像。迁移官网的三种方法，1-ORB描述子蛮力匹配　Brute-Force Matching with ORB Descriptors；2－使用SIFT描述子和比率检测蛮力匹配 Brute-Force Matching with SIFT Descriptors and Ratio Test; 3-基于FLANN的匹配器 FLANN based Matcher
   
    Paras:
    img_1 - 待匹配图像1路径
    img_2 - 待匹配图像2路径
    method - 参数为:'ORB','SIFT','FLANN'
    '''
    plt.figure(figsize=(30,15))
    img1 = cv.imread(img_1_fp,cv.IMREAD_GRAYSCALE) # queryImage
    img2 = cv.imread(img_2_fp,cv.IMREAD_GRAYSCALE) # trainImage
    if method=='ORB':
        # Initiate ORB detector
        orb = cv.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)
        
        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        # Draw first 10 matches.
        img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3),plt.show()
        
    if method=='SIFT':        
        # Initiate SIFT detector
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good[0:int(1*len(good)):int(0.1*len(good))],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3),plt.show()        
    
    if method=='FLANN':
        # Initiate SIFT detector
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask,
                           flags = cv.DrawMatchesFlags_DEFAULT)
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        plt.imshow(img3,),plt.show()
        
feature_matching(img_1_fp,img_2_fp,method='ORB',)    
```


    
<a href=""><img src="./imgs/17_09.png" height='auto' width='auto' title="caDesign"></a>
    



```python
feature_matching(img_1_fp,img_2_fp,method='SIFT',)  
```


    
<a href=""><img src="./imgs/17_10.png" height='auto' width='auto' title="caDesign"></a>
    



```python
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)        
feature_matching(img_1_fp,img_2_fp,index_params=index_params,method='FLANN',)    
```


    
<a href=""><img src="./imgs/17_11.png" height='auto' width='auto' title="caDesign"></a>
    


### 1.3 KITTI，动态街景视觉感知
上述解释尺度空间是从算法数理逻辑层面进行的解释，如果从感性上来理解，不同尺度的图像，可以模拟人眼观察事物从远及近的过程，这个过程中可以认为尺度空间满足视觉不变性，即图像的分析不受图像灰度、对比度的影响，并满足平移不变性、尺度不变性、欧几里得不变性和仿射不变性。因此可以借助计算机视觉分析技术来分析研究城市街道景观的感知变化。为了保持街道景观分析数据的统一性，使用KITTI数据，这样可以保持图像的连续性，拍摄高度以及视角的不变性，提示数据分析结果的可靠性。

#### 1.3.1 [The KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/index.php)

KITTI数据集是用于无人驾驶场景下计算机视觉算法评测数据集，连续记录有车行路线下的连续城市景观图像，以及GPS定位等信息，在这里并不是用KITTI数据集研究无人驾驶，但是无人驾驶项目的数据却可以为城市规划带来用于研究城市新的数据来源。随着无人驾驶技术的发展，以及实际的推广，该部分的信息量将会成倍的增加。此次实验使用[2011_09_29_drive_0071 (4.1 GB) ](http://www.cvlibs.net/datasets/kitti/raw_data.php)标识数据，为城市内的街巷空间。

查看'oxts/dataformat.txt'文件，可以获知'oxts/data'下文件的数据格式，其信息罗列如下：

| Column1      | Column2                                                                              |
|--------------|--------------------------------------------------------------------------------------|
| lat          |    latitude of the oxts-unit (deg)                                                   |
| lon          |    longitude of the oxts-unit (deg)                                                  |
| alt          |    altitude of the oxts-unit (m)                                                     |
| roll         |   roll angle (rad),    0 = level, positive = left side up,      range: -pi   .. +pi  |
| pitch        |  pitch angle (rad),   0 = level, positive = front down,        range: -pi/2 .. +pi/2 |
| yaw          |    heading (rad),       0 = east,  positive = counter clockwise, range: -pi   .. +pi |
| vn           |     velocity towards north (m/s)                                                     |
| ve           |     velocity towards east (m/s)                                                      |
| vf           |     forward velocity, i.e. parallel to earth-surface (m/s)                           |
| vl           |     leftward velocity, i.e. parallel to earth-surface (m/s)                          |
| vu           |     upward velocity, i.e. perpendicular to earth-surface (m/s)                       |
| ax           |     acceleration in x, i.e. in direction of vehicle front (m/s^2)                    |
| ay           |     acceleration in y, i.e. in direction of vehicle left (m/s^2)                     |
| ay           |     acceleration in z, i.e. in direction of vehicle top (m/s^2)                      |
| af           |     forward acceleration (m/s^2)                                                     |
| al           |     leftward acceleration (m/s^2)                                                    |
| au           |     upward acceleration (m/s^2)                                                      |
| wx           |     angular rate around x (rad/s)                                                    |
| wy           |     angular rate around y (rad/s)                                                    |
| wz           |     angular rate around z (rad/s)                                                    |
| wf           |     angular rate around forward axis (rad/s)                                         |
| wl           |     angular rate around leftward axis (rad/s)                                        |
| wu           |     angular rate around upward axis (rad/s)                                          |
| pos_accuracy |   velocity accuracy (north/east in m)                                                |
| vel_accuracy |   velocity accuracy (north/east in m/s)                                              |
| navstat      |        navigation status (see navstat_to_string)                                     |
| numsats      |        number of satellites tracked by primary GPS receiver                          |
| posmode      |        position mode of primary GPS receiver (see gps_mode_to_string)                |
| velmode      |        velocity mode of primary GPS receiver (see gps_mode_to_string)                |
| orimode      |        orientation mode of primary GPS receiver (see gps_mode_to_string)             |


> @ARTICLE{Geiger2013IJRR,
  author = {Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun},
  title = {Vision meets Robotics: The KITTI Dataset},
  journal = {International Journal of Robotics Research (IJRR)},
  year = {2013}
} 


```python
def KITTI_info(KITTI_info_fp,timestamps_fp):
    import util
    '''
    function - 读取KITTI文件信息，1-包括经纬度，惯性导航系统信息等的.txt文件，2-包含时间戳的.txt文件
    '''

    import pandas as pd
    import util
    import os

    drive_fp=util.filePath_extraction(KITTI_info_fp,['txt'])
    '''展平列表函数'''
    flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]
    drive_fp_list=flatten_lst([[os.path.join(k,f) for f in drive_fp[k]] for k,v in drive_fp.items()])

    columns=["lat","lon","alt","roll","pitch","yaw","vn","ve","vf","vl","vu","ax","ay","ay","af","al","au","wx","wy","wz","wf","wl","wu","pos_accuracy","vel_accuracy","navstat","numsats","posmode","velmode","orimode"]
    drive_info=pd.concat([pd.read_csv(item,delimiter=' ',header=None) for item in drive_fp_list],axis=0)
    drive_info.columns=columns
    drive_info=drive_info.reset_index()
    
    timestamps=pd.read_csv(timestamps_fp,header=None)
    timestamps.columns=['timestamps_']
    drive_info=pd.concat([drive_info,timestamps],axis=1,sort=False)
    #drive_29_0071_info.index=pd.to_datetime(drive_29_0071_info["timestamps_"]) #用时间戳作为行(row)索引
    return drive_info

import util
KITTI_info_fp=r'D:\dataset\KITTI\2011_09_29_drive_0071_sync\oxts\data'
timestamps_fp=r'D:\dataset\KITTI\2011_09_29_drive_0071_sync\image_03\timestamps.txt'
drive_29_0071_info=KITTI_info(KITTI_info_fp,timestamps_fp)
util.print_html(drive_29_0071_info)
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>lat</th>
      <th>lon</th>
      <th>alt</th>
      <th>roll</th>
      <th>pitch</th>
      <th>yaw</th>
      <th>vn</th>
      <th>ve</th>
      <th>vf</th>
      <th>vl</th>
      <th>vu</th>
      <th>ax</th>
      <th>ay</th>
      <th>ay</th>
      <th>af</th>
      <th>al</th>
      <th>au</th>
      <th>wx</th>
      <th>wy</th>
      <th>wz</th>
      <th>wf</th>
      <th>wl</th>
      <th>wu</th>
      <th>pos_accuracy</th>
      <th>vel_accuracy</th>
      <th>navstat</th>
      <th>numsats</th>
      <th>posmode</th>
      <th>velmode</th>
      <th>orimode</th>
      <th>timestamps_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>49.008645</td>
      <td>8.398104</td>
      <td>112.990593</td>
      <td>0.031074</td>
      <td>0.022152</td>
      <td>2.600568</td>
      <td>0.343806</td>
      <td>-0.608220</td>
      <td>0.698425</td>
      <td>-0.018258</td>
      <td>-0.002050</td>
      <td>0.185589</td>
      <td>0.465920</td>
      <td>9.818093</td>
      <td>0.398583</td>
      <td>0.157497</td>
      <td>9.821557</td>
      <td>-0.005760</td>
      <td>0.007926</td>
      <td>-0.006220</td>
      <td>-0.005889</td>
      <td>0.008117</td>
      <td>-0.005841</td>
      <td>0.260931</td>
      <td>0.031113</td>
      <td>4.0</td>
      <td>10.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>2011-09-29 13:54:59.990872576</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>49.008645</td>
      <td>8.398103</td>
      <td>112.982414</td>
      <td>0.029396</td>
      <td>0.023378</td>
      <td>2.600133</td>
      <td>0.360348</td>
      <td>-0.638403</td>
      <td>0.732806</td>
      <td>-0.020087</td>
      <td>-0.005521</td>
      <td>0.013390</td>
      <td>0.220784</td>
      <td>9.751132</td>
      <td>0.235504</td>
      <td>-0.072489</td>
      <td>9.750655</td>
      <td>-0.012038</td>
      <td>0.010221</td>
      <td>-0.002413</td>
      <td>-0.012081</td>
      <td>0.010291</td>
      <td>-0.001830</td>
      <td>0.260931</td>
      <td>0.031113</td>
      <td>4.0</td>
      <td>10.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>2011-09-29 13:55:00.094612992</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>49.008646</td>
      <td>8.398102</td>
      <td>112.977341</td>
      <td>0.028140</td>
      <td>0.024028</td>
      <td>2.600907</td>
      <td>0.385173</td>
      <td>-0.647590</td>
      <td>0.753470</td>
      <td>-0.003373</td>
      <td>-0.008993</td>
      <td>-0.036309</td>
      <td>0.126375</td>
      <td>9.827835</td>
      <td>0.195970</td>
      <td>-0.156881</td>
      <td>9.825555</td>
      <td>-0.012341</td>
      <td>0.006952</td>
      <td>0.007101</td>
      <td>-0.012162</td>
      <td>0.006748</td>
      <td>0.007589</td>
      <td>0.260931</td>
      <td>0.031113</td>
      <td>4.0</td>
      <td>10.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>2011-09-29 13:55:00.198486528</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>49.008646</td>
      <td>8.398101</td>
      <td>112.975266</td>
      <td>0.028846</td>
      <td>0.025269</td>
      <td>2.601397</td>
      <td>0.401536</td>
      <td>-0.665305</td>
      <td>0.777082</td>
      <td>0.002034</td>
      <td>0.003541</td>
      <td>-0.005085</td>
      <td>0.330036</td>
      <td>9.951577</td>
      <td>0.238956</td>
      <td>0.045568</td>
      <td>9.954079</td>
      <td>0.004590</td>
      <td>0.013375</td>
      <td>0.003898</td>
      <td>0.004692</td>
      <td>0.013260</td>
      <td>0.004161</td>
      <td>0.260931</td>
      <td>0.031113</td>
      <td>4.0</td>
      <td>10.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>2011-09-29 13:55:00.302340864</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>49.008646</td>
      <td>8.398100</td>
      <td>112.974319</td>
      <td>0.029025</td>
      <td>0.027787</td>
      <td>2.601377</td>
      <td>0.406576</td>
      <td>-0.681044</td>
      <td>0.793172</td>
      <td>-0.001648</td>
      <td>0.010464</td>
      <td>-0.238262</td>
      <td>0.240819</td>
      <td>9.833897</td>
      <td>0.022946</td>
      <td>-0.043796</td>
      <td>9.839752</td>
      <td>0.001368</td>
      <td>0.021714</td>
      <td>0.001375</td>
      <td>0.001421</td>
      <td>0.021666</td>
      <td>0.001967</td>
      <td>0.260931</td>
      <td>0.031113</td>
      <td>4.0</td>
      <td>10.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>2011-09-29 13:55:00.406079232</td>
    </tr>
  </tbody>
</table>



通过提取的数据，应用Plotly库打印包含地图信息的图表，观察数据在现实世界里具体的位置。


```python
def plotly_scatterMapbox(df,**kwargs):
    import pandas as pd
    import plotly.graph_objects as go
    '''
    function - 使用plotly的go.Scattermapbox方法，在地图上显示点及其连线，坐标为经纬度
    
    Paras:
    df - DataFrame格式数据，含经纬度
    field - 'lon':df的longitude列名，'lat'：为df的latitude列名，'center_lon':地图显示中心精经度定位，"center_lat":地图显示中心维度定位，"zoom"：为地图缩放
    '''
    field={'lon':'lon','lat':'lat',"center_lon":8.398104,"center_lat":49.008645,"zoom":16}
    field.update(kwargs) 
    
    fig=go.Figure(go.Scattermapbox(mode = "markers+lines",lat=df[field['lat']], lon=df[field['lon']],marker = {'size': 10}))  #亦可以选择列，通过size=""配置增加显示信息  
    fig.update_layout(
        margin ={'l':0,'t':0,'b':0,'r':0},
        mapbox = {
            'center': {'lon': 10, 'lat': 10},
            'style': "stamen-terrain",
            'center': {'lon': field['center_lon'], 'lat':field['center_lat']},
            'zoom': 16})    
    fig.show()
    
plotly_scatterMapbox(drive_29_0071_info)    
```

<a href=""><img src="./imgs/17_12.png" height='auto' width='auto' title="caDesign"></a>

读取连续的图像，动态显示观察。


```python
drive_29_0071_img_fp=util.filePath_extraction(r'D:\dataset\KITTI\2011_09_29_drive_0071_sync\image_03\data',['png'])
drive_29_0071_img_fp_list=util.flatten_lst([[os.path.join(k,f) for f in drive_29_0071_img_fp[k]] for k,v in drive_29_0071_img_fp.items()])
```


```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from IPython.display import HTML

fig=plt.figure(figsize=(20,10))
ims=[[plt.imshow(mpimg.imread(f),animated=True)] for f in drive_29_0071_img_fp_list[:200]]
print("finished reading the imgs.")

ani=animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=1000)
#ani.save(r'./imgs/drive_29_0071_imgs.mp4')
#print(".mp4 saved.")
HTML(ani.to_html5_video())
```

    finished reading the imgs.
    
    
<video width='auto' height='auto' controls><source src="./imgs/17_13.mp4" height='auto' width='auto' title="caDesign" type='video/mp4'></video>

#### 1.3.2 动态街景视觉感知
图像匹配可以返回两幅图像中特征点基本相似的关键点，那么对于无人驾驶拍摄的连续影像而言，假设确定一个固定位置，即选择该位置上的一张影像，将其分别与之后的所有影像进行图像匹配，返回关键点，计算每一对的特征点匹配数量。这个特征点匹配数量的变化体现了当前位置的图像与之后顺序影像相似的程度。因为后一帧（后一位置）影像包含前一影像的一部分，当距离越近，两者匹配返回的特征点匹配数量越多；反之，离当前位置越远，匹配的数量越少，而且这个过程基本上是逐渐减少的。因此可以由上述计算推测一条街道视觉感知变化的情况，找到感知消失的距离；同时，也可以比较不同街道感知变化的差异；甚至，可以比较不同街道感知的相似度。

代码实现上主要包括三个部分，第一个是定义批量特征提取及返回匹配点数量的类`dynamicStreetView_visualPerception`；二是匹配点数量是随距离增加而逐渐降低的，需要找到降低后，基本不再变化的位置点，即感知消失的位置，定义类`movingAverage_inflection`;最后计算这个感知消失的距离，定义函数`vanishing_position_length`。

* A - 返回特征匹配点数量 

返回特征点匹配数量的类，是使用Star方法提取关键点，这个方法可以降低直接使用SIFT产生的噪音，然后再用SIFT根据Star提取的关键点来返回描述子，最后利用描述子进行图像匹配，返回特征点匹配的数量。


```python
class dynamicStreetView_visualPerception:
    '''
    class - 应用Star提取图像关键点，结合SIFT获得描述子，根据特征匹配分析特征变化（视觉感知变化），即动态街景视觉感知
    
    Paras:
    imgs_fp - 图像路径列表
    knnMatch_ratio - 图像匹配比例，默认为0.75
    '''
    def __init__(self,imgs_fp,knnMatch_ratio=0.75):
        self.knnMatch_ratio=knnMatch_ratio
        self.imgs_fp=imgs_fp
    
    def kp_descriptor(self,img_fp):
        import cv2 as cv
        '''
        function - 提取关键点和获取描述子
        '''
        img=cv.imread(img_fp)
        star_detector=cv.xfeatures2d.StarDetector_create()        
        key_points=star_detector.detect(img) #应用处理Star特征检测相关函数，返回检测出的特征关键点
        img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) #将图像转为灰度
        kp,des=cv.xfeatures2d.SIFT_create().compute(img_gray, key_points) #SIFT特征提取器提取特征
        return kp,des
        
     
    def feature_matching(self,des_1,des_2,kp_1=None,kp_2=None):
        import cv2 as cv
        '''
        function - 图像匹配
        '''
        bf=cv.BFMatcher()
        matches=bf.knnMatch(des_1,des_2,k=2)
        
        '''
        可以由匹配matches返回关键点（train,query）的位置索引，train图像的索引，及描述子之间的距离
        DMatch.distance - Distance between descriptors. The lower, the better it is.
        DMatch.trainIdx - Index of the descriptor in train descriptors
        DMatch.queryIdx - Index of the descriptor in query descriptors
        DMatch.imgIdx - Index of the train image.
        '''
        '''
        if kp_1 !=None and kp_2 != None:
            kp1_list=[kp_1[mat[0].queryIdx].pt for mat in matches]
            kp2_list=[kp_2[mat[0].trainIdx].pt for mat in matches]
            des_distance=[(mat[0].distance,mat[1].distance) for mat in matches]
            print(des_distance[:5])
        '''
        
        good=[]
        for m,n in matches:
            if m.distance < self.knnMatch_ratio*n.distance:
                good.append(m) 
        #good_num=len(good)
        return good #,good_num
    
    
    def sequence_statistics(self):
        from tqdm import tqdm
        '''
        function - 序列图像匹配计算，每一位置图像与后续所有位置匹配分析
        '''        
        des_list=[]
        print("计算关键点和描述子...")
        for f in tqdm(self.imgs_fp):        
            _,des=self.kp_descriptor(f)
            des_list.append(des)
        matches_sequence={}
        print("计算序列图像匹配数...")
        for i in tqdm(range(len(des_list)-1)):
            matches_temp=[]
            for j_des in des_list[i:]:
                matches_temp.append(self.feature_matching(des_list[i],j_des))
            matches_sequence[i]=matches_temp
        matches_num={k:[len(v) for v in val] for k,val in matches_sequence.items()}
        return matches_num  

dsv_vp=dynamicStreetView_visualPerception(drive_29_0071_img_fp_list) #[:200]
#kp1,des1=dsv_vp.kp_descriptor(drive_29_0071_img_fp_list[10])
#kp2,des2=dsv_vp.kp_descriptor(drive_29_0071_img_fp_list[50])
#dsv_vp.feature_matching(des1,des2,kp1,kp2)
matches_num=dsv_vp.sequence_statistics()
```

      0%|          | 2/1059 [00:00<00:53, 19.64it/s]

    计算关键点和描述子...
    

    100%|██████████| 1059/1059 [01:08<00:00, 15.51it/s]
      0%|          | 0/1058 [00:00<?, ?it/s]

    计算序列图像匹配数...
    

    100%|██████████| 1058/1058 [28:15<00:00,  1.60s/it]
    

* B - 寻找拐点，及跳变稳定的区域（视觉感知变化的位置，及消失的位置）

寻找拐点，即寻找变化最快的位置，在该点之前特征点匹配的数量快速减低，该点之后变化开始缓慢。即可以理解为一开始视觉信息快速的流失，因为此时不同影像重叠的信息明确，但随距离增加，明确重叠的信息在减少，因此特征点数量变化也会变得缓慢。下述以第一张影像为位置点，分析其与之后所有的影像，可以得知其拐点为12，因为位置图索引为0，因此12即为索引为13影像所在的位置。


```python
idx=0   
x=range(idx,idx+len(matches_num[idx])) 
y=matches_num[idx]
util.kneed_lineGraph(x,y)
```

    曲线拐点（凸）： 12
    曲线拐点（凹）： 12
    


    
<a href=""><img src="./imgs/17_14.png" height='auto' width='auto' title="caDesign"></a>
    


上述方法找到的拐点，是曲线变化最快的点。但是需要确定在哪个位置，视觉感知的联系降为最低（即特征点匹配的数量基本不再降低）的位置。这个位置是曲线变化基本为0的位置，就是计算每一点与前一点的差值，差值越小说明变化越小。如果这个差值变化基本保持不变，就说明已经找到了这个位置点，即保证差值的差值与前一差值保持相等。例如如果有一组数据为[20,10,8,1,1,1]，做第一次差值(取绝对值)结果为[10,2,7,0,0]，做差值的差值，结果为[8,5,7,0]，那么如果`diff(x) == diff(diff(x))`，即差值的差值第3个数7，与第一次差值的第3个数7相等，说明该位置之后曲线基本保持水平一致不再变化。同时满足`diff(x) != 0`，如果为0，即为数据相等，此时曲线已经保持基本不变。

下述图表绘制，对曲线做了平滑处理，降低噪声，以便找到跳变基本为0的位置点。同时，给出了置信区间，以及标注的异常点，便于观察数据平滑后的变化。平滑后的曲线尽量要在置信区间内，虽然在曲线降低的区段出现异常点，但是因为不在跳变稳定的区段，所以这个异常点错误对结果并没有影响。


```python
class movingAverage_inflection:
    import pandas as pd
    
    '''
    class - 曲线（数据）平滑，与寻找曲线水平和纵向的斜率变化点
    
    Paras:
    series - pandas 的Series格式数据
    window - 滑动窗口大小，值越大，平滑程度越大
    plot_intervals - 是否打印置信区间，某人为False 
    scale - 偏差比例，默认为1.96, 
    plot_anomalies - 是否打印异常值，默认为False,
    figsize - 打印窗口大小，默认为(15,5),
    threshold - 拐点阈值，默认为0
    '''
    def __init__(self,series, window, plot_intervals=False, scale=1.96, plot_anomalies=False,figsize=(15,5),threshold=0):
        self.series=series
        self.window=window
        self.plot_intervals=plot_intervals
        self.scale=scale
        self.plot_anomalies=plot_anomalies
        self.figsize=figsize
        
        self.threshold=threshold
        self.rolling_mean=self.movingAverage()
    
    def masks(self,vec):
        '''
        function - 寻找曲线水平和纵向的斜率变化，参考 https://stackoverflow.com/questions/47342447/find-locations-on-a-curve-where-the-slope-changes
        '''
        d=np.diff(vec)
        dd=np.diff(d)

        # Mask of locations where graph goes to vertical or horizontal, depending on vec
        to_mask=((d[:-1] != self.threshold) & (d[:-1] == -dd-self.threshold))
        # Mask of locations where graph comes from vertical or horizontal, depending on vec
        from_mask=((d[1:] != self.threshold) & (d[1:] == dd-self.threshold))
        return to_mask, from_mask
        
    def apply_mask(self,mask, x, y):
        return x[1:-1][mask], y[1:-1][mask]   
    
    def knee_elbow(self):
        '''
        function - 返回拐点的起末位置
        '''        
        x_r=np.array(self.rolling_mean.index)
        y_r=np.array(self.rolling_mean)
        to_vert_mask, from_vert_mask=self.masks(x_r)
        to_horiz_mask, from_horiz_mask=self.masks(y_r)     

        to_vert_t, to_vert_v=self.apply_mask(to_vert_mask, x_r, y_r)
        from_vert_t, from_vert_v=self.apply_mask(from_vert_mask, x_r, y_r)
        to_horiz_t, to_horiz_v=self.apply_mask(to_horiz_mask, x_r, y_r)
        from_horiz_t, from_horiz_v=self.apply_mask(from_horiz_mask, x_r, y_r)    
        return x_r,y_r,to_vert_t, to_vert_v,from_vert_t, from_vert_v,to_horiz_t, to_horiz_v,from_horiz_t, from_horiz_v

    def movingAverage(self):
        rolling_mean=self.series.rolling(window=self.window).mean()        
        return rolling_mean        

    def plot_movingAverage(self,inflection=False):
        import numpy as np
        from sklearn.metrics import median_absolute_error, mean_absolute_error
        import matplotlib.pyplot as plt
        """
        function - 打印移动平衡/滑动窗口，及拐点
        """

        plt.figure(figsize=self.figsize)
        plt.title("Moving average\n window size = {}".format(self.window))
        plt.plot(self.rolling_mean, "g", label="Rolling mean trend")

        #打印置信区间，Plot confidence intervals for smoothed values
        if self.plot_intervals:
            mae=mean_absolute_error(self.series[self.window:], self.rolling_mean[self.window:])
            deviation=np.std(self.series[self.window:] - self.rolling_mean[self.window:])
            lower_bond=self.rolling_mean - (mae + self.scale * deviation)
            upper_bond=self.rolling_mean + (mae + self.scale * deviation)
            plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
            plt.plot(lower_bond, "r--")

            # 显示异常值，Having the intervals, find abnormal values
            if self.plot_anomalies:
                anomalies=pd.DataFrame(index=self.series.index, columns=self.series.to_frame().columns)
                anomalies[self.series<lower_bond]=self.series[self.series<lower_bond].to_frame()
                anomalies[self.series>upper_bond]=self.series[self.series>upper_bond].to_frame()
                plt.plot(anomalies, "ro", markersize=10)
                
        if inflection:
            x_r,y_r,to_vert_t, to_vert_v,from_vert_t, from_vert_v,to_horiz_t, to_horiz_v,from_horiz_t, from_horiz_v=self.knee_elbow()
            plt.plot(x_r, y_r, 'b-')
            plt.plot(to_vert_t, to_vert_v, 'r^', label='Plot goes vertical')
            plt.plot(from_vert_t, from_vert_v, 'kv', label='Plot stops being vertical')
            plt.plot(to_horiz_t, to_horiz_v, 'r>', label='Plot goes horizontal')
            plt.plot(from_horiz_t, from_horiz_v, 'k<', label='Plot stops being horizontal')     
            

        plt.plot(self.series[self.window:], label="Actual values")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.xticks(rotation='vertical')
        plt.show()


import numpy as np
import pandas as pd
idx=0   
x=np.array(range(idx,idx+len(matches_num[idx]))) 
y=np.array(matches_num[idx])
y_=pd.Series(y,index=x)
MAI=movingAverage_inflection(y_, window=15,plot_intervals=True,scale=1.96, plot_anomalies=True,figsize=(15*2,5*2),threshold=0)
MAI.plot_movingAverage(inflection=True)
```


    
<a href=""><img src="./imgs/17_15.png" height='auto' width='auto' title="caDesign"></a>
    


* C - 计算感知消失的距离

定位到第一个跳变为0的位置，就是曲线变化基本平缓的位置，获取其索引值。然后找出观察位置与第一个跳变为0位置之间所有图像的坐标，定义其为地理空间数据'GeoDataFrame'的数据格式。因为研究区域在德国，因此找到德国的EPSG编号，用于投影，然后计算路径的长度，并统计相关量。由结果可知，视觉感知消失距离的均值约为29m，16-46之间的数量约占到62%。


```python
def vanishing_position_length(matches_num,coordi_df,epsg,**kwargs):
    from shapely.geometry import Point, LineString, shape
    import geopandas as gpd
    import pyproj
    '''
    function - 计算图像匹配特征点几乎无关联的距离，即对特定位置视觉随距离远去而感知消失的距离
    
    Paras:
    matches_num - 由类dynamicStreetView_visualPerception计算的特征关键点匹配数量
    coordi_df - 包含经纬度的DataFrame，其列名为：lon,lat
    **kwargs - 同类movingAverage_inflection配置参数
    '''
    MAI_paras={'window':15,'plot_intervals':True,'scale':1.96, 'plot_anomalies':True,'figsize':(15*2,5*2),'threshold':0}
    MAI_paras.update(kwargs)   
    #print( MAI_paras)
    
    vanishing_position={}
    for idx in range(len(matches_num)): 
        x=np.array(range(idx,idx+len(matches_num[idx]))) 
        y=np.array(matches_num[idx])
        y_=pd.Series(y,index=x)   
        MAI=movingAverage_inflection(y_, window=MAI_paras['window'],plot_intervals=MAI_paras['plot_intervals'],scale=MAI_paras['scale'], plot_anomalies=MAI_paras['plot_anomalies'],figsize=MAI_paras['figsize'],threshold=MAI_paras['threshold'])   
        _,_,_,_,from_vert_t, _,_, _,from_horiz_t,_=MAI.knee_elbow()
        if np.any(from_horiz_t!= None) :
            vanishing_position[idx]=(idx,from_horiz_t[0])
        else:
            vanishing_position[idx]=(idx,idx)
    vanishing_position_df=pd.DataFrame.from_dict(vanishing_position,orient='index',columns=['start_idx','end_idx'])
    
    vanishing_position_df['geometry']=vanishing_position_df.apply(lambda idx:LineString(coordi_df[idx.start_idx:idx.end_idx][['lon','lat']].apply(lambda row:Point(row.lon,row.lat), axis=1).values.tolist()), axis=1)
    crs_4326={'init': 'epsg:4326'}
    vanishing_position_gdf=gpd.GeoDataFrame(vanishing_position_df,geometry='geometry',crs=crs_4326)
    
    crs_=pyproj.CRS(epsg) 
    vanishing_position_gdf_reproj=vanishing_position_gdf.to_crs(crs_)
    vanishing_position_gdf_reproj['length']=vanishing_position_gdf_reproj.geometry.length
    return vanishing_position_gdf_reproj
    
     
coordi_df=drive_29_0071_info
vanishing_gpd=vanishing_position_length(matches_num,coordi_df,epsg="EPSG:3857",threshold=0)
print("感知消失距离统计:","_"*50,"\n")
print(vanishing_gpd[vanishing_gpd["length"] >1].length.describe())
print("频数统计：","_"*50,"\n")
print(vanishing_gpd[vanishing_gpd["length"] >1]["length"].value_counts(bins=5))

util.print_html(vanishing_gpd)
```

    感知消失距离统计: __________________________________________________ 
    
    count    983.000000
    mean      29.147147
    std       15.663416
    min        1.034268
    25%       18.190853
    50%       29.592061
    75%       40.271488
    max       76.957715
    dtype: float64
    频数统计： __________________________________________________ 
    
    (16.219, 31.404]    315
    (31.404, 46.588]    303
    (0.957, 16.219]     220
    (46.588, 61.773]    130
    (61.773, 76.958]     15
    Name: length, dtype: int64
    

    C:\Users\richi\Anaconda3\envs\openCVpytorch\lib\site-packages\pyproj\crs\crs.py:53: FutureWarning:
    
    '+init=<authority>:<code>' syntax is deprecated. '<authority>:<code>' is the preferred initialization method. When making the change, be mindful of axis order changes: https://pyproj4.github.io/pyproj/stable/gotchas.html#axis-order-changes-in-proj-6
    
    




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
      <td>LINESTRING (934872.661 6276328.368, 934872.550 6276328.441, 934872.449 6276328.505, 934872.346 6276328.570, 934872.232 6276328.643, 934872.132 6276328.713, 934872.039 6276328.778, 934871.942 6276328.832, 934871.859 6276328.879, 934871.774 6276328.932, 934871.701 6276328.976, 934871.634 6276329.009, 934871.567 6276329.039, 934871.516 6276329.063, 934871.471 6276329.087, 934871.423 6276329.115, 934871.365 6276329.146, 934871.288 6276329.185, 934871.209 6276329.230, 934871.118 6276329.279, 934871.012 6276329.331, 934870.903 6276329.378, 934870.773 6276329.436, 934870.646 6276329.492, 934870.511 6276329.557, 934870.349 6276329.636, 934870.189 6276329.718, 934870.015 6276329.807, 934869.801 6276329.909, 934869.586 6276330.005, 934869.330 6276330.121, 934869.078 6276330.236, 934868.809 6276330.358, 934868.493 6276330.500, 934868.185 6276330.625, 934867.830 6276330.779, 934867.495 6276330.927, 934867.144 6276331.074, 934866.738 6276331.226, 934866.354 6276331.372, 934865.914 6276331.540, 934865.500 6276331.706, 934865.073 6276331.873, 934864.587 6276332.052, 934864.141 6276332.218, 934863.683 6276332.386, 934863.170 6276332.570, 934862.695 6276332.735, 934862.171 6276332.920, 934861.692 6276333.099, 934861.210 6276333.279, 934860.675 6276333.480, 934860.190 6276333.668, 934859.661 6276333.882, 934859.183 6276334.078, 934858.704 6276334.271, 934858.173 6276334.486, 934857.682 6276334.686, 934857.141 6276334.912, 934856.654 6276335.123, 934856.177 6276335.337, 934855.656 6276335.579, 934855.182 6276335.798, 934854.660 6276336.041, 934854.188 6276336.253, 934853.726 6276336.452, 934853.237 6276336.658, 934852.819 6276336.843, 934852.424 6276337.020, 934852.018 6276337.206, 934851.673 6276337.370, 934851.319 6276337.533, 934851.019 6276337.665, 934850.742 6276337.781, 934850.471 6276337.902, 934850.254 6276337.997, 934850.068 6276338.081, 934849.892 6276338.152, 934849.762 6276338.205, 934849.636 6276338.255, 934849.527 6276338.300, 934849.418 6276338.347, 934849.294 6276338.401, 934849.178 6276338.450, 934849.048 6276338.506, 934848.928 6276338.557, 934848.806 6276338.610, 934848.669 6276338.671, 934848.541 6276338.727, 934848.409 6276338.784, 934848.256 6276338.848)</td>
      <td>26.590485</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>81</td>
      <td>LINESTRING (934872.550 6276328.441, 934872.449 6276328.505, 934872.346 6276328.570, 934872.232 6276328.643, 934872.132 6276328.713, 934872.039 6276328.778, 934871.942 6276328.832, 934871.859 6276328.879, 934871.774 6276328.932, 934871.701 6276328.976, 934871.634 6276329.009, 934871.567 6276329.039, 934871.516 6276329.063, 934871.471 6276329.087, 934871.423 6276329.115, 934871.365 6276329.146, 934871.288 6276329.185, 934871.209 6276329.230, 934871.118 6276329.279, 934871.012 6276329.331, 934870.903 6276329.378, 934870.773 6276329.436, 934870.646 6276329.492, 934870.511 6276329.557, 934870.349 6276329.636, 934870.189 6276329.718, 934870.015 6276329.807, 934869.801 6276329.909, 934869.586 6276330.005, 934869.330 6276330.121, 934869.078 6276330.236, 934868.809 6276330.358, 934868.493 6276330.500, 934868.185 6276330.625, 934867.830 6276330.779, 934867.495 6276330.927, 934867.144 6276331.074, 934866.738 6276331.226, 934866.354 6276331.372, 934865.914 6276331.540, 934865.500 6276331.706, 934865.073 6276331.873, 934864.587 6276332.052, 934864.141 6276332.218, 934863.683 6276332.386, 934863.170 6276332.570, 934862.695 6276332.735, 934862.171 6276332.920, 934861.692 6276333.099, 934861.210 6276333.279, 934860.675 6276333.480, 934860.190 6276333.668, 934859.661 6276333.882, 934859.183 6276334.078, 934858.704 6276334.271, 934858.173 6276334.486, 934857.682 6276334.686, 934857.141 6276334.912, 934856.654 6276335.123, 934856.177 6276335.337, 934855.656 6276335.579, 934855.182 6276335.798, 934854.660 6276336.041, 934854.188 6276336.253, 934853.726 6276336.452, 934853.237 6276336.658, 934852.819 6276336.843, 934852.424 6276337.020, 934852.018 6276337.206, 934851.673 6276337.370, 934851.319 6276337.533, 934851.019 6276337.665, 934850.742 6276337.781, 934850.471 6276337.902, 934850.254 6276337.997, 934850.068 6276338.081, 934849.892 6276338.152, 934849.762 6276338.205, 934849.636 6276338.255, 934849.527 6276338.300)</td>
      <td>25.073412</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>81</td>
      <td>LINESTRING (934872.449 6276328.505, 934872.346 6276328.570, 934872.232 6276328.643, 934872.132 6276328.713, 934872.039 6276328.778, 934871.942 6276328.832, 934871.859 6276328.879, 934871.774 6276328.932, 934871.701 6276328.976, 934871.634 6276329.009, 934871.567 6276329.039, 934871.516 6276329.063, 934871.471 6276329.087, 934871.423 6276329.115, 934871.365 6276329.146, 934871.288 6276329.185, 934871.209 6276329.230, 934871.118 6276329.279, 934871.012 6276329.331, 934870.903 6276329.378, 934870.773 6276329.436, 934870.646 6276329.492, 934870.511 6276329.557, 934870.349 6276329.636, 934870.189 6276329.718, 934870.015 6276329.807, 934869.801 6276329.909, 934869.586 6276330.005, 934869.330 6276330.121, 934869.078 6276330.236, 934868.809 6276330.358, 934868.493 6276330.500, 934868.185 6276330.625, 934867.830 6276330.779, 934867.495 6276330.927, 934867.144 6276331.074, 934866.738 6276331.226, 934866.354 6276331.372, 934865.914 6276331.540, 934865.500 6276331.706, 934865.073 6276331.873, 934864.587 6276332.052, 934864.141 6276332.218, 934863.683 6276332.386, 934863.170 6276332.570, 934862.695 6276332.735, 934862.171 6276332.920, 934861.692 6276333.099, 934861.210 6276333.279, 934860.675 6276333.480, 934860.190 6276333.668, 934859.661 6276333.882, 934859.183 6276334.078, 934858.704 6276334.271, 934858.173 6276334.486, 934857.682 6276334.686, 934857.141 6276334.912, 934856.654 6276335.123, 934856.177 6276335.337, 934855.656 6276335.579, 934855.182 6276335.798, 934854.660 6276336.041, 934854.188 6276336.253, 934853.726 6276336.452, 934853.237 6276336.658, 934852.819 6276336.843, 934852.424 6276337.020, 934852.018 6276337.206, 934851.673 6276337.370, 934851.319 6276337.533, 934851.019 6276337.665, 934850.742 6276337.781, 934850.471 6276337.902, 934850.254 6276337.997, 934850.068 6276338.081, 934849.892 6276338.152, 934849.762 6276338.205, 934849.636 6276338.255, 934849.527 6276338.300)</td>
      <td>24.953489</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>81</td>
      <td>LINESTRING (934872.346 6276328.570, 934872.232 6276328.643, 934872.132 6276328.713, 934872.039 6276328.778, 934871.942 6276328.832, 934871.859 6276328.879, 934871.774 6276328.932, 934871.701 6276328.976, 934871.634 6276329.009, 934871.567 6276329.039, 934871.516 6276329.063, 934871.471 6276329.087, 934871.423 6276329.115, 934871.365 6276329.146, 934871.288 6276329.185, 934871.209 6276329.230, 934871.118 6276329.279, 934871.012 6276329.331, 934870.903 6276329.378, 934870.773 6276329.436, 934870.646 6276329.492, 934870.511 6276329.557, 934870.349 6276329.636, 934870.189 6276329.718, 934870.015 6276329.807, 934869.801 6276329.909, 934869.586 6276330.005, 934869.330 6276330.121, 934869.078 6276330.236, 934868.809 6276330.358, 934868.493 6276330.500, 934868.185 6276330.625, 934867.830 6276330.779, 934867.495 6276330.927, 934867.144 6276331.074, 934866.738 6276331.226, 934866.354 6276331.372, 934865.914 6276331.540, 934865.500 6276331.706, 934865.073 6276331.873, 934864.587 6276332.052, 934864.141 6276332.218, 934863.683 6276332.386, 934863.170 6276332.570, 934862.695 6276332.735, 934862.171 6276332.920, 934861.692 6276333.099, 934861.210 6276333.279, 934860.675 6276333.480, 934860.190 6276333.668, 934859.661 6276333.882, 934859.183 6276334.078, 934858.704 6276334.271, 934858.173 6276334.486, 934857.682 6276334.686, 934857.141 6276334.912, 934856.654 6276335.123, 934856.177 6276335.337, 934855.656 6276335.579, 934855.182 6276335.798, 934854.660 6276336.041, 934854.188 6276336.253, 934853.726 6276336.452, 934853.237 6276336.658, 934852.819 6276336.843, 934852.424 6276337.020, 934852.018 6276337.206, 934851.673 6276337.370, 934851.319 6276337.533, 934851.019 6276337.665, 934850.742 6276337.781, 934850.471 6276337.902, 934850.254 6276337.997, 934850.068 6276338.081, 934849.892 6276338.152, 934849.762 6276338.205, 934849.636 6276338.255, 934849.527 6276338.300)</td>
      <td>24.832104</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>82</td>
      <td>LINESTRING (934872.232 6276328.643, 934872.132 6276328.713, 934872.039 6276328.778, 934871.942 6276328.832, 934871.859 6276328.879, 934871.774 6276328.932, 934871.701 6276328.976, 934871.634 6276329.009, 934871.567 6276329.039, 934871.516 6276329.063, 934871.471 6276329.087, 934871.423 6276329.115, 934871.365 6276329.146, 934871.288 6276329.185, 934871.209 6276329.230, 934871.118 6276329.279, 934871.012 6276329.331, 934870.903 6276329.378, 934870.773 6276329.436, 934870.646 6276329.492, 934870.511 6276329.557, 934870.349 6276329.636, 934870.189 6276329.718, 934870.015 6276329.807, 934869.801 6276329.909, 934869.586 6276330.005, 934869.330 6276330.121, 934869.078 6276330.236, 934868.809 6276330.358, 934868.493 6276330.500, 934868.185 6276330.625, 934867.830 6276330.779, 934867.495 6276330.927, 934867.144 6276331.074, 934866.738 6276331.226, 934866.354 6276331.372, 934865.914 6276331.540, 934865.500 6276331.706, 934865.073 6276331.873, 934864.587 6276332.052, 934864.141 6276332.218, 934863.683 6276332.386, 934863.170 6276332.570, 934862.695 6276332.735, 934862.171 6276332.920, 934861.692 6276333.099, 934861.210 6276333.279, 934860.675 6276333.480, 934860.190 6276333.668, 934859.661 6276333.882, 934859.183 6276334.078, 934858.704 6276334.271, 934858.173 6276334.486, 934857.682 6276334.686, 934857.141 6276334.912, 934856.654 6276335.123, 934856.177 6276335.337, 934855.656 6276335.579, 934855.182 6276335.798, 934854.660 6276336.041, 934854.188 6276336.253, 934853.726 6276336.452, 934853.237 6276336.658, 934852.819 6276336.843, 934852.424 6276337.020, 934852.018 6276337.206, 934851.673 6276337.370, 934851.319 6276337.533, 934851.019 6276337.665, 934850.742 6276337.781, 934850.471 6276337.902, 934850.254 6276337.997, 934850.068 6276338.081, 934849.892 6276338.152, 934849.762 6276338.205, 934849.636 6276338.255, 934849.527 6276338.300, 934849.418 6276338.347)</td>
      <td>24.815814</td>
    </tr>
  </tbody>
</table>




```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axs=plt.subplots(ncols=2,nrows=1,figsize =(30, 15))
starting_idx=vanishing_gpd.iloc[0].start_idx
ending_idx=vanishing_gpd.iloc[0].end_idx
axs[0].imshow(mpimg.imread(drive_29_0071_img_fp_list[starting_idx]))
axs[0].set_title("starting position ")
axs[1].imshow(mpimg.imread(drive_29_0071_img_fp_list[ending_idx]))
axs[1].set_title("ending position ")

plt.show()
```


    
<a href=""><img src="./imgs/17_16.png" height='auto' width='auto' title="caDesign"></a>
    



```python
FLANN_INDEX_KDTREE = 1
im_1,im_2=drive_29_0071_img_fp_list[starting_idx],drive_29_0071_img_fp_list[ending_idx]
index_params=dict(algorithm=FLANN_INDEX_KDTREE, trees=5)        
feature_matching(im_1,im_2,index_params=index_params,method='FLANN',) 
```


    
<a href=""><img src="./imgs/17_17.png" height='auto' width='auto' title="caDesign"></a>
    


* D - 比较两个街区的变化

与上述方法相同，选取了自然景观偏多的另一个街区。比较计算结果为：

| 统计量\数值 | 街区_26_0009 | 街区_29_0071 |
|--------|------------|------------|
| count  | 983        | 388        |
| mean   | 29.147147  | 64.207892  |
| std    | 15.663416  | 32.833346  |
| min    | 1.034268   | 1.065737   |
| 25%    | 18.190853  | 42.206708  |
| 50%    | 29.592061  | 61.216303  |
| 75%    | 40.271488  | 84.249092  |
| max    | 76.957715  | 190.549947 |


从计算的结果可以观察到，不同的街区视觉感知消失的距离不同。城市内的街道小巷视觉感知消失的距离较之较为开阔自然植被相对较多的街道要小，标准差也偏小，即分布的离散程度要小。就是说，在变化丰富的小巷中行走时，视觉感知变化会比较快，有琳琅满目的感觉；但是在开阔区域行走时，视觉感知变化就会很慢，这里是慢了约2倍的距离。当然对于同是小巷而言，不同的设计，这个视觉变化也会不同，例如推断景观单一的小巷，视觉感知变化应该比较慢，注意这里并没有数据计算验证。


```python
KITTI_info_fp=r'D:\dataset\KITTI\2011_09_26_drive_0009_sync\oxts\data'
timestamps_fp=r'D:\dataset\KITTI\2011_09_26_drive_0009_sync\image_03\timestamps.txt'
drive_26_0009_info=KITTI_info(KITTI_info_fp,timestamps_fp)
util.print_html(drive_26_0009_info)
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>lat</th>
      <th>lon</th>
      <th>alt</th>
      <th>roll</th>
      <th>pitch</th>
      <th>yaw</th>
      <th>vn</th>
      <th>ve</th>
      <th>vf</th>
      <th>vl</th>
      <th>vu</th>
      <th>ax</th>
      <th>ay</th>
      <th>ay</th>
      <th>af</th>
      <th>al</th>
      <th>au</th>
      <th>wx</th>
      <th>wy</th>
      <th>wz</th>
      <th>wf</th>
      <th>wl</th>
      <th>wu</th>
      <th>pos_accuracy</th>
      <th>vel_accuracy</th>
      <th>navstat</th>
      <th>numsats</th>
      <th>posmode</th>
      <th>velmode</th>
      <th>orimode</th>
      <th>timestamps_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>49.009348</td>
      <td>8.437134</td>
      <td>114.456520</td>
      <td>0.072242</td>
      <td>0.001566</td>
      <td>-0.269727</td>
      <td>-2.930182</td>
      <td>10.286884</td>
      <td>10.695751</td>
      <td>0.082894</td>
      <td>-0.075015</td>
      <td>0.005953</td>
      <td>0.845164</td>
      <td>9.914973</td>
      <td>0.041978</td>
      <td>0.136528</td>
      <td>9.950167</td>
      <td>0.016663</td>
      <td>-0.036832</td>
      <td>0.003312</td>
      <td>0.016662</td>
      <td>-0.036976</td>
      <td>0.000604</td>
      <td>0.428562</td>
      <td>0.057983</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>2011-09-26 13:08:24.967313152</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>49.009345</td>
      <td>8.437150</td>
      <td>114.453873</td>
      <td>0.073139</td>
      <td>-0.000124</td>
      <td>-0.269609</td>
      <td>-2.919715</td>
      <td>10.304161</td>
      <td>10.709611</td>
      <td>0.068613</td>
      <td>-0.036820</td>
      <td>0.081271</td>
      <td>0.990642</td>
      <td>10.267646</td>
      <td>0.087095</td>
      <td>0.239172</td>
      <td>10.312656</td>
      <td>0.002734</td>
      <td>-0.012582</td>
      <td>0.002888</td>
      <td>0.002734</td>
      <td>-0.012760</td>
      <td>0.001956</td>
      <td>0.428562</td>
      <td>0.057983</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>2011-09-26 13:08:25.070703872</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>49.009342</td>
      <td>8.437164</td>
      <td>114.458748</td>
      <td>0.069770</td>
      <td>0.002527</td>
      <td>-0.269658</td>
      <td>-2.895219</td>
      <td>10.327819</td>
      <td>10.725885</td>
      <td>0.039142</td>
      <td>0.018783</td>
      <td>0.178470</td>
      <td>0.832624</td>
      <td>10.211571</td>
      <td>0.190807</td>
      <td>0.103002</td>
      <td>10.245165</td>
      <td>-0.026792</td>
      <td>0.022694</td>
      <td>-0.001178</td>
      <td>-0.026791</td>
      <td>0.022720</td>
      <td>0.000459</td>
      <td>0.428562</td>
      <td>0.057983</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>2011-09-26 13:08:25.174198784</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>49.009340</td>
      <td>8.437178</td>
      <td>114.462502</td>
      <td>0.069280</td>
      <td>0.006679</td>
      <td>-0.269507</td>
      <td>-2.905630</td>
      <td>10.325249</td>
      <td>10.726179</td>
      <td>0.050174</td>
      <td>0.005620</td>
      <td>-0.133723</td>
      <td>0.461374</td>
      <td>9.323450</td>
      <td>-0.091298</td>
      <td>-0.187775</td>
      <td>9.334138</td>
      <td>-0.004250</td>
      <td>0.035865</td>
      <td>0.000067</td>
      <td>-0.004234</td>
      <td>0.035775</td>
      <td>0.002543</td>
      <td>0.428562</td>
      <td>0.057983</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>2011-09-26 13:08:25.277587456</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>49.009337</td>
      <td>8.437194</td>
      <td>114.461815</td>
      <td>0.068783</td>
      <td>0.008822</td>
      <td>-0.269148</td>
      <td>-2.912686</td>
      <td>10.314180</td>
      <td>10.717372</td>
      <td>0.062980</td>
      <td>-0.029065</td>
      <td>-0.172930</td>
      <td>0.626856</td>
      <td>9.475164</td>
      <td>-0.101285</td>
      <td>-0.028623</td>
      <td>9.497079</td>
      <td>-0.005153</td>
      <td>0.022360</td>
      <td>0.002338</td>
      <td>-0.005122</td>
      <td>0.022146</td>
      <td>0.003912</td>
      <td>0.428562</td>
      <td>0.057983</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>2011-09-26 13:08:25.381086208</td>
    </tr>
  </tbody>
</table>




```python
#plotly_scatterMapbox(drive_26_0009_info,center_lon=8.437134,center_lat=49.009348)  
```


```python
import util,os
drive_26_0009_img_fp=util.filePath_extraction(r'D:\dataset\KITTI\2011_09_26_drive_0009_sync\image_03\data',['png'])
drive_26_0009_img_fp_list=util.flatten_lst([[os.path.join(k,f) for f in drive_26_0009_img_fp[k]] for k,v in drive_26_0009_img_fp.items()])
```


```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from IPython.display import HTML

fig=plt.figure(figsize=(20,10))
ims=[[plt.imshow(mpimg.imread(f),animated=True)] for f in drive_26_0009_img_fp_list] #[:200]
print("finished reading the imgs.")

ani=animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=1000)
ani.save(r'./imgs/drive_26_0009_info.mp4')
#print(".mp4 saved.")
HTML(ani.to_html5_video())
```

    finished reading the imgs.
    
    
<video width='auto' height='auto' controls><source src="./imgs/17_18.mp4" height='auto' width='auto' title="caDesign" type='video/mp4'></video>
    

```python
dsv_vp_=dynamicStreetView_visualPerception(drive_26_0009_img_fp_list) #[:200]
#kp1,des1=dsv_vp.kp_descriptor(drive_29_0071_img_fp_list[10]) 
#kp2,des2=dsv_vp.kp_descriptor(drive_29_0071_img_fp_list[50])
#dsv_vp.feature_matching(des1,des2,kp1,kp2)
matches_num_=dsv_vp_.sequence_statistics()
```

      0%|          | 2/447 [00:00<00:23, 18.79it/s]

    计算关键点和描述子...
    

    100%|██████████| 447/447 [00:25<00:00, 17.26it/s]
      0%|          | 0/446 [00:00<?, ?it/s]

    计算序列图像匹配数...
    

    100%|██████████| 446/446 [03:35<00:00,  2.07it/s]
    


```python
import numpy as np
import pandas as pd
idx=0   
x=np.array(range(idx,idx+len(matches_num_[idx]))) 
y=np.array(matches_num_[idx])
y_=pd.Series(y,index=x)
MAI_=movingAverage_inflection(y_, window=15,plot_intervals=True,scale=1.96, plot_anomalies=True,figsize=(15*2,5*2),threshold=0)
MAI_.plot_movingAverage(inflection=True)
```


    
<a href=""><img src="./imgs/17_19.png" height='auto' width='auto' title="caDesign"></a>
    



```python
coordi_df_=drive_26_0009_info
vanishing_gpd_=vanishing_position_length(matches_num_,coordi_df_,epsg="EPSG:3857",threshold=0)
print("感知消失距离统计:","_"*50,"\n")
print(vanishing_gpd_[vanishing_gpd_["length"] >1].length.describe())
print("频数统计：","_"*50,"\n")
print(vanishing_gpd_[vanishing_gpd_["length"] >1]["length"].value_counts(bins=5))

util.print_html(vanishing_gpd_)
```

    感知消失距离统计: __________________________________________________ 
    
    count    388.000000
    mean      64.207892
    std       32.833346
    min        1.065737
    25%       42.206708
    50%       61.216303
    75%       84.249092
    max      190.549947
    dtype: float64
    频数统计： __________________________________________________ 
    
    (38.963, 76.859]      185
    (76.859, 114.756]      89
    (0.875, 38.963]        81
    (114.756, 152.653]     30
    (152.653, 190.55]       3
    Name: length, dtype: int64
    

    C:\Users\richi\Anaconda3\envs\openCVpytorch\lib\site-packages\pyproj\crs\crs.py:53: FutureWarning:
    
    '+init=<authority>:<code>' syntax is deprecated. '<authority>:<code>' is the preferred initialization method. When making the change, be mindful of axis order changes: https://pyproj4.github.io/pyproj/stable/gotchas.html#axis-order-changes-in-proj-6
    
    




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
      <td>43</td>
      <td>LINESTRING (939217.509 6276447.662, 939219.233 6276447.172, 939220.806 6276446.729, 939222.376 6276446.284, 939224.102 6276445.790, 939225.668 6276445.344, 939227.389 6276444.855, 939228.960 6276444.418, 939230.544 6276443.992, 939232.286 6276443.526, 939233.857 6276443.085, 939235.421 6276442.609, 939237.129 6276442.105, 939238.677 6276441.639, 939240.219 6276441.184, 939241.907 6276440.686, 939243.435 6276440.239, 939244.958 6276439.793, 939246.626 6276439.303, 939248.136 6276438.865, 939249.638 6276438.428, 939251.290 6276437.955, 939252.788 6276437.520, 939254.282 6276437.088, 939255.924 6276436.613, 939257.415 6276436.120, 939258.919 6276435.686, 939260.575 6276435.207, 939262.084 6276434.773, 939263.616 6276434.270, 939265.294 6276433.802, 939266.816 6276433.383, 939268.490 6276432.911, 939270.035 6276432.434, 939271.582 6276431.972, 939273.291 6276431.462, 939274.844 6276431.001, 939276.398 6276430.538, 939278.108 6276430.023, 939279.653 6276429.561, 939281.210 6276429.099, 939282.944 6276428.582, 939284.498 6276428.064)</td>
      <td>69.802844</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>40</td>
      <td>LINESTRING (939219.233 6276447.172, 939220.806 6276446.729, 939222.376 6276446.284, 939224.102 6276445.790, 939225.668 6276445.344, 939227.389 6276444.855, 939228.960 6276444.418, 939230.544 6276443.992, 939232.286 6276443.526, 939233.857 6276443.085, 939235.421 6276442.609, 939237.129 6276442.105, 939238.677 6276441.639, 939240.219 6276441.184, 939241.907 6276440.686, 939243.435 6276440.239, 939244.958 6276439.793, 939246.626 6276439.303, 939248.136 6276438.865, 939249.638 6276438.428, 939251.290 6276437.955, 939252.788 6276437.520, 939254.282 6276437.088, 939255.924 6276436.613, 939257.415 6276436.120, 939258.919 6276435.686, 939260.575 6276435.207, 939262.084 6276434.773, 939263.616 6276434.270, 939265.294 6276433.802, 939266.816 6276433.383, 939268.490 6276432.911, 939270.035 6276432.434, 939271.582 6276431.972, 939273.291 6276431.462, 939274.844 6276431.001, 939276.398 6276430.538, 939278.108 6276430.023, 939279.653 6276429.561)</td>
      <td>62.938444</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>59</td>
      <td>LINESTRING (939220.806 6276446.729, 939222.376 6276446.284, 939224.102 6276445.790, 939225.668 6276445.344, 939227.389 6276444.855, 939228.960 6276444.418, 939230.544 6276443.992, 939232.286 6276443.526, 939233.857 6276443.085, 939235.421 6276442.609, 939237.129 6276442.105, 939238.677 6276441.639, 939240.219 6276441.184, 939241.907 6276440.686, 939243.435 6276440.239, 939244.958 6276439.793, 939246.626 6276439.303, 939248.136 6276438.865, 939249.638 6276438.428, 939251.290 6276437.955, 939252.788 6276437.520, 939254.282 6276437.088, 939255.924 6276436.613, 939257.415 6276436.120, 939258.919 6276435.686, 939260.575 6276435.207, 939262.084 6276434.773, 939263.616 6276434.270, 939265.294 6276433.802, 939266.816 6276433.383, 939268.490 6276432.911, 939270.035 6276432.434, 939271.582 6276431.972, 939273.291 6276431.462, 939274.844 6276431.001, 939276.398 6276430.538, 939278.108 6276430.023, 939279.653 6276429.561, 939281.210 6276429.099, 939282.944 6276428.582, 939284.498 6276428.064, 939286.066 6276427.547, 939287.805 6276426.966, 939289.397 6276426.443, 939290.989 6276425.912, 939292.743 6276425.322, 939294.346 6276424.793, 939295.954 6276424.267, 939297.722 6276423.706, 939299.329 6276423.194, 939300.937 6276422.686, 939302.710 6276422.158, 939304.327 6276421.674, 939306.101 6276421.160, 939307.712 6276420.675, 939309.326 6276420.186, 939310.942 6276419.700)</td>
      <td>94.113985</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>36</td>
      <td>LINESTRING (939222.376 6276446.284, 939224.102 6276445.790, 939225.668 6276445.344, 939227.389 6276444.855, 939228.960 6276444.418, 939230.544 6276443.992, 939232.286 6276443.526, 939233.857 6276443.085, 939235.421 6276442.609, 939237.129 6276442.105, 939238.677 6276441.639, 939240.219 6276441.184, 939241.907 6276440.686, 939243.435 6276440.239, 939244.958 6276439.793, 939246.626 6276439.303, 939248.136 6276438.865, 939249.638 6276438.428, 939251.290 6276437.955, 939252.788 6276437.520, 939254.282 6276437.088, 939255.924 6276436.613, 939257.415 6276436.120, 939258.919 6276435.686, 939260.575 6276435.207, 939262.084 6276434.773, 939263.616 6276434.270, 939265.294 6276433.802, 939266.816 6276433.383, 939268.490 6276432.911, 939270.035 6276432.434, 939271.582 6276431.972, 939273.291 6276431.462)</td>
      <td>53.033050</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>37</td>
      <td>LINESTRING (939224.102 6276445.790, 939225.668 6276445.344, 939227.389 6276444.855, 939228.960 6276444.418, 939230.544 6276443.992, 939232.286 6276443.526, 939233.857 6276443.085, 939235.421 6276442.609, 939237.129 6276442.105, 939238.677 6276441.639, 939240.219 6276441.184, 939241.907 6276440.686, 939243.435 6276440.239, 939244.958 6276439.793, 939246.626 6276439.303, 939248.136 6276438.865, 939249.638 6276438.428, 939251.290 6276437.955, 939252.788 6276437.520, 939254.282 6276437.088, 939255.924 6276436.613, 939257.415 6276436.120, 939258.919 6276435.686, 939260.575 6276435.207, 939262.084 6276434.773, 939263.616 6276434.270, 939265.294 6276433.802, 939266.816 6276433.383, 939268.490 6276432.911, 939270.035 6276432.434, 939271.582 6276431.972, 939273.291 6276431.462, 939274.844 6276431.001)</td>
      <td>52.856469</td>
    </tr>
  </tbody>
</table>




```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

idx_=30
fig, axs=plt.subplots(ncols=2,nrows=1,figsize =(30, 15))
starting_idx_=vanishing_gpd_.iloc[idx_].start_idx
ending_idx_=vanishing_gpd_.iloc[idx_].end_idx
axs[0].imshow(mpimg.imread(drive_26_0009_img_fp_list[starting_idx_]))
axs[0].set_title("starting position ")
axs[1].imshow(mpimg.imread(drive_26_0009_img_fp_list[ending_idx_]))
axs[1].set_title("ending position ")

plt.show()
```


    
<a href=""><img src="./imgs/17_20.png" height='auto' width='auto' title="caDesign"></a>
    



```python
FLANN_INDEX_KDTREE = 1
im_1_,im_2_=drive_26_0009_img_fp_list[starting_idx_],drive_26_0009_img_fp_list[ending_idx_]
index_params=dict(algorithm=FLANN_INDEX_KDTREE, trees=5)        
feature_matching(im_1_,im_2_,index_params=index_params,method='FLANN',) 
```


    
<a href=""><img src="./imgs/17_21.png" height='auto' width='auto' title="caDesign"></a>
    


### 1.4 要点
#### 1.4.1 数据处理技术

* 应用OpenCV计算机视觉开源库分析影像，图像处理，特征提取，图像匹配

* SIFT与尺度空间（scale space）

* KITTI 无人驾驶场景下计算机视觉算法评测数据集

* 寻找曲线拐点，以及跳变稳定的区域

#### 1.4.2 新建立的函数

* function - 应用OpenCV计算高斯模糊，并给定滑动条调节参数, `Gaussion_blur(img_fp)`

* function - SIFT(scale invariant feature transform) 尺度不变特征变换)特征点检测, `SIFT_detection(img_fp,save=False)`

* function - 使用Star特征检测器提取图像特征, `STAR_detection(img_fp,save=False)`

* function - OpenCV 根据图像特征匹配图像。迁移官网的三种方法，1-ORB描述子蛮力匹配　Brute-Force Matching with ORB Descriptors；2－使用SIFT描述子和比率检测蛮力匹配 Brute-Force Matching with SIFT Descriptors and Ratio Test; 3-基于FLANN的匹配器 FLANN based Matcher, `feature_matching(img_1_fp,img_2_fp,index_params=None,method='FLANN')`

* function - 读取KITTI文件信息，1-包括经纬度，惯性导航系统信息等的.txt文件，2-包含时间戳的.txt文件，`KITTI_info(KITTI_info_fp,timestamps_fp`

*  function - 使用plotly的go.Scattermapbox方法，在地图上显示点及其连线，坐标为经纬度, `plotly_scatterMapbox(df,**kwargs)`

* class - 应用Star提取图像关键点，结合SIFT获得描述子，根据特征匹配分析特征变化（视觉感知变化），即动态街景视觉感知, `dynamicStreetView_visualPerception`

* class - 曲线（数据）平滑，与寻找曲线水平和纵向的斜率变化点, `movingAverage_inflection`

* function - 计算图像匹配特征点几乎无关联的距离，即对特定位置视觉随距离远去而感知消失的距离, `vanishing_position_length(matches_num,coordi_df,epsg,**kwargs)`

#### 1.4.3 所调用的库


```python
import cv2 as cv
import sys,os
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.image as mpimg
from IPython.display import HTML
import plotly.graph_objects as go

from scipy.stats import norm
from scipy.stats import multivariate_normal
from sklearn.metrics import median_absolute_error, mean_absolute_error

from shapely.geometry import Point, LineString, shape
import geopandas as gpd
import pyproj
```

#### 1.4.4 参考文献
1. D.Lowe.Distinctive Image Features from Scale-Invariant Keypoints[J], 2004. SIFT提出者

