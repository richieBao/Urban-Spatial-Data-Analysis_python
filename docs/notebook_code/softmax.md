> Created on Thu Jan  7 14\32\53 2021 @author: Richie Bao-caDesign设计(cadesign.cn)

## 1. 伯努利分布(Bernouli distribution)，似然函数，最大/极大似然估计(Maximum Likelihood Estimation, MLE)，逻辑回归(Logistic Regression, LR)二分类，SoftMax回归多分类
SoftMax回归多分类是逻辑回归二分类的一种推广，逻辑回归通过最大似然估计更新参数（实际上通常使用梯度下降法），最大似然估计是似然函数联合概率分布对应的最大值，而似然函数描述伯努利分布中概率（参数$p$）的概率分布，伯努利分布为离散型概率分布。因此对于SoftMax的理解最好是从伯努利分布开始逐步的层层剥离。

### 1.1 伯努利分布(Bernouli distribution)
伯努利分布(Bernoulli distribution)，又名两点分布或者0-1分布，是一个离散型概率分布（为纪念瑞士科学家Jakob I. Bernoulli而命名）。其实验对象只有两种可能结果，例如从不知比例只有黑色和白色球的罐子里取球(取出后需要放回后，再取球)，白色为1，黑色为0。为白的概率记作$p(0 \leq p \leq 1)$，则为黑的概率为$q=1-p$。其概率质量函数为：$f_{x} = p^{x} (1-p) ^{1-x}=\begin{cases}p & if \quad x = 1\\q & if \quad x = 0\end{cases}  $。其期望值为：$E[X]= \sum_{i=0}^1 x_{i}   f_{X} (x)=0+p=p$。其方差为：$var[X]= \sum_{i=0}^1  ( x_{i}-E[X] ) ^{2}  f_{X} (x)= (0-p)^{2}(1-p)+ (1-p)^{2}p=p(1-p)=pq$。

指定一种结果（例如为白球）的概率$p=0.3$，可以使用scipy.stats库中的bernoulli进行相关计算，例如打印概率质量函数曲线，生成符合概率$p$的随机数组等。如果定义$p=0.3$，则生成的随机数组包含0和1两个元素，其比例围绕7:3上下浮动。

> 概率质量函数（probability mass function, pmf），是离散随机变量在各特定取值上的概率。概率质量函数和概率密度函数不同之处在于：概率质量函数是对离散随机变量定义的，本身代表该值的概率；概率密度函数是对连续随机变量定义的，本身不是概率，只是对连续随机变量的概率密度函数在某区间内进行积分后才是概率。

> 期望值：在概率论和统计学中，一个离散型随机变量的期望值（或数学期望，亦简称期望）是试验中每次可能结果乘以其结果概率的总和。期望值是该变量输出值的加权平均，可能与每一个结果都不相等，并不一定包含于其分布值域，也并不一定等于值域均值。


```python
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import numpy as np

fig, ax=plt.subplots(1, 1)
#
p=0.3
mean,var,skew,kurt=bernoulli.stats(p,moments='mvsk')
#打印概率质量函数/分布，pmf
x=np.arange(bernoulli.ppf(0.01,p),bernoulli.ppf(0.99,p))
ax.plot(x,bernoulli.pmf(x,p),'bo',ms=8,label='bernoulli pmf')
ax.vlines(x,0,bernoulli.pmf(x,p),colors='b',lw=5,alpha=0.5)

#Freeze the distribution and display the frozen
rv=bernoulli(p)
ax.vlines(x,0,rv.pmf(x),colors='k',linestyles='-',lw=1,label='frozen pmf')
ax.legend(loc='best',frameon=False)
plt.show()

#查看cdf和ppf的精度 Check accuracy of cdf and ppf。 cdf(k,p,loc=0),Cumulative distribution function;ppf(q, p, loc=0),Percent point function (inverse of cdf — percentiles).
prob=bernoulli.cdf(x,p)
print(".ppf(.cdf)={}".format(np.allclose(x,bernoulli.ppf(prob,p))))

#生成符合指定概率p的随机数组 Generate random numbers
r=bernoulli.rvs(p, size=100)
unique_elements, counts_elements=np.unique(r, return_counts=True)
print("生成符合伯努利分布的数组(p={})：\n{}\n包含元素为：{}，对应频数为：{}".format(p,r,unique_elements, counts_elements))
```


    
<a href=""><img src="./imgs/21_04.png" height="auto" width="auto" title="caDesign"></a>
    


    .ppf(.cdf)=True
    生成符合伯努利分布的数组(p=0.3)：
    [0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 1 0 0 0 0 1 0 1 0 1 1 1 1 0 0 1 0 0 0 1 0
     0 1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1 0 1 1 0 0 1 1 0 0 0 0 0 0
     0 0 0 0 0 0 0 1 0 1 0 0 1 0 1 0 1 0 0 1 1 0 0 1 0 0]
    包含元素为：[0 1]，对应频数为：[71 29]
    

### 1.2 似然函数(Likelihood function)
在伯努利分布中，应用`scipy.stats.bernoulli.rvs`生成符合$p=0.3$，包含100个要素的随机数组。这个过程可以用实际案例描述为黑罐子里取黑球和白球（两种球的数量比例满足0.3，和1-0.3=07）的过程（或者抛掷硬币的过程，如果是抛掷硬币，为1，即正面朝上的概率为0.3时，说明硬币分布不均匀，正面部分的质量更高，更容易让背面朝上），取得白球，即值为1时的概率为0.3；而取得黑球，即值为0时的概率为1-0.3=0.7。即，已知参数$p=0.3$，罐子里白球和黑球的比例大约为3:7（球的总数为任意数），可以根据已知参数求得随机变量的输出结果的概率，例如随机抽取100次后，出现白球的概率是多少？（或黑球的概率是多少？），这个过程就是概率的描述；对于上述过程的近似反过程，如只知道指定数量的随机数组，即已知随机变量输出结果（随机抽取100次后，以数组表述黑白球的抽取结果），此时并不知道参数$p$，而需要求得参数$p$取值的概率分布（即估计参数$p$的可能性，$p$的取值范围在$0 \leq p \leq 1$），这个过程就是似然的描述。概率函数和似然函数具有一样的模型，区别在于将谁看作变量，将谁看作参数。如果概率函数记作：$P(X |   p_{x}  )$，其中已知$p_{x}=0.3$；则似然函数为：$L(p_{x} | X)$，其中已知$X$，为一个包含值1和0（两种情况结果）的随机数组。

概率描述的过程，是应用概率质量函数/分布的公式，代入取得的白球概率$p=0.3$，计算黑球（x=0）的概率为$q= 0.3^{0}  (1-0.3)^{1-0} =0.7$，可以推断出在随机取球100次之后，所取得白、黑球（事件）的概率为0.3和0.7。而似然描述的过程，是根据随机抽取100次后，用表述黑白球抽取结果的数组中的值（1，和0），即$x$的值，逐一代入概率质量函数/分布的公式后求积。因为$x$值为1或为0，因此会出现两种结果，当$x=0$时，结果为$p^{0}  (1-p)^{1-0} =1-p$；当$x=1$时，结果为$p^{1}  (1-p)^{1-1} =p$。如果随机抽取的球里，白球有29个，黑球有71个，则似然函数为$L(p_{x} | X)=p^{29}  (1-p)^{71} $。参数$p$位于0到1之间，因此生成0-1连续$p$参数值(就是抽取黑白球的概率值)，将其代入似然函数，可以绘制抽取黑白球概率值的概率分布。


```python
import sympy
x_, p_=sympy.symbols('x p', positive=True)
pmf=p_**x_*(1-p_)**(1-x_) #构建概率质量函数/分布的公式
print("(离散型)概率质量函数，pmf={}".format(pmf))
L=np.prod([pmf.subs(x_, i) for i in r]) #推导似然函数/分布公式。np.prod()用于计算所有元素的乘积
print("似然函数,L={}".format(L))

fig, ax=plt.subplots(1, 1)
p_x=np.arange(0,1,0.01) #生成0-1连续的概率值
L_=sympy.lambdify(p_,L,"numpy")
ax.plot(p_x,L_(p_x),'b-',ms=8,label='Likelihood function distribution')

ax.legend(loc='best',frameon=False)
plt.show()
```

    (离散型)概率质量函数，pmf=p**x*(1 - p)**(1 - x)
    似然函数,L=p**29*(1 - p)**71
    


    
<a href=""><img src="./imgs/21_05.png" height="auto" width="auto" title="caDesign"></a>
    


### 1.3 最大/极大似然估计(Maximum Likelihood Estimation, MLE)
给定一个概率分布$D$，已知其概率密度函数（连续分布，Probability Density Function,pdf）或概率质量函数（离散分布，Probability Mass Function,pmf）为$f_{D} $，以及一个分布参数$\theta $(即上述$p$值)。则可以从这个分布中抽取$n$个采样值，$X_{1} ,X_{2}, \ldots ,X_{n}  $，利用$f_{D} $计算其似然函数：$L( \theta  |  x_{1}, x_{2},\ldots, x_{n} )= f_{ \theta } (x_{1}, x_{2},\ldots, x_{n} )$。若$D$是离散分布，$f_{ \theta }$即是在参数为$\theta$时观测到这一采用的概率。若其是连续分布，$f_{ \theta }$则为，$X_{1} ,X_{2}, \ldots ,X_{n}  $联合分布的概率密度函数在观测值处的取值。一旦获得，$X_{1} ,X_{2}, \ldots ,X_{n}  $，就可以求得一个关于$\theta$的估计，这正式上述描述的似然函数。似然函数可进一步表示为$L( \theta  | X)=P(X |  \theta )= \prod_{i=1}^n P( x_{i}  |  \theta ) $，其中$\prod_{i=1}^n$表示为元素积。而最大似然估计是寻找关于$\theta$的最可能的值，即在所有$\theta$取值中（0-1），寻找一个值使这个采样的‘可能性’最大化，就是在$\theta$的所有可能取值中寻找一个值使得似然函数取到最大值。这个使可能性最大的$\widehat{ \theta } $值即称为$\theta$的最大似然估计。最大似然估计是样本的函数。

求最大似然估计就是对似然函数求导，并令其导函数为0时的解，即曲线变化趋于0的位置。其导函数为$-71 \times  p^{29}  \times  (1-p)^{70}+29 \times  p^{28} \times (1-p)^{71}$，计算相对复杂。而最大化一个似然函数同最大化它的自然对数是等价的，因此对似然函数取对数，其似然函数对数的导函数为$\frac{-71}{1-p}+ \frac{29}{p}  $，可见计算量得以大幅度减少；而由似然函数积的方式，转换为似然函数对数和的形式，也进一步减少了计算量。

最后求得的最大似然估计值为29/100，与生成符合指定概率$p=0.3$的随机数组中的$p$值基本一致。


```python
print("似然函数的导函数={}".format(sympy.diff(L,p_)))
L_max_=sympy.solve(sympy.diff(L,p_),p_)
print("令似然函数的导数为0时，解得最大似然估计={}".format(L_max_))

L_log=sympy.expand_log(sympy.log(L)) #sympy.expand_log来简化数学表达式中的对数项
print("似然函数的对数={}".format(L_log))
print("似然函数对数的导函数={}".format(sympy.diff(L_log,p_)))
L_log_=sympy.lambdify(p_,L_log,"numpy") 
L_max=sympy.solve(sympy.diff(L_log,p_),p_) #sympy.solve令所有方程等于0，解方程或方程组。即求曲线最大值的位置
print("令似然函数对数的导数为0时，解得最大似然估计={}".format(L_max))

fig, ax=plt.subplots(1, 1)
L_log_values=L_log_(p_x)
ax.plot(p_x,L_log_values,'r-',ms=8,label='log of Likelihood function distribution')

def rescale_linear(array, new_min, new_max):
    """
    function - 按指定区间缩放/映射数组/Rescale an arrary linearly.
    
    Paras:
    new_min - 映射区间的最小值
    new_max - 映射区间的最大值
    """
    minimum, maximum=np.min(array), np.max(array)
    m=(new_max - new_min) / (maximum - minimum)
    b=new_min - m * minimum
    return m * array + b

L_log_values=L_log_values[np.isfinite(L_log_values)]
ax.plot(p_x,rescale_linear(L_(p_x),L_log_values.min(),L_log_values.max()),'b-',ms=8,label='Likelihood function distribution mapping')

ax.legend(loc='best',frameon=False)
plt.show()
```

    似然函数的导函数=-71*p**29*(1 - p)**70 + 29*p**28*(1 - p)**71
    令似然函数的导数为0时，解得最大似然估计=[29/100, 1]
    似然函数的对数=29*log(p) + log((1 - p)**71)
    似然函数对数的导函数=-71/(1 - p) + 29/p
    令似然函数对数的导数为0时，解得最大似然估计=[29/100]
    

    <lambdifygenerated-44>:2: RuntimeWarning: divide by zero encountered in log
      return (29*log(p) + log((1 - p)**71))
    


    
<a href=""><img src="./imgs/21_06.png" height="auto" width="auto" title="caDesign"></a>
    


### 1.4 逻辑回归(Logistic Regression, LR)
逻辑回归模型为：$p(x)= \sigma (t)= \frac{1}{1+ e^{-t} } = \frac{1}{1+ e^{-(  \beta _{0}+ \beta _{1} x_{1}+ \beta _{2} x_{2}+ \ldots + \beta _{2} x_{n})} }= \frac{1}{1+ e^{- w^{T} x} }  $，其中$ \sigma (t)$就是sigmoid函数(在深度学习中用于激活函数)，sigmoid函数可以将线性模型$f(x)=\beta _{0}+ \beta _{1} x_{1}+ \beta _{2} x_{2}+ \ldots + \beta _{2} x_{n}= w^{T} x$，归一化到[0,1]之间。而因为为二元分类，类标只有两类值1和0,特征值经过加权和（参数全部初始化为1）喂入sigmoid函数后所得到的值位于[0,1]区间，则该值与类标的差值和就为逻辑回归的损失函数，通过随机梯度下降法更新初始化的参数（并未使用最大似然估计，可能无法解析求解），最终训练所得的参数应该使得逻辑回归模型的预测结果趋近于类标1或0。逻辑回归模型是线性模型和sigmoid函数的组合，这与解释”反向传播“部分所构建的隐含层和输出层的网络结构是一样的，可以互相印证理解。

<a href=""><img src="./imgs/21_01.jpg" height='auto' width='700' title="caDesign"></a>

随机梯度下降`stocGradAscent1`方法的定义之间迁移*Machine learning in action*书中的代码，作者解释了优化的随机梯度下降，主要改进包括：1，每次迭代时，调整更新步长alpha值（即是学习率lr），学习率会越来越小，从而缓解系数的高频波动。同时为了避免迭代不断减小至0，约束学习率大于一个稍微大点的常数项，对应代码为`alpha = 4/(1.0+j+i)+0.0001 `。同时，改变样本的优化顺序，即是随机选择样本来更新回归系数。这样可以减少周期性的波动。对应的代码为`randIndex = int(np.random.uniform(0,len(dataIndex)))`。

当训练求得参数（权重值）后，通过逻辑回归模型预测的结果是位于0到1之间的浮点数，以0.5为界，大于0.5返回类标为1，否则类标为0作为分类的输出。重新生成分类数据集，用于测试逻辑回归模型精度。


> 参考文献：
1. Peter Harrington.Machine learning in action[M].NY:Manning Publications; 1st edition (November 24, 2020)
2. Cavin Hackeling. Mastering Machine Learning with scikit-learn[M].Packt Publishing Ltd.July 2017.Second published. 中文版为：Cavin Hackeling.张浩然译.scikit-learning 机器学习[M].人民邮电出版社.2019.2


```python
'''
Created on Oct 27, 2010
Logistic Regression Working Module  /https://github.com/pbharrin/machinelearninginaction/blob/master/Ch05/logRegres.py
@author: Peter

updated on Fri Jan  8 17:42:57 2021 @author: Richie Bao-caDesign设计(cadesign.cn)
'''
class LogisticRegression:
    '''
    class - 自定义逻辑回归(二元分类)
    '''
    def __init__(self):
        pass        
    
    def generate_dataset_linear(self,slop,intercept,num=100,multiple=10,magnitude=50):        
        import numpy as np
        '''
        function - 根据一元一次函数构建二维离散型分类数据集，类标为1和0
        '''
        X_1=np.random.random(num)-0.5
        X_1*=multiple
        X_2=slop*X_1+intercept
        mag_random_val=np.random.random(num)-0.5
        mag_random_val*=magnitude
        mask=mag_random_val>=0
        y=mask*1
        X_2+=mag_random_val
        X=np.stack((np.ones(len(X_1)),X_1,X_2),axis=1)
        return X,y
        
    def make_classification_dataset(self,n_features=2,n_classes=2,n_samples=100,n_informative=2,n_redundant=0,n_clusters_per_class=1):
        from sklearn.datasets import make_classification
        '''
        function - 使用Sklearn提供的make_classification方法，直接构建离散型分类数据集
        
        Paras:
        参数查看sklearn库make_classification方法
        '''        
        X,y=make_classification(n_samples=n_samples,n_features=n_features,n_classes=n_classes, n_redundant=n_redundant, n_informative=n_informative,n_clusters_per_class=n_clusters_per_class)
        X=np.append(np.ones(len(X)).reshape(-1,1),X,axis=1)
        return X,y
    
    def plot(self,X,y,weights,figsize=(10,10)):
        import matplotlib.pyplot as plt
        '''
        function - 绘制逻辑回归结果
        '''
        fig=plt.figure(figsize=figsize)
        ax=fig.add_subplot(111)
        ax.scatter(X[y==1][:,1],X[y==1][:,2],s=30,c='red',marker='s',label='label=1')
        ax.scatter(X[y==0][:,1],X[y==0][:,2],s=30,c='green',label='label=0')
        
        x=np.arange(-3.0,3.0,0.1)
        y=(-weights[0]-weights[1]*x)/weights[2]
        ax.plot(x, y,label='best fit')
        
        ax.legend(loc='best',frameon=False)
        plt.show()        
        
    def sigmoid(self,x):
        '''
        function - 逻辑回归函数_sigmoid函数
        '''
        return 1.0/(1+np.exp(-x))
        
    def stocGradAscent1(self,dataMatrix, classLabels, numIter=150):
        from tqdm import tqdm
        '''
        function - 随机梯度下降（优化）
        '''
        m,n = np.shape(dataMatrix)
        weights = np.ones(n)   #initialize to all ones
        for j in tqdm(range(numIter)):
            dataIndex = list(range(m))
            for i in range(m):
                alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
                randIndex = int(np.random.uniform(0,len(dataIndex)))#go to 0 because of the constant
                h = self.sigmoid(sum(dataMatrix[randIndex]*weights))
                error = classLabels[randIndex] - h
                weights = weights + alpha * error * dataMatrix[randIndex]
                del(dataIndex[randIndex])
        return weights        
    
    def classifyVector(self,inX, weights):
        '''
        function - 代入梯度下降法训练的权重值于逻辑回归函数（sigmoid），预测样本特征对应的类标
        '''
        prob = self.sigmoid(sum(inX*weights))
        if prob > 0.5: return 1.0
        else: return 0.0
        
    def test_accuracy(self,X_,y_,weights):
        from tqdm import tqdm
        '''
        function - 测试训练模型
        '''
        m,n=np.shape(X_)
        matches_num=0
        for i in tqdm(range(m)):
            pred=self.classifyVector(X_[i,:],weights)            
            if pred==np.bool(y_[i]):
                matches_num+=1
        accuracy=float(matches_num)/m
        print("测试数据集精度：{}".format(accuracy))
        return accuracy
    

LR=LogisticRegression()
X,y=LR.make_classification_dataset(n_features=2,n_classes=2,n_samples=100)
weights=LR.stocGradAscent1(X,y,numIter=150)

X_,y_=LR.generate_dataset_linear(slop=5,intercept=5,num=100,multiple=10,magnitude=50) #建立测试数据集，LR.make_classification_dataset和LR.generate_dataset_linear均可以生成测试数据集
accuracy=LR.test_accuracy(X_,y_,weights)
LR.plot(X,y,weights)
```

    100%|██████████| 150/150 [00:00<00:00, 818.86it/s]
    100%|██████████| 100/100 [00:00<?, ?it/s]
    

    测试数据集精度：0.85
    


    
<a href=""><img src="./imgs/21_07.png" height="auto" width="auto" title="caDesign"></a>
    


* sklearn库LogisticRegression方法实现

LogisticRegression方法与上述自定义方法的逻辑回归模型训练的结果保持一致。精度均为0.85，通过不同多次运行（获取不同的训练数据集和测试数据集)，各自结果保持一致。混淆矩阵分类精度计算结果显示，总共100个样本，真实值为0的类标计55个，误判为类标1的为14个；真实值为1的类标计45个，误判为类标1的为1个。精确度、召回率以及调和平均值的结果由`classification_report`方法计算。


```python
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt

logreg=LogisticRegression ()
logreg.fit(X,y)
predictions=logreg.predict(X_)
CM=confusion_matrix(y_,predictions,labels=logreg.classes_)
print("confusion_matrix:\n{}".format(CM))
plt.matshow(CM)
plt.title('confusion matrix')
plt.ylabel('True lable')
plt.xlabel('Predicted label')
plt.colorbar()
plt.show()

print(classification_report(y_,predictions))
```

    confusion_matrix:
    [[41 14]
     [ 1 44]]
    


    
<a href=""><img src="./imgs/21_08.png" height="auto" width="auto" title="caDesign"></a>
    


                  precision    recall  f1-score   support
    
               0       0.98      0.75      0.85        55
               1       0.76      0.98      0.85        45
    
        accuracy                           0.85       100
       macro avg       0.87      0.86      0.85       100
    weighted avg       0.88      0.85      0.85       100
    
    

### 1.5 SoftMax回归/函数/归一化指数函数
#### 1.5.1 自定义SoftMax回归多分类

* 定义线性函数——`linear_weights(self,X)`

用于SoftMax函数输入的对象$z$，是由线性函数计算求取的结果。例如假设有两个特征，每个特征下包含5个样本，$X= \begin{bmatrix}1 & 2 \\4 & 7\\5&6\\3&6\\12&3 \end{bmatrix} $；并假设有3个类标，$y= \begin{bmatrix}0 & 1&2\end{bmatrix} $。根据类标数量（3）和特征数（2）随机初始化权重值为$W=\begin{bmatrix}7 & 5&2 \\3 & 4&9 \end{bmatrix} $，为2(特征数)行3（类标数）列。将每一个样本（包含两个特征值，例如第一个样本特征值为1，2），分别对应权重矩阵每一列的两个权值（例如第1列，7，3）计算加权和，结果为$1 \times 7+2 \times 3=13$，以此类推，对于第一个样本特征值其余对应的权重值计算加权和结果为：13，20。其它样本同上。这个计算的过程就是数组的点积计算，可以由`np.dot`方法实现。由此根据每个类标对应一个初始化的权值，计算了每个样本对应3个类标的输出值，用于SoftMax函数的输入。


```python
X=np.array([[1,2],[4,7],[5,6],[3,6],[12,3]])
print("X=",X)
W=np.array([[7,5,2],[3,4,9]])
print("y=",W)
z=np.dot(X,W)
print("z=",z)
```

    X= [[ 1  2]
     [ 4  7]
     [ 5  6]
     [ 3  6]
     [12  3]]
    y= [[7 5 2]
     [3 4 9]]
    z= [[13 13 20]
     [49 48 71]
     [53 49 64]
     [39 39 60]
     [93 72 51]]
    

* 定义SoftMax函数——`softmax(self,z)`

SoftMax回归/函数，或称为归一化指数函数，是逻辑回归/函数(Logistic Regression)的一种推广。将一个含任意实数的K维向量z"压缩"到另一个K维实向量$\sigma (z)$中，使得每一个元素的范围都在(0,1)之间，并且所有元素的和为1.该函数的形式通常为：$ \sigma (z)_{j} =\frac{ e_{j}^{z}  }{ \sum_{k=1}^K   e_{k}^{z}   } \quad for \quad j =1, \ldots ,K$。SoftMax函数实际上是有限项离散概率分布的梯度对数归一化，广泛应用于多分类问题方法中。在多项逻辑回归和线性判别分析中，函数的输入是从K个不同的线性函数得到的结果，而样本向量$x$属于第$j$个分类的几率为：$P( y=j | x) =\frac{ e^{ x^{T} w_{j}  }  }{ \sum_{k=1}^K   e^{ x^{T} w_{k}  }   }$，这可以被视作K个线性函数$x \mapsto  x^{T} w_{1}  , \ldots ,x \mapsto  x^{T} w_{k} $，SoftMax函数的复合。亦可以表述为，对于输入数据$\{( x_{1}, y_{1}  ),( x_{2}, y_{2}), \ldots ,( x_{n}, y_{n}  )  \}$，有K个类别，即$y_{i}  \in \{1,2, \ldots ,k\}$,那么SoftMax回归主要估算输入数据$x_{i} $会属于每一类的概率，即$h_{ \theta } ( x_{i} )= \begin{bmatrix} p( y_{i}=1 \mid  x_{i}; \theta)\\ p( y_{i}=2 \mid  x_{i}; \theta)\\  \vdots  \\p( y_{i}=k \mid  x_{i}; \theta)  \end{bmatrix} = \frac{1}{ \sum_{k=1}^K  e^{   \theta _{k} ^{T} } x_{i}   }  \begin{bmatrix}e^{   \theta _{1} ^{T} } x_{i} \\e^{   \theta _{2} ^{T} } x_{i} \\ \vdots \\ e^{   \theta _{k} ^{T} } x_{i} \end{bmatrix} $，其中$ \theta _{1} , \theta _{2} , \ldots , \theta _{k}  \in  \theta $是模型的参数， 乘以$ \frac{1}{ \sum_{k=1}^K  e^{   \theta _{k} ^{T} } x_{i}   } $是为了让概率位于[0,1]并且概率之和为1，SoftMax回归将输入数据$x_{i}$归属于类别$j$的概率为：$p( y_{i}=j |  x_{i} ; \theta  )=\frac{e^{   \theta _{j} ^{T} } x_{i}}{ \sum_{k=1}^K  e^{   \theta _{k} ^{T} } x_{i}   }$。

不同的作者可能对同一问题的公式表述有所差异，例如不同的变量符号，或者表述规则，例如上述两种阐述中，模型的权值分别被表述为$w$和$\theta$，亦或者以矩阵的方式表述的方程组等。不管公式的表达方式如何，其核心的算法始终是保持一致的，可以根据自己容易理解的方式来书写表达式。

<a href=""><img src="./imgs/21_03.jpg" height='auto' width='700' title="caDesign"></a>

> 参考文献：
1. http://rinterested.github.io/statistics/softmax.html; https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/;  https://zhuanlan.zhihu.com/p/98061179
2. Aston Zhang,Zack C. Lipton,Mu Li,etc.Dive into Deep Learning[M].；中文版-阿斯顿.张,李沐,扎卡里.C. 立顿,亚历山大.J. 斯莫拉.动手深度学习[M].人民邮电出版社,北京,2019-06-01

* 定义损失函数/交叉熵(cross entropy)损失函数——`loss(self,y_one_hot,y_probs)`

SoftMax运算符将输出变换为一个合法的类标预测分布（联合预测概率，预测概率分布），真实标签通过独热编码转换后的形式也是一种类别分布表达（0值为概率为0，1值为概率为1），二者形状保持一致。评估预测概率和真实标签的分布，只需要估计对应正确类别的预测概率。这样衡量两个概率分布差异的测量函数，可以使用交叉熵(cross entropy)，公式为：$H( y^{(i)},  \widehat{y} ^{(i)}  )=- \sum_{j=1}^q   y^{(i)}_{j}log   \widehat{y} ^{(i)}_{j} $，其中带下标的$y^{(i)}_{j}$是向量$y^{(i)}$中非0即1的元素（独热编码）；$\widehat{y} ^{(i)}_{j}$为联合预测概率$ \widehat{y} ^{(i)}$中对应类标（对应正确类别）的预测概率。

根据交叉熵的定义，SoftMax回归的损失函数（代价函数）计算公式为：$L( \theta )=- \frac{1}{m} [ \sum_{i=1}^m  \sum_{j=1}^k 1 \{ y_{i}=j \}log \frac{ e^{   \theta_{j}  ^{T} } x_{i}  }{ \sum_{l=1}^k e^{   \theta_{l}  ^{T} } x_{i}  } ]$，其中$m$为样本数，$k$代表类标数，$1\{ \bullet \}$是示性函数，1{值为真的表达式}=1，1{值为假的表达式}=0。SoftMax回归的输出值的形状为(m,k)，每一样本（总共m个样本）对应有k（类标数）个概率值，每一类标对应的概率值为$p_{k}$(联合预测概率)。$p_{k}$概率值位于[0,1]之间，当取对数后，对应值转换到[-inf,0]之间。概率越大的值，即越趋近于1，当取对数后其值越趋近于0。示性函数对应数据集类标列的独热编码(One-Hot Encoding)。例如类标有4类[0,1,2,3]，假设一个样本对应的类标为3，经过独热编码后，表示为[0. 0. 0. 1.]，可理解为对应类标位置的值取1，而没有对应的位置均为0。而该样本经过SoftMax计算后结果为[0.31132239 0.2897376  0.19737688 0.20156313]，即该样本对应各个类标的概率值$p_{k},k=0,1,2,3$。对$p_{k}$取对数后，结果为[-1.16692628 -1.23877959 -1.62264029 -1.60165265]，概率越大的值，例如0.31132239，其对数结果越趋近于0，对应-1.16692628。将$p_{k}$取对数后的值与该样本类标的独热编码相乘，结果为[-0. -0. -0. -1.60165265]，即仅保留该样本对应类标位置概率值的对数，用以表述概率值的误差（如果概率值为1，即100%的正确，其对数为0，即误差为0；如果概率值为0，即100%的错误，其对数为负无穷，即误差无限大）。计算所有样本的误差值后求和平均，即为SoftMax损失函数的的结果，表示预测误差的大小。

> 示性函数（特征函数，Characteristic function）可以代表不同的概念，最通常且多数统称为指示函数，方程式为:$1_{A} :X \mapsto \{0,1\}$，其中在集合$X$中，任意子集$A$内一点含值1，于集合$X-A$内一点含值0。

* 定义梯度下降法，权值更新——`gradient(self,X,y_one_hot,y_probs,lr,lambda_)`

利用梯度下降法最小化损失函数，求解$\theta$的梯度，同时因为如果训练数据不多，容易出现过拟合现象，而增加了一个正则项（具体解释见正则化部分），求解梯度公式为（未给推断过程）：$\frac{ \partial L( \theta )}{ \partial   \theta _{j} } =- \frac{1}{m}[ \sum_{i=1}^m  x_{i}(1\{ 
y_{i}=j  \}-p(y_{i}=j |  x_{i}; \theta  ))  ] + \lambda \theta _{j}$，对应的代码为`grad_loss_L=-(1.0/self.m_samples)*np.dot((y_one_hot-y_probs).T,X)+lambda_*self.weights`。


```python
def make_classification_dataset_(n_features=2,n_classes=2,n_samples=100,n_informative=2,n_redundant=0,n_clusters_per_class=1,figsize=(10,10),plot=True):
    from sklearn.datasets import make_classification
    import matplotlib.pyplot as plt
    import numpy as np
    '''
    function - 使用Sklearn提供的make_classification方法，直接构建离散型分类数据集;并打印查看

    Paras:
    参数查看sklearn库make_classification方法
    '''        
    X,y=make_classification(n_samples=n_samples,n_features=n_features,n_classes=n_classes, n_redundant=n_redundant, n_informative=n_informative,n_clusters_per_class=n_clusters_per_class)
    print("类标：",np.unique(y))
    
    if plot:
        fig=plt.figure(figsize=figsize)
        ax=fig.add_subplot(111)
        for label in np.unique(y):
            ax.scatter(X[y==label][:,0],X[y==label][:,1],s=30,label='label={}'.format(label))

        ax.legend(loc='best',frameon=False)
        plt.show()    

    return X,y

X,y=make_classification_dataset_(n_features=2, n_redundant=0, n_informative=2,n_clusters_per_class=1, n_classes=4,n_samples=200)
```

    类标： [0 1 2 3]
    


    
<a href=""><img src="./imgs/21_09.png" height="auto" width="auto" title="caDesign"></a>
    



```python
class softmax_multiClass_classification_UDF:
    '''
    class - 自定义softmax回归多分类
    '''
    def __init__(self,m_samples,n_features,n_classes):
        self.m_samples,self.n_features=m_samples,n_features
        self.n_classes=n_classes
        self.weights=np.random.rand(self.n_classes,self.n_features)
        self.loss_overall=list()
        
    
    def softmax(self,Z):
        '''
        function - 定义softmax函数
        '''
        exp_sum=np.sum(np.exp(Z),axis=1,keepdims=True)
        return np.exp(Z)/exp_sum
    
    def linear_weights(self,X):
        '''
        function - 定义线性函数
        '''
        return np.dot(X,self.weights.T)
    
    def loss(self,y_one_hot,y_probs):
        '''
        function - 定义损失函数
        '''
        return -(1.0/self.m_samples)*np.sum(y_one_hot*np.log(y_probs))
    
    def one_hot(self,y):
        '''
        function - numpy实现one-hot-code
        '''
        return np.squeeze(np.eye(self.n_classes)[y.reshape(-1)])
    
    def gradient(self,X,y_one_hot,y_probs,lr,lambda_):
        '''
        function - 定义梯度下降法，权值更新
        '''
        #求解梯度
        grad_loss_L=-(1.0/self.m_samples)*np.dot((y_one_hot-y_probs).T,X)+lambda_*self.weights
        #更新权值     
        grad_loss_L[:,0]=grad_loss_L[:,0]-lambda_*self.weights[:,0]
        self.weights=self.weights-lr*grad_loss_L    
            
    def predict_test(self,X_,y_):
        '''
        function - 预测
        '''
        y_probs_=self.softmax(self.linear_weights(X_))
        y_pred=np.argmax(y_probs_,axis=1).reshape((-1,1))
        accuracy=np.sum(y_pred==y_.reshape((-1,1)))/len(X_)
        print("accuracy:%.5f"%accuracy)
        
        return y_pred
        
    def loss_curve(self,figsize=(8,5)):
        import matplotlib.pyplot as plt
        '''
        function - 更新打印损失曲线
        '''
        fig=plt.figure(figsize=figsize)
        plt.plot(np.arange(len(self.loss_overall)),self.loss_overall)
        plt.title("loss curve")
        plt.xlabel('epoch')
        plt.ylabel('LOSS')
        plt.show()
        
    def train(self,X,y,epochs,lr=0.1,lambda_=0.01):
        from tqdm import tqdm
        '''
        function - 训练模型
        '''        
        for epoch in tqdm(range(epochs)):
            #执行加权和与softmax函数
            y_probs=self.softmax(self.linear_weights(X))
            #计算损失函数
            y_one_hot=self.one_hot(y.reshape((-1,1)))
            self.loss_overall.append(self.loss(y_one_hot,y_probs))
            #求解梯度与权值更新
            self.gradient(X,y_one_hot,y_probs,lr,lambda_)        
        
            
import numpy as np    
m_samples,n_features,n_classes=X.shape[0],X.shape[1],len(np.unique(y))
sm_classi=softmax_multiClass_classification_UDF(m_samples,n_features,n_classes)  
epochs=1000
sm_classi.train(X,y,epochs,lr=0.1,lambda_=0.01,)
sm_classi.loss_curve()
```

    100%|██████████| 1000/1000 [00:00<00:00, 14936.18it/s]
    


    
<a href=""><img src="./imgs/21_10.png" height="auto" width="auto" title="caDesign"></a>
    



```python
X_,y_=make_classification_dataset_(n_features=2, n_redundant=0, n_informative=2,n_clusters_per_class=1, n_classes=4,n_samples=100,plot=False)
y_pred=sm_classi.predict_test(X_,y_)
```

    类标： [0 1 2 3]
    accuracy:0.86000
    

#### 1.5.2 sklearn机器学习库实现SoftMax回归多分类

SoftMax回归多分类sklearn库提供的方法同上述的二分类，应用`LogisticRegression `实现，只是需要配置`multi_class`参数（default=’auto’，因此也无需配置，会自动识别）；参数`solver`需要配置为求解多分类的优化项。其测试数据集的计算结果同上述自定义类的精度接近。


```python
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt

logreg_multiClassi=LogisticRegression (solver='lbfgs',multi_class="multinomial")
logreg_multiClassi.fit(X,y)
predictions=logreg_multiClassi.predict(X_)
CM=confusion_matrix(y_,predictions,labels=logreg_multiClassi.classes_)
print("confusion_matrix:\n{}".format(CM))
plt.matshow(CM)
plt.title('confusion matrix')
plt.ylabel('True lable')
plt.xlabel('Predicted label')
plt.colorbar()
plt.show()

print(classification_report(y_,predictions))
```

    confusion_matrix:
    [[23  2  0  0]
     [ 2 22  0  1]
     [ 2  7 16  0]
     [ 0  1  0 24]]
    


    
<a href=""><img src="./imgs/21_11.png" height="auto" width="auto" title="caDesign"></a>
    


                  precision    recall  f1-score   support
    
               0       0.85      0.92      0.88        25
               1       0.69      0.88      0.77        25
               2       1.00      0.64      0.78        25
               3       0.96      0.96      0.96        25
    
        accuracy                           0.85       100
       macro avg       0.87      0.85      0.85       100
    weighted avg       0.87      0.85      0.85       100
    
    

#### 1.5.3 PyTorch深度学习库实现SoftMax回归多分类——自定义
应用`torch.utils.data`库中的`TensorDataset`可以很方便的建立张量数据集；应用`random_split`切分数据集为训练、验证和测试数据集；应用`DataLoader`将数据集转换为可迭代对象，用于模型训练的数据加载。并可以定义在每轮迭代中随机均匀采样多个样本组成一个小批量（指定batch_size参数值），使用小批量计算梯度。也可配置参数`num_workers`，使用多进程加速数据读取。如果`num_workers=0`则为不使用额外的进程来加速读取数据。

PyTorch自定义SoftMax回归的算法同上述阐述，算法保持一致，只是在代码的书写过程中，需要根据PyTorch的语法规则做出调整，尤其PyTorch张量运算自动求梯度的方式，大幅度减轻了代码编写的复杂程度。


```python
X,y=make_classification_dataset_(n_features=2, n_redundant=0, n_informative=2,n_clusters_per_class=1, n_classes=3,n_samples=1000)
import torch.utils.data as data_utils
#01-建立数据集
data_iter=data_utils.TensorDataset(torch.from_numpy(X).double(),torch.from_numpy(y).long())
#02-切分数据集
train_set, val_set=data_utils.random_split(data_iter,[800,200])
#03-配置每批大小，batch size
train_data_loader=data_utils.DataLoader(train_set,batch_size=50,shuffle=True,num_workers=2)
val_data_loader=data_utils.DataLoader(val_set,batch_size=50,shuffle=True,num_workers=2)
```

    类标： [0 1 2]
    


    
<a href=""><img src="./imgs/21_12.png" height="auto" width="auto" title="caDesign"></a>
    



```python
class softmax_multiClass_classification_UDF_pytorch:
    '''
    class - 自定义softmax回归多分类_PyTorch版
    '''
    def __init__(self,num_inputs,num_outputs):
        print("num_inputs/feature={},num_outputs/label={}".format(num_inputs,num_outputs))
        self.num_inputs=num_inputs
        self.num_outputs=num_outputs
        self.W=torch.tensor(np.random.normal(0, 0.01, (self.num_inputs, self.num_outputs)), dtype=torch.double,requires_grad=True) #需要对权值求梯度     
        self.b=torch.zeros(num_outputs,dtype=torch.float,requires_grad=True) #需要对偏置求梯度
        self.params=[self.W,self.b]
            
    def SoftMax(self,Z):
        '''
        function - 定义SoftMax函数
        '''
        Z_exp=Z.exp()
        exp_sum=Z_exp.sum(dim=1,keepdim=True)
        return Z_exp/exp_sum
        
    def net(self,X):
        '''
        function - 定义模型，含线性模型输入，SoftMax回归输出
        '''
        return self.SoftMax(torch.mm(X.view((-1,self.num_inputs)),self.W)+self.b)
    
    def cross_entropy(self,y_pred,y):
        '''
        function - 定义交叉熵损失函数
        '''
        return -torch.log(y_pred.gather(1,y.view(-1,1))) #torch.gather，收集输入的特定维度指定位置的数值。即提取出对应正确类别的预测概率
    
    def accuracy(self,y_pred,y):
        '''
        function - 定义分类准确率，即正确预测数量与总预测数量之比
        '''
        return (y_pred.argmax(dim=1)==y).float().mean().item()
    
    def evaluate_accuracy(self,data_iter):
        '''
        funtion - 平均模型net在数据集data_iter上的准确率
        '''
        accu_sum,n=0.0,0
        for X,y in data_iter:
            accu_sum+=(self.net(X).argmax(dim=1)==y).float().sum().item()
            n+=y.shape[0]
        return accu_sum/n
    
    def sgd(self,lr):
        '''
        funtion - 梯度下降
        '''
        for param in self.params:
            param.data-=lr*param.grad    
    
    def train(self,train_iter,epochs,lr,test_iter=None):
        from tqdm.auto import tqdm
        '''
        function - 训练模型
        '''        
        for epoch in tqdm(range(epochs)):   
            train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
            for X,y in train_iter:
                #01-线性模型输入，SoftMax回归输出
                y_pred=self.net(X)
                #02-计算损失函数
                l=self.cross_entropy(y_pred,y).sum()
                #03-参数梯度清零
                if self.params is not None and self.params[0].grad is not None:
                    for param in self.params:
                        param.grad.data.zero_()
                #04-计算给定张量的梯度和，此处为损失函数的反向传播
                l.backward()
                #05-求梯度
                self.sgd(lr)
                #05-每批误差和
                train_l_sum += l.item()
                #06-每批正确率
                train_acc_sum += (y_pred.argmax(dim=1) == y).sum().item()
                n += y.shape[0]
            if test_iter is not None:
                test_acc=self.evaluate_accuracy(test_iter)
            if epoch%100==0:
                print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'% (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))   
                

import torch
import torchvision
import numpy as np

num_inputs,num_outputs=X.shape[1],len(np.unique(y))
sm_classi_pytorch=softmax_multiClass_classification_UDF_pytorch(num_inputs,num_outputs)
epochs,lr=1000,0.1
sm_classi_pytorch.train(train_data_loader, epochs,lr,val_data_loader)
```

    num_inputs/feature=2,num_outputs/label=3
    


    HBox(children=(HTML(value=''), FloatProgress(value=0.0, max=1000.0), HTML(value='')))


    epoch 1, loss 0.3359, train acc 0.871, test acc 0.905
    epoch 101, loss 0.2705, train acc 0.910, test acc 0.895
    epoch 201, loss 0.2767, train acc 0.907, test acc 0.900
    epoch 301, loss 0.2828, train acc 0.910, test acc 0.890
    epoch 401, loss 0.2780, train acc 0.916, test acc 0.810
    epoch 501, loss 0.2718, train acc 0.915, test acc 0.900
    epoch 601, loss 0.2660, train acc 0.914, test acc 0.880
    epoch 701, loss 0.2821, train acc 0.906, test acc 0.885
    epoch 801, loss 0.2689, train acc 0.921, test acc 0.900
    epoch 901, loss 0.2888, train acc 0.907, test acc 0.895
    
    

#### 1.5.4 PyTorch深度学习库实现SoftMax回归多分类——调用已有算法
注意上述分开定义SoftMax回归和交叉熵损失函数可能会造成数值不稳定，而PyTorch提供的`torch.nn.CrossEntropyLoss`方法，结合了`nn.LogSoftmax()`，和`nn.NLLLoss()`于单独的一个类中。因此，在定义net网络时，仅含有一个线性模型，而自定义的`flattenLayer`是用于转换特征输入向量的形状，也置于net网络中。

此时定义的SoftMax回归多分类方法并未以类的形式出现，而是为单独的函数，可以将`flattenLayer`,`evaluate_accuracy`,`sgd_v1`以及`train_v1`函数置于'util.py'工具文件中，方便日后直接调用。


```python
import torch
from torch import nn
import numpy as np
from collections import OrderedDict

#将每批次样本X的形状转换为(batch_size,-1)
class flattenLayer(nn.Module):
    def __init__(self):
        super(flattenLayer,self).__init__()
    def forward(self,x):
        return x.view(x.shape[0],-1)

#定义模型
net=nn.Sequential(
    OrderedDict([
        ('flatten',flattenLayer()),
        ('linear',nn.Linear(num_inputs,num_outputs))        
    ])
    )

#初始化模型的权重参数
nn.init.normal_(net.linear.weight,mean=0,std=0.01)
nn.init.constant_(net.linear.bias,val=0)

#SoftMax,交叉熵损失函数CrossEntropyLossntropyLoss()
loss = nn.CrossEntropyLoss() 

#优化算法
optimizer=torch.optim.SGD(net.parameters(),lr=0.1)

def evaluate_accuracy(data_iter, net):
    '''
    funtion - 平均模型net在数据集data_iter上的准确率
    '''
    accu_sum,n=0.0,0
    for X,y in data_iter:
        accu_sum+=(net(X).argmax(dim=1)==y).float().sum().item()
        n+=y.shape[0]
    return accu_sum/n

def sgd_v1(params,lr):
    '''
    funtion - 梯度下降，v1版
    '''
    for param in params:
        param.data-=lr*param.grad  

def train_v1(net,train_iter, test_iter, loss, num_epochs,params=None, lr=None, optimizer=None,interval_print=100):
    from tqdm.auto import tqdm
    '''
    function - 训练模型，v1版
    
    Paras:
    net - 构建的模型结构
    train_iter - 可迭代训练数据集
    test_iter - 可迭代测试数据集
    loss - 损失函数
    num_epochs - 训练迭代次数
    params=None - 初始化模型参数，以列表形式表述，例如[W,b]
    lr=None, - 学习率
    optimizer=None - 优化函数
    interval_print=100 - 打印反馈信息间隔周期
    '''
    for epoch in tqdm(range(num_epochs)):
        train_l_sum, train_acc_sum, n=0.0, 0.0, 0
        for X, y in train_iter:
            y_pred=net(X)
            l=loss(y_pred,y).sum()
            
            #梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is None:
                sgd_v1(params,lr) #应用自定义的梯度下降法
            else:
                optimizer.step() #应用torch.optim.SGD库的梯度下降法
                
            train_l_sum += l.item()
            train_acc_sum += (y_pred.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        if test_iter is not None:        
            test_acc = evaluate_accuracy(test_iter, net)
        if epoch%interval_print==0:
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'%(epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))          

X,y=make_classification_dataset_(n_features=2, n_redundant=0, n_informative=2,n_clusters_per_class=1, n_classes=3,n_samples=1000,plot=False)
import torch.utils.data as data_utils
#01-建立数据集
data_iter=data_utils.TensorDataset(torch.from_numpy(X).float(),torch.from_numpy(y).long())
#02-切分数据集
train_set, val_set=data_utils.random_split(data_iter,[800,200])
#03-配置每批大小，batch size
train_data_loader=data_utils.DataLoader(train_set,batch_size=50,shuffle=True,num_workers=2)
val_data_loader=data_utils.DataLoader(val_set,batch_size=50,shuffle=True,num_workers=2)    
    
num_epochs=500
train_v1(net=net,train_iter=train_data_loader, test_iter=val_data_loader, loss=loss, num_epochs=num_epochs,params=None, lr=None, optimizer=optimizer)
```

    类标： [0 1 2]
    


    HBox(children=(HTML(value=''), FloatProgress(value=0.0, max=500.0), HTML(value='')))


    epoch 1, loss 0.0165, train acc 0.829, test acc 0.885
    epoch 101, loss 0.0079, train acc 0.871, test acc 0.875
    epoch 201, loss 0.0079, train acc 0.873, test acc 0.875
    epoch 301, loss 0.0079, train acc 0.871, test acc 0.875
    epoch 401, loss 0.0079, train acc 0.874, test acc 0.875
    
    

#### 1.5.5 PyTorch_SoftMax回归多分类，用于图像数据集Fashion-MNIST

* 图像数据集[Fashion-MNIST](https://www.kaggle.com/zalando-research/fashionmnist)

Fashion-MNIST包括60,000个例子的训练集和10,000个例子的测试集。每个示例都是一个$28 \times 28$灰度图像，总共784像素，与来自10个类的标签相关联，可用于替代原始[MNIST手写数字数据集](http://yann.lecun.com/exdb/mnist/)，对机器学习算法进行基准测试。 图像的像素都有一个与之相关联的像素值，表示该像素的明度和暗度，数字越大表示越暗。像素值为0到255的整数。训练和测试数据集有785列，第一列由标签组成，表示服装的物品。其余的列包含关联图像的像素值。如果要定位一幅图像一个像素的位置，可以应用$x=i \times 28+j$定位，其中$i,j$是0到27之间的整数，像素($x$)位于一个$28 \times 28$矩阵的第i行和第j列。

图像标签（类标）为：

0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

服务于PyTorch深度学习框架，主要用来构建计算机视觉模型的[torchvision](https://pytorch.org/docs/stable/torchvision/index.html)包中，`torchvision.datasets`可以用来加载常用的数据集；`torchvision.models`包含常用的模型结果；`torchvision.transforms`用于图片的变换；`torchvision.utils`含有用的工具等。

`.ToTensor()`将所有图像数据shape(h/高,w/宽,c/通道数)，像素值位于[0，255]的PIL图片（或者数据类型为np.unit8的NumPy数组）,转换为tensor张量shape(c,h,w)，数据类型为torch.float32，像素值位于[0.0,1.0]。


```python
import torchvision
import torchvision.transforms as transforms
mnist_train=torchvision.datasets.FashionMNIST(root='./datasets/', train=True, download=True, transform=transforms.ToTensor()) 
mnist_test=torchvision.datasets.FashionMNIST(root='./datasets/', train=False, download=True, transform=transforms.ToTensor())
```

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ./datasets/FashionMNIST\raw\train-images-idx3-ubyte.gz
    


    HBox(children=(HTML(value=''), FloatProgress(value=1.0, bar_style='info', layout=Layout(width='20px'), max=1.0…


    Extracting ./datasets/FashionMNIST\raw\train-images-idx3-ubyte.gz to ./datasets/FashionMNIST\raw
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ./datasets/FashionMNIST\raw\train-labels-idx1-ubyte.gz
    


    HBox(children=(HTML(value=''), FloatProgress(value=1.0, bar_style='info', layout=Layout(width='20px'), max=1.0…


    Extracting ./datasets/FashionMNIST\raw\train-labels-idx1-ubyte.gz to ./datasets/FashionMNIST\raw
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ./datasets/FashionMNIST\raw\t10k-images-idx3-ubyte.gz
    


    HBox(children=(HTML(value=''), FloatProgress(value=1.0, bar_style='info', layout=Layout(width='20px'), max=1.0…


    Extracting ./datasets/FashionMNIST\raw\t10k-images-idx3-ubyte.gz to ./datasets/FashionMNIST\raw
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ./datasets/FashionMNIST\raw\t10k-labels-idx1-ubyte.gz
    
    
    


    HBox(children=(HTML(value=''), FloatProgress(value=1.0, bar_style='info', layout=Layout(width='20px'), max=1.0…


    Extracting ./datasets/FashionMNIST\raw\t10k-labels-idx1-ubyte.gz to ./datasets/FashionMNIST\raw
    Processing...
    Done!
    

    C:\Users\richi\Anaconda3\envs\openCVpytorch\lib\site-packages\torchvision\datasets\mnist.py:480: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ..\torch\csrc\utils\tensor_numpy.cpp:141.)
      return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)
    


```python
print("mnist-数据类型：{}\nmnist_train大小={},mnist_test大小={}".format(type(mnist_train),len(mnist_train),len(mnist_test)))
feature, label=mnist_train[0]

def fashionMNIST_label_num2text(labels_int):
    '''
    function - 将Fashion-MNIST数据集，整型类标转换为名称
    '''
    labels_text=['t-shirt', 'trouser', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels_text[int(i)] for i in labels_int]

print("feature.shape={},lable={}，{}".format(feature.shape,label,fashionMNIST_label_num2text([label])))

def fashionMNIST_show(imgs,labels,figsize=(12, 12)):
    import matplotlib.pyplot as plt
    from IPython import display
    '''
    function - 打印显示Fashion-MNIST数据集图像
    '''
    display.set_matplotlib_formats('svg') #以svg格式打印显示
    _, axs=plt.subplots(1, len(imgs), figsize=figsize)
    for ax,img,label in zip(axs,imgs,labels):
        ax.imshow(img.view((28,28)).numpy())
        ax.set_title(label)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    plt.show()
imgs_len=10    
X=[mnist_train[i][0] for i in range(imgs_len)]
y=[mnist_train[i][1] for i in range(imgs_len)]
fashionMNIST_show(imgs=X,labels=fashionMNIST_label_num2text(y),figsize=(12, 12))

#读取小批量
import torch.utils.data as data_utils
batch_size=256
num_workers=4
train_iter=data_utils.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter=data_utils.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
```

    mnist-数据类型：<class 'torchvision.datasets.mnist.FashionMNIST'>
    mnist_train大小=60000,mnist_test大小=10000
    feature.shape=torch.Size([1, 28, 28]),lable=9，['ankle boot']
    
    
    


    
<a href=""><img src="./imgs/21_13.png" height="auto" width="auto" title="caDesign"></a>
    


将上述常用的自定义函数放置于'util.py'文件中，方便调用。外部仅定义了net模型,loss损失函数和optimizer优化函数。


```python
import util
from torch import nn
from collections import OrderedDict
import torch

#定义模型
num_inputs,num_outputs=28*28,10
net=nn.Sequential(
    OrderedDict([
        ('flatten',util.flattenLayer()),
        ('linear',nn.Linear(num_inputs,num_outputs))        
    ])
    )

#SoftMax,交叉熵损失函数CrossEntropyLossntropyLoss()
loss=nn.CrossEntropyLoss() 

#优化算法
optimizer=torch.optim.SGD(net.parameters(),lr=0.1)

num_epochs=500
util.train_v1(net,train_iter, test_iter, loss, num_epochs,params=None, lr=None, optimizer=optimizer,interval_print=100)
```


    HBox(children=(HTML(value=''), FloatProgress(value=0.0, max=500.0), HTML(value='')))


    epoch 1, loss 0.0031, train acc 0.747, test acc 0.790
    epoch 101, loss 0.0015, train acc 0.868, test acc 0.845
    epoch 201, loss 0.0014, train acc 0.872, test acc 0.842
    epoch 301, loss 0.0014, train acc 0.873, test acc 0.820
    epoch 401, loss 0.0014, train acc 0.875, test acc 0.844
    
    

### 1.5 要点
#### 1.5.1 数据处理技术

* `from tqdm.auto import tqdm`，可以固定打印进度条于一个位置，不会跳转

* `scipy.stats.bernoulli`，伯努利分布

* `sklearn.datasets.make_classification`，生成随机分类数据集

#### 1.5.2 新建立的函数

* function - 按指定区间缩放/映射数组/Rescale an arrary linearly. `rescale_linear(array, new_min, new_max)`

* class - 自定义逻辑回归(二元分类). `LogisticRegression:`

包括：

* function - 根据一元一次函数构建二维离散型分类数据集，类标为1和0. `generate_dataset_linear(self,slop,intercept,num=100,multiple=10,magnitude=50)`

* function - 使用Sklearn提供的make_classification方法，直接构建离散型分类数据集. `make_classification_dataset(self,n_features=2,n_classes=2,n_samples=100,n_informative=2,n_redundant=0,n_clusters_per_class=1)`

* function - 绘制逻辑回归结果. `plot(self,X,y,weights,figsize=(10,10))`

* function - 逻辑回归函数_sigmoid函数. `sigmoid(self,x)`

* function - 随机梯度下降（优化）. `stocGradAscent1(self,dataMatrix, classLabels, numIter=150)`

* function - 代入梯度下降法训练的权重值于逻辑回归函数（sigmoid），预测样本特征对应的类标. `classifyVector(self,inX, weights)`

* function - 测试训练模型. `test_accuracy(self,X_,y_,weights)`

---

* function - 使用Sklearn提供的make_classification方法，直接构建离散型分类数据集;并打印查看. `make_classification_dataset_(n_features=2,n_classes=2,n_samples=100,n_informative=2,n_redundant=0,n_clusters_per_class=1,figsize=(10,10),plot=True)`


* class - 自定义softmax回归多分类

包括：

* function - 定义softmax函数. `softmax(self,Z)`

* function - 定义线性函数. `linear_weights(self,X)`

* function - 定义损失函数. `closs(self,y_one_hot,y_probs)`

* function - numpy实现one-hot-code. `one_hot(self,y)`

* function - 定义梯度下降法，权值更新. `gradient(self,X,y_one_hot,y_probs,lr,lambda_)`

* function - 预测. `predict_test(self,X_,y_`

* function - 更新打印损失曲线. `loss_curve(self,figsize=(8,5))`

* function - 训练模型. `train(self,X,y,epochs,lr=0.1,lambda_=0.01)`

---

* class - 自定义softmax回归多分类_PyTorch版. `softmax_multiClass_classification_UDF_pytorch`

包括：

* function - 定义SoftMax函数. `SoftMax(self,Z)`

* function - 定义模型，含线性模型输入，SoftMax回归输出. `net(self,X)`

* function - 定义交叉熵损失函数. `cross_entropy(self,y_pred,y)`

* function - 定义分类准确率，即正确预测数量与总预测数量之比. `accuracy(self,y_pred,y)`

* funtion - 平均模型net在数据集data_iter上的准确率. `evaluate_accuracy(self,data_iter)`

* funtion - 梯度下降. `sgd(self,lr)`

* function - 训练模型. `train(self,train_iter,epochs,lr,test_iter=None)`

---

* class - 将每批次样本X的形状转换为(batch_size,-1). `flattenLayer(nn.Module)`

* funtion - 平均模型net在数据集data_iter上的准确率. `evaluate_accuracy(data_iter, net)`

* funtion - 梯度下降，v1版. `sgd_v1(params,lr)`

* function - 训练模型，v1版. `train_v1(net,train_iter, test_iter, loss, num_epochs,params=None, lr=None, optimizer=None,interval_print=100)`

* function - 将Fashion-MNIST数据集，整型类标转换为名称. `fashionMNIST_label_num2text(labels_int)`

* function - 打印显示Fashion-MNIST数据集图像. `fashionMNIST_show(imgs,labels,figsize=(12, 12))`

#### 1.5.3 所调用的库


```python
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import numpy as np
import sympy
from tqdm import tqdm
from collections import OrderedDict
from IPython import display

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix,classification_report

import torch.utils.data as data_utils
import torch
import torchvision
from torch import nn
```

#### 1.5.4 参考文献
1. Peter Harrington.Machine learning in action[M].NY:Manning Publications; 1st edition (November 24, 2020)
2. Cavin Hackeling. Mastering Machine Learning with scikit-learn[M].Packt Publishing Ltd.July 2017.Second published. 中文版为：Cavin Hackeling.张浩然译.scikit-learning 机器学习[M].人民邮电出版社.2019.2
3. http://rinterested.github.io/statistics/softmax.html; https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/; https://zhuanlan.zhihu.com/p/98061179
4. Aston Zhang,Zack C. Lipton,Mu Li,etc.Dive into Deep Learning[M].；中文版-阿斯顿.张,李沐,扎卡里.C. 立顿,亚历山大.J. 斯莫拉.动手深度学习[M].人民邮电出版社,北京,2019-06-01
