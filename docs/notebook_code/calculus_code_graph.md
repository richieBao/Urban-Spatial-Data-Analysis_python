> Created on Sat Nov  7 22/18/56 2020 @author: Richie Bao-caDesign设计(cadesign.cn) 

## 1. 微积分基础的代码表述
微积分（Calculus），是研究极限、微分、积分和无穷级数等的一个数学分支。更本质的讲，微积分是一门研究变化的学问。在本书中多处涉及到微积分的知识，例如阐述回归的部分对于残差平方和关于回归系数求微分，并另微分结果为0解方程组得回归系数值，构建回归方程；在梯度下降法中梯度就是分别对每个变量求偏微分；在SIR传播模型的阐述中通过建立易感人群、恢复人群和受感人群的微分方程建立SIR传播模型。可见微积分在数据分析中具有重要的作用，因此有必要以代码的途径结合图表表述阐述微积分的基础知识,为相关数据分析提供预备。以《7天搞定微积分》和《漫画微积分》的内容为讲述的结构，主要使用[SymPy(Calculus)](https://docs.sympy.org/latest/tutorial/calculus.html)库解释微积分。

> 该部分参考文献
> 1. [日]石山平,大上丈彦著.李巧丽译.7天搞定微积分[M].南海出版公司.海口.2010.8.
2. [日]小岛宽之著,十神 真漫画绘制,株式会社BECOM漫画制作,张仲恒译.漫画微积分[M].科学出版社.北京.2009.8.

### 1.1 导数（Derivative）与微分（Differentiation）

* 导数

导数是用来分析变化的，即曲线（函数图像）在某点处的斜率，表示倾斜的程度。对于直线函数求导会得到直线的斜率，对曲线函数求导则能得到各点的斜率（即瞬间斜率）。下述代码使用Sympy库的diff方法计算了曲线上采样点各处的斜率，其具体的过程是先建立曲线图形的函数表达式为：$y=sin(2 \pi x)$， 由diff方法关于x求微分结果为：$y'(x_{0}) =2 \pi cos(2 \pi x)$，通过该微分方程，给定任意一点的横坐标，就可以计算获得曲线对应点的斜率。为了清晰表述采样点各处斜率的变化情况，由导数（斜率值）derivative_value，采样点行坐标sample_x，采样点纵坐标sample_y，假设采样点横坐标的固定变化值为delta_x=0.1，计算sample_x+delta_x处的纵坐标，从而绘制各点处的切线。为了能够清晰的看到各个采样点处导数的倾斜程度，即变化趋势的强弱，对齐所有切线原点于横坐标上，保持各点横轴变化量不变，计算各个结束点的纵坐标，通过向量长度的变化可以确定各点变化趋势的大小，通过向量方向的变化可以确定各点变化趋势的走势。

* 极限

上述计算斜率的方法是直接使用了Sympy库，为了更清晰的理解计算的过程，需要首先了解什么是极限。极限可以描述一个序列的指标愈来愈大时，数列中元素的性质变化趋势，也可以描述函数的自变量接近某一个值时，相对应函数值变化的趋势。例如对于数列（sequence）$a_{n} = \frac{1}{n} $，随着$n$的增大，$a_{n}$从0的右侧越来约接近0，于是可以认为0是这个序列的极限。对于函数的极限可以假设$f(x)$是一个实函数，$c$是一个实数，那么$\lim_{x \rightarrow c} f(x)=L$, 表示当$x$充分靠近$c$时，$f(x)$可以任意的靠近$L$，即$x$趋向$c$时，函数$f(x)$的极限是$L$。用数学算式表示极限，例如$\lim_{n \rightarrow 1} (1-n)$表示使$n$无限接近1时，$1-n$无限接近1-1，即无限接近0。又如$\lim_{n \rightarrow 1}  \frac{ n^{2}-3n+2 }{n-1}= \lim_{n \rightarrow 1}   \frac{(n-1)(n-2)}{n-1}= \lim_{n \rightarrow 1} (n-2)$，即$n$无限接近1时，$\lim_{n \rightarrow 1}  \frac{ n^{2}-3n+2 }{n-1}$无限接近2。

如果要求函数$f(x)$图形点$A$的斜率，点$A$的坐标为$(x_{0},f(x_{0}))$，将点$A$向右移动$\triangle x$，即横向长度差，则纵向长度差为$f(x_{0}+ \triangle  x)-f(x_{0})$，过点$A$的斜率，即$f(x)$在$x_{0}$处的导数（导函数）为：$f' ( x_{0} )$，即：$f' ( x_{0} )= \lim_{ \triangle x \rightarrow 0}  \frac{ \triangle y}{ \triangle x} =\lim_{ \triangle x \rightarrow 0} \frac{f( x_{0}+ \triangle x )-f( x_{0} )}{ \triangle x} $，也可记作$y'( x_{0} ), { \frac{dy}{dx} |} _{x= x_{0} } , \frac{dy}{dx} ( x_{0} ),{ \frac{df}{dx} |} _{x= x_{0} }$等，读作‘对y(f(x))关于x求导’，$d$是derivative(导数)的第一个字母。$\frac{d}{dx} $表示是一个整体，是‘求关于x的导数’的求导计算。

已知曲线图形的函数表达式为$y=sin(2 \pi x)$，在Sympy下建立表达式，根据上述求导极限方程可以得到函数在点$x_{0}$处的求导表达式(导函数)，即下述代码中变量`limit_x_0_expr`为，$\frac{sin( \pi (2d+1.6))-sin(1.6 \pi )}{d} $，其中$d$为横向长度差$\triangle x$，当$\triangle x \mapsto 0$时的极限值即为导数/斜率，计算结果为$2.0 \pi cos(1.6 \pi )$。这与使用Sympy的diff方法求得的关于$x$求导方程$2.0 \pi cos(2 \pi x)$，代入$x_{0}=0.8$的结果一致。

* 误差率

获得$f(x)$导函数，可以求任一点的斜率，例如在$x_{0}=0.8$点处斜率为$2.0 \pi cos(1.6 \pi )$，欲建立该点处的切线方程，需要求得截距，根据$g(x)=ax+b$，其中$a$为斜率，$b$为截距，则有$b=ax-g(x)$，此时$g(x)=f(x)$，而斜率$a$已知，$x=x_{0}=0.8$，求得截距则可建立该点的切线方程$g(x)$。误差率则是以$x$为起点进行变化时，$f(x)$和$g(x)$的值之间的差异，占$x$的变化量的百分比，即$误差率= \frac{(f和g的差异)}{(x的变化量)}$。离$x_{0}$越近，误差率越小。所谓近似成一次函数，就是令原函数的误差率局部为0的情况。所以在讨论局部性质时，可以用一次函数替代原函数进而推导处正确的结论。

* 求导的基本公式

1. $p' =0(p为常数)$;
2. $(px)' =0(p为常数)$;
3. $(af(x))'=a f' (x) $ #常系数微分
4. $\{f(x)+g(x)\} ' =f '(x) +g '(x)$；#和的微分
5. $( x^{n} )' =n x^{n-1} $; #幂函数的导数
6. $\{f(x)g(x)\}'= f(x)'g(x) + f(x)g(x)'  $; #积的微分
7. $\{\frac{f(x)}{f(x)} \}' = \frac{ g'(x)f(x)-g(x) f' (x) }{ f(x)^{2} } $ #商的微分
8. $\{g(f(x))\}' = g'(f(x)) f' (x) $ # 复合函数的微分
9. $\{ f^{-1} (x)\}' = \frac{1}{ f' (x)} $ #反函数的微分方程

* 微分

对于微分的理解可以拓展为函数图像中，某点切线的斜率和函数的变化率。微分是对函数的局部变化率的一种线性描述。其可以近似的描述当函数自变量的取值足够小的改变时，函数的值是怎样变化的。

* 由‘微分=0’可知极值

极大点和极小点是函数增减性发生变化的地方，对研究函数的性质来说是很重要的。极大点、极小点常常会变成最大点、最小点，是求解某些（最优解）问题时十分关键的点。极值条件：$y=f(x)$在$x=a$处为极大点或极小点，则有$f' (a)=0$。即求极大点和极小点，只需找到满足$f' (a)=0$的$a$即可。

增减性的判断条件：当$f' (a)>0$时，所近似的一次函数在$x=a$处呈现递增的趋势，因此可知$f(x)$同样呈现递增趋势；同样当$f' (a)<0$时，$f(x)$处于下降的状态，即不在顶端也不在谷底。

* 平均值定理

对于$a,b(a<b)$来说，存在一个$\zeta $,当$a<\zeta<b$时，满足$f(b)= f' ( \zeta ) (b-a)+f(a)$。


```python
import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy import diff,pprint,limit

x=sympy.symbols('x')
curvilinear_expr=sympy.sin(2*sympy.pi*x) #定义曲线函数

#A-使用Sympy库diff方法求导
derivative_curvilinear_expr=diff(curvilinear_expr,x) #curvilinear_expr 关于x求微分/导数方程
print("curvilinear_expr 关于x求微分/导数方程:")
pprint(derivative_curvilinear_expr,use_unicode=True) 

curvilinear_expr_=sympy.lambdify(x,curvilinear_expr,"numpy")
derivative_expr=sympy.lambdify(x,derivative_curvilinear_expr,"numpy")

t=np.arange(0.0,1.25,0.01)
y=curvilinear_expr_(t)

fig, axs=plt.subplots(1,3,figsize=(26,8))
axs[0].plot(t, y,label="curvilinear graph")
axs[0].set_title(r'derivative', fontsize=20)
axs[0].text(1, -0.6, r'$y\'(x_{0}) =2 \pi cos(2 \pi x)$', fontsize=20)  #$y'(x_{0}) =2 \pi cos(2 \pi x)$    $\sum_{i=0}^\infty x_i$
axs[0].text(0.6, 0.6, r'$y=sin(2 \pi x)$',fontsize=20)
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)

#采样原点
sample_x=t[::5]
sample_y=curvilinear_expr_(sample_x)
axs[0].plot(sample_x,sample_y,'o',label='sample points',color='black')

#采样终点
derivative_value=derivative_expr(sample_x) #求各个采样点的导数（斜率）
delta_x=0.1 #x向变化量
sample_x_endPts=sample_x+delta_x
sample_y_endPts=derivative_value*delta_x+sample_y

def demo_con_style_multiple(a_coordi,b_coordi,ax,connectionstyle):
    '''
    function - 在matplotlib的子图中绘制多个连接线
    reference - matplotlib官网Connectionstyle Demo :https://matplotlib.org/3.3.2/gallery/userdemo/connectionstyle_demo.html#sphx-glr-gallery-userdemo-connectionstyle-demo-py

    Paras:
    a_coordi - 起始点点的x，y坐标
    b_coordi - 结束点的x，y坐标
    ax - 子图
    connectionstyle - 连接线的形式
    '''
    x1, y1=a_coordi[0],a_coordi[1]
    x2, y2=b_coordi[0],b_coordi[1]

    ax.plot([x1, x2], [y1, y2], ".")
    for i in range(len(x1)):
        ax.annotate("",
                    xy=(x1[i], y1[i]), xycoords='data',
                    xytext=(x2[i], y2[i]), textcoords='data',
                    arrowprops=dict(arrowstyle="<-", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle=connectionstyle,
                                    ),
                    )
    
demo_con_style_multiple((sample_x,sample_y),(sample_x_endPts,sample_y_endPts),axs[0],"arc3,rad=0.")    
demo_con_style_multiple((sample_x,sample_y*0),(sample_x_endPts,sample_y_endPts-sample_y),axs[1],"arc3,rad=0.")   

#B-使用极限方法求导
axs[2].set_title(r'$y=f(x)=sin(2 \pi x)$', fontsize=20)
axs[2].plot(t, y,label="curvilinear graph")
import util
dx=0.15
x_0=0.8
util.demo_con_style((x_0,curvilinear_expr_(x_0)),(x_0+dx,curvilinear_expr_(x_0+dx)),axs[2],"angle,angleA=-90,angleB=180,rad=0")    
axs[2].text(x_0+0.05, curvilinear_expr_(x_0)-0.1, "△ x", family="monospace",size=20)
axs[2].text(x_0+0.2,curvilinear_expr_(x_0+dx)-0.3, r"$△ y=f(x_{0}+△ x)-f(x_{0})$", family="monospace",size=20)
color = 'blue'
axs[2].annotate(r'$A:(x_{0},f(x_{0}))$', xy=(x_0, curvilinear_expr_(x_0)), xycoords='data',xytext=(x_0-0.15, curvilinear_expr_(x_0)+0.2), textcoords='data',weight='bold', color=color,arrowprops=dict(arrowstyle='->',connectionstyle="arc3",color=color))
axs[2].annotate(r'$f(x_{0}+△ x)$', xy=(x_0+dx, curvilinear_expr_(x_0+dx)), xycoords='data',xytext=(x_0+dx+0.1, curvilinear_expr_(x_0+dx)+0.2), textcoords='data',weight='bold', color=color,arrowprops=dict(arrowstyle='->',connectionstyle="arc3",color=color))
axs[2].set_xlabel('x')
axs[2].set_ylabel('y')
axs[2].spines['right'].set_visible(False)
axs[2].spines['top'].set_visible(False)

d=sympy.symbols('d')
limit_x_0_expr=(curvilinear_expr.subs(x,x_0+d)-curvilinear_expr.subs(x,x_0))/d #函数f(x)在点x_0处的极限方程
print("f(x)在x_0处的求导方程：")
pprint(limit_x_0_expr)
limit_x_0=limit(limit_x_0_expr,d,0)
print(r"f(x)在x_0处的导数/斜率为：")
pprint(limit_x_0)

t_=np.arange(0.6,1.2,0.01)
intercept=curvilinear_expr_(x_0)-limit_x_0*x_0
axs[2].plot(t_,limit_x_0*t_+intercept,'--r',label="derivative of f(x) at x_0") #limit_x_0*t_+intercept即为x_0处的切线方程

#C-（x_0）误差率
gx=limit_x_0*x+intercept
x_1=x_0+dx
err=(curvilinear_expr.subs(x,x_1)-gx.subs(x,x_1))/dx
print("x_0点导函数，在x_1点的误差率：%.2f"%err)

axs[0].legend(loc='lower left', frameon=False)
axs[2].legend(loc='lower left', frameon=False)
plt.show()
```

    curvilinear_expr 关于x求微分/导数方程:
    2⋅π⋅cos(2⋅π⋅x)
    f(x)在x_0处的求导方程：
    sin(π⋅(2⋅d + 1.6)) - sin(1.6⋅π)
    ───────────────────────────────
                   d               
    f(x)在x_0处的导数/斜率为：
    2.0⋅π⋅cos(1.6⋅π)
    x_0点导函数，在x_1点的误差率：2.34
    


    
<a href=""><img src="./imgs/14_01.png" height="auto" width="auto" title="caDesign"></a>
    


### 1.2 积分(Integrate)
积分是导数的逆运算（针对计算方式而言），利用积分可以求出变化的规律和不规整图形的面积。积分和导数通常配套使用，合成为微积分。积分通常分为定积分和不定积分两种。对于定积分，给定一个正实值函数$f(x)$,$f(x)$在一个实数区间$[a,b]$上的定积分为$\int_a^b f(x)dx $,可以在数值上理解为在$o_{xy} $坐标平面上，由曲线$（x,f(x)）(x \in [a,b])$，直线$x=a,x=b$以及$x$轴围成的曲边梯形的面积值（一种确定的实数值）。$f(x)$的不定积分（或原函数）是指任何满足导数是函数$f(x)$的函数$F(x)$。一个函数$f(x)$的不定积份不是唯一的：只要$F(x)$是$f(x)$的不定积分，那么与之相差一个常数的函数$F(x)+C$也是$f$的不定积分。$  \int_ a^ b f(x)dx $读作求函数$f(x)$关于$x$的积分，$  \int_ a^ b f(x)dy $读作求函数$f(x)$关于$y$的积分，因为积分是导数的逆运算，因此可以理解为‘关于$x$求导得到$f(x)$的原函数即为积分’。因此，对于表述‘计算求导后得$f(x)$的函数’，‘求$f(x)$的不定积分’，‘求$f(x)$的原函数’，这三种表达方式意思相同。

$f(x)$是基础函数，$f' ( x )= \lim_{ \triangle x \rightarrow 0}  \frac{ \triangle y}{ \triangle x} =\lim_{ \triangle x \rightarrow 0} \frac{f( x+ \triangle x )-f( x )}{ \triangle x}=\frac{df(x)}{dx} 
$是$f(x)$导函数的各种表述，$F(x)= \int f' ( x )dx$，为$f'(x)$的不定积分，即原函数(如果确定了常数C，即为$f(x)$)。通常$\int_a^b f(x)dx=F(x) |_a^b   $表示定积分，$\int f(x)dx=F(x) $表示不定积分。

* 不定积分、定积分和面积

$\int f(x)dx$实际上表示将$f(x) \times dx$进行$\int$（积分），而$\int$是'summantion（合计）'的开头字母的变形，表示对$f(x) \times dx$的合计之意。$f(x)$是‘与$x$对应的$y$轴坐标’，$dx$表示延$x$轴的最小增量。因此$f(x) \times dx$就是变化横轴增量$dx$下矩形的面积。当对所有位于区间$[a,b]$下变化增量为$dx$的矩形面积求积分（合计）后（宽度极限小的长方形的集合），即为区间为$[a,b]$的横轴与曲线围合的面积。

* 区分求积法

对于函数$f(x)$，给定区间$[a,b]$，假设进行$n$次分割，长方形从左向右依次为$x_{1} ,x_{2},x_{3}, \ldots ,x_{k},\ldots ,x_{n} $，单个矩形的面积为$\frac{b-a}{n} \times f( x_{k} ) $，将全部的长方形面积加起来为$S_{a-b} =\frac{b-a}{n} \times f( x_{1} )+ \frac{b-a}{n} \times f( x_{2} )+ \ldots +\frac{b-a}{n} \times f( x_{n} )=\frac{b-a}{n}\{f(x_{1})+fx_{2}+ \ldots +fx_{n}\}= \lim_{n \rightarrow  \infty } \frac{b-a}{n}\{f(x_{1})+fx_{2}+ \ldots +fx_{n}\}= \lim_{n \rightarrow \infty}  \frac{b-a}{n} \sum_{k=1}^n f(x_{k}) $(或$\lim_{n \rightarrow \infty}  \frac{b-a}{n} \sum_{k=0}^{n-1}f(x_{k})$)。

下述代码在使用区分求积法时，给定的函数为$f' (x)= x^{2}  $（是函数$f(x)= \frac{ x^{3} }{3}$ 的导函数），已知区间为$[a,b]$，依据上述公式则有$S_{a-b}=\lim_{n \rightarrow \infty}  \frac{b-a}{n} \sum_{k=0}^{n-1}f(x_{k})=\lim_{n \rightarrow \infty}  \frac{b-a}{n}\sum_{k=0}^{n-1} (a+k \times \frac{b-a}{n} )^{2} $。在使用极限计算求和公式时，需要使用`doit()`方法计算不被默认计算的对象（极限、积分、求和及乘积等），否则不能计算极限。

定积分求给定区间的面积，直接使用Sympy提供的integrate方法，给定区间计算结果约为0.29，与区分求积法计算结果相同。

* 换元积分公式

对于$f(x)$，将变量$x$替换为一个关于变量$y$的函数，即$x=g(y)$时，对于$f(x)$的定积分$S= \int_a^b f(x)dx$的值，用$y$表示为：$\int_a^b f(x)dx= \int_ \alpha ^ \beta f(g(y)) g'(y)dy   $。


```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import sympy
from sympy import diff,pprint,integrate,oo,Sum,limit #oo 为正无穷

x,n,k=sympy.symbols('x n k')
curvilinear_expr=x**3/3 #定义曲线函数

derivative_curvilinear_expr=diff(curvilinear_expr,x) #curvilinear_expr 关于x求微分/导数方程
print("曲线函数curvilinear_expr为：")
pprint(curvilinear_expr)
print("curvilinear_expr 关于x求微分/导数，导函数为:")
pprint(derivative_curvilinear_expr,use_unicode=True) 

integrate_derivative_curvilinear_expr=integrate(derivative_curvilinear_expr,x)
print("curvilinear_expr导函数的积分：")
pprint(integrate_derivative_curvilinear_expr)

curvilinear_expr_=sympy.lambdify(x,curvilinear_expr,"numpy")
derivative_expr=sympy.lambdify(x,derivative_curvilinear_expr,"numpy")

t=np.arange(-1.25,1.25,0.01)
y=curvilinear_expr_(t)

fig=plt.figure(figsize=(26,8))
ax=fig.add_subplot(111)
ax.plot(t, y,label="curvilinear graph")
ax.plot(t, derivative_expr(t),'--c',label="derivative graph")
ax.set_title(r'Integrate and derivative', fontsize=20)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.axhline(0,color='black',linewidth=0.5)

a,b=0.5,1.0 #定义区间
ix=np.linspace(a,b,10)
iy=derivative_expr(ix)
ax.plot(ix,iy,'o',label='ab division',color='black')
verts=[(a, 0)]+list(zip(ix, iy))+[(b, 0)]
poly=Polygon(verts, facecolor='0.9', edgecolor='0.5') #绘制面积区域
ax.add_patch(poly)
plt.text(0.5 * (a + b), 0.2, r"$\int_a^b f(x)\mathrm{d}x$",horizontalalignment='center', fontsize=20)
ax.set_xticks((a, b))
ax.set_xticklabels(('$a$', '$b$'))
ax.stem(ix,iy,'-.')

#A-使用区分求积法求取面积
Sum_ab=(b-a)/n*Sum(derivative_curvilinear_expr.subs(x,a+k*(b-a)/n),(k,0,n-1)) #面积求和公式
print("所有长方形面积之和的公式：\n")
pprint(Sum_ab)
print("doit():\n")
pprint(Sum_ab.doit())
S_ab=limit(Sum_ab.doit(),n,oo)
print("区分求积法计算的面积=",S_ab)

#B-使用定积分求面积（函数）
S_ab_integrate=integrate(derivative_curvilinear_expr,(x,a,b))
print("定积分计算的面积=",S_ab_integrate)


ax.legend(loc='lower left', frameon=False)
plt.xticks(fontsize=20)
plt.show()
```

    曲线函数curvilinear_expr为：
     3
    x 
    ──
    3 
    curvilinear_expr 关于x求微分/导数，导函数为:
     2
    x 
    curvilinear_expr导函数的积分：
     3
    x 
    ──
    3 
    所有长方形面积之和的公式：
    
        n - 1              
         ____              
         ╲                 
          ╲               2
           ╲       ⎛k    ⎞ 
    0.5⋅   ╱  0.25⋅⎜─ + 1⎟ 
          ╱        ⎝n    ⎠ 
         ╱                 
         ‾‾‾‾              
        k = 0              
    ───────────────────────
               n           
    doit():
    
        ⎛             ⎛ 2    ⎞        ⎛ 3    2    ⎞⎞
        ⎜             ⎜n    n⎟        ⎜n    n    n⎟⎟
        ⎜         0.5⋅⎜── - ─⎟   0.25⋅⎜── - ── + ─⎟⎟
        ⎜             ⎝2    2⎠        ⎝3    2    6⎠⎟
    0.5⋅⎜0.25⋅n + ──────────── + ──────────────────⎟
        ⎜              n                  2        ⎟
        ⎝                                n         ⎠
    ────────────────────────────────────────────────
                           n                        
    区分求积法计算的面积= 0.291666666666667
    定积分计算的面积= 0.291666666666667
    


    
<a href=""><img src="./imgs/14_02.png" height="auto" width="auto" title="caDesign"></a>
    


### 1.3 泰勒展开式（Taylor expansion）
在上述阐述误差率时，在曲线局部使用一次函数替代（近似）曲线，例如对于函数$f(x)$，令$p= f' (a),q=f(a)$,则在距$x=a$很近的地方，能够将$f(x)$近似为$f(x) \sim q+p(x-a)$。使用一次函数其误差率相对较高，如果近似为二次函数或者三次函数，是否可以降低误差率？泰勒展开就是将复杂的函数改写成多项式。

一般函数$f(x)$(要能够无限次的进行微分)，则可以表示成如下形式，$f(x)= a_{0}+  a_{1}x+ a_{2} x^{2} + a_{3} x^{3}+ \ldots + a_{n} x^{n}+ \ldots $，右边称为$f(x)$的泰勒展开。这个公式在包含$x=0$的某个限制区域内，才意味着函数$f(x)$同无限次多项式时完全一致的，然而，一旦超出这个无限制区域，右边将会变成一个无法确定的数。例如对于函数$f(x)= \frac{1}{1-x} $，有$(f(x)=) \frac{1}{1-x}=1+x+ x^{2}  + x^{3} + x^{4} + \ldots $，令$x=0.1$，有$f(0.1)= \frac{1}{1-0.1} = \frac{1}{0.9} = \frac{10}{9}  = 1.111 \ldots $，及右边为$1+0.1+ 0.1^{2}  +0.1^{3} + 0.1^{4} + \ldots=1.1111111 \ldots $，因此左右相等。但是如果令$x=2$是，左右则不会相等。对于上述函数，只有满足$-1 < x < 1$的$x$才成立。

* 泰勒展示的求解方法——确定系数

对于$f(x)= a_{0}+  a_{1}x+ a_{2} x^{2} + a_{3} x^{3}+ \ldots + a_{n} x^{n}+ \ldots$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;式(1)

首先，带入$x=0$，由$f(0)= a_{0} $,可知0次常数项$a_{0}$为$f(0)$。------（A）

然后对式(1)进行微分，$ f' (x)= a_{1}+  2a_{2} x +3 a_{3} x^{2}+ \ldots + na_{n} x^{n-1}+ \ldots$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;式(2)

将$x=0$带入式(2),由$f' (0)=  a_{1} $，可知一次系数$ a_{1}$为$f' (0)$。------（B）

继续对式(2)进行微分，$ f'' (x)= 2a_{2}+6 a_{3} x+ \ldots + n(n-1)a_{n} x^{n-2}+ \ldots$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;式(3)

代入$x=0$，可知二次系数$a_{2}$为$\frac{1}{2}  f''(0) $。------（C）

对式(3)进行微分，$f''' (x)=6 a_{3}+ \ldots + n(n-1)(n-2)a_{n} x^{n-3}+ \ldots$， 

由此可知，三次系数$a_{3}$为$\frac{1}{6}  f'''(0) $。

持续进行这种运算，$n$次微分后，就应该得到，$f^{(n)}  (x)=n(n-1) \ldots  \times 2 \times 1 a_{n} + \ldots $，其中$f^{(n)}  (x)$表示$n$次微分后的$f(x)$。

由此可知，$n$次系数$a_{n}$为$\frac{1}{n!}   f^{(n)} (0)$。 $n!$，读作‘$n$的阶乘’，它表示$n \times (n-1) \times (n-2) \times  \ldots  \times 2 \times 1$。

---

对$f(x)$进行泰勒展开，便有

$f(x)=f(0)+ \frac{1}{1!}  f' (0)x+\frac{1}{2!}  f'' (0)  x^{2} +\frac{1}{3!}  f''' (0)  x^{3}+ \ldots +\frac{1}{n!}  f^{(n)}  (0)  x^{n}+ \ldots $

上述公式中，

$f(0)$  <------0次的常数项， 即$ a_{0} =f(0)$ ------（A）

$f' (0)x$ <------1次项， 即$ a_{1} =f' (0)$------（B）

$\frac{1}{2!}  f'' (0)  x^{2}$<------2次项， 即$ a_{2} =\frac{1}{2!}  f'' (0)$------（C）

$\frac{1}{3!}  f''' (0)  x^{3}$<------3次项， 即 a_{3} =\frac{1}{3!}  f''' (0) ------（D）


泰勒展开，不一定非要从$x=0$的地方开始，也可以从$( x_{0},f(x_{0}) )$处开始，此时只需要将$0$替换为$x_{0}$,展开方法同上，得：

$f(x)=f( x_{0} )+ \frac{1}{1!}  f' (x_{0} )(x-x_{0} )+\frac{1}{2!}  f'' (x_{0} )  (x-x_{0} )^{2} +\frac{1}{3!}  f''' (x_{0} )  (x-x_{0} )^{3}+ \ldots +\frac{1}{n!}  f^{(n)}  (x_{0} )  (x-x_{0} )^{n}+ \ldots $

* 误差项

$ R_{n}(x) = \frac{  f^{(n+1)}( \xi ) }{(n+1)!} (x- x_{0} ) ^{n+1} $ （推导过程略）


```python
import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy import pprint,solve,diff,factorial

x,a_1,b_1,x_i=sympy.symbols('x a_1 b_1 x_i')

#定义原函数
cos_curve=sympy.cos(x)
cos_curve_=sympy.lambdify(x,cos_curve,"numpy")
#定义区间
a,b=0,6 
ix=np.linspace(a,b,100)

fig=plt.figure(figsize=(26,8))
ax=fig.add_subplot(111)
ax.plot(ix,cos_curve_(ix) ,label="cosx graph")

#A-x=0位置点近似多项式
x_0=0
#近似曲线系数计算
a_0=cos_curve.subs(x,x_0)
a_1=diff(cos_curve,x).subs(x,x_0)/factorial(1)
a_2=diff(cos_curve,x,x).subs(x,x_0)/factorial(2)
a_3=diff(cos_curve,x,x,x).subs(x,x_0)/factorial(3)
a_4=diff(cos_curve,x,x,x,x).subs(x,x_0)/factorial(4)

ix_=ix[:30]
#1阶近似
f_1=a_1*x+a_0
print("1阶函数：",f_1)
ax.plot(ix_,[1]*len(ix_),label="1 order")

#2阶近似
f_2=a_2*x**2+a_1*x+a_0
f_2_=sympy.lambdify(x,f_2,"numpy")
ax.plot(ix_,f_2_(ix_),label="2 order")
print("2阶函数：")
pprint(f_2)

#3阶近似
f_3=a_3*x**3+a_2*x**2+a_1*x+a_0
f_3_=sympy.lambdify(x,f_3,"numpy")
ax.plot(ix_,f_3_(ix_),'--',label="3 order")
print("3阶函数：")
pprint(f_3)

#4阶近似
f_4=a_4*x**4+a_3*x**3+a_2*x**2+a_1*x+a_0
f_4_=sympy.lambdify(x,f_4,"numpy")
ax.plot(ix_,f_4_(ix_),label="4 order")
print("4阶函数：")
pprint(f_4)

#B-x任一点近似多项式(3阶为例)
x_1=5
f_x=cos_curve.subs(x,x_i)+diff(cos_curve,x).subs(x,x_i)*(x-x_i)/factorial(1)+diff(cos_curve,x,x).subs(x,x_i)*(x-x_i)**2/factorial(2)+diff(cos_curve,x,x,x).subs(x,x_i)*(x-x_i)**3/factorial(3) #近似多项式
f_x_1=f_x.subs(x_i,x_1)
f_x_1_=sympy.lambdify(x,f_x_1,"numpy")
ax.plot(ix[60:],f_x_1_(ix[60:]),label="3 order_x_1",c='red',ls='-.')
ax.plot(x_1,cos_curve.subs(x,x_1),'o')
ax.annotate(r'$x_1$', xy=(x_1, cos_curve.subs(x,x_1)), xycoords='data',xytext=(x_1-0.1,cos_curve.subs(x,x_1)+0.3), textcoords='data',weight='bold', color='red',arrowprops=dict(arrowstyle='->',connectionstyle="arc3",color='red'),fontsize=25)

#误差项(3阶)
xi=x_1+0.25
c,d=x_1-0.5,x_1+0.5
error=diff(cos_curve,x,x,x,x).subs(x,xi)*(d-c)**4/factorial(4)
print("3阶多项式区间[%.2f,%.2f]内位置点%.2f的误差为：%.2f"%(c,d,xi,error))

ax.set_title(r'Taylor expansion', fontsize=20)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.axhline(0,color='black',linewidth=0.5)

ax.legend(loc='lower left', frameon=False)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()
```

    1阶函数： 1
    2阶函数：
         2
        x 
    1 - ──
        2 
    3阶函数：
         2
        x 
    1 - ──
        2 
    4阶函数：
     4    2    
    x    x     
    ── - ── + 1
    24   2     
    3阶多项式区间[4.50,5.50]内位置点5.25的误差为：0.02
    


    
<a href=""><img src="./imgs/14_03.png" height="auto" width="auto" title="caDesign"></a>
    


### 1.4 偏微分（偏导数 Partial derivative）

* 偏微分

函数$z=f(x,y)$，在某个邻域内的所有点$(x,y)$都可以关于$x$进行偏微分时，在点$(x,y)$处，关于$x$的偏微分系数$f_{x} (x,y)$所对应的函数$(x,y) \mapsto f_{x} (x,y)$被称为$z=f(x,y)$关于$x$的偏导数。可表示为：$f_{x}, f_{x}(x,y), \frac{ \partial f}{ \partial x} ,\frac{ \partial z}{ \partial x}$。

同样，在这个邻域内的所有点$(x,y)$都可以关于$y$进行偏微分时，所对应的$(x,y) \mapsto f_{y} (x,y)$被称为$z=f(x,y)$关于$y$的偏导数。可表示为：$f_{y}, f_{y}(x,y), \frac{ \partial f}{ \partial y} ,\frac{ \partial z}{ \partial y}$。求偏导数的过程叫做偏微分。

偏微分计算直接使用Sympy库的diff方法。

* 全微分

由$z=f(x,y)$在$(x,y)=(a,b)$处的近似一次函数可知

$f(x,y) \sim  f_{x}(a,b)(x-a)+ f_{y}(a,b)(x-b)+f(a,b)$，可以将其改写为：


$f(x,y)-f(a,b) \sim  \frac{ \partial f}{ \partial x} (a,b)(x-a)+ \frac{ \partial f}{ \partial y}(a,b)(x-b)$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;式(1)

$f(x,y)-f(a,b)$意味着，当点$(a,b)$向$(x,y)$变化时，高度$z(=f(x,y))$的增量，效仿一元函数的情况写作$\triangle z$。另外，$x-a$为$\triangle x$，$y-b$为$\triangle y$。

此时，式(1)可以写作

$\triangle z \sim  \frac{ \partial z}{ \partial x}  \triangle x+ \frac{ \partial z}{ \partial y}  \triangle y$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;式(2)，$(x \sim a,y \sim b)$时，这个式子意味着：对于函数$z=f(x,y)$，当$x$由$a$增加了$\triangle x$，$y$由$b$增加了$\triangle y$后，$z$就相应增加了$ \frac{ \partial z}{ \partial x}  \triangle x+ \frac{ \partial z}{ \partial y}  \triangle y$。

$ \frac{ \partial z}{ \partial x}  \triangle x$表示'$y$固定在$b$时$x$方向上的增量'，

$ \frac{ \partial z}{ \partial y}  \triangle y$表示'$x$固定在$a$时$y$方向上的增量'。

说明‘$z(=f(x,y))$’的增量可以分解为$x$方向上的增量与$y$方向上的增量之和。

将式(2)作理想化（瞬时化）处理了，得，

$dz= \frac{ \partial z}{ \partial x} dx+\frac{ \partial z}{ \partial y} dy$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;式(3)

或者，$df= f_{x} dx+ f_{y} dy$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;式(4)

式(3)(4)被称为全微分公式。即，$(曲面高度的增量)=（x方向上的微分系数） \times (x方向上的增量)+（y方向上的微分系数） \times (y方向上的增量)$

* 链式法则公式(Chain rule)

当$z=f(x,y),x=a(t),y=b(t)$时，$\frac{dz}{dt}= \frac{ \partial f}{ \partial x} \frac{da}{dt}   + \frac{ \partial f}{ \partial y} \frac{db}{dt}$。（推导过程略）

> （偏）微分在机器学习领域中广泛应用，具体可以参看‘梯度下降法’部分，对寻找极值的解释。


```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sympy
from sympy import pprint,diff

x,y=sympy.symbols('x y')
f=-(x**2+x*y+y**2)/10
f_=sympy.lambdify([x,y],f,"numpy")

fig=plt.figure(figsize=(12,12))
ax=fig.add_subplot(111,projection='3d')

x_i=np.arange(-5,5,0.1)
y_i=np.arange(-5,5,0.1)
x_mesh,y_mesh=np.meshgrid(x_i,y_i)
mesh=f_(x_mesh,y_mesh)
surf=ax.plot_surface(x_mesh,y_mesh ,mesh, cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=0.5,)

#偏微分
px=diff(f,x)
py=diff(f,y)
print("偏微分x：∂𝑓/∂𝑥=")
pprint(px)
print("偏微分y：∂𝑓/∂y=")
pprint(py)

x_0,y_0=-3,3
z_0=f.subs([(x,x_0),(y,y_0)])
ax.plot(x_0, y_0,z_0, marker = "o",color = "red",label = "x_0")

#平行于xz面，绘制点（x_0,y_0）的切线。关于x的偏导数
xz_dx=3 
xz_dz=px.subs([(x,x_0),(y,y_0)])*xz_dx
ax.plot((x_0,x_0+xz_dx),(y_0,y_0),(z_0,z_0+xz_dz),label="xz_tangent")

#平行于yz面，绘制点（x_0,y_0）的切线。关于y的偏导数
yz_dy=3
yz_dz=py.subs([(x,x_0),(y,y_0)])*yz_dy
ax.plot((x_0,x_0),(y_0,y_0+yz_dy),(z_0,z_0+xz_dz),label="yz_tangent")

ax.view_init(30,100) #可以旋转图形的角度，方便观察
ax.legend()
plt.show()  
```

    偏微分x：∂𝑓/∂𝑥=
      x   y 
    - ─ - ──
      5   10
    偏微分y：∂𝑓/∂y=
      x    y
    - ── - ─
      10   5
    


    
<a href=""><img src="./imgs/14_04.png" height="auto" width="auto" title="caDesign"></a>
    


### 1.5 要点
#### 1.5.1 数据处理技术

* 使用Sympy计算微积分

#### 1.5.2 新建立的函数

* function - 在matplotlib的子图中绘制多个连接线，`demo_con_style_multiple(a_coordi,b_coordi,ax,connectionstyle)`

#### 1.5.3 所调用的库


```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sympy
from sympy import diff,pprint,limit,integrate,oo,Sum,factorial,solve
import util
from matplotlib.patches import Polygon
```

#### 1.5.4 参考文献
1. [日]石山平,大上丈彦著.李巧丽译.7天搞定微积分[M].南海出版公司.海口.2010.8.
2. [日]小岛宽之著,十神 真漫画绘制,株式会社BECOM漫画制作,张仲恒译.漫画微积分[M].科学出版社.北京.2009.8.
