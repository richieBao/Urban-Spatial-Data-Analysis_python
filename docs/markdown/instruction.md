# 指南
## 注重如何将知识讲授明白的阶段
如果作者自身一知半解，或仅观一井之天，那么是无论如何不能把问题解释清楚的，因此第1阶段（炼虚期）花费了不少时间来实验积累，终能勉强一窥全貌，迈入了第2个阶段（合体期）。在该阶段，除了继续累积实验外，重点是如何把知识讲授清楚，这也是不在完成全书之后上线社区，而是在成书过程中就以互动社区与伙伴们交流的动机。

## 面向的读者
书名《城市空间数据分析方法——PYTHON语言实现》反映了研究专业方向，城市规划/风景园林/建筑/生态规划/地理信息等专业；而人群，高校在校学生和研究者、工作一族，均未尝不可；对于python语言基础要求，需要有最低的入门基础，而这个问题，我们也试图尝试在该阶段就能够给予解决，以单独的热身形式出现，或者其它。

## 如何学习
这是读者最为关心的问题，也是作者需要解决的问题。我们搭建了系统的资源平台来满足基本的数据获取、代码运行、互动讨论来辅助有效的学习，而最为关键的是如何阐释，这个会在该阶段不断实验、调整达到所期冀的要求，这个与社区里伙伴们的反馈亦栖息相关。

## 必要的知识补充
在该书的阐述中，因为设计/规划专业的特殊性和学科发展的程度，很多知识在大学中并未配置相关课程，因此城市空间数据分析过程中涉及到的很多知识点将结合相关文献，融入到各项实验当中，一定程度上减轻读者再次查找相关知识点的负担。并由简单的数据入手，逐步的到复杂数据的分析，可以更清楚的理解分析的内容。<穿插的简单数据示例以🍅标识，返回实验数据以🐨标识。——可能放弃这种标识>

## 由案例来学习
千言万语不如一个简单的案例示范。通常查看某一个函数或者方法及属性的功用，最好的方法是直接搜索案例来查看数据结构的变化，往往不需要阅读相关解释文字，就可以从数据的前后变化中获悉该方法的作用，进而辅助阅读文字说明，进一步深入理解或者找到不明地方的解释。

## 充分利用搜索引擎
代码的世界摸不到边际，更是不可能记住所有方法函数，而且代码库在不断的更新完善，记住的不一定是对的。学习代码关键的仍旧是养成写代码的习惯，训练写代码的思维，最终是借助这个工具来解决实际的问题。那么，那些无数的库、函数、方法、属性等内容，通常是在写代码过程中，充分利用搜索引擎来查找，其中往往能够找到答案的一个平台是[StackOverflow](https://stackoverflow.com/)，以及对应库的官网在线文档等。而经常用到的方法函数会在不知不觉中记住，或有意识的对经常用到的方法函数强化记忆，从而更快的记住，在下次用时将不用再搜索查找。

## 库的安装
Anaconda集成环境，使得python库的安装，尤其库之间的依赖关系不再是让coder们扰心的事情。在开始本书代码之前，不需要一气安装所有需要的库在一个环境之下，而是跟随着代码的编写，需要提示有什么库没有安装，再行安装最好。因为，虽然Anaconda提供了非常友好的库管理环境，但是还是会有个别库之间的冲突，在安装过程中也有可能造成失败，连同原以安装的库也会出现问题，从而无形之中增加了时间成本。同时，对于项目开发，最好仅建立支持本项目的环境，使之轻便，网络环境部署时也不会有多余的部分。但是，对于自行练习或者不是很繁重的项目任务，或者项目的前期，只要环境能够支持不崩溃应该就可以。

## 电脑的配置
本书涉及到深度学习部分，为了增加运算速度，用到GPU，关于何种GPU可以查看相关信息。作者所使用的配置为(仅作参考)：
```
OMEN by HP Laptop 15
Processor - Intel(R) Core(TM) i9-9880H CPU @2.30GHz 
Installed RAM - 32.0GB(31.9GB usable)
System type - 64-bit operating system, x64-based processor

Display adapter - NVIDA GeForce RTX 2080 with Max-Q Design
```

> 建议内存最低32.0GB,最好64.0GB以上，甚至更高。如果内存较低，则考虑分批读取处理数据后合并等方法，善用内存管理，数据读写方式（例如用HDF5格式等）。

城市空间数据分析通常涉及到海量的数据处理，例如芝加哥城区域三维激光雷达数据约有1.41TB, 因此如果使用的是笔记本，最好配置容量大的外置硬盘，虽然会影响到读取和存储速度，但是避免笔记本自身存储空间的占用，影响运算速度。

## 善用`print()`
`print()`是python语言中使用最为频繁的语句，在代码编写、调试过程中，要不断的用该语句来查看数据的值、数据的变化、数据的结构、变量所代表的内容、监测程序进程、以及显示结果等等，`print()`实时查看数据反馈，才可知道代码编写的目的是否达到，并做出反馈，善用`print()`是用python写代码的基础。

## 库的学习
本书的内容是城市空间数据分析方法的python语言实现，在各类实验中使用有大量的相关库，例如[SciPy](https://www.scipy.org/)，[NumPy](https://numpy.org/)，[pandas](https://pandas.pydata.org/)，[scikit-learn](https://scikit-learn.org/stable/)，[PyTorch](https://pytorch.org/)，[statsmodels](https://www.statsmodels.org/stable/index.html)，[SymPy](https://www.sympy.org/en/index.html)，[Pygame](https://www.pygame.org/news)，[matplotlib](https://matplotlib.org/)，[Plotly](https://plotly.com/)，[seaborn](https://seaborn.pydata.org/)，[bokeh](https://docs.bokeh.org/en/latest/index.html)，[GeoPandas](https://geopandas.org/)，[GDAL](https://gdal.org/)，[rasterstats](https://pythonhosted.org/rasterstats/)，[Shapely](https://shapely.readthedocs.io/en/latest/manual.html)，[pathlib](https://docs.python.org/3/library/pathlib.html)，[PySAL](https://pysal.org/)，[NetworkX](https://networkx.github.io/)，[rasterio](https://rasterio.readthedocs.io/en/latest/)，[EarthPy](https://earthpy.readthedocs.io/en/latest/)，[PDAL](https://pdal.io/)，[scikit-imge](https://scikit-image.org/)，[VTK](https://vtk.org/)，[flask](https://flask.palletsprojects.com/en/1.1.x/)，[sqlite3](https://docs.python.org/3/library/sqlite3.html)，[cv2](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html)，[re](https://docs.python.org/3/library/re.html)，[itertools](https://docs.python.org/3/library/itertools.html)，[urllib](https://docs.python.org/3/library/urllib.html)，[Pillow](https://python-pillow.org/)，[tkinter](https://docs.python.org/3/library/tkinter.html)等，上述库为本书主要使用的库，很多库都是大体量，每个库单独学习都会花费一定时间。如果很多库之前没有接触过，一种方式是，在阅读到相关部分用到该库时，提前快速的学习各个库官方提供的教程(tutorial)，不是手册（manual），从而快速的掌握库的主要结构，再继续阅读本书实验；另一种方式是，为了不打断阅读的连续性，可以只查找用到该库的类或函数的内容。每个库的内容一般都会非常多，不需要一一学习，用到时根据需要有针对性，和有目的性的查阅即可。但是有些库是需要系统的看下其教程，例如scikit-learn、PyTorch等极其重量型的库。

## 避免重复造轮子
“避免重复造轮子”是程序员的一个‘座右铭’，当然每个人所持的观点不尽相同，但是可以肯定的是，是没有必要从0开始搭建所有项目内容的，这如同再写python的所有库，甚至python核心内容的东西，例如scikit-learn、PyTorch集成了大量的模型算法，我们只是直接使用，不会再重复写这些代码。但是在本书阐述过程中，会有意识的对最为基本的部分按照计算公式重写代码，目的是认识清楚这究竟是怎么回事，而真正的项目则无需如此。

## 从数学-公式到代码，图形：用Python语言学习数学的知识点
就像书面表达音乐的最佳方式是乐谱，那么最能巧妙展示数学特点的就是数学公式。很多人，尤其规划、建筑和景观专业，因为专业本身形象思维的特点，大学多年的图式思维训练，对逻辑思维有所抵触，实际上好的设计规划，除了具有一定的审美意识，空间设计能力，好的逻辑思维会让设计师在空间形式设计能力上亦有所提升，能够感觉到这种空间能力的变化，更别提，建筑大专业本身就是工科体系，是关乎到人们生命安全的严谨的事情。

当开始接触公式时，可能不适应，但是当你逐步的用公式代替所要阐述的方法时， 你会慢慢喜欢上这种方式，即使我们在小学就开始学习公式。

再者，很多时候公式，甚至文字内容让人费解，怎么也读不懂作者所阐述的究竟是什么，尤其包含大部分公式，而案例都是“白纸黑字”的论述时。这是因为我们只是看的到，但是摸不到，你很难去调整参数再实验作者的案例，但是在python等语言的世界中，数据、代码都是实实在在的东西，你可以用print()查看每一步数据的变化，尤其不易理解的地方，一比较前后的数据变化，和所用的什么数据，一切都会迎刃而解。这也是，为什么本书的所有阐述都是基于代码的，公式看不懂，没有问题；文字看不懂，也没有问题，我们只要逐行的运行代码，似乎就会明朗开来，而且尽可能的将数据以图示的方式表述，来增加理解的方式，让问题更容易浮出水面。所有有章节都有一个.ipynb的Jupyter文件，可以互动操作代码，实现的边学边学边实验的目的。

## 参考文献及书的推荐
城市空间数据分析方法需要多学科的融合，规划、建筑和景观的专业知识，python等语言知识，数学（统计学、微积分、线性代数、数据库等），以及地理信息系统，生态学等等大类，而其下则为更为繁杂的小类细分。实际上，我们只是用到在解决问题时所用到的知识，当在阅读时，如果该处的知识点不是很明白，可以查找相关文献来获取补充，在每一部分时，也提供了所参考的文献，书籍，有必要时可以翻看印证。

## 知识点的跳跃-没有方程式的推导
很多公式并不会引其推导过程，主要是专业目的性，一个专业做了所有专业的事，那是非常疯的，不现实，不实际，不可行，只有学科间不断的融合，才能快速的成长，在有必要就本专业研究问题寻找解决方法时，需要自行推导再用心做这件事，而已有的公式所代表的研究成果，例如概率密度函数公式、最小二乘法、判定系数等等，是不建议从头开始逐步推导。

## 对于那些绕弯的代码
读代码不是读代码本身，而是读代码所要传递的信息。有些代码比较简单，例如数组的加减乘除；但是有些代码“不知其所以然”，但却解决了问题，那么这里面可能就有一些非常巧妙的处理手段，例如梯度下降法、主成分分析、生成对抗网络，以及各类算法等。很多时候解决问题的方法越巧妙，其方式越不同于常规，因此阅读起来就不那么容易，但是只要“灵感一闪”，似乎很快的就会明朗，因此对于这些绕弯的代码，除了必要的不断查看数据变化，甚至结合图表分析，以及网络搜索最好，最易懂的解释文字外，需要耐下心和静下心来，找到作者所发现的方法，甚至很多时候需要休息下，松弛下脑筋，待改个时间再探索，反而不容易陷入其中，迷糊了答案。

## 对于“千行+”复杂代码的学习
百千行的代码能够一下子看懂的可能性不大，尤其这些代码以类的形式出现时，各类参数、变量，甚至函数自身在各个函数下不断的跳跃，乃至分支、合并、并不断产生新的变量。捋清楚初始化的条件，以及初始化参数传输的路径是首先需要做的事，这个过程视代码的复杂程度难易变化。一旦清楚了代码运行的流程，即语句执行的顺序（有顺序，也有并行），可以将其记录下来，方便进一步的分析，和避免遗忘；进而开始第2阶段的代码分析，因为已经清楚了流程，那么就从开始逐行的运行代码，打印数据，理解每一行代码所要解决的是哪些问题，当然这个过程与捋清代码运行流程可以同时进行，就是边梳理运行行，边分析代码是如果解决问题的；在逐行分析代码时，必定遇到那些绕弯的代码，如果一时无法理解，不影响后续代码行前提下，可以先绕过，直接获取用于后续代码行的数据，最后再重点分析这个这些绕弯的代码。如果无法绕过的话，就只能花费精力先来解决这个问题。

## 为什么有时要理解算法
库中的方法通常是解决某一问题算法的体现，一般而言是直接调用该方法，避免重复造轮子，这毋庸置疑。但是，很多算法能够启发我们，对于某些问题提出新的解决策略，而这只有理解了算法自身才能过做到。例如对于图像提出的尺度空间（scale space），规划领域空间研究上（尤其针对遥感影像等空间数据）上是值得借鉴并提出分析空间的新方法。

而此时，因为不是成书状态，我们可以弹性调整适应，有共同学习的环境基因，而不是独自秉烛夜读，这个共同成长的经历会为读者和作者带来共鸣，也相信社区的互动会激发学习的兴趣，乃至激情。

