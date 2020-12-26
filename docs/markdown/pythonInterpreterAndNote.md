# python解释器与笔记
“工欲善其事，必先利其器”，这不仅包括python解释器，同时包括作为新进一员的程序“猿”（coder）,还应该具备哪些“保命”的手段。用代码来解决专业的问题，代码是工具，而灵魂是解决问题的方法，作为工具，除了python本身的各类方法之外，无以计数的库，以及各类库所提供的方法更是数不胜数。而解决问题的方法却也不局限于一种，当在代码的湖海中畅游时，都会不间断的汲取这里的养分，将其用于、融于自身解决问题的逻辑中，当谱写了千万行的代码之后，回过头来，却不时的发现，早已不记得应用了哪些，或者借助了哪些库中的方法来解决哪些问题，待另一个问题需要同样的方法解决时，模糊的印象无法找到最初方法出现在什么地方，不得不再次苦苦搜寻，这是为什么每一个认真的程序员都会有一个自己的代码库，正如，为什么本书的代码会托管于GitHub中一样，需要之时，只需在代码仓库中搜索一下，迁移应用。

## python解释器
本书的代码几乎全部在anadonda（开源python包管理器）中实现，使用其中的spyder和JupyterLab（交互式笔记本，支持众多语言）完成。

序号 |解释器名称| 免费与否|推荐说明|
------------ |:-------------|:-------------|:-------------|
1 |[python 官方](https://www.python.org/downloads/)|免费|**不推荐**，但轻便，可以安装，偶尔用于随手简单代码的验证，乃至当计算器使用| 
2 |[anaconda](https://www.anaconda.com/)|个人版(Individual Edition)免费；<em>团队版(Team Edition)和企业版(Enterprise Edition)付费</em> |集成了众多科学包及其依赖项，**强烈推荐**，因为自行解决依赖项是一件十分令人苦恼的事情。其中包含Spyder和JupyterLab为本书所使用|
3 |[Visual Studio Code(VSC)](https://code.visualstudio.com/)|免费|**推荐使用**，用于查看代码非常方便，并支持多种语言格式；同时本书用docsify部署网页版时，即使用VSC实现|
4 |[PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows)|付费|**一般推荐**，通常只在部署网页时使用，本书“实验用网络应用平台的部署部分”用到该平台|
5 |[Notepad++](https://notepad-plus-plus.org/)|免费|仅用于查看代码、文本编辑，**推荐使用**，轻便的纯文本代码编辑器，可以用于代码查看，尤其兼容众多文件格式，当有些文件乱码时，不妨尝试用此编辑器|

## 逆向工程

## 笔记
做笔记的目的，是为了梳理知识和日后查看，其分为线上（云端）和本地。大多数编辑是本地和线上同步，因为每人拥有多部电子设备，需要在不同平台下切换工作，而无需随身带上所有设备就可以工作；同时，方便多人协作，这需要一个共同的云上环境；再者，丢设备的情况时而有之，更别谈设备崩溃至无法修复，云端和本地组合都是最好的选择，本书花费几年心血所书写的代码以及实验研究记录，因为设备问题而丢失，对于个人而言，这个损失是无法承担的。所以，对于工作中的伙伴们，尤其搞研究的学者，必定要做好防范措施。

本书书写过程中使用[GitHub](https://github.com/richieBao)托管代码，和书写markdown说明文档，在本地和云端同步是直接应用GitHub提供的GitHub Desktop来推送(push)和拉取(pull)代码以及相关文档。有时也会直接应用[StackEdit](https://stackedit.io/)直接书写推送markdown文档到云端。当然亦可以直接在GitHub上在线编辑。

为了学习代码，能够直接交互操作运行验证代码，并同时记录说明文字，则可以使用[JupyterLab或者Jupyter Notebook](https://jupyter.org/)，这个方式实际上更适合于教学、交流，当项目繁杂庞大时，应用Jupyter工具时，效率会大幅度下降，代码运行效率低，同时，不适合建立代码结构层次关系，代码的书写还是要在anaconda下的spyder中完成。另一个本地书写markdown文档的工具推荐使用[VNote](https://tamlok.github.io/vnote/en_us/)，这个仅作参考。

每个人都会有自己做笔记的习惯，而作代码的笔记，与我们常规使用OneNote等工具有所不同，要高亮显示代码格式，最好能运行查看结果，因此需要结合自身情况来选择笔记工具，使得学习代码这件事事半功倍。

<a href="https://www.anaconda.com/"><img src="./imgs/anacondaimg.jpg" height="50" width="auto" title="caDesign">
<a href="https://www.python.org/downloads/"><img src="./imgs/python-logo@2x.png" height="50" width="auto" title="caDesign">
<a href="https://code.visualstudio.com/"><img src="./imgs/vsc.jpg" height="35" width="auto" title="caDesign" align="top">
<a href="https://www.jetbrains.com/pycharm/download/#section=windows"><img src="./imgs/PyCharm_Logo1.png" height="65" width="auto" title="caDesign" align="top">
<a href="https://notepad-plus-plus.org/"><img src="./imgs/notepadlogo.png" height="50" width="auto" title="caDesign">
<a href="https://jupyter.org/"><img src="./imgs/logo_svg.png" height="45" width="auto" title="caDesign">
