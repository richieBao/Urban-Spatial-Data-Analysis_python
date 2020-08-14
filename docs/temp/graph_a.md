```mermaid
classDiagram

GUI包含功能 --> 5_影像数据处理 : a.数据读取与预处理
5_影像数据处理 : 1. 数组格式图像(numpy-array)
5_影像数据处理 : 2. 图片数据(.jpg,.png...)
5_影像数据处理 --> class_main_window : a.

GUI包含功能 : b. 显示背景影像，可缩放
GUI包含功能 : c. 选择分类，在图像中采样
GUI包含功能 : d. 显示当前采样分类状态
GUI包含功能 : e. 计算采样位置坐标
GUI包含功能 : a. 影像数据处理()
GUI包含功能 : -. python控制台信息打印()

GUI包含功能 --> 1_显示背景影像_可缩放 : b.图像显示
1_显示背景影像_可缩放 --> 显示背景影像_可缩放部分 : b.
1_显示背景影像_可缩放 : 1. 在画布（canvas）中显示影像
1_显示背景影像_可缩放 : 2. 增加滚动条
1_显示背景影像_可缩放 : 3. 鼠标滚轮缩放

class_CanvasImage o-- class_main_window
class_CanvasImage --o class_AutoScrollbar
class_CanvasImage : 基本包括所有功能处理函数
class_CanvasImage : def __init__()
class_CanvasImage : def compute_samplePosition()
class_CanvasImage : def click_xy_collection()
class_CanvasImage : def __scroll_x()
class_CanvasImage : def _scroll_y()
class_CanvasImage : def __move_from()
class_CanvasImage : def __move_to()
class_CanvasImage : def __wheel()
class_CanvasImage : def show_image()

class_CanvasImage <|-- 显示背景影像_可缩放部分: b.
显示背景影像_可缩放部分 : 滚动条拖动
显示背景影像_可缩放部分 : 左键拖动
显示背景影像_可缩放部分 : 鼠标滚动

显示背景影像_可缩放部分 : 
显示背景影像_可缩放部分 :  def __scroll_x()
显示背景影像_可缩放部分 : def _scroll_y()
显示背景影像_可缩放部分 : def __move_from()
显示背景影像_可缩放部分 :  def __move_to()
显示背景影像_可缩放部分 : def __wheel()
显示背景影像_可缩放部分 : def show_image()

class_main_window *-- 外部调用
class_main_window : 主程序
class_main_window : 图像读取与预处理
class_main_window : def __init__()
class_main_window : def landsat_stack_array2img()

外部调用 : 定义工作空间-workspace
外部调用 : 实例化调用-app=main_window
外部调用 : 保存采样数据-pickle

class_AutoScrollbar : 滚动条默认隐藏配置
class_AutoScrollbar : 异常处理
class_AutoScrollbar : def set()
class_AutoScrollbar : def pack()
class_AutoScrollbar : def place)
class_AutoScrollbar -- 显示背景影像_可缩放部分 : b.

GUI包含功能 --> 2_选择分类_在图像中采样 : c.分类采样
2_选择分类_在图像中采样 : 1. 布局3个单选按钮-Radiobutton
2_选择分类_在图像中采样 --> 左键点击采样 : c.

GUI包含功能 --> 3_显示当前采样分类状态 : d.分类状态
3_显示当前采样分类状态 : 1. 布局1个标签-Label

2_选择分类_在图像中采样 --> 3_显示当前采样分类状态 : c-d

class_CanvasImage <|-- 左键点击采样 : c.
左键点击采样 : def click_xy_collection()
左键点击采样 : 根据分类存储点击坐标

class_CanvasImage <|-- 初始化 : 综合
初始化 : 所有处理在此汇总
初始化 : def __init__()
3_显示当前采样分类状态 --> 初始化 : d.
2_选择分类_在图像中采样 --> 初始化 : c.
1_显示背景影像_可缩放 --> 初始化 : b.
class_main_window --> 初始化 : a.

GUI包含功能 --> 4_计算采样位置坐标 : e.采样坐标变换
4_计算采样位置坐标 : 1. 还原画布移动缩放后采样点的坐标

4_计算采样位置坐标 --> 坐标变换 : e.
坐标变换 : def compute_samplePosition()
class_CanvasImage <|-- 坐标变换 : e.
4_计算采样位置坐标 --> 初始化 : e.

```
