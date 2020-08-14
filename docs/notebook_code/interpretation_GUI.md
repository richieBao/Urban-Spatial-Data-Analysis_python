> Created on Sat Aug  8 19/52/37 2020  @author: Richie Bao-caDesign设计(cadesign.cn)

## 1. 遥感影像解译（基于NDVI），建立采样工具（GUI_tkinter），混淆矩阵
如果只是分析城市的绿地、裸地和水体，不涉及到更精细的土地覆盖分类，例如灌丛、草地、裸地、居民地、园地、耕地、河流，湖泊等等，则可以自行利用Landsat系列遥感影像通过NDVI、NDWI和NDBI等手段提取绿地（可进一步细分耕地和林地）、裸地和水体等；对于精细的分类推荐使用eCognition等平台工具解译。基于NDVI的遥感影像解译，首先是读取能够反映不同季节绿地情况的Landsate不同季节的影像，根据研究的目的范围裁切，裁切的边界可以在QGIS中完成;->然后计算不同季节的NDVI;->再通过使用交互的plotly图表，分析NDVI取值范围，判断不同土地覆盖阈值范围，解译影像；->如果要判断解译的精度需要给出采样，即随机提取的点的真实土地覆盖类型，这个过程是一个需要手工操作的过程，而python的内嵌库[tkinter](https://docs.python.org/3/library/tkinter.html)的图形用户界面(Graphical User Interface,GUI)能够方便的帮助我们快速的建立交互操作平台。从而完成采样工作；->最后计算混淆矩阵和百分比精度，判断解译的精度。

### 1.1 影像数据处理
为了方便影像的处理，通常将一些常用的工具构建为函数，例如下述的影像裁切函数`raster_clip`。影像处理最为需要关心的就是坐标投影系统，通常Landsat影像都包含对应的投影系统，例如UTM,DATUM:WGS84,UTM_ZONE16，因此最好的选择就是直接统一为该坐标投影系统，不建议转换为其它投影。对于其它地理文件而言，如果是栅格数据通常需要转换投影，即栅格文件本书通常是有自己投影的。而对于.shp的矢量地理文件，一般保持为WGS84，即EPSG:4326，通常不含投影，方便用于不同的坐标投影系统平台，以及作为中转的数据格式类型。因此，在QGIS等平台下建立裁切边界时，仅保持其坐标系为WGS84，而不配置投影。在所定义的裁切函数中，根据所读取的Landsat影像的投影系统，再进行定义，方便数据的处理。

影像数据一般都很大，因此影像数据的处理，尤其高分辨的影像尤其花费时间，因此有必要将处理后的数据即刻保存到硬盘空间下，当需要时直接从硬盘中读取，避免再次花费时间计算。


```python
import util
import os
workspace=r"F:\data_02_Chicago\9_landsat\data_processing"
Landsat_fp={
        "w_180310":r"F:\data_02_Chicago\9_landsat\LC08_L1TP_023031_20180310_20180320_01_T1", #冬季
        "s_190820":r"F:\data_02_Chicago\9_landsat\LC08_L1TP_023031_20190804_20190820_01_T1", #夏季
        "a_191018":r"F:\data_02_Chicago\9_landsat\LC08_L1TP_023031_20191007_20191018_01_T1" #秋季   
        }

w_180310_band_fp_dic,w_180310_Landsat_para=util.LandsatMTL_info(Landsat_fp["w_180310"]) #LandsatMTL_info(fp)函数，在Landsat遥感影像处理部分阐述，将其放置于util.py文件中后调用
s_190820_band_fp_dic,s_190820_Landsat_para=util.LandsatMTL_info(Landsat_fp["s_190820"])
a_191018_band_fp_dic,a_191018_Landsat_para=util.LandsatMTL_info(Landsat_fp["a_191018"])

print("w_180310-MAP_PROJECTION:%s,DATUM:%s,UTM_ZONE%s"%(w_180310_Landsat_para['MAP_PROJECTION'],w_180310_Landsat_para['DATUM'],w_180310_Landsat_para['UTM_ZONE']))
print("s_190820-MAP_PROJECTION:%s,DATUM:%s,UTM_ZONE%s"%(s_190820_Landsat_para['MAP_PROJECTION'],s_190820_Landsat_para['DATUM'],s_190820_Landsat_para['UTM_ZONE']))
print("a_191018-MAP_PROJECTION:%s,DATUM:%s,UTM_ZONE%s"%(a_191018_Landsat_para['MAP_PROJECTION'],a_191018_Landsat_para['DATUM'],a_191018_Landsat_para['UTM_ZONE']))

def raster_clip(raster_fp,clip_boundary_fp,save_path):
    import earthpy.spatial as es
    import geopandas as gpd
    from pyproj import CRS
    import rasterio as rio
    '''
    function - 给定裁切边界，批量裁切栅格数据
    
    Paras:
    raster_fp - 待裁切的栅格数据文件路径（.tif）,具有相同的坐标投影系统
    clip_boundary - 用于裁切的边界（.shp，WGS84，无投影）
    
    return:
    rasterClipped_pathList - 裁切后的文件路径列表
    '''
    clip_bound=gpd.read_file(clip_boundary_fp)
    with rio.open(raster_fp[0]) as raster_crs:
        raster_profile=raster_crs.profile
        clip_bound_proj=clip_bound.to_crs(raster_profile["crs"])
    
    rasterClipped_pathList=es.crop_all(raster_fp, save_path, clip_bound_proj, overwrite=True) #对所有波段band执行裁切
    print("finished clipping.")
    return rasterClipped_pathList
    
clip_boundary_fp=r".\data\geoData\LandsatChicago_boundary.shp"
save_path=r"F:\data_02_Chicago\9_landsat\data_processing\s_190820"    
s_190820_clipped_fp=raster_clip(list(s_190820_band_fp_dic.values()),clip_boundary_fp,save_path)    
```

    w_180310-MAP_PROJECTION:UTM,DATUM:WGS84,UTM_ZONE16
    s_190820-MAP_PROJECTION:UTM,DATUM:WGS84,UTM_ZONE16
    a_191018-MAP_PROJECTION:UTM,DATUM:WGS84,UTM_ZONE16
    finished clipping.
    

通过定义的裁切函数，直接计算冬季和秋季影像。


```python
save_path=r"F:\data_02_Chicago\9_landsat\data_processing\w_180310"    
w_180310_clipped_fp=raster_clip(list(w_180310_band_fp_dic.values()),clip_boundary_fp,save_path) 

save_path=r"F:\data_02_Chicago\9_landsat\data_processing\a_191018"    
a_191018_clipped_fp=raster_clip(list(a_191018_band_fp_dic.values()),clip_boundary_fp,save_path) 
```

    finished clipping.
    finished clipping.
    

从美国地质调查局下载的Landsat影像，各个波段是单独的文件。为了避免每次读取单个文件，则使用earthpy的.stack方法将所有波段放置于一个文件下，即数组形式，方便数据处理。


```python
import earthpy.spatial as es
w_180310_array, w_180310_raster_prof=es.stack(w_180310_clipped_fp[:7], out_path=os.path.join(workspace,r"w_180310_stack.tif"))
print("finished stacking_1...")

s_190820_array, s_190820_raster_prof=es.stack(s_190820_clipped_fp[:7], out_path=os.path.join(workspace,r"s_190820_stack.tif"))
print("finished stacking_2...")

a_191018_array, a_191018_raster_prof=es.stack(a_191018_clipped_fp[:7], out_path=os.path.join(workspace,r"a_191018_stack.tif"))
print("finished stacking_3...")
```

    finished stacking_1...
    finished stacking_2...
    finished stacking_3...
    

只有将数据显示出来，才能更好的判断地理空间数据处理的结果是否正确。建立`bands_show`影像波段显示函数，查看影像。可以通过`band_num`输入参数，确定合成显示的波段。


```python
def bands_show(img_stack_list,band_num):
    import matplotlib.pyplot as plt
    from rasterio.plot import plotting_extent
    import earthpy.plot as ep
    '''
    function - 指定波段，同时显示多个遥感影像
    
    Paras:
    img_stack_list - 影像列表
    band_num - 显示的层
    '''
    
    def variable_name(var):
        '''
        function - 将变量名转换为字符串
        
        Paras:
        var - 变量名
        '''
        return [tpl[0] for tpl in filter(lambda x: var is x[1], globals().items())][0]

    plt.rcParams.update({'font.size': 12})
    img_num=len(img_stack_list)
    fig, axs = plt.subplots(1,img_num,figsize=(12*img_num, 12))
    i=0
    for img in img_stack_list:
        ep.plot_rgb(
                    img,
                    rgb=band_num,
                    stretch=True,
                    str_clip=0.5,
                    title="%s"%variable_name(img),
                    ax=axs[i]
                )
        i+=1
    plt.show()
img_stack_list=[w_180310_array,s_190820_array,a_191018_array]
band_num=[3,2,1]
bands_show(img_stack_list,band_num)
```


<a href=""><img src="./imgs/11_07.png" height="auto" width="auto" title="caDesign"></a>


从上述直接显示的合成波段影像来看，显示效果偏暗，不利于细节的观察。借助[scikit-image](https://scikit-image.org/)图像处理库中的exposure方法，可以拉伸图像，调亮与增强对比度。处理时需要注意要逐波段的处理，然后再合并。


```python
def image_exposure(img_bands,percentile=(2,98)):
    from skimage import exposure
    import numpy as np
    '''
    function - 拉伸图像 contract stretching
    
    Paras:
    img_bands - landsat stack后的波段
    percentile - 百分位数
    
    return:
    img_bands_exposure - 返回拉伸后的影像
    '''
    bands_temp=[]
    for band in img_bands:
        p2, p98=np.percentile(band, (2, 98))
        bands_temp.append(exposure.rescale_intensity(band, in_range=(p2,p98)))
    img_bands_exposure=np.concatenate([np.expand_dims(b,axis=0) for b in bands_temp],axis=0)
    print("finished exposure.")
    return img_bands_exposure

w_180310_exposure=image_exposure(w_180310_array)
s_190820_exposure=image_exposure(s_190820_array)
a_191018_exposure=image_exposure(a_191018_array)
```

    finished exposure.
    finished exposure.
    finished exposure.
    

显示处理后的遥感影像，对比原显示效果，可见有明显的提升。


```python
img_stack_list=[w_180310_exposure,s_190820_exposure,a_191018_exposure]
band_num=[3,2,1]
bands_show(img_stack_list,band_num)
```


<a href=""><img src="./imgs/11_08.png" height="auto" width="auto" title="caDesign"></a>


### 1.2 计算NDVI，交互图像与解译
调用在阐述影像波段计算指数时，所定义的NDVI计算函数`NDVI(RED_band,NIR_band)`，计算NDVI归一化植被指数。同时查看计算结果的最大和最小值，即值的区间，可以看到s_190820_NDVI数据（夏季），明显有错误，通过查看上述图像，也能够发现所下载的夏季的Landsat影像在城区中有很多云，初步判断可能是造成数据异常的原因。


```python
import util
import rasterio as rio
import os
workspace=r"F:\data_02_Chicago\9_landsat\data_processing"
w_180310=rio.open(os.path.join(workspace,r"w_180310_stack.tif"))
s_190820=rio.open(os.path.join(workspace,r"s_190820_stack.tif"))
a_191018=rio.open(os.path.join(workspace,r"a_191018_stack.tif"))

w_180310_NDVI=util.NDVI(w_180310.read(4),w_180310.read(5))
s_190820_NDVI=util.NDVI(s_190820.read(4),s_190820.read(5))
a_191018_NDVI=util.NDVI(a_191018.read(4),a_191018.read(5))
```

    NDVI_min:0.000000,max:15.654902
    NDVI_min:-9999.000000,max:8483.000000
    NDVI_min:0.000000,max:15.768559
    

因为NDVI的数据为一个维度数组，可以直接使用在异常值处理部分所定义的`is_outlier(data,threshold=3.5)`函数，处理异常值，并打印NDVI图像，查看处理结果，能够较为清晰的看到NDVI对绿地植被的识别，可以较好的区分水体和裸地。


```python
import matplotlib.pyplot as plt
import earthpy.plot as ep
fig, axs=plt.subplots(1,3,figsize=(30, 12))
ep.plot_bands(w_180310_NDVI, cbar=False, title="w_180310_NDVI", ax=axs[0],cmap='flag')

s_190820_NDVI=util.NDVI(s_190820.read(4),s_190820.read(1)) #对异常值采取了原地取代的方式，因此如果多次运行异常值检测，需要重新计算NDVI，保持原始数据不变 
is_outlier_bool,_=util.is_outlier(s_190820_NDVI,threshold=3)
s_190820_NDVI[is_outlier_bool]=0 #原地取代
ep.plot_bands(s_190820_NDVI, cbar=False, title="s_190820_NDVI", ax=axs[1],cmap='terrain')

ep.plot_bands(a_191018_NDVI, cbar=False, title="a_191018_NDVI", ax=axs[2],cmap='flag')
plt.show()
```

    NDVI_min:0.000000,max:64105.000000
    


<a href=""><img src="./imgs/11_09.png" height="auto" width="auto" title="caDesign"></a>


直接处理形式为(4900, 4604)的NDVI数组，包含$4900 \times 4604$多个数据，如果计算机硬件条件可以，保证在所能够承担的计算时间长度下，可以不用压缩数据。否则，在不影响最终计算结果对分析的影响，可以适当的压缩影像，直接使用[scikit-image](https://scikit-image.org/)图像处理库中的rescale方法，同时也调入了resize, downscale_local_mean方法，可以自行查看其功能。一般计算的方法是求取局部均值

打印夏季影像计算所得的NDVI，按阈值显示颜色的图像，方便查看不同阈值区间的区域范围，便于确定阈值，从而解译土地用地分类。


```python
import plotly.express as px
from skimage.transform import rescale, resize, downscale_local_mean
s_190820_NDVI_rescaled=rescale(s_190820_NDVI, 0.1, anti_aliasing=False)
fig=px.imshow(img=s_190820_NDVI_rescaled,zmin=s_190820_NDVI_rescaled.min(),zmax=s_190820_NDVI_rescaled.max(),width=800,height=800,color_continuous_scale=px.colors.sequential.speed)
fig.show()
```

<a href=""><img src="./imgs/11_10.png" height="auto" width="800" title="caDesign"></a>

所计算的NDVI值是一个无量纲，维度为1的数组，值通常为浮点型小数，在使用matplotlib，plotly，seaborn和bokeh等图表库时，一般会使用RGB，RGBA，十六进制形式，以及浮点型等表示颜色的数据格式。对于所计算的NDVI，定义函数data_division，将其按照分类的阈值转换为RGB的颜色格式方便图像打印。因为NDVI本身不是颜色数据，首先根据分类阈值计算百分位数（.percentile），然后根据其百分位数使用.digitize方法，返回数值对应区间的索引（整数）。由区间数量即唯一的索引数定义不同的位于0-255的随机整数（每个值对应三个随机整数值），代表颜色。

```python
from skimage.transform import rescale, resize, downscale_local_mean
def data_division(data,division,right=True):
    import numpy as np
    import pandas as pd
    '''
    function - 将数据按照给定的百分数划分，并给定固定的值,整数值或RGB色彩值
    
    Paras:
    data - 待划分的numpy数组
    division - 百分数列表
    
    return：
    data_digitize - 返回整数值
    data_rgb - 返回RGB，颜色值
    '''
    percentile=np.percentile(data,np.array(division))
    data_digitize=np.digitize(data,percentile,right)
    
    unique_digitize=np.unique(data_digitize)
    random_color_dict=[{k:np.random.randint(low=0,high=255,size=1)for k in unique_digitize} for i in range(3)]
    data_color=[pd.DataFrame(data_digitize).replace(random_color_dict[i]).to_numpy() for i in range(3)]
    data_rgb=np.concatenate([np.expand_dims(i,axis=-1) for i in data_color],axis=-1)
    
    return data_digitize,data_rgb

w_180310_NDVI_rescaled=rescale(w_180310_NDVI, 0.1, anti_aliasing=False)
s_190820_NDVI_rescaled=rescale(s_190820_NDVI, 0.1, anti_aliasing=False)
a_191018_NDVI_rescaled=rescale(a_191018_NDVI, 0.1, anti_aliasing=False)
print("finished rescale .")
```

    finished rescale .
    
```python
division=[0,35,85]
_,s_190820_NDVI_RGB=data_division(s_190820_NDVI_rescaled,division)
fig=px.imshow(img=s_190820_NDVI_RGB,zmin=s_190820_NDVI_RGB.min(),zmax=s_190820_NDVI_RGB.max(),width=800,height=800,color_continuous_scale=px.colors.sequential.speed)
fig.show()
```

<a href=""><img src="./imgs/11_06.png" height="auto" width="800" title="caDesign"></a>


能够辅助解译，确定阈值，可以通过直方图（频数分布）查看NDVI的数据分布情况。因为地物的数量不同，某些转折，或断裂的位置点则可能代表不同的地物。

```python
fig, axs=plt.subplots(1,3,figsize=(20, 6))

count, bins, ignored = axs[0].hist(w_180310_NDVI.flatten(), bins=100,density=True) 
count, bins, ignored = axs[1].hist(s_190820_NDVI.flatten(), bins=100,density=True) 
count, bins, ignored = axs[2].hist(a_191018_NDVI.flatten(), bins=100,density=True) 
plt.show()
```

<a href=""><img src="./imgs/11_11.png" height="auto" width="auto" title="caDesign"></a>

虽然上述代码可以打印，按阈值显示NDVI图像，但是不能够交互的操作，即通过不断的调整阈值区间，即刻的查看阈值图像，从而判断适合于分类地物的阈值。plotly图表工具提供了简单的交互式工具，可以通过增加滑条窗口工具（widgets.IntSlier），以及下拉栏（widgets.Dropdown），实现交互。这里定义了三个滑动条来定义阈值区间，一个下拉栏栏定义可选择的三个不同季节的NDVI数据，调用上述定义的`data_division`函数，按阈值区间配置颜色显示图像。通过不断调整阈值，与波段3，2，1组合的真彩色图像比较，判断三个季节土地用地分类阈值区间为[0,35,85]。注意通常不同影像的NDVI分类阈值可能不同。

```python
def percentile_slider(season_dic):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from ipywidgets import widgets
    from IPython.display import display
    
    '''
    function - 多个栅格数据，给定百分比，变化观察
    
    Paras:
    season_dic -  多个栅格字典
    '''
    
    p_1_slider=widgets.IntSlider(min=0, max=100, value=10, step=1, description="percentile_1")
    p_2_slider=widgets.IntSlider(min=0, max=100, value=30, step=1, description="percentile_2")
    p_3_slider=widgets.IntSlider(min=0, max=100, value=50, step=1, description="percentile_3")

    
    season_keys=list(season_dic.keys())

    season=widgets.Dropdown(
        description='season',
        value=season_keys[0],
        options=season_keys
    )

    season_val=season_dic[season_keys[0]]
    _,img=data_division(season_val,division=[10,30,50],right=True)
    trace1=go.Image(z=img)

    g=go.FigureWidget(data=[trace1,],
                      layout=go.Layout(
                      title=dict(
                      text='NDVI interpretation'
                            ),
                      width=800,
                      height=800
                        ))

    def validate():
        if season.value in season_keys:
            return True
        else:
            return False

    def response(change):
        if validate():
            division=[p_1_slider.value,p_2_slider.value,p_3_slider.value]
            _,img_=data_division(season_dic[season.value],division,right=True)
            with g.batch_update():
                g.data[0].z=img_
    p_1_slider.observe(response, names="value")            
    p_2_slider.observe(response, names="value")    
    p_3_slider.observe(response, names="value")    
    season.observe(response, names="value")    

    container=widgets.HBox([p_1_slider,p_2_slider,p_3_slider,season])
    box=widgets.VBox([container,
                  g])
    display(box)
    
season_dic={"w_180310":w_180310_NDVI_rescaled,"s_190820":s_190820_NDVI_rescaled,"a_191018":a_191018_NDVI_rescaled}    
percentile_slider(season_dic)  
```

<a href=""><img src="./imgs/11_04.png" height="auto" width="auto" title="caDesign"></a>

确定[0,35,85]的阈值区间后，就可计算所有季节的NDVI土地用地分类。

```python
division=[0,35,85]
w_interpretation,_=data_division(w_180310_NDVI_rescaled,division,right=True)
s_interpretation,_=data_division(s_190820_NDVI_rescaled,division,right=True)
a_interpretation,_=data_division(a_191018_NDVI_rescaled,division,right=True)
```

获得三个不同季节NDVI的解译结果后，根据冬季大部分农田无种植，秋季部分农田收割，夏季大部分农田生长；以及落叶树木夏季茂盛，常绿树木四季常青的特点。可以解译出农田、常绿、落叶和裸地、水体等土地覆盖类型，这里则直接将只要各季节为绿色的就划分为绿地（.logical_or），值为2；而水体中湖泊的阈值区分较好，河流的则不是很清晰，则确定只有都为水体才为水体的判断（.logical_and），值为3；其余的则为裸地，值为0和1。


```python
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

fig=make_subplots(rows=1, cols=4,shared_xaxes=False,subplot_titles=('w_interpretation',  's_interpretation', 'a_interpretation', 'green_water_bareLand'))

fig1=px.imshow(img=np.flip(w_interpretation,0),zmin=w_interpretation.min(),zmax=w_interpretation.max(),width=800,height=800,color_continuous_scale=px.colors.sequential.deep,title="winter")
fig2=px.imshow(img=np.flip(s_interpretation,0),zmin=s_interpretation.min(),zmax=s_interpretation.max(),width=800,height=800,color_continuous_scale=px.colors.sequential.haline)
fig3=px.imshow(img=np.flip(a_interpretation,0),zmin=a_interpretation.min(),zmax=a_interpretation.max(),width=800,height=800,color_continuous_scale=px.colors.sequential.haline)

green=np.logical_or(w_interpretation==2,s_interpretation==2,a_interpretation==2) #只要为绿地（值为2），就为2
water=np.logical_and(w_interpretation==3,s_interpretation==3,a_interpretation==3) #只有都为水体（值为3），才为3
green_v=np.where(green==True,2,0)
water_v=np.where(water==True,3,0)
green_water_bareLand=green_v+water_v
fig4=px.imshow(img=np.flip(green_water_bareLand,0),zmin=green_water_bareLand.min(),zmax=green_water_bareLand.max(),width=800,height=800,color_continuous_scale=px.colors.sequential.haline)

trace1=fig1['data'][0]
trace2=fig2['data'][0]
trace3=fig3['data'][0]
trace4=fig4['data'][0]

fig.add_trace(trace1, row=1, col=1)
fig.add_trace(trace2, row=1, col=2)
fig.add_trace(trace3, row=1, col=3)
fig.add_trace(trace4, row=1, col=4)

fig.update_layout(height=500, width=1600, title_text="interpretation")      
fig.show()
```

<a href=""><img src="./imgs/11_12.png" height="auto" width="auto" title="caDesign"></a>

### 1.3 采样交互操作平台的建立，与精度计算
#### 1.3.1  使用tkinter建立采样交互操作平台
对于采样的工作，可以借助于QGIQ等平台，建立.shp点数据格式，随机生成一定数量的点，由目视确定每个点的土地覆盖类型；或者直接手工点取。然后将该采样数据在python下读取，再进行后续的精度分析。该方法需要数据在不同平台下转换，操作稍许繁琐。此处，我们来通过tkinter自行建立可交互的GUI采样工具。采样的参照底图选择了秋季的影像，将该影像单独保存，方便调用。


```python
import matplotlib.pyplot as plt
import earthpy.plot as ep
from skimage.transform import rescale, resize, downscale_local_mean
import os
a_191018_exposure_rescaled=np.concatenate([np.expand_dims(rescale(b, 0.1, anti_aliasing=False),axis=0) for b in a_191018_exposure],axis=0)
fig, ax= plt.subplots(figsize=(12, 12))
ep.plot_rgb(
            a_191018_exposure_rescaled,
            rgb=[3,2,1],
            stretch=True,
            #extent=extent,
            str_clip=0.5,
            title="a_191018_exposure_rescaled",
            ax=ax
        )
plt.show()
print("a_191018_exposure_rescaled shape:",a_191018_exposure_rescaled.shape)
save_path=r"C:\Users\richi\omen-richiebao\omen_github\Urban-Spatial-Data-Analysis_python\notebook\BaiduMapPOIcollection_ipynb\data"
np.save(os.path.join(save_path,'a_191018_exposure_rescaled.npy'),a_191018_exposure_rescaled)
```

<a href=""><img src="./imgs/11_13.png" height="auto" width="auto" title="caDesign"></a>

* 样本的大小

使用抽样方法进行精度估计，样本愈小，产生的总体估计量误差就愈大。在应用时需要计算在允许的误差范围内，一组样本应包含的最少样本个数。其公式为：$n= \frac{pq z^{2} }{ d^{2} }$，其中$p$和$q$分别为解译图判读正确和错误的百分比，或表示为$p(1-p)$;z为对应于置信水平的双侧临界值；$d$误差允许范围。一般情况下，先假定解译精度为90%，置信水平为95%，并通过scipy.stats的norm工具计算置信水平为95%时双侧临界值为1.95996，带入公式可求得样本数约为138。


```python
from scipy.stats import norm
import math
val=norm.ppf((1+0.95)/2)
n=val**2*0.9*0.1/0.05**2
print("样本数量为：",n)
```

    样本数量为： 138.2925175449885

采样交互平台的代码结构（在markdown中显示）

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

完成一个能够处理一个或者多个任务的综合性代码，一般是要结合类来梳理代码结构，如果仅依靠函数，虽然一样能完成任务，但是代码结构松散，不方便代码的编写，以及代码查看。在开始一个任务之前最好先捋清楚所要实现的功能，例如采样GUI工具需要实现如下功能：

1. 影像数据处理
2. 显示背景影像，可缩放
3. 选择分类，在图像中采样
4. 显示当前采样分类状态
5. 计算采样位置坐标
6. python控制台信息打印

对于一个不大的功能开发，例如可能一开始只是想实现一个采样的工具，对于一些细节也许考虑的并不清晰，在开始任务后，随着代码编写的深入再做出调整。不过，如果一开始，尤其功能复杂的工具开发，就将可能的问题尽量考虑清晰，尤其细节上的实现，和综合结构，那么会提升代码编写的效率，并避免由于考虑不周带来的代码调整，甚至造成的重新编写。当然即使考虑的再清晰，尤其对于新手来讲，问题仍旧会层出不穷，因为代码本身就是一个不断调试的过程。

写代码的过程要不断的调试，而读代码则要捋清楚代码的结构，尤其各项功能实现的先后顺序，以及之间的数据关联。上述代码的结构图，能够很好的帮助我们识别代码的结构，如果没有这个结构图，则需要一步步从头运行程序，结合print()打印需要查看的变量数据，推导出整个代码的流程。首先确定“GUI包含功能”，由所要实现的功能出发，捋清楚脉络。按照实现的顺序，由a,b,c,d,e字母标识，可以延着字母标识顺序查看所调用的类和函数。这个图表并没有给出对应的具体代码行，不过根据这个顺序已经能够把握住整个代码的结构流程。对应的代码行，根据给出的对应函数，可以方便找到。

这个随手工具的开发，有两个关键点需要注意。一个是，图像缩放功能；另一个是，采样点坐标变换。对于第一个问题，迁移了[stack overflow-tkinter canvas zoom+ move/pan](https://stackoverflow.com/questions/41656176/tkinter-canvas-zoom-move-pan)给出的代码，因为原代码是直接读取图像文件，而这里需要读取数组结构的遥感影像波段文件，因此需要增加图像处理的部分代码，并修改源代码适应增加的部分。源代码图像缩放的核心部分不需要做出调整，包括滚动条配置，左键拖动配置和鼠标滚动缩放部分，而最为核心的则是图像显示函数`show_image(self, event=None)`部分。因为图像移动和缩放引发了画布（canvas）的比例缩放，滚动条（_scroll_x/__scroll_y）和鼠标拖动（_move_from/__move_to）引发的是移动，而鼠标滑轮（__wheel）引发了画布的缩放。缩放需要控制图像的大小，当缩小到一定程度，图像不再缩小，这个由图像缩放比例因子（imscale）参数控制，初始值为1.0。而滚轮每滚动一次的缩放比例则由delta参数控制。可以尝试修改不同的参数值，观察图像变化情况。因为画布的移动缩放，图像显示需要对应做出调整，读取当前图像位置（bbox1），画布可视区域位置(bbox2)，计算滚动条区域(bbox)，确定图像是部分还是整个位于画布可视区域，如果部分则计算可视区域图像的平铺坐标（$ x_{1} , y_{1} , x_{2} , y_{2}$），并裁切图像，否则为整个图像。图像是随画布缩放`self.canvas.scale('all', x, y, scale, scale) `，即缩放画布及画布上所有物件。对于该部分的理解最好的方法是运行代码，打印相关变量，查看数据变化。

因为图像移动缩放，引发了第2个问题，采样点坐标的变换。采样的基本思想是，确定所要采样的类别，由3个单选按钮控制，点击鼠标触发`click_xy_collection`函数，绘制点（实际上是圆形），并按照类别保存点的索引至字典xy。而采样点的坐标是由当前画布确定，画布是移动和缩放的，采样点的坐标随画布的变化而变化，那么如何获得对应实际图像大小下（图像是数组，由行列表示，那么一个像素的坐标可以表示为$(  x_{i}, y_{j}   )$，注意图像$x_{i}$为列，$y_{j}$为行，与numpy数组正好相反，即numpy数组$x_{i}$为行，$y_{j}$为列），采样点的坐标？图像的缩放实际上正是在‘线性代数基础的代码表述’部分所阐述的内容，直接线性变换就可以返回到原始坐标。要实现线性变换，需要获得比例缩放因子，即scale参数。为了确定该参数，也可以在画布未移动缩放时，自动生成两个点，计算其坐标，即为实际的坐标，以其为参照。这两个点随后会随其它采样点跟着画布移动缩放，坐标值也发生了变换。在新的画布空间下，获取当前坐标，计算这两个点前后的距离比值，即为缩放比例，其中与scale参数保持一致。因此，可以将1/scale作为线性变换的比例因子，返回缩放前的状态。采用点坐标变换，通过点击按钮（button_computePosition）调用函数`compute_samplePosition`(GUI中显示的文本为，calculate sampling position)实现。线性变换用np.matmul()计算，即两个矩阵之积。最好将计算的采样点坐标根据给定的文件位置保存。用于后续精度分析。

> 注意，tkinter编写的GUI不能在Jupyter（Lab）下运行，需要在spyder等解释器下打开运行。可以新建.py文件，将下述代码复制于该文件，再运行。如果自行编写基于tkintet开发的GUI工具，也需要在spyder等解释其下编写调试。

```python
import math,os,random
import warnings
import tkinter as tk
import numpy as np
from tkinter import ttk
from PIL import Image, ImageTk
from skimage.util import img_as_ubyte

class AutoScrollbar(ttk.Scrollbar):
    '''滚动条默认时隐藏'''
    def set(self,low,high):
        if float(low)<=0 and float(high)>=1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self,low,high)
    
    def pack(self,**kw):
        raise tk.TclError("Cannot use pack with the widget"+self.__class_.__name__)

    def place(self,**kw):
        raise tk.TclError("Cannot use pack with the widget"+self.__class_.__name__)
```


```python
class CanvasImage(ttk.Frame):
    '''显示图像，可缩放'''
    def __init__(self,mainframe,img):
        '''初始化Frame框架'''
        ttk.Frame.__init__(self,master=mainframe)
        self.master.title("pixel sampling of remote sensing image")  
        self.img=img
        self.master.geometry('%dx%d'%self.img.size) 
        self.width, self.height = self.img.size

        #增加水平、垂直滚动条
        hbar=AutoScrollbar(self.master, orient='horizontal')
        vbar = AutoScrollbar(self.master, orient='vertical')
        hbar.grid(row=1, column=0,columnspan=4, sticky='we')
        vbar.grid(row=0, column=4, sticky='ns')
        #创建画布并绑定滚动条
        self.canvas = tk.Canvas(self.master, highlightthickness=0, xscrollcommand=hbar.set, yscrollcommand=vbar.set,width=self.width,height=self.height)        
        self.canvas.config(scrollregion=self.canvas.bbox('all')) 
        self.canvas.grid(row=0,column=0,columnspan=4,sticky='nswe')
        self.canvas.update() #更新画布
        hbar.configure(command=self.__scroll_x) #绑定滚动条于画布
        vbar.configure(command=self.__scroll_y)
     
        self.master.rowconfigure(0,weight=1) #使得画布（显示图像）可扩展
        self.master.columnconfigure(0,weight=1)              
        
        #于画布绑定事件（events）
        self.canvas.bind('<Configure>', lambda event: self.show_image())  #调整画布大小
        self.canvas.bind('<ButtonPress-1>', self.__move_from) #原画布位置
        self.canvas.bind('<B1-Motion>', self.__move_to) #移动画布到新的位置
        self.canvas.bind('<MouseWheel>', self.__wheel) #Windows和MacOS下缩放，不适用于Linux
        self.canvas.bind('<Button-5>', self.__wheel) #Linux下，向下滚动缩放
        self.canvas.bind('<Button-4>',   self.__wheel) #Linux下，向上滚动缩放
        #处理空闲状态下的击键，因为太多击键，会使得性能低的电脑运行缓慢
        self.canvas.bind('<Key>', lambda event: self.canvas.after_idle(self.__keystroke, event))
        
        self.imscale=1.0 #图像缩放比例
        self.delta=1.2 #滑轮，画布缩放量级        
        
        #将图像置于矩形容器中，宽高等于图像的大小
        
        self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=0)
      
        self.show_image()     
        
        self.xy={"water":[],"vegetation":[],"bareland":[]}
        self.canvas.bind('<Button-1>',self.click_xy_collection)

        self.xy_rec={"water":[],"vegetation":[],"bareland":[]}
        
        #配置按钮，用于选择样本，以及计算样本位置
        button_frame=tk.Frame(self.master,bg='white', width=5000, height=30, pady=3).grid(row=2,sticky='NW')
        button_computePosition=tk.Button(button_frame,text='calculate sampling position',fg='black',width=25, height=1,command=self.compute_samplePosition).grid(row=2,column=0,sticky='w')
        
        self.info_class=tk.StringVar(value='empty')
        button_green=tk.Radiobutton(button_frame,text="vegetation",variable=self.info_class,value='vegetation').grid(row=2,column=1,sticky='w')
        button_bareland=tk.Radiobutton(button_frame,text="bareland",variable=self.info_class,value='bareland').grid(row=2,column=2,sticky='w')    
        button_water=tk.Radiobutton(button_frame,text="water",variable=self.info_class,value='water').grid(row=2,column=3,sticky='w') 

        self.info=tk.Label(self.master,bg='white',textvariable=self.info_class,fg='black',text='empty',font=('Arial', 12), width=10, height=1).grid(row=0,padx=5,pady=5,sticky='nw')
        self.scale_=1
        
        #绘制一个参考点
        self.ref_pts=[self.canvas.create_oval((0,0,1.5,1.5),fill='white'), self.canvas.create_oval((self.width,self.height,self.width-0.5, self.height-0.5),fill='white')] 
        
        self.ref_coordi={'ref_pts':[((self.canvas.coords(i)[2]+self.canvas.coords(i)[0])/2,(self.canvas.coords(i)[3]+self.canvas.coords(i)[1])/2) for i in self.ref_pts]}
        self.sample_coordi_recover={}
        

    def compute_samplePosition(self):
        self.xy_rec.update({'ref_pts':self.ref_pts})
        #print(self.xy_rec)
        sample_coordi={key:[((self.canvas.coords(i)[2]+self.canvas.coords(i)[0])/2,(self.canvas.coords(i)[3]+self.canvas.coords(i)[1])/2) for i in self.xy_rec[key]] for key in self.xy_rec.keys()}
        print("+"*50)
        print("sample coordi:",sample_coordi)
        print("_"*50)
        print(self.ref_coordi)
        print("image size:",self.width, self.height )
        print("_"*50)
        distance=lambda p1,p2:math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
        scale_byDistance=distance(sample_coordi['ref_pts'][0],sample_coordi['ref_pts'][1])/distance(self.ref_coordi['ref_pts'][0],self.ref_coordi['ref_pts'][1])
        print("scale_byDistance:",scale_byDistance)
        print("scale_by_self.scale_:",self.scale_)
        
        #缩放回原始坐标系
        
        #x_distance=sample_coordi['ref_pts'][0][0]-self.ref_coordi['ref_pts'][0][0]
        #y_distance=sample_coordi['ref_pts'][0][1]-self.ref_coordi['ref_pts'][0][1]
        f_scale=np.array([[1/scale_byDistance,0],[0,1/scale_byDistance]])
        #f_scale=np.array([[scale_byDistance,0,x_distance],[0,scale_byDistance,y_distance],[0,0,scale_byDistance]])
        #print("x_distance,y_distance:",np.array([x_distance,y_distance]))
        
        sample_coordi_recover={key:np.matmul(np.array(sample_coordi[key]),f_scale) for key in sample_coordi.keys() if sample_coordi[key]!=[]}
        print("sample_coordi_recove",sample_coordi_recover)
        relative_coordi=np.array(sample_coordi_recover['ref_pts'][0])-1.5/2
        sample_coordi_recover={key:sample_coordi_recover[key]-relative_coordi for key in sample_coordi_recover.keys() }
        
        print("sample_coordi_recove",sample_coordi_recover)
        self.sample_coordi_recover=sample_coordi_recover
    
    def click_xy_collection(self,event):
        multiple=self.imscale
        length=1.5*multiple #根据图像缩放比例的变化调节所绘制矩形的大小，保持大小一致
        
        event2canvas=lambda e,c:(c.canvasx(e.x),c.canvasy(e.y)) 
        cx,cy=event2canvas(event,self.canvas) #cx,cy=event2canvas(event,self.canvas)        
        print(cx,cy)         
        if self.info_class.get()=='vegetation':      
            self.xy["vegetation"].append((cx,cy))  
            rec=self.canvas.create_oval((cx-length,cy-length,cx+length,cy+length),fill='yellow')
            self.xy_rec["vegetation"].append(rec)
        elif self.info_class.get()=='bareland':
            self.xy["bareland"].append((cx,cy))  
            rec=self.canvas.create_oval((cx-length,cy-length,cx+length,cy+length),fill='red')
            self.xy_rec["bareland"].append(rec)        
        elif self.info_class.get()=='water':
            self.xy["water"].append((cx,cy))  
            rec=self.canvas.create_oval((cx-length,cy-length,cx+length,cy+length),fill='aquamarine')
            self.xy_rec["water"].append(rec)    
            
        print("_"*50)
        print("sampling count",{key:len(self.xy_rec[key]) for key in self.xy_rec.keys()})    
        print("total:",sum([len(self.xy_rec[key]) for key in self.xy_rec.keys()]) )
        
    def __scroll_x(self,*args,**kwargs):
        '''水平滚动画布，并重画图像'''
        self.canvas.xview(*args,**kwargs)#滚动水平条
        self.show_image() #重画图像
        
    def __scroll_y(self, *args, **kwargs):
        """ 垂直滚动画布，并重画图像"""
        self.canvas.yview(*args,**kwargs)  #垂直滚动
        self.show_image()  #重画图像      

    def __move_from(self, event):
        ''' 鼠标滚动，前一坐标 '''
        self.canvas.scan_mark(event.x, event.y)

    def __move_to(self, event):
        ''' 鼠标滚动，下一坐标'''
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.show_image()  #重画图像 

    def __wheel(self, event):
        ''' 鼠标滚轮缩放 '''
        x=self.canvas.canvasx(event.x)
        y=self.canvas.canvasy(event.y)
        bbox = self.canvas.bbox(self.container)  # 图像区域
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]: pass  #鼠标如果在图像区域内部
        else: return  # 只有鼠标在图像内才可以滚动缩放
        scale=1.0
        # 响应Linux (event.num)或Windows (event.delta)滚轮事件
        if event.num==5 or event.delta == -120:  # 向下滚动
            i=min(self.width, self.height)
            if int(i * self.imscale) < 30: return  # 图像小于30 pixels
            self.imscale /= self.delta
            scale/= self.delta
        if event.num==4 or event.delta == 120:  # 向上滚动
            i=min(self.canvas.winfo_width(), self.canvas.winfo_height())
            if i < self.imscale: return  # 如果1个像素大于可视图像区域
            self.imscale *= self.delta
            scale*= self.delta
        self.canvas.scale('all', x, y, scale, scale)  # 缩放画布上的所有对象
        self.show_image()
        self.scale_=scale*self.scale_

    def show_image(self, event=None):
        ''' 在画布上显示图像'''
        bbox1=self.canvas.bbox(self.container)  #获得图像区域
        # 在bbox1的两侧移除1个像素的移动
        bbox1 = (bbox1[0] + 1, bbox1[1] + 1, bbox1[2] - 1, bbox1[3] - 1)
        bbox2 = (self.canvas.canvasx(0),  # 获得画布上的可见区域
                 self.canvas.canvasy(0),
                 self.canvas.canvasx(self.canvas.winfo_width()),
                 self.canvas.canvasy(self.canvas.winfo_height()))
        bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),  #获取滚动区域框
                max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]
        if bbox[0] == bbox2[0] and bbox[2] == bbox2[2]:  # 整个图像在可见区域
            bbox[0] = bbox1[0]
            bbox[2] = bbox1[2]
        if bbox[1] == bbox2[1] and bbox[3] == bbox2[3]:  # 整个图像在可见区域
            bbox[1] = bbox1[1]
            bbox[3] = bbox1[3]
        self.canvas.configure(scrollregion=bbox)  # 设置滚动区域
        x1 = max(bbox2[0] - bbox1[0], 0)  # 得到图像平铺的坐标(x1,y1,x2,y2)
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # 显示图像，如果它在可见的区域
            x = min(int(x2 / self.imscale), self.width)   # 有时大于1个像素...
            y = min(int(y2 / self.imscale), self.height)  # ...有时不是
            image = self.img.crop((int(x1 / self.imscale), int(y1 / self.imscale), x, y))
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1))))
            imageid = self.canvas.create_image(max(bbox2[0], bbox1[0]), max(bbox2[1], bbox1[1]),anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # 将图像设置为背景
            self.canvas.imagetk=imagetk  # keep an extra reference to prevent garbage-collection

```

```python
class main_window:    
    '''主窗口类'''
    def __init__(self,mainframe,rgb_band,img_path=0,landsat_stack=0,):
        '''读取图像'''
        if img_path:
            self.img_path=img_path
            self.__image=Image.open(self.img_path)
        if rgb_band:
            self.rgb_band=rgb_band    
        if type(landsat_stack) is np.ndarray:
            self.landsat_stack=landsat_stack
            self.__image=self.landsat_stack_array2img(self.landsat_stack,self.rgb_band)

        self.MW=CanvasImage(mainframe,self.__image)
 
    def landsat_stack_array2img(self,landsat_stack,rgb_band):
        r,g,b=self.rgb_band
        landsat_stack_rgb=np.dstack((landsat_stack[r],landsat_stack[g],landsat_stack[b]))  #合并三个波段
        landsat_stack_rgb_255=img_as_ubyte(landsat_stack_rgb) #使用skimage提供的方法，将float等浮点型色彩，转换为0-255整型
        landsat_image=Image.fromarray(landsat_stack_rgb_255)
        return landsat_image

```

```python
if __name__ == "__main__":
    img_path=r'C:\Users\richi\Pictures\n.png' 
    
    workspace=r"C:\Users\richi\omen-richiebao\omen_github\Urban-Spatial-Data-Analysis_python\notebook\BaiduMapPOIcollection_ipynb\data"
    img_fp=os.path.join(workspace,'a_191018_exposure_rescaled.npy')
    landsat_stack=np.load(img_fp)        
    
    rgb_band=[3,2,1]          
    mainframe=tk.Tk()
    app=main_window(mainframe, rgb_band=rgb_band,landsat_stack=landsat_stack) #img_path=img_path,landsat_stack=landsat_stack,
    #app=main_window(mainframe, rgb_band=rgb_band,img_path=img_path)
    mainframe.mainloop()
    
    #保存采样点
    import pickle as pkl
    with open(os.path.join(workspace,r'sampling_position.pkl'),'wb') as handle:
        pkl.dump(app.MW.sample_coordi_recover,handle)
```

<a href=""><img src="./imgs/11_03.png" height="auto" width="800" title="caDesign"></a>

#### 1.3.2 分类精度计算
混淆矩阵（confusion matrix），每一行（列）代表一个类的实例预测，而每一列（行）代表一个实际的类的实例，是一种特殊的，具有两个维度（实际值和预测值）的列联表（contingency table），并且两个维度中都有着一样的类别的集合。例如：$\begin{bmatrix}&&预测的类别 & &\\&&baraland&vegetation&water\\实际的类别 &bareland &51&2&0\\&vegetation&10&54&0\\ &water&0&2&19  \end{bmatrix} $，可以解读为，总共138个样本（采样点），真实裸地为53个，误判为绿地的2个；真实绿地为64个，误判为裸地的10个；真实水体为21个，误判为绿地的2个。混淆矩阵的计算使用Sklearn库提供的'confusion_matrix'方法计算。

通过计算混淆矩阵，分析获知绿地中误判为裸地的相对较多，一方面是在解译过程中，%35的阈值界限可以适当调大，使部分绿地划分到裸地类别；另一方面，可能是在采样过程中，采样点设置的比较大，同时覆盖了绿地和裸地，不能确定最终坐标是落于绿地还是落于裸地，造成采样上的错误，此时可以调小采样点直径，更精确的定位。

除了计算混淆矩阵，同时计算百分比精度，即正确的分类占总样本数的比值。


```python
import pickle as pkl
import os
import pandas as pd
workspace=r"./data"
with open(os.path.join(workspace,r'sampling_position_138.pkl'),'rb') as handle:
    sampling_position=pkl.load(handle)
sampling_position_int={key:sampling_position[key].astype(int) for key in sampling_position.keys() if key !='ref_pts'}
i=0
sampling_position_int_={}
for key in sampling_position_int.keys():
    for j in sampling_position_int[key]:
        sampling_position_int_.update({i:[j.tolist(),key]})
        i+=1
sampling_position_df=pd.DataFrame.from_dict(sampling_position_int_,columns=['coordi','category'],orient='index') #转换为pandas的DataFrame数据格式，方便数据处理
sampling_position_df['interpretation']=[green_water_bareLand[coordi[1]][coordi[0]] for coordi in sampling_position_df.coordi]
interpretation_mapping={1:"bareland",2:"vegetation",3:"water",0:"bareland"}
sampling_position_df.interpretation=sampling_position_df.interpretation.replace(interpretation_mapping)
sampling_position_df.to_pickle(os.path.join(workspace,r'sampling_position_df.pkl'))
```

```python
from sklearn.metrics import confusion_matrix
precision_confusionMatrix=confusion_matrix(sampling_position_df.category, sampling_position_df.interpretation)
precision_percent=np.sum(precision_confusionMatrix.diagonal())/len(sampling_position_df)
print("precision - confusion Matrix:\n",precision_confusionMatrix)     
print("precision - percent:",precision_percent)
```

### 1.4 要点
#### 1.4.1 数据处理技术

* 配合使用rasterio,geopandas, earthpy处理遥感影像

* 使用plotly构建可交互图表

* 使用tkinter构建交互的GUI采样工具

#### 1.4.2 新建立的函数

* function - 给定裁切边界，批量裁切栅格数据，`raster_clip(raster_fp,clip_boundary_fp,save_path)`

* function - 指定波段，同时显示多个遥感影像，`w_180310_array(img_stack_list,band_num)`

* function - 将变量名转换为字符串， `variable_name(var)`

* function - 拉伸图像 contract stretching，`image_exposure(img_bands,percentile=(2,98))`

* function - 将数据按照给定的百分数划分，并给定固定的值,整数值或RGB色彩值，`data_division(data,division,right=True)`

* function - 多个栅格数据，给定百分比，变化观察，`percentile_slider(season_dic)`

* 基于tkinter，开发交互式GUI采样工具

#### 1.4.3 所调用的库

```python 
import os,random

import earthpy.spatial as es
import earthpy.plot as ep

import geopandas as gpd
from pyproj import CRS
import rasterio as rio
from rasterio.plot import plotting_extent

import matplotlib.pyplot as plt

from skimage import exposure
from skimage.transform import rescale, resize, downscale_local_mean

import numpy as np
import pandas as pd
from scipy.stats import norm
import math

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import widgets
from IPython.display import display

import warnings
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from skimage.util import img_as_ubyte

import pickle as pkl
from sklearn.metrics import confusion_matrix
```

#### 1.4.4 参考文献
-
