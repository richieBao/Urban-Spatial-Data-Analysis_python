# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:02:19 2020

@author: richieBao-caDesign设计(cadesign.cn)
"""

#位于百度地图POI数据的抓取与地理空间点地图/1. 百度地图POI数据的抓取-单个分类实现与地理空间点地图/1.1 单个分类POI爬取
def baiduPOI_dataCrawler(query_dic,bound_coordinate,partition,page_num_range,poi_fn_list=False):
    #所调用的库及辅助文件
    import coordinate_transformation as cc    
    import urllib, json, csv,os,pathlib
    
    '''function-百度地图开放平台POI数据爬取'''
    urlRoot='http://api.map.baidu.com/place/v2/search?' #数据下载网址，查询百度地图服务文档
    #切分检索区域
    xDis=(bound_coordinate['rightTop'][0]-bound_coordinate['leftBottom'][0])/partition
    yDis=(bound_coordinate['rightTop'][1]-bound_coordinate['leftBottom'][1])/partition    
    #判断是否要写入文件
    if poi_fn_list:
        for file_path in poi_fn_list:
            fP=pathlib.Path(file_path)
            if fP.suffix=='.csv':
                poi_csv=open(fP,'w',encoding='utf-8')
                csv_writer=csv.writer(poi_csv)    
            elif fP.suffix=='.json':
                poi_json=open(fP,'w',encoding='utf-8')
    num=0
    jsonDS=[] #存储读取的数据，用于.json格式数据的保存
    #循环切分的检索区域，逐区下载数据
    print("Start downloading data...")
    for i in range(partition):
        for j in range(partition):
            leftBottomCoordi=[bound_coordinate['leftBottom'][0]+i*xDis,bound_coordinate['leftBottom'][1]+j*yDis]
            rightTopCoordi=[bound_coordinate['leftBottom'][0]+(i+1)*xDis,bound_coordinate['leftBottom'][1]+(j+1)*yDis]
            for p in page_num_range:  
                #更新请求参数
                query_dic.update({'page_num':str(p),
                                  'bounds':str(leftBottomCoordi[1]) + ',' + str(leftBottomCoordi[0]) + ','+str(rightTopCoordi[1]) + ',' + str(rightTopCoordi[0]),
                                  'output':'json',
                                 })

                url=urlRoot+urllib.parse.urlencode(query_dic)
                data=urllib.request.urlopen(url)
                responseOfLoad=json.loads(data.read()) 
                if responseOfLoad.get("message")=='ok':
                    results=responseOfLoad.get("results") 
                    for row in range(len(results)):
                        subData=results[row]
                        baidu_coordinateSystem=[subData.get('location').get('lng'),subData.get('location').get('lat')] #获取百度坐标系
                        Mars_coordinateSystem=cc.bd09togcj02(baidu_coordinateSystem[0], baidu_coordinateSystem[1]) #百度坐标系-->火星坐标系
                        WGS84_coordinateSystem=cc.gcj02towgs84(Mars_coordinateSystem[0],Mars_coordinateSystem[1]) #火星坐标系-->WGS84
                        
                        #更新坐标
                        subData['location']['lat']=WGS84_coordinateSystem[1]
                        subData['detail_info']['lat']=WGS84_coordinateSystem[1]
                        subData['location']['lng']=WGS84_coordinateSystem[0]
                        subData['detail_info']['lng']=WGS84_coordinateSystem[0]  
                        
                        if csv_writer:
                            csv_writer.writerow([subData]) #逐行写入.csv文件
                        jsonDS.append(subData)
            num+=1       
            print("No."+str(num)+" was written to the .csv file.")
    if poi_json:       
        json.dump(jsonDS,poi_json)
        poi_json.write('\n')
        poi_json.close()
    if poi_csv:
        poi_csv.close()
    print("The download is complete.")
    return jsonDS


#位于百度地图POI数据的抓取与地理空间点地图/1. 百度地图POI数据的抓取-单个分类实现与地理空间点地图/1.2 将.csv格式的POI数据转换为pandas的DataFrame
def csv2df(poi_fn_csv):
    import pandas as pd
    from benedict import benedict #benedict库是dict的子类，支持键列表（keylist）/键路径（keypath），应用该库的flatten方法展平嵌套的字典，准备用于DataFrame数据结构
    import csv
    '''function-转换.csv格式的POI数据为pandas的DataFrame'''
    n=0
    with open(poi_fn_csv, newline='',encoding='utf-8') as csvfile:
        poi_reader=csv.reader(csvfile)
        poi_dict={}    
        poiExceptions_dict={}
        for row in poi_reader:    
            if row:
                try:
                    row_benedict=benedict(eval(row[0])) #用eval方法，将字符串字典"{}"转换为字典{}
                    flatten_dict=row_benedict.flatten(separator='_') #展平嵌套字典
                    poi_dict[n]=flatten_dict
                except:                    
                    print("incorrect format of data_row number:%s"%n)                    
                    poiExceptions_dict[n]=row
            n+=1
            #if n==5:break #因为循环次数比较多，在调试代码时，可以设置停止的条件，节省时间与方便数据查看
    poi_df=pd.concat([pd.DataFrame(poi_dict[d_k].values(),index=poi_dict[d_k].keys(),columns=[d_k]).T for d_k in poi_dict.keys()], sort=True,axis=0)
    # print("_"*50)
    for col in poi_df.columns:
        try:
            poi_df[col]=pd.to_numeric(poi_df[col])
        except:
            pass
            #print("%s data type is not converted..."%(col))
    print("_"*50)
    print(".csv to DataFrame is completed!")
    #print(poi_df.head()) #查看最终DataFrame格式下POI数据
    #print(poi_df.dtypes) #查看数据类型
    return poi_df


#位于百度地图POI数据的抓取与地理空间点地图/1.1 多个分类POI爬取|1.2 批量转换.csv格式数据为GeoDataFrame/1.2.1 定义提取文件夹下所有文件路径的函数
def filePath_extraction(dirpath,fileType):
    import os
    '''以所在文件夹路径为键，值为包含该文件夹下所有文件名的列表。文件类型可以自行定义 '''
    filePath_Info={}
    i=0
    for dirpath,dirNames,fileNames in os.walk(dirpath): #os.walk()遍历目录，使用help(os.walk)查看返回值解释
       i+=1
       if fileNames: #仅当文件夹中有文件时才提取
           tempList=[f for f in fileNames if f.split('.')[-1] in fileType]
           if tempList: #剔除文件名列表为空的情况,即文件夹下存在不为指定文件类型的文件时，上一步列表会返回空列表[]
               filePath_Info.setdefault(dirpath,tempList)
    return filePath_Info


#频数分布计算
def frequency_bins(df,bins):
    import pandas as pd
    '''function-频数分布计算'''

    #A-组织数据
    column_name=df.columns[0]
    column_bins_name=df.columns[0]+'_bins'
    df[column_bins_name]=pd.cut(x=df[column_name],bins=bins,right=False) #参数right=False指定为包含左边值，不包括右边值。
    df_bins=df.sort_values(by=[column_name]) #按照分割区间排序
    df_bins.set_index([column_bins_name,df_bins.index],drop=False,inplace=True) #以price_bins和原索引值设置多重索引，同时配置drop=False参数保留原列。
    #print(df_bins.head(10))

    #B-频数计算
    dfBins_frequency=df_bins[column_bins_name].value_counts() #dropna=False  
    dfBins_relativeFrequency=df_bins[column_bins_name].value_counts(normalize=True) #参数normalize=True将计算相对频数(次数) dividing all values by the sum of values
    dfBins_freqANDrelFreq=pd.DataFrame({'fre':dfBins_frequency,'relFre':dfBins_relativeFrequency})
    #print(dfBins_freqANDrelFreq)

    #C-组中值计算
    dfBins_median=df_bins.median(level=0)
    dfBins_median.rename(columns={column_name:'median'},inplace=True)
    #print(dfBins_median)

    #D-合并分割区间、频数计算和组中值的DataFrame格式数据。
    df_fre=dfBins_freqANDrelFreq.join(dfBins_median).sort_index().reset_index() #在合并时会自动匹配index
    #print(ranmen_fre)

    #E-计算频数比例
    df_fre['fre_percent%']=df_fre.apply(lambda row:row['fre']/df_fre.fre.sum()*100,axis=1)

    return df_fre


#convert points .shp to raster 将点数据写入为raster数据。使用raster.SetGeoTransform,栅格化数据。参考GDAL官方代码
def pts2raster(pts_shp,raster_path,cellSize,field_name=False):
    from osgeo import gdal, ogr,osr
    '''
    function - 将.shp格式的点数据转换为.tif栅格(raster)
    
    Paras:
    pts_shp - .shp格式点数据文件路径
    raster_path - 保存的栅格文件路径
    cellSize - 栅格单元大小
    field_name - 写入栅格的.shp点数据属性字段
    '''
    #定义空值（没有数据）的栅格数值 Define NoData value of new raster
    NoData_value=-9999
    
    #打开.shp点数据，并返回地理区域范围 Open the data source and read in the extent
    source_ds=ogr.Open(pts_shp)
    source_layer=source_ds.GetLayer()
    x_min, x_max, y_min, y_max=source_layer.GetExtent()
    
    #使用GDAL库建立栅格 Create the destination data source
    x_res=int((x_max - x_min) / cellSize)
    y_res=int((y_max - y_min) / cellSize)
    target_ds=gdal.GetDriverByName('GTiff').Create(raster_path, x_res, y_res, 1, gdal.GDT_Float64) #gdal的数据类型 gdal.GDT_Float64,gdal.GDT_Int32...
    target_ds.SetGeoTransform((x_min, cellSize, 0, y_max, 0, -cellSize))
    outband=target_ds.GetRasterBand(1)
    outband.SetNoDataValue(NoData_value)

    #向栅格层中写入数据
    if field_name:
        gdal.RasterizeLayer(target_ds,[1], source_layer,options=["ATTRIBUTE={0}".format(field_name)])
    else:
        gdal.RasterizeLayer(target_ds,[1], source_layer,burn_values=[-1])   
        
    #配置投影坐标系统
    spatialRef=source_layer.GetSpatialRef()
    target_ds.SetProjection(spatialRef.ExportToWkt())       
        
    outband.FlushCache()
    return gdal.Open(raster_path).ReadAsArray()


def pts_geoDF2raster(pts_geoDF,raster_path,cellSize,scale):
    from osgeo import gdal,ogr,osr
    import numpy as np
    from scipy import stats
    '''
    function - 将GeoDaraFrame格式的点数据转换为栅格数据
    
    Paras:
    pts_geoDF - GeoDaraFrame格式的点数据
    raster_path - 保存的栅格文件路径
    cellSize - 栅格单元大小
    scale - 缩放核密度估计值
    '''
    #定义空值（没有数据）的栅格数值 Define NoData value of new raster
    NoData_value=-9999
    x_min, y_min,x_max, y_max=pts_geoDF.geometry.total_bounds

    #使用GDAL库建立栅格 Create the destination data source
    x_res=int((x_max - x_min) / cellSize)
    y_res=int((y_max - y_min) / cellSize)
    target_ds=gdal.GetDriverByName('GTiff').Create(raster_path, x_res, y_res, 1, gdal.GDT_Float64 )
    target_ds.SetGeoTransform((x_min, cellSize, 0, y_max, 0, -cellSize))
    outband=target_ds.GetRasterBand(1)
    outband.SetNoDataValue(NoData_value)   
    
    #配置投影坐标系统
    spatialRef = osr.SpatialReference()
    epsg=int(pts_geoDF.crs.srs.split(":")[-1])
    spatialRef.ImportFromEPSG(epsg)  
    target_ds.SetProjection(spatialRef.ExportToWkt())
    
    #向栅格层中写入数据
    #print(x_res,y_res)
    X, Y = np.meshgrid(np.linspace(x_min,x_max,x_res), np.linspace(y_min,y_max,y_res))
    positions=np.vstack([X.ravel(), Y.ravel()])
    values=np.vstack([pts_geoDF.geometry.x, pts_geoDF.geometry.y])    
    print("Start calculating kde...")
    kernel=stats.gaussian_kde(values)
    Z=np.reshape(kernel(positions).T, X.shape)
    print("Finish calculating kde!")
    #print(values)
        
    outband.WriteArray(np.flip(Z,0)*scale)        
    outband.FlushCache()
    print("conversion complete!")
    return gdal.Open(raster_path).ReadAsArray()


def start_time():
    import datetime
    '''
    function-计算当前时间
    '''
    start_time=datetime.datetime.now()
    print("start time:",start_time)
    return start_time

def duration(start_time):
    import datetime
    '''
    function-计算持续时间

    Paras:
    start_time - 开始时间
    '''
    end_time=datetime.datetime.now()
    print("end time:",end_time)
    duration=(end_time-start_time).seconds/60
    print("Total time spend:%.2f minutes"%duration)
    
    
def is_outlier(data,threshold=3.5):
   import numpy as np
   '''
   function-判断异常值

   Params:
   data - 待分析的数据，列表或者一维数组
   threshold - 判断是否为异常值的边界条件    
   '''
   MAD=np.median(abs(data-np.median(data)))
   modified_ZScore=0.6745*(data-np.median(data))/MAD
   #print(modified_ZScore)
   is_outlier_bool=modified_ZScore>threshold    
   return is_outlier_bool,data[~is_outlier_bool]   


def print_html(df,row_numbers=5):
    from IPython.display import HTML
    '''
    function - 在Jupyter中打印DataFrame格式数据为HTML

    Paras:
    df - 需要打印的DataFrame或GeoDataFrame格式数据
    row_numbers - 打印的行数，如果为正，从开始打印如果为负，从末尾打印
     '''
    if row_numbers>0:
        return HTML(df.head(row_numbers).to_html())
    else:
        return HTML(df.tail(abs(row_numbers)).to_html())


def coefficient_of_determination(observed_vals,predicted_vals):
    import pandas as pd
    import numpy as np
    import math
    '''
    function - 回归方程的判定系数
    
    Paras:
    observed_vals - 观测值（实测值）
    predicted_vals - 预测值
    '''
    vals_df=pd.DataFrame({'obs':observed_vals,'pre':predicted_vals})
    #观测值的离差平方和(总平方和，或总的离差平方和)
    obs_mean=vals_df.obs.mean()
    SS_tot=vals_df.obs.apply(lambda row:(row-obs_mean)**2).sum()
    #预测值的离差平方和
    pre_mean=vals_df.pre.mean()
    SS_reg=vals_df.pre.apply(lambda row:(row-pre_mean)**2).sum()
    #观测值和预测值的离差积和
    SS_obs_pre=vals_df.apply(lambda row:(row.obs-obs_mean)*(row.pre-pre_mean), axis=1).sum()
    
    #残差平方和
    SS_res=vals_df.apply(lambda row:(row.obs-row.pre)**2,axis=1).sum()
    
    #判断系数
    R_square_a=(SS_obs_pre/math.sqrt(SS_tot*SS_reg))**2
    R_square_b=1-SS_res/SS_tot
            
    return R_square_a,R_square_b


def vector_plot_3d(ax_3d,C,origin_vector,vector,color='r',label='vector',arrow_length_ratio=0.1):
    '''
    funciton - 转换SymPy的vector及Matrix数据格式为matplotlib可以打印的数据格式

    Paras:
    ax_3d - matplotlib的3d格式子图
    C - /coordinate_system - SymPy下定义的坐标系
    origin_vector - 如果是固定向量，给定向量的起点（使用向量，即表示从坐标原点所指向的位置），如果是自由向量，起点设置为坐标原点
    vector - 所要打印的向量
    color - 向量色彩
    label - 向量标签
    arrow_length_ratio - 向量箭头大小     
    '''
    origin_vector_matrix=origin_vector.to_matrix(C)
    x=origin_vector_matrix.row(0)[0]
    y=origin_vector_matrix.row(1)[0]
    z=origin_vector_matrix.row(2)[0]

    vector_matrix=vector.to_matrix(C)
    u=vector_matrix.row(0)[0]
    v=vector_matrix.row(1)[0]
    w=vector_matrix.row(2)[0]
    ax_3d.quiver(x,y,z,u,v,w,color=color,label=label,arrow_length_ratio=arrow_length_ratio)
    
    
def move_alongVectors(vector_list,coeffi_list,C,ax,):
    import random
    import sympy
    '''
    function - 给定向量，及对应系数，延向量绘制

    Paras:
    vector_list - 向量列表，按移动顺序
    coeffi_list - 向量的系数    
    C - SymPy下定义的坐标系
    ax - 子图
    '''
    colors=[color[0] for color in mcolors.TABLEAU_COLORS.items()]  #mcolors.BASE_COLORS, mcolors.TABLEAU_COLORS,mcolors.CSS4_COLORS
    colors__random_selection=random.sample(colors,len(vector_list)-1)
    v_accumulation=[]
    v_accumulation.append(vector_list[0])
    #每个向量绘制以之前所有向量之和为起点
    for expr in vector_list[1:]:
        v_accumulation.append(expr+v_accumulation[-1])

    v_accumulation=v_accumulation[:-1]   
    for i in range(1,len(vector_list)):
        vector_plot_3d(ax,C,v_accumulation[i-1].subs(coeffi_list),vector_list[i].subs(coeffi_list),color=colors__random_selection[i-1],label='v_%s'%coeffi_list[i-1][0],arrow_length_ratio=0.2)    
    
    
def LandsatMTL_info(fp):
    import os
    import re
    '''
    function - 读取landsat *_MTL.txt文件，提取需要的信息

    Paras:
    fp - Landsat 文件根目录

    return:
    band_fp_dic - 返回各个波段的路径字典
    Landsat_para - 返回Landsat 参数 
    '''
    fps=[os.path.join(root,file) for root, dirs, files in os.walk(fp) for file in files] #提取文件夹下所有文件的路径
    MTLPattern=re.compile(r'_MTL.txt',re.S) #匹配对象模式，提取_MTL.txt遥感影像的元数据文件
    MTLFn=[fn for fn in fps if re.findall(MTLPattern,fn)][0]
    with open(MTLFn,'r') as f: #读取所有元数据文件信息
        MTLText=f.read()
    bandFn_Pattern=re.compile(r'FILE_NAME_BAND_[0-9]\d* = "(.*?)"\n',re.S)  #Landsat 波段文件
    band_fn=re.findall(bandFn_Pattern,MTLText)
    band_fp=[[(re.findall(r'B[0-9]\d*',fn)[0], re.findall(r'.*?%s$'%fn,f)[0]) for f in fps if re.findall(r'.*?%s$'%fn,f)] for fn in band_fn] #(文件名，文件路径)
    band_fp_dic={i[0][0]:i[0][1] for i in band_fp}
    #需要数据的提取标签/根据需要读取元数据信息
    values_fields=["RADIANCE_ADD_BAND_10",
                   "RADIANCE_ADD_BAND_11",
                   "RADIANCE_MULT_BAND_10",
                   "RADIANCE_MULT_BAND_11",
                   "K1_CONSTANT_BAND_10",
                   "K2_CONSTANT_BAND_10",
                   "K1_CONSTANT_BAND_11",
                   "K2_CONSTANT_BAND_11",
                   "DATE_ACQUIRED",
                   "SCENE_CENTER_TIME",
                   "MAP_PROJECTION",
                   "DATUM",
                   "UTM_ZONE"]    
    Landsat_para={field:re.findall(re.compile(r'%s = "*(.*?)"*\n'%field),MTLText)[0] for field in values_fields} #（参数名，参数值）
    return  band_fp_dic,Landsat_para #返回所有波段路径和需要的参数值    

def NDVI(RED_band,NIR_band):
    import numpy as np
    '''
    function - 计算NDVI指数

    Paras:
    RED_band - 红色波段
    NIR_band - 近红外波段
    '''
    RED_band=np.ma.masked_where(NIR_band+RED_band==0,RED_band)
    NDVI=(NIR_band-RED_band)/(NIR_band+RED_band)
    NDVI=NDVI.filled(-9999)
    print("NDVI"+"_min:%f,max:%f"%(NDVI.min(),NDVI.max()))
    return NDVI

def las_2_DSM_Classification_raster_(las_fp,save_path):
    import pdal
    '''
    function - 转换单个.las点云数据为分类栅格数据，和DSM栅格数据
    
    Paras:
    las_fp - .las格式文件路径
    save_path - 保存路径列表，分类DSM存储与不同路径下
    
    '''
    
    #pipeline-用于建立分类栅格
    json_classification="""
        {
        "pipeline": [
                     "%s",
                     {
                     "filename":"%s",
                     "type":"writers.gdal",
                     "dimension":"Classification",
                     "data_type":"uint16_t",
                     "output_type":"mean",  
                     "resolution": 1
                     }               
        ]        
        }"""%(las_fp,save_path[0])
    
    #pipeline-用于建立DSM栅格数据
    json_DSM="""
        {
        "pipeline": [
                     "%s",
                     {
                     "filename":"%s",
                     "gdaldriver":"GTiff",
                     "type":"writers.gdal",
                     "output_type":"mean",  
                     "resolution": 1
                     }               
        ]        
        }"""%(las_fp,save_path[1])    
    
    json_combo=[json_classification,json_DSM] #配置选择性输出
    for json in json_combo:
        pipeline=pdal.Pipeline(json)
        pipeline.loglevel=8 #日志级别配置
        if pipeline.validate(): #检查json选项是否正确
            #print(pipeline.validate())
            try:
                count=pipeline.execute()
            except:
                print("\n An exception occurred,the file name:%s"%las_fp)
                print(pipeline.log) #如果出现错误，打印日志查看，以便修正错误代码
                
        else:
            print("pipeline unvalidate!!!")
            
def  las_info_extraction(las_fp,json_combo):
    import pdal
    '''
    function - 转换单个.las点云数据为分类栅格数据，和DSM栅格数据
    
    Paras:
    las_fp - .las格式文件路径
    save_path - 保存路径列表，分类DSM存储与不同路径下
    
    '''
    pipeline_list=[]
    if 'json_classification' in json_combo.keys():
        #pipeline-用于建立分类栅格
        json_classification="""
            {
            "pipeline": [
                         "%s",
                         {
                         "filename":"%s",
                         "type":"writers.gdal",
                         "dimension":"Classification",
                         "data_type":"uint16_t",
                         "output_type":"mean",  
                         "resolution": 1
                         }               
            ]        
            }"""%(las_fp,json_combo['json_classification'])
        pipeline_list.append(json_classification)
        
    elif 'json_DSM' in json_combo.keys():
        #pipeline-用于建立DSM栅格数据
        json_DSM="""
            {
            "pipeline": [
                         "%s",
                         {
                         "filename":"%s",
                         "gdaldriver":"GTiff",
                         "type":"writers.gdal",
                         "output_type":"mean",  
                         "resolution": 1
                         }               
            ]        
            }"""%(las_fp,json_combo['json_DSM']) 
        pipeline_list.append(json_DSM)
        
    elif 'json_ground' in json_combo.keys():
        #pipelin-用于提取ground地表
        json_ground="""
            {
            "pipeline": [
                         "%s",
                         {
                         "type":"filters.range",
                         "limits":"Classification[2:2]"                     
                         },
                         {
                         "filename":"%s",
                         "gdaldriver":"GTiff",
                         "type":"writers.gdal",
                         "output_type":"mean",  
                         "resolution": 1
                         }               
            ]        
            }"""%(las_fp,json_combo['json_ground'])   
        pipeline_list.append(json_ground)
    
    
    for json in pipeline_list:
        pipeline=pdal.Pipeline(json)
        pipeline.loglevel=8 #日志级别配置
        if pipeline.validate(): #检查json选项是否正确
            #print(pipeline.validate())
            try:
                count=pipeline.execute()
            except:
                print("\n An exception occurred,the file name:%s"%las_fp)
                print(pipeline.log) #如果出现错误，打印日志查看，以便修正错误代码
                
        else:
            print("pipeline unvalidate!!!")
    #print("finished conversion...")     

def las_info_extraction_combo(las_dirPath,json_combo_):
    import util,os,re
    from tqdm import tqdm
    '''
    function - 批量转换.las点云数据为DTM和分类栅格
    
    Paras:
    las_dirPath - LAS文件路径
    save_path - 保存路径
    
    return:
        
    '''
    file_type=['las']
    las_fn=util.filePath_extraction(las_dirPath,file_type)
    '''展平列表函数'''
    flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]
    las_fn_list=flatten_lst([[os.path.join(k,las_fn[k][i]) for i in range(len(las_fn[k]))] for k in las_fn.keys()])
    pattern=re.compile(r'[_](.*?)[.]', re.S) 
    for i in tqdm(las_fn_list):  
        fn_num=re.findall(pattern, i.split("\\")[-1])[0] #提取文件名字符串中的数字
        #注意文件名路径中"\"和"/"，不同库支持的类型可能有所不同，需自行调整
        json_combo={key:os.path.join(json_combo_[key],"%s_%s.tif"%(os.path.split(json_combo_[key])[-1],fn_num)).replace("\\","/") for key in json_combo_.keys()}        
        util.las_info_extraction_combo(i.replace("\\","/"),json_combo)
    
       
def raster_mosaic(dir_path,out_fp,):
    import rasterio,glob,os
    from rasterio.merge import merge
    '''
    function - 合并多个栅格为一个
    
    Paras:
    dir_path - 栅格根目录
    out-fp - 保存路径
    
    return:
    out_trans - 返回变换信息
    '''
    
    #迁移rasterio提供的定义数组最小数据类型的函数
    def get_minimum_int_dtype(values):
        """
        Uses range checking to determine the minimum integer data type required
        to represent values.

        :param values: numpy array
        :return: named data type that can be later used to create a numpy dtype
        """

        min_value = values.min()
        max_value = values.max()

        if min_value >= 0:
            if max_value <= 255:
                return rasterio.uint8
            elif max_value <= 65535:
                return rasterio.uint16
            elif max_value <= 4294967295:
                return rasterio.uint32
        elif min_value >= -32768 and max_value <= 32767:
            return rasterio.int16
        elif min_value >= -2147483648 and max_value <= 2147483647:
            return rasterio.int32
    
    search_criteria = "*.tif" #搜寻所要合并的栅格.tif文件
    fp_pattern=os.path.join(dir_path, search_criteria)
    fps=glob.glob(fp_pattern) #使用glob库搜索指定模式的文件
    src_files_to_mosaic=[]
    for fp in fps:
        src=rasterio.open(fp)
        src_files_to_mosaic.append(src)    
    mosaic,out_trans=merge(src_files_to_mosaic)  #merge函数返回一个栅格数组，以及转换信息   
    
    #获得元数据
    out_meta=src.meta.copy()
    #更新元数据
    data_type=get_minimum_int_dtype(mosaic)
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     #通过压缩和配置存储类型，减小存储文件大小
                     "compress":'lzw',
                     "dtype":get_minimum_int_dtype(mosaic), 
                      }
                    )       
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(mosaic.astype(data_type))     
    
    return out_trans            

def las_classification_plotWithLegend(las_fp):  
    import rasterio as rio
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib import colors
    from matplotlib.patches import Rectangle
    '''
    function - 显示由.las文件生成的分类栅格文件，并显示图例
    
    Paras:
    las_fp - 分类文件路径
    '''    
    with rio.open(las_fp) as classi_src:
        classi_array=classi_src.read(1)

    las_classi_colorName={0:'black',1:'white',2:'beige',3:'palegreen',4:'lime',5:'green',6:'tomato',7:'silver',8:'grey',9:'lightskyblue',10:'purple',11:'slategray',12:'grey',13:'cadetblue',14:'lightsteelblue',15:'brown',16:'indianred',17:'darkkhaki',18:'azure',9999:'pink'}
    las_classi_colorRGB=pd.DataFrame({key:colors.hex2color(colors.cnames[las_classi_colorName[key]]) for key in las_classi_colorName.keys()})
    classi_array_color=[pd.DataFrame(classi_array).replace(las_classi_colorRGB.iloc[idx]).to_numpy() for idx in las_classi_colorRGB.index]
    classi_array_color_=np.concatenate([np.expand_dims(i,axis=-1) for i in classi_array_color],axis=-1)
    fig, ax=plt.subplots(figsize=(12, 12))
    im=ax.imshow(classi_array_color_, )
    ax.set_title(
        "LAS_classification",
        fontsize=14,
    )

    #增加图例
    color_legend=pd.DataFrame(las_classi_colorName.items(),columns=["id","color"])
    las_classi_name={0:'never classified',1:'unassigned',2:'ground',3:'low vegetation',4:'medium vegetation',5:'high vegetation',6:'building',7:'low point',8:'reserved',9:'water',10:'rail',11:'road surface',12:'reserved',13:'wire-guard(shield)',14:'wire-conductor(phase)',15:'transimission',16:'wire-structure connector(insulator)',17:'bridge deck',18:'high noise',9999:'null'}
    color_legend['label']=las_classi_name.values()
    classi_lengend=[Rectangle((0, 0), 1, 1, color=c) for c in color_legend['color']]

    ax.legend(classi_lengend,color_legend.label,mode='expand',ncol=3)
    plt.tight_layout()
    plt.show()
    
def las_classification_plotWithLegend_(classi_array):  
    import rasterio as rio
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib import colors
    from matplotlib.patches import Rectangle
    '''
    function - 显示由.las文件生成的分类栅格文件，并显示图例
    
    Paras:
    las_fp - 分类文件路径
    '''    
    las_classi_colorName={0:'black',1:'white',2:'beige',3:'palegreen',4:'lime',5:'green',6:'tomato',7:'silver',8:'grey',9:'lightskyblue',10:'purple',11:'slategray',12:'grey',13:'cadetblue',14:'lightsteelblue',15:'brown',16:'indianred',17:'darkkhaki',18:'azure',9999:'white'}
    las_classi_colorRGB=pd.DataFrame({key:colors.hex2color(colors.cnames[las_classi_colorName[key]]) for key in las_classi_colorName.keys()})
    classi_array_color=[pd.DataFrame(classi_array).replace(las_classi_colorRGB.iloc[idx]).to_numpy() for idx in las_classi_colorRGB.index]
    classi_array_color_=np.concatenate([np.expand_dims(i,axis=-1) for i in classi_array_color],axis=-1)
    fig, ax=plt.subplots(figsize=(12, 12))
    im=ax.imshow(classi_array_color_, )
    ax.set_title(
        "LAS_classification",
        fontsize=14,
    )

    #增加图例
    color_legend=pd.DataFrame(las_classi_colorName.items(),columns=["id","color"])
    las_classi_name={0:'never classified',1:'unassigned',2:'ground',3:'low vegetation',4:'medium vegetation',5:'high vegetation',6:'building',7:'low point',8:'reserved',9:'water',10:'rail',11:'road surface',12:'reserved',13:'wire-guard(shield)',14:'wire-conductor(phase)',15:'transimission',16:'wire-structure connector(insulator)',17:'bridge deck',18:'high noise',9999:'null'}
    color_legend['label']=las_classi_name.values()
    classi_lengend=[Rectangle((0, 0), 1, 1, color=c) for c in color_legend['color']]

    ax.legend(classi_lengend,color_legend.label,mode='expand',ncol=3)
    plt.tight_layout()
    plt.show()
    
def raster_reprojection(raster_fp,dst_crs,save_path):
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    import rasterio as rio
    '''
    function - 转换栅格投影
    
    Paras:
    raster_fp - 待转换投影的栅格
    dst_crs - 目标投影
    save_path - 保存路径
    '''
    with rio.open(raster_fp) as src:
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rio.open(save_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)      
    print("finished reprojecting...")    
    
def get_crs_raster(raster_fp):
    import rasterio as rio
    '''
    function - 获取给定栅格的投影坐标-crs.
    
    Paras:
    raster)fp - 给定栅格文件的路径
    '''
    with rio.open(raster_fp) as raster_crs:
        raster_profile=raster_crs.profile
        return raster_profile['crs']    
    
def animated_gif_show(gif_fp,figsize=(8,8)):
    from PIL import Image, ImageSequence
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML
    '''
    function - 读入.gif，并动态显示
    
    Paras:
    gif_fp - GIF文件路径
    figsize - 图表大小
    '''
    gif=Image.open(gif_fp,'r')
    frames=[np.array(frame.getdata(),dtype=np.uint8).reshape(gif.size[0],gif.size[1]) for frame in ImageSequence.Iterator(gif)] #dtype=np.uint8

    fig=plt.figure(figsize=figsize)
    imgs=[(plt.imshow(img,animated=True),) for img in frames]

    anim=animation.ArtistAnimation(fig, imgs, interval=300,repeat_delay=3000, blit=True)
    return HTML(anim.to_html5_video())

def demo_con_style(a_coordi,b_coordi,ax,connectionstyle):
    '''
    function - 在matplotlib的子图中绘制连接线
    reference - matplotlib官网Connectionstyle Demo

    Paras:
    a_coordi - a点的x，y坐标
    b_coordi - b点的x，y坐标
    ax - 子图
    connectionstyle - 连接线的形式
    '''
    x1, y1=a_coordi[0],a_coordi[1]
    x2, y2=b_coordi[0],b_coordi[1]

    ax.plot([x1, x2], [y1, y2], ".")
    ax.annotate("",
                xy=(x1, y1), xycoords='data',
                xytext=(x2, y2), textcoords='data',
                arrowprops=dict(arrowstyle="->", color="0.5",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle=connectionstyle,
                                ),
                )

    ax.text(.05, .95, connectionstyle.replace(",", ",\n"),
            transform=ax.transAxes, ha="left", va="top")
