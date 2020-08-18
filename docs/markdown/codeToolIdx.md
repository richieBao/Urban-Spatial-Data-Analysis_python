# 代码工具索引
> 本书的目的不仅在于一般意义上的教材，以及专业上相关实验的阐述，同时，使大量实验所编写代码片段形成一个个可以方便迁移使用的工具，形成工具包。

## 1. 单个分类POI数据爬取与地理空间点地图
1. function-百度地图开放平台POI数据爬取，baiduPOI_dataCrawler(query_dic,bound_coordinate,partition,page_num_range,poi_fn_list=False);
2. function-转换.csv格式的POI数据为pandas的DataFrame,csv2df(poi_fn_csv)；

## 2. 多个分类POI数据爬取与描述性统计
3. function-百度地图开放平台POI数据批量爬取，baiduPOI_batchCrawler(poi_config_para)。需要调用单个分类POI爬取函数baiduPOI_dataCrawler(query_dic,4.bound_coordinate,partition,page_num_range,poi_fn_list=False)'
4. funciton-以所在文件夹路径为键，值为包含该文件夹下所有文件名的列表。文件类型可以自行定义 filePath_extraction(dirpath,fileType)
5. funciton-.csv格式POI数据批量转换为GeoDataFrame，poi_csv2GeoDF_batch(poi_paths,fields_extraction,save_path)。需要调用转换.csv格式的POI数据为pandas的DataFrame函数csv2df(poi_fn_csv)
6. funciton-使用plotly以表格形式显示DataFrame格式数据,ployly_table(df,column_extraction)
7. function-频数分布计算，frequency_bins(df,bins)

## 3. 正态分布与概率密度函数，异常值处理
8. funciton-数据集z-score概率密度函数分布曲线(即观察值/实验值 observed/empirical data)与标准正态分布(即理论值 theoretical set)比较，comparisonOFdistribution(df,field,bins=100)
9. function-判断异常值，is_outlier(data,threshold=3.5)

## 4. OpenStreetMap（OSM）数据处理
10. function-转换shape的polygon为osmium的polygon数据格式（.txt），用于.osm地图数据的裁切，shpPolygon2OsmosisTxt(shape_polygon_fp,osmosis_txt_fp)
11. class-通过继承osmium类 class osmium.SimpleHandler读取.osm数据, osmHandler(osm.SimpleHandler)
12. function-根据条件逐个保存读取的osm数据（node, way and area）,save_osm(osm_handler,osm_type,save_path=r"./data/",fileType="GPKG")
13. function-计算当前时间，start_time()
14. function-计算持续时间, duration(start_time)

## 5.核密度估计与地理空间点密度分布
15. function - 将.shp格式的点数据转换为.tif栅格(raster)，pts2raster(pts_shp,raster_path,cellSize,field_name=False)
16. function - 将GeoDaraFrame格式的点数据转换为栅格数据， pts_geoDF2raster(pts_geoDF,raster_path,cellSize,scale)

## 6.标准误，中心极限定理，t分布，统计显著性，效应量，置信区间；公共健康数据的地理空间分布与相关性分析
17. function - 在Jupyter中打印DataFrame格式数据为HTML, print_html(df,row_numbers=5
18. function - 生成颜色列表或者字典， generate_colors()

## 7.简单回归，多元回归
19. function - 在matplotlib的子图中绘制连接线，demo_con_style(a_coordi,b_coordi,ax,connectionstyle)
20. function - 回归方程的判定系数， coefficient_of_determination(observed_vals,predicted_vals)
21. function - 简单线性回归方程-回归显著性检验（回归系数检验）， ANOVA(observed_vals,predicted_vals,df_reg,df_res)
22. function - 简单线性回归置信区间估计，以及预测区间， confidenceInterval_estimator_LR(x,sample_num,X,y,model,confidence=0.05)
23. function - DataFrame数据格式，成组计算pearsonr相关系数，correlationAnalysis_multivarialbe(df)
24. function - 回归方程的修正自由度的判定系数， coefficient_of_determination_correction(observed_vals,predicted_vals,independent_variable_n)
25. function - 多元线性回归方程-回归显著性检验（回归系数检验），全部回归系数的总体检验，以及单个回归系数的检验， ANOVA_multivarialbe(observed_vals,predicted_vals,independent_variable_n,a_i,X)
26. function - 多元线性回归置信区间估计，以及预测区间， confidenceInterval_estimator_LR_multivariable(X,y,model,confidence=0.05)

## 8. 回归公共健康数据，与梯度下降法
27. function - 返回指定邻近数目的最近点坐标，k_neighbors_entire(xy,k=3)
28. function - 多项式回归degree次数选择，及正则化, PolynomialFeatures_regularization(X,y,regularization='linear')
29. 梯度下降法 - 定义模型，定义损失函数，定义梯度下降函数，定义训练模型函数

## 9. 线性代数基础的代码表述
30. funciton - 转换SymPy的vector及Matrix数据格式为matplotlib可以打印的数据格式，vector_plot_3d(ax_3d,C,origin_vector,vector,color='r',label='vector',arrow_length_ratio=0.1)
31. function - 给定向量，及对应系数，延向量绘制，move_alongVectors(vector_list,coeffi_list,C,ax,)
32. function - 将向量集合转换为向量矩阵，并计算简化的行阶梯形矩阵，vector2matrix_rref(v_list,C)

## 10. Landsat遥感影像处理，数字高程，主成成分分析
33. function - 按照文件名中的数字排序文件列表， fp_sort(fp_list,str_pattern,prefix="")
34. function - 读取landsat *_MTL.txt文件，提取需要的信息，LandsatMTL_info(fp)
35. funciton - 转换SymPy的vector及Matrix数据格式为matplotlib可以打印的数据格式， vector_plot_2d(ax_2d,C,origin_vector,vector,color='r',label='vector',width=0.022)
36. function - 给定圆心，半径，划分份数，计算所有直径的首尾坐标， circle_lines(center,radius,division)
37. function - 计算二维点到直线上的投影，point_Proj2Line(line_endpts,point)
38. function - 计算NDVI指数， NDVI(RED_band,NIR_band)

## 11.遥感影像解译（基于NDVI），建立采样工具（GUI_tkinter），混淆矩阵
39. function - 给定裁切边界，批量裁切栅格数据，raster_clip(raster_fp,clip_boundary_fp,save_path)
40. function - 指定波段，同时显示多个遥感影像，w_180310_array(img_stack_list,band_num)
41. function - 将变量名转换为字符串， variable_name(var)
42. function - 拉伸图像 contract stretching，image_exposure(img_bands,percentile=(2,98))
43. function - 将数据按照给定的百分数划分，并给定固定的值,整数值或RGB色彩值，data_division(data,division,right=True)
44. function - 多个栅格数据，给定百分比，变化观察，percentile_slider(season_dic)
45. 基于tkinter，开发交互式GUI采样工具

## 12. 点云数据（激光雷达）处理——分类数据，DSM，建筑高度提取，插值
46. function - 转换单个.las点云数据为分类栅格数据，和DSM栅格数据等
47. function - 显示由.las文件生成的分类栅格文件，并显示图例
48. function - 批量转换.las点云数据为DSM和分类栅格
49. function - 合并多个栅格为一个
50. function - 迁移rasterio提供的定义数组最小数据类型的函数
51. function - 获取给定栅格的投影坐标-crs
52. function - 转换栅格投影
53. function - 根据给定栅格的范围，裁切.shp格式数据，并定义投影同给定栅格
54. function - 使用rasterio.fill的插值方法，补全缺失的数据
55. function - 使用earthpy库显示遥感影像（一个波段）