> Created on  Jan 14 2019 @author: Richie Bao-caDesign设计(cadesign.cn)__+updated on Sun Dec 20 09:41:18 2020

## 1. 视觉词袋（BOW），决策树（Decision trees）->随机森林（Random forests），交叉验证 cross_val_score，视觉感知-图像分类_识别器，网络实验平台服务器端部署
### 1.1 视觉词袋与构建图像映射特征
计算每一张图像的关键点描述子（一个关键点描述子有128维，一幅图像有多个关键点），把所有图像的关键点描述子集合在一起，就是所有图像特征的集合。因为每一关键点描述子均不同，因此使用聚类的方法指定聚类的数量（32）聚合特征，即聚合所有图像的关键描述子为指定数量构造码本/BOW,Bag of Words（码本是将所有可用的码/聚类放在一起，组成类似字典的表，用序号给不同的码编号/或者是列表的排序。如果要对一个单词/或一句话编码，则可以应用码本编号），并建立了聚类模型。该码本就是所有图像特征（聚类后）的集合，因为每一图像的关键点描述子都可以在该码本中找到对应的特征（32个聚类的1个或多个聚类组合），因此可以用码本编码每一图像（用聚类模型预测）。保持码本的形状（shape，32），将一幅图像的所有关键点描述子对应到码本的编号上，因为聚类的结果，一个编号通常对应有多个关键点，计算所有对应到编号上关键点的频数（有的编码频数也会为0，即没有对应的关键点），这个对应码本编号频数的结果就反映了该图像特征。

<a href=""><img src="./imgs/19_01.jpg" height='auto' width='1200' title="caDesign"></a>

读取'视觉感知-基于图像的空间分类:问卷调研'的数据库结果。因为数据库中存储的图像路径为相对路径，而此时的代码可能位于他处，因此将其转换为绝对路径。


```python
import sqlite3
import pandas as pd
db_fp=r'C:\Users\richi\omen-richiebao_s\omen_github\caDesign_ExperimentPlatform\data-dev.sqlite'
vp_classification=pd.read_sql_table('vp_classification', 'sqlite:///%s'%db_fp) #pd.read_sql_table从数据库中读取指定的表

import util
import os
vp_classification['absolute_imags_fp']=vp_classification.apply(lambda row:os.path.join(r'C:\Users\richi\omen-richiebao_s\omen_github\caDesign_ExperimentPlatform\app\static',row.imgs_fp),axis=1)
util.print_html(vp_classification)
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>imgs_fp</th>
      <th>c_1</th>
      <th>c_2</th>
      <th>c_3</th>
      <th>c_4</th>
      <th>c_5</th>
      <th>c_6</th>
      <th>c_7</th>
      <th>c_8</th>
      <th>classification</th>
      <th>timestamp</th>
      <th>index</th>
      <th>absolute_imags_fp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>KITTI/2011_09_26_drive_0009_sync/0000000000.png</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>宽木</td>
      <td>2020-12-20 13:10:36.725045</td>
      <td>1</td>
      <td>C:\Users\richi\omen-richiebao_s\omen_github\caDesign_ExperimentPlatform\app\static\KITTI/2011_09_26_drive_0009_sync/0000000000.png</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>KITTI/2011_09_26_drive_0009_sync/0000000010.png</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>宽木</td>
      <td>2020-12-20 13:10:42.138661</td>
      <td>2</td>
      <td>C:\Users\richi\omen-richiebao_s\omen_github\caDesign_ExperimentPlatform\app\static\KITTI/2011_09_26_drive_0009_sync/0000000010.png</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>KITTI/2011_09_26_drive_0009_sync/0000000020.png</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>宽木</td>
      <td>2020-12-20 13:10:46.728137</td>
      <td>3</td>
      <td>C:\Users\richi\omen-richiebao_s\omen_github\caDesign_ExperimentPlatform\app\static\KITTI/2011_09_26_drive_0009_sync/0000000020.png</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>KITTI/2011_09_26_drive_0009_sync/0000000030.png</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>宽木</td>
      <td>2020-12-20 13:10:55.079670</td>
      <td>4</td>
      <td>C:\Users\richi\omen-richiebao_s\omen_github\caDesign_ExperimentPlatform\app\static\KITTI/2011_09_26_drive_0009_sync/0000000030.png</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>KITTI/2011_09_26_drive_0009_sync/0000000040.png</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>宽木</td>
      <td>2020-12-20 13:10:58.771853</td>
      <td>5</td>
      <td>C:\Users\richi\omen-richiebao_s\omen_github\caDesign_ExperimentPlatform\app\static\KITTI/2011_09_26_drive_0009_sync/0000000040.png</td>
    </tr>
  </tbody>
</table>




```python
print("分类频数统计：\n",vp_classification.classification.value_counts())
```

    分类频数统计：
     阔木    189
    窄建    188
    宽木    104
    开阔     85
    窄木     78
    宽建     60
    阔建     47
    林荫     47
    Name: classification, dtype: int64
    

以一张图像的分类结果为键，以图像绝对路径为名建立对应到一张图像的字典。然后将所有图像的字典放置于一个列表中，用于构造所有图像的码本和计算每一图像的特征映射。


```python
def load_training_data(imgs_df,classi_field,classi_list,path_field):
    import pandas as pd
    '''
    function - 按照分类提取图像路径与规律
    
    Paras:
    imgs_df - 由pandas读取的数据库表数据，含分类信息
    '''
    imgs_group=imgs_df.groupby(['classification'])
    training_data=[[{'object_class':classi,'image_path':row} for row in imgs_group.get_group(classi)[path_field].tolist()] for classi in classi_list]
    flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]
    return flatten_lst(training_data)

classi_list=['林荫','窄建','窄木','宽建','宽木','阔建','阔木','开阔']   
classi_field='classification'    
path_field='absolute_imags_fp'
training_data=load_training_data(vp_classification,classi_field,classi_list,path_field)    
print(training_data[:5])
```

    [{'object_class': '林荫', 'image_path': 'C:\\Users\\richi\\omen-richiebao_s\\omen_github\\caDesign_ExperimentPlatform\\app\\static\\KITTI/2011_09_26_drive_0117_sync/0000000657.png'}, {'object_class': '林荫', 'image_path': 'C:\\Users\\richi\\omen-richiebao_s\\omen_github\\caDesign_ExperimentPlatform\\app\\static\\KITTI/2011_09_26_drive_0117_sync/0000000036.png'}, {'object_class': '林荫', 'image_path': 'C:\\Users\\richi\\omen-richiebao_s\\omen_github\\caDesign_ExperimentPlatform\\app\\static\\KITTI/2011_09_26_drive_0117_sync/0000000354.png'}, {'object_class': '林荫', 'image_path': 'C:\\Users\\richi\\omen-richiebao_s\\omen_github\\caDesign_ExperimentPlatform\\app\\static\\KITTI/2011_09_26_drive_0117_sync/0000000038.png'}, {'object_class': '林荫', 'image_path': 'C:\\Users\\richi\\omen-richiebao_s\\omen_github\\caDesign_ExperimentPlatform\\app\\static\\KITTI/2011_09_26_drive_0117_sync/0000000196.png'}]
    

如果代码量比较大，逐行的分析代码之间互相调用的关系是费时和费力的事情，因此借助逆向工程。代码逆向工程的工具很多，这里使用了[Sourcetrail](https://www.sourcetrail.com/#intro)工具。定义`feature_builder_BOW`构造视觉码本和提取图像映射特征的类之后，主要调用了两个方法，先用`feature_builder_BOW().get_visual_BOW(training_data,)`构造码本，再用 `feature_builder_BOW().get_feature_map(training_data,kmeans)`返回每一图像映射特征。从下述逆向工程分析结果能够很清楚的梳理出两次类方法的调用所关联调用的其它类中函数。`.get_visual_BOW`函数内调用有`extract_features`和`visual_BOW`两个函数，同时显示了调用的内嵌方法，print,enumerate,tqdm等；`.get_feature_map`则调用有`extract_features`和`normalize`两个函数，同时使用了类变量` self.num_clusters`。同时，也可以查看`extract_features`分布被`.get_visual_BOW`和`.get_feature_map`调用。

<a href=""><img src="./imgs/19_02.png" height='auto' width='1200' title="caDesign"></a>


```python
class feature_builder_BOW:
    '''
    class - 根据所有图像关键点描述子聚类建立图像视觉词袋，获取每一图像的特征（码本）映射的频数统计
    '''   
    def __init__(self,num_cluster=32):
        self.num_clusters=num_cluster

    def extract_features(self,img):
        import cv2 as cv
        '''
        function - 提取图像特征
        
        Paras:
        img - 读取的图像
        '''
        star_detector=cv.xfeatures2d.StarDetector_create()
        key_points=star_detector.detect(img)
        img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        kp,des=cv.xfeatures2d.SIFT_create().compute(img_gray, key_points) #SIFT特征提取器提取特征
        return des
    
    def visual_BOW(self,des_all):
        from sklearn.cluster import KMeans
        '''
        function - 聚类所有图像的特征（描述子/SIFT），建立视觉词袋
        
        des_all - 所有图像的关键点描述子
        '''
        print("start KMean...")
        kmeans=KMeans(self.num_clusters)
        kmeans=kmeans.fit(des_all)
        #centroids=kmeans.cluster_centers_
        print("end KMean...")
        return kmeans         
    
    def get_visual_BOW(self,training_data):
        import cv2 as cv
        from tqdm import tqdm
        '''
        function - 提取图像特征，返回所有图像关键点聚类视觉词袋
        
        Paras:
        training_data - 训练数据集
        '''
        des_all=[]
        #i=0        
        for item in tqdm(training_data):

            
            classi_judge=item['object_class']
            img=cv.imread(item['image_path'])
            des=self.extract_features(img)
            des_all.extend(des)           
            #print(des.shape)

            #if i==10:break
            #i+=1        
        kmeans=self.visual_BOW(des_all)      
        return kmeans
    
    def normalize(self,input_data):
        import numpy as np
        '''
        fuction - 归一化数据
        
        input_data - 待归一化的数组
        '''
        sum_input=np.sum(input_data)
        if sum_input>0:
            return input_data/sum_input #单一数值/总体数值之和，最终数值范围[0,1]
        else:
            return input_data               
    
    def construct_feature(self,img,kmeans):
        import numpy as np
        '''
        function - 使用聚类的视觉词袋构建图像特征（构造码本）
        
        Paras:
        img - 读取的单张图像
        kmeans - 已训练的聚类模型
        '''
        des=self.extract_features(img)
        labels=kmeans.predict(des.astype(np.float)) #对特征执行聚类预测类标
        feature_vector=np.zeros(self.num_clusters)
        for i,item in enumerate(feature_vector): #计算特征聚类出现的频数/直方图
            feature_vector[labels[i]]+=1
        feature_vector_=np.reshape(feature_vector,((1,feature_vector.shape[0])))
        return self.normalize(feature_vector_)
    
    def get_feature_map(self,training_data,kmeans):
        import cv2 as cv
        '''
        function - 返回每个图像的特征映射（码本映射）
        Paras:
        training_data - 训练数据集
        kmeans - 已训练的聚类模型
        '''
        feature_map=[]
        for item in training_data:
            temp_dict={}
            temp_dict['object_class']=item['object_class']
            #print("Extracting feature for",item['image_path'])
            img=cv.imread(item['image_path'])
            temp_dict['feature_vector']=self.construct_feature(img,kmeans)
            if temp_dict['feature_vector'] is not None:
                feature_map.append(temp_dict)
        #print(feature_map[0]['feature_vector'].shape,feature_map[0])
        return feature_map
import util
s_t=util.start_time()

kmeans=feature_builder_BOW().get_visual_BOW(training_data,)       
print("_"*50)

import pickle
with open('./data/visual_BOW.pkl','wb') as f: # 使用with结构避免手动的文件关闭操作
    pickle.dump(kmeans,f) #存储kmeans聚类模型
    
feature_map=feature_builder_BOW().get_feature_map(training_data,kmeans)    
with open('./data/visual_feature.pkl','wb') as f:
    pickle.dump(feature_map,f) #存储图像特征

util.duration(s_t)
```

      0%|          | 2/798 [00:00<00:51, 15.32it/s]

    start time: 2020-12-21 11:18:32.306435
    

    100%|██████████| 798/798 [00:48<00:00, 16.46it/s]
    

    start KMean...
    end KMean...
    __________________________________________________
    end time: 2020-12-21 11:24:29.873511
    Total time spend:5.95 minutes
    

### 1.2 决策树（Decision trees）->随机森林（Random forests）

#### 1.2.1 决策树
*Mastering Machine Learning with scikit-learn* 对决策树和随机森林算法有清晰易于理解的阐述，这里以其为蓝本，并应用其案例数据，辅助python计算和应用Sklearn库加以说明。首先录入猫狗分类的特征数据集。所录入的数据为文本类型，将其通过pandas库提供的`pd.get_dummies`方法转换为独热编码(One-Hot Encoding)，主要应用于数据集的特征列；同时也应用了sklearn库的`preprocessing.LabelEncoder()`将其转换为整数编码，主要用于数据集的类标列。

决策树（见下文中定义函数`decisionTree_structure`计算的流程图）测试特征的内部节点，用盒子表示，节点之间通过边来连接，边表示了测试的可能输出，（根据阈值）将训练实例分到不同的子集中。子节点应用特征值继续测试训练实例的子集，直到满足一个停止标准。分类任务中，决策树中不再分支的节点为叶节点，表示类别（如果是在回归任务中，一个叶节点包含多个实例，这些实例对应的响应变量值可以通过求均值来估计这个叶节点对应的响应变量）；带有分支的节点通常称为分支节点（或子节点）。在决策树构建完成后，对于一个测试实例进行预测，只需要从根节点顺着对应的边到达某个叶节点。

训练决策树的优化算法，使用Ross Quinlan发明的迭代二叉树3代(Iterative Dichotomiser 3 (ID3)的算法。

> 参考文献
> 1. Cavin Hackeling. Mastering Machine Learning with scikit-learn[M].Packt Publishing Ltd.July 2017.Second published. 中文版为：Cavin Hackeling.张浩然译.scikit-learning 机器学习[M].人民邮电出版社.2019.2


```python
import pandas as pd
catDog_trainingData_df=pd.DataFrame({'plays_Fetch':['Yes','No','No','No','No','No','No','No','No','Yes','Yes','No','Yes','Yes'],
                                  'is_grumpy':['No','Yes','Yes','Yes','No','Yes','Yes','No','Yes','No','No','No','Yes','Yes'],
                                  'fvorite_food':['bacon','dog_food','cat_food','bacon','cat_food','bacon','cat_food','dog_food','cat_food','dog_food','bacon','cat_food','cat_food','bacon'],
                                  'species':['dog','dog','cat','cat','cat','cat','cat','dog','cat','dog','dog','cat','cat','dog']})
print("原始数据：\n",catDog_trainingData_df)
```

    原始数据：
        plays_Fetch is_grumpy fvorite_food species
    0          Yes        No        bacon     dog
    1           No       Yes     dog_food     dog
    2           No       Yes     cat_food     cat
    3           No       Yes        bacon     cat
    4           No        No     cat_food     cat
    5           No       Yes        bacon     cat
    6           No       Yes     cat_food     cat
    7           No        No     dog_food     dog
    8           No       Yes     cat_food     cat
    9          Yes        No     dog_food     dog
    10         Yes        No        bacon     dog
    11          No        No     cat_food     cat
    12         Yes       Yes     cat_food     cat
    13         Yes       Yes        bacon     dog
    

数据集的特征值完成独热编码后，或增加相应的列，对于仅存在有两个分类的列（例如'Yes'和'No'），虽然增加了新的一列，但是与单独一列实质上并没有区别（非1即0），因此在决策树中使用二者中的任何一列都是一样的（例如play_No或play_Yes）。但是分类在3类及其以上者，当用独热编码完成转换后，新增列之间是不能互相替换的，例如food_bacon,food_cat food,food_dog food。


```python
def df_multiColumns_LabelEncoder(df,columns=None):
    from sklearn import preprocessing
    '''
    function - 根据指定的（多个）列，将分类转换为整数表示，区间为[0,分类数-1]
    
    Paras:
    df - DataFrame格式数据
    columns - 指定待转换的列名列表
    '''
    output=df.copy()
    if columns is not None:
        for col in columns:
            output[col]=preprocessing.LabelEncoder().fit_transform(output[col])
    else:
        for column_name, col in output.iteritems():
            output[column_name]=preprocessing.LabelEncoder().fit_transform(col)
            
    return output    
    
    
catDog_trainingData_df_encoder=df_multiColumns_LabelEncoder(df=catDog_trainingData_df,columns=['plays_Fetch','is_grumpy','fvorite_food','species'])    
print("encoder of each column\n",catDog_trainingData_df_encoder)

catDog_trainingData_dummies=pd.get_dummies(catDog_trainingData_df,prefix=['play','grumpy','food','species'])
print("fequency of each column:\n",catDog_trainingData_dummies.apply(pd.Series.value_counts))
print('one-hot(dummies):\n')
import util
util.print_html(catDog_trainingData_dummies,row_numbers=14)
```

    encoder of each column
         plays_Fetch  is_grumpy  fvorite_food  species
    0             1          0             0        1
    1             0          1             2        1
    2             0          1             1        0
    3             0          1             0        0
    4             0          0             1        0
    5             0          1             0        0
    6             0          1             1        0
    7             0          0             2        1
    8             0          1             1        0
    9             1          0             2        1
    10            1          0             0        1
    11            0          0             1        0
    12            1          1             1        0
    13            1          1             0        1
    fequency of each column:
        play_No  play_Yes  grumpy_No  grumpy_Yes  food_bacon  food_cat_food  \
    0        5         9          8           6           9              8   
    1        9         5          6           8           5              6   
    
       food_dog_food  species_cat  species_dog  
    0             11            6            8  
    1              3            8            6  
    one-hot(dummies):
    
    




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>play_No</th>
      <th>play_Yes</th>
      <th>grumpy_No</th>
      <th>grumpy_Yes</th>
      <th>food_bacon</th>
      <th>food_cat_food</th>
      <th>food_dog_food</th>
      <th>species_cat</th>
      <th>species_dog</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>



决策树通过检测一个特征序列的值来估计响应变量的值。即能产出只包含猫和只包含狗的子集的检测，要优于一个产出中同时包含猫狗的检测。因为一个子集中的成员同时包含不同的类，无法确定实例的分类。对于这个检测可以使用熵（可以参考‘连续距离聚类与业态分布结构计算-对于信息熵和均衡度分析部分’）的衡量方式量化不确定性的程度（单位为比特，bits）。其公式为：$H(X)=- \sum_{i=1}^n P( x_{i})  log_{b} P( x_{i})$，其中$P( x_{i})$ 是输出$i$的概率，$b$常见值为2,e和10。由于一个小于1的数值的对数为负数，求和为负数，因此取反。为了方便计算一个对象的熵来查看计算的流程，定义`entropy_compomnent`函数实现。

找出对分类动物最有帮助的特征，即找出能把熵降到最低的特征（熵值越大，类别分布越均匀；熵值越小，类别分布越集中，即分布不均匀）。下述代码的过程是按照下图的执行指向过程计算，在层-A根节点中，根据类标计算猫（8只）狗（6只）熵值为0.985。先利用'food_cat food'特征列，左子节点不吃猫食（0）对应的狗分类为6只，猫分类为2只，信息熵为0.811；右子节点吃猫食（1）对应的狗分类为0只，猫分类为6只，信息熵为0.0；因为左子节点信息熵0.811>0.5，因此需要对左子节点继续检测。选择'grumpy_Yes'特征列对应左子节点，即'food_cat food'中值为0（不吃猫食）的行（实例），包括6只狗和2只猫。这8个实例可以根据'grumpy_Yes'特征列，即是否脾气暴躁（grumpy）划分为两类，其中脾气暴躁（1）的为4个实例（右子节点），而脾气不暴燥（0）也为4个实例（左子节点）。将其分布对应到类标列，可以得知脾气暴躁的4个实例中，为猫分类的有2个，为狗分类的亦有2个，信息熵为1；而脾气不暴躁的4个实例中，为猫分类的为0，为狗的分类为4，信息熵为0。就是说明，如果不吃猫食，而脾气不暴躁的实例分类为狗。

上述的决策树，根据两个特征列，'food_cat food'和'grumpy_Yes'判断物种分类（类标），因为层-C存在熵值为1的右子节点，需要继续删选特征，继续检测。例如想判断第12行实例（样本）的类标，已知'food_cat food'的值为1（吃猫食），'grumpy_Yes'的值为1（脾气暴躁），通过层-B可以判断出，该实例会被分配到右子节点(吃猫食)，该层右子节点信息熵为0，因此可以判断，该实例为猫，与实际相符。再例如，对于第10行实例，已知'food_cat food'的值为0（不吃猫食），'grumpy_Yes'的值为0（脾气不暴躁），通过层-B可以判断出，该实例会被分配到左子节点(不吃猫食)，因为该节点信息熵大于0.5，因此继续用层-C检测，由脾气不暴躁，将其指向左子节点，该节点信息熵为0，因此可以判定该实例为狗，与实际相符。

<a href=""><img src="./imgs/19_03.png" height='auto' width='1000' title="caDesign"></a>

* 信息增益(Information gain)

上述的计算直接选择了两个特征列，实际上是需要判断选择哪些特征列用于检测，以减少分类的不确定性。在层-B中，产生了两个子集（子节点），熵分别为0.811和0.0，其平均熵为$(0.811+0.0)/2=0.4055$，而根节点的熵为0.985，最大不确定性的熵为1。衡量熵的减少可以使用信息增益的指标，其公式为：$IG(T,a)=H(T)- \sum_{v \in vals(a)}^{}  \frac{ | \{x \in T |  x_{a}= \upsilon  \} | }{ |  T| }H(\{x \in T | x_{a}= \upsilon \})  $，其中$X_{a}  \in vals(a)$表示实例$x$对应的特征$a$的值，$x \in T | x_{a}= \upsilon$表示特征$a$的值等于$\upsilon$的实例数量，$H(\{x \in T | x_{a}= \upsilon \})$是特征$a$的值等于$\upsilon$的实例的子集的熵。

自定义信息增益函数`IG`，计算结果中food_cat_food特征列的值0.463587为所有特征列里最小信息熵，因此用该特征列检测。

* 基尼不纯度(Gini impurity)

除了通过创建能产生最大信息增益的节点来创建一个决策树，还可以用启发性算法基尼不纯度(Gini impurity)衡量一个集合中类的比例，其公式为：$Gini(t)=1- \sum_{i=1}^j  P(i | t)^{2} $，其中$j$是类的数量，$t$是节点对应的实例子集，$ P(i | t$是从节点的子集中选择一个属于类$i$元素的概率。当集合中所有元素都属于同一类时，选择任一元素属于这个类的概率均为1，因此Gini值为0。和熵一样，当每个被选择的类概率都相等时Gini达到最大值，其最大值依赖可能类的数量，公式为：$Gini_{max}=1- \frac{1}{n}  $。如果分类问题包括两个类，Gini的最大值等于1/2。

Sklearn库中DecisionTreeClassifier算法在参数criterion中，给出了上述两种方法`{“gini”, “entropy”}`，可以自行配置。


```python
def entropy_compomnent(numerator,denominator):
    import math
    '''
    function - 计算信息熵分量
    '''
    if numerator!=0:
        return -numerator/denominator*math.log2(numerator/denominator)    
    elif numerator==0:
        return 0

print('层-A 根节点- 类标species(8,6):entropy={entropy}'.format(entropy=(entropy_compomnent(6,14)+entropy_compomnent(8,14))))
print('层-B - food_cat food(8,6):左子节点8(2,6)-Left_entropy={L_entropy};右子节点6(6,0)-Right_entropy={R_entropy};'.format(L_entropy=(entropy_compomnent(2,8)+entropy_compomnent(6,8)),R_entropy=(entropy_compomnent(0,6)+entropy_compomnent(6,6))))
print('层-C - grumpy_Yes-8(4,4):左子节点4(0,4)-Left_entropy={L_entropy};右子节点4(2,2)-Right_entropy={R_entropy};'.format(L_entropy=(entropy_compomnent(0,4)+entropy_compomnent(4,4)),R_entropy=(entropy_compomnent(2,4)+entropy_compomnent(2,4))))
```

    层-A 根节点- 类标species(8,6):entropy=0.9852281360342516
    层-B - food_cat food(8,6):左子节点8(2,6)-Left_entropy=0.8112781244591328;右子节点6(6,0)-Right_entropy=0.0;
    层-C - grumpy_Yes-8(4,4):左子节点4(0,4)-Left_entropy=0.0;右子节点4(2,2)-Right_entropy=1.0;
    


```python
def IG(df_dummies):
    import pandas as pd
    import util
    '''
    function - 计算信息增量(IG)
    
    para:
    df_dummies - DataFrame格式，独热编码的特征值
    '''
    weighted_frequency=df_dummies.apply(pd.Series.value_counts)
    #print(weighted_frequency)
    weighted_sum=weighted_frequency.sum(axis=0)
    #print(weighted_sum)
    feature_columns=weighted_frequency.columns.tolist()
    #print(weighted_sum.loc[feature_columns[0]])
    Parent_entropy=entropy_compomnent(weighted_frequency[feature_columns[-1]][0],14)+entropy_compomnent(weighted_frequency[feature_columns[-1]][1],14)
    #print(Parent_entropy)
    
    cal_info=[]
    for feature in feature_columns[:-2]:        
        v_0_frequency=df_dummies.query('%s==0'%feature).iloc[:,-1].value_counts().reindex(df_dummies[feature].unique(),fill_value=0) #频数可能为0，如果为0则会被舍弃（value_counts），因此需要补回（.reindex）
        v_1_frequency=df_dummies.query('%s==1'%feature).iloc[:,-1].value_counts().reindex(df_dummies[feature].unique(),fill_value=0)
        first_child_entropy=entropy_compomnent(v_0_frequency[0], v_0_frequency.sum(axis=0))+entropy_compomnent(v_0_frequency[1], v_0_frequency.sum(axis=0)) 
        second_child_entropy=entropy_compomnent(v_1_frequency[0], v_1_frequency.sum(axis=0))+entropy_compomnent(v_1_frequency[1], v_1_frequency.sum(axis=0))

        cal_dic={'test':feature,
                 'Parent_entropu':Parent_entropy,
                 'first_child_entropy':first_child_entropy,
                 'second_child_entropy':second_child_entropy,
                 'Weighted_average_expression':'%f*%d/%d+%f*%d/%d'%(first_child_entropy,weighted_frequency[feature][0],weighted_sum.loc[feature],second_child_entropy,weighted_frequency[feature][1],weighted_sum.loc[feature]),
                 'IG':first_child_entropy*(weighted_frequency[feature][0]/weighted_sum.loc[feature])+second_child_entropy*(weighted_frequency[feature][1]/weighted_sum.loc[feature])
                 
                }
        cal_info.append(cal_dic)
    cal_info_df=pd.DataFrame.from_dict(cal_info)    
    return cal_info_df

cal_info_df=IG(df_dummies=catDog_trainingData_dummies,)
util.print_html(cal_info_df,row_numbers=7)
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test</th>
      <th>Parent_entropu</th>
      <th>first_child_entropy</th>
      <th>second_child_entropy</th>
      <th>Weighted_average_expression</th>
      <th>IG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>play_No</td>
      <td>0.985228</td>
      <td>0.721928</td>
      <td>0.764205</td>
      <td>0.721928*5/14+0.764205*9/14</td>
      <td>0.749106</td>
    </tr>
    <tr>
      <th>1</th>
      <td>play_Yes</td>
      <td>0.985228</td>
      <td>0.764205</td>
      <td>0.721928</td>
      <td>0.764205*9/14+0.721928*5/14</td>
      <td>0.749106</td>
    </tr>
    <tr>
      <th>2</th>
      <td>grumpy_No</td>
      <td>0.985228</td>
      <td>0.811278</td>
      <td>0.918296</td>
      <td>0.811278*8/14+0.918296*6/14</td>
      <td>0.857143</td>
    </tr>
    <tr>
      <th>3</th>
      <td>grumpy_Yes</td>
      <td>0.985228</td>
      <td>0.918296</td>
      <td>0.811278</td>
      <td>0.918296*6/14+0.811278*8/14</td>
      <td>0.857143</td>
    </tr>
    <tr>
      <th>4</th>
      <td>food_bacon</td>
      <td>0.985228</td>
      <td>0.918296</td>
      <td>0.970951</td>
      <td>0.918296*9/14+0.970951*5/14</td>
      <td>0.937101</td>
    </tr>
    <tr>
      <th>5</th>
      <td>food_cat_food</td>
      <td>0.985228</td>
      <td>0.811278</td>
      <td>0.000000</td>
      <td>0.811278*8/14+0.000000*6/14</td>
      <td>0.463587</td>
    </tr>
    <tr>
      <th>6</th>
      <td>food_dog_food</td>
      <td>0.985228</td>
      <td>0.845351</td>
      <td>0.000000</td>
      <td>0.845351*11/14+0.000000*3/14</td>
      <td>0.664204</td>
    </tr>
  </tbody>
</table>



* 使用sklearn库的DecisionTreeClassifier实现决策树以及打印决策树流程图表。

* [交叉验证 cross_val_score](cross_val_score)

训练和测试数据集如果相同，即在相同的数据上训练和测试，模型（估计器，estimator）只会重复它刚刚看到的样本标签，可能会获得完美的分数，但无法正确预测其它的数据，这种情况称之为过拟合。因此在训练机器学习模型时，通常的做法是将数据集分为训练和测试数据集。但是手动配置超参（数）时，因为参数可以调整以使估计器达到最优，使得存在测试集过拟合的风险。这样，关于测试集的“知识”学习就会“泄露”到模型中，评估指标就不再报告泛化性能。为了解决这个问题，数据集被切分为训练数据集、验证数据集和测试数据集。然而，将数据集划分为三个集合，大大减少了学习模型的样本数量，以及训练和验证数据集组合随机选择的机会。解决这一问题的方案可以应用交叉验证（cross val score,CV）的方法。

在CV方法中，测试数据集仍然用于最终的模型评估，但是不再需要验证数据集。被称为k-fold（k-倍/重/折）的CV中，数据集被分成k个更小的集合，对于每k个folds，k-1个folds用于训练集，1个fold用于测试集。得到的模型在剩余的数据上进行验证(也就是说，它被用作一个测试集来计算性能度量，比如精度)。如下图所示(颜色表示数据集类型)。

<a href=""><img src="./imgs/19_04.png" height='auto' width='500' title="caDesign">引自：[Sklearn官网]( https://scikit-learn.org/stable/modules/cross_validation.html)</a>


```python
X=catDog_trainingData_dummies[catDog_trainingData_dummies.columns[:-2]].to_numpy()
y=catDog_trainingData_df_encoder[catDog_trainingData_df_encoder.columns[-1]].to_numpy()
```


```python
def decisionTree_structure(X,y,criterion='entropy',cv=None,figsize=(6, 6)):
    import numpy as np
    from matplotlib import pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    from sklearn.model_selection import cross_val_score
    '''
    function - 使用决策树分类，并打印决策树流程图表。迁移于Sklearn的'Understanding the decision tree structure', https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    
    Paras:
    X - 数据集-特征值（解释变量）
    y- 数据集-类标/标签(响应变量)
    criterion - DecisionTreeClassifier 参数，衡量拆分的质量，即衡量哪一项检测最能减少分类的不确定性
    cv - cross_val_score参数，确定交叉验证分割策略，默认值为None，即5-fole(折)的交叉验证
    
    '''
    #X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=0)
    X_train,y_train=X,y
    clf=DecisionTreeClassifier(criterion=criterion,max_leaf_nodes=3, random_state=0)
    clf.fit(X_train, y_train)        
    
    n_nodes=clf.tree_.node_count
    children_left=clf.tree_.children_left
    children_right=clf.tree_.children_right
    feature=clf.tree_.feature
    threshold=clf.tree_.threshold    
    print("n_nodes:{n_nodes},\nchildren_left:{children_left},\nchildren_right={children_right},\nthreshold={threshold}".format(n_nodes=n_nodes,children_left=children_left,children_right=children_right,threshold=threshold))
    print("_"*50)
    
    node_depth=np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves=np.zeros(shape=n_nodes, dtype=bool)
    stack=[(0, 0)]  # start with the root node id (0) and its depth (0)    

    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has {n} nodes and has "
          "the following tree structure:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}node={node} is a leaf node.".format(
                space=node_depth[i] * "\t", node=i))
        else:
            print("{space}node={node} is a split node: "
                  "go to node {left} if X[:, {feature}] <= {threshold} "
                  "else to node {right}.".format(
                      space=node_depth[i] * "\t",
                      node=i,
                      left=children_left[i],
                      feature=feature[i],
                      threshold=threshold[i],
                      right=children_right[i]))   
            
    plt.figure(figsize=figsize)      
    tree.plot_tree(clf)
    plt.show()
    
    CV_scores=cross_val_score(clf,X,y, cv=cv)
    print('cross_val_score:\n',CV_scores) #交叉验证每次运行的估计器得分数组   
    print("%0.2f accuracy with a standard deviation of %0.2f" % (CV_scores.mean(), CV_scores.std())) #同时给出了平均得分，和标准差
    return clf
    
clf=decisionTree_structure(X,y)       
```

    n_nodes:5,
    children_left:[ 1  3 -1 -1 -1],
    children_right=[ 2  4 -1 -1 -1],
    threshold=[ 0.5  0.5 -2.  -2.  -2. ]
    __________________________________________________
    The binary tree structure has 5 nodes and has the following tree structure:
    
    node=0 is a split node: go to node 1 if X[:, 5] <= 0.5 else to node 2.
    	node=1 is a split node: go to node 3 if X[:, 3] <= 0.5 else to node 4.
    	node=2 is a leaf node.
    		node=3 is a leaf node.
    		node=4 is a leaf node.
    


    
<a href=""><img src="./imgs/19_06.png" height='auto' width='500' title="caDesign"></a>
    


    cross_val_score:
     [0.66666667 0.66666667 0.66666667 1.         0.5       ]
    0.70 accuracy with a standard deviation of 0.16
    

#### 1.2.2 随机森林（Random forests）
决策树属于勤奋学习模型(eager learners)，与之相反的是KNN算法这样的惰性学习模型（lazy learners）。决策树学习算法会产生出完美拟合每一个训练实例的巨型复杂的决策树模型，而无法对真实的关系进行泛化，即容易过拟合。解决决策树过拟合的方法可以通过剪枝的方法移除决策树中过深的节点和叶子，或者从训练数据和特征的子集中创建多棵决策树构成多个模型的集成（集成是指多个估计器的组合），被称为随机森林的决策树集合。而创建集成的方法有套袋法(bagging)，推进法(boosting)，和堆叠法(stacking)。

* classification_report

`sklearn.metrics.classification_report`用于显示分类指标的文本报告，在报告中显示有每个类的精确度(precision)，召回率(recall)，F1分数(f1-score)等信息。精确度和召回率可以从Wikipedia的[Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)中获取详细的解释，给出的阐释图表可以一目了然的理解计算方法。实心小圆代表类标狗总计12只，空心小圆代表类标猫总计10只。通过假设的模型预测，得到正确预测为狗的有5只，正确预测为猫的为3只，其它的均为错误预测。因此对于分类狗而言，$precision= \frac{5}{8} $，$recall= \frac{5}{12} $。f1-score是精确度和召回率的调和平均值，$\frac{2}{ F_{1} } = \frac{1}{P}+ \frac{1}{R}    \longrightarrow F_{1} =2 \frac{P \times R}{P+R} $。

support字段为每个标签出现的次数。avg行为均值和加权均值（参数sample_weight 配置权重值，默认为None），avg行对应的support为标签总和。

从计算结果可知，使用随机森林算法的各项指标均高于决策树算法的结果。

<a href=""><img src="./imgs/19_05.png" height='auto' width='400' title="caDesign">引自：[Wikipedia]( https://en.wikipedia.org/wiki/Precision_and_recall)</a>


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X,y=make_classification(n_samples=1000,n_features=100,n_informative=20,n_clusters_per_class=2,random_state=11)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=11)
clf_DTC=DecisionTreeClassifier(random_state=11)
clf_DTC.fit(X_train,y_train)
predictions_DTC=clf_DTC.predict(X_test)
print(classification_report(y_test,predictions_DTC))
```

                  precision    recall  f1-score   support
    
               0       0.73      0.66      0.69       127
               1       0.68      0.75      0.71       123
    
        accuracy                           0.70       250
       macro avg       0.71      0.70      0.70       250
    weighted avg       0.71      0.70      0.70       250
    
    


```python
clf_RFC=RandomForestClassifier(n_estimators=10,random_state=11)
clf_RFC.fit(X_train,y_train)
predictions_RFC=clf_RFC.predict(X_test)
print(classification_report(y_test,predictions_RFC))
```

                  precision    recall  f1-score   support
    
               0       0.74      0.83      0.79       127
               1       0.80      0.70      0.75       123
    
        accuracy                           0.77       250
       macro avg       0.77      0.77      0.77       250
    weighted avg       0.77      0.77      0.77       250
    
    

### 1.3 视觉感知-图像分类_识别器
#### 1.3.1 图像分类器
应用极端随机森林(extremely randomized trees, Extra-Tress)训练分类模型，注意应用了`preprocessing.LabelEncoder()`方法编码，如果要映射回原来的类标，可以执行`.inverse_transform`实现。从对估计器的评测结果来看，f1-score的平均得分为0.62，分项得分中'开阔 '、'窄建 '和'阔建'分类的预测得分大于0.7，相对较好；而'林荫'、'窄木'、'宽建'和'宽木 '都小于0.5，预测精度并不理想。这里的一个主要原因是图像分类的确定，这几个分类中有很多图像并不容易区分之间的差异，也就导致了对图像的分类选择并不专业，最终致使建立的预测模型的预测精度并不是很好，可以尝试从新建立分类标准，提供数据集的类标精度。


```python
import pickle
import numpy as np

class ERF_trainer:
    '''
    class - 用极端随机森林训练图像分类器
    '''
    def __init__(self,X,label_words,save_path):
        from sklearn import preprocessing
        from sklearn.ensemble import ExtraTreesClassifier
        import os,pickle
        
        print('Start training...')
        self.le=preprocessing.LabelEncoder()
        self.clf=ExtraTreesClassifier(n_estimators=100,max_depth=16,random_state=0) #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
        y=self.encode_labels(label_words)
        self.clf.fit(np.asarray(X),y)
        with open(os.path.join(save_path,'ERF_clf.pkl'), 'wb') as f:  #存储训练好的图像分类器模型
            pickle.dump(self.clf, f)   
        print("end training and saved estimator.")
            
    def  encode_labels(self,label_words):
        '''
        function - 对标签编码，及训练分类器
        '''
        self.le.fit(label_words)
        return np.array(self.le.transform(label_words),dtype=np.float64)
    
    def classify(self,X):
        '''
        function - 对未知数据的预测分类
        '''
        label_nums=self.clf.predict(np.asarray(X))
        label_words=self.le.inverse_transform([int(x) for x in label_nums])
        return label_words

feature_map_fp='./data/visual_feature.pkl'            
with open(feature_map_fp,'rb') as f:
    feature_map=pickle.load(f) #读取存储的图像特征
label_words=[x['object_class'] for x in feature_map]    
dim_size=feature_map[0]['feature_vector'].shape[1]   
X=[np.reshape(x['feature_vector'],(dim_size,)) for x in feature_map]
save_path='./data'

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, label_words,test_size=0.3, random_state=42)
erf=ERF_trainer(X_train,y_train,save_path)      
```

    Start training...
    end training and saved estimator.
    


```python
from sklearn.metrics import classification_report
print(classification_report(y_test,erf.classify(X_test)))
```

                  precision    recall  f1-score   support
    
              宽建       0.56      0.26      0.36        19
              宽木       0.44      0.29      0.35        28
              开阔       1.00      0.68      0.81        28
              林荫       0.25      0.17      0.20        12
              窄建       0.73      0.93      0.82        61
              窄木       0.33      0.18      0.24        22
              阔建       0.73      0.73      0.73        11
              阔木       0.53      0.76      0.62        59
    
        accuracy                           0.62       240
       macro avg       0.57      0.50      0.52       240
    weighted avg       0.60      0.62      0.59       240
    
    

#### 1.3.2 图像识别器
图像识别器需要调用3个已经保存的文件，第一个是'ERF_clf.pkl'，保存有应用极端随机森林算法训练的图像分类器模型，用于类别预测；第二个是'visual_BOW.pkl'，保存有视觉词袋KMeans聚类模型，用于构建图像特征（位于前述的feature_builder_BOW类中）的参数输入；第三个是'visual_feature.pkl'，存储的是图像特征，主要是读取类标转换为整型编码，然后应用`inverse_transform`方法将预测的整型编码转换为原始的类标。


```python
class ImageTag_extractor:
    '''
    class - 图像识别器，基于图像分类模型，视觉词袋以及图像特征
    '''
    def __init__(self, ERF_clf_fp, visual_BOW_fp,visual_feature_fp):
        from sklearn import preprocessing
        import pickle
        with open(ERF_clf_fp,'rb') as f:  #读取存储的图像分类器模型
            self.clf=pickle.load(f)

        with open(visual_BOW_fp,'rb') as f:  #读取存储的聚类模型和聚类中心点
            self.kmeans=pickle.load(f)

        '''对标签编码'''
        with open(visual_feature_fp, 'rb') as f:
            self.feature_map=pickle.load(f)
        self.label_words=[x['object_class'] for x in self.feature_map]
        self.le=preprocessing.LabelEncoder()
        self.le.fit(self.label_words)   
        
    def predict(self,img):
        import util
        import numpy as np
        feature_vector=util.feature_builder_BOW().construct_feature(img,self.kmeans)  #提取图像特征，之前定义的feature_builder_BOW()类，可放置于util.py文件中，方便调用
        label_nums=self.clf.predict(np.asarray(feature_vector)) #进行图像识别/分类
        image_tag=self.le.inverse_transform([int(x) for x in label_nums])[0] #获取图像分类标签
        return image_tag
    
ERF_clf_fp=r'./data/ERF_clf.pkl'
visual_BOW_fp='./data/visual_BOW.pkl'
visual_feature_fp='./data/visual_feature.pkl'

import cv2 as cv
import os
imgs_fp=r'C:\Users\richi\omen-richiebao_s\omen_github\Urban-Spatial-Data-Analysis_python\notebook\BaiduMapPOIcollection_ipynb\imgs\kitti'
imgs_=[os.path.join(imgs_fp,f) for f in os.listdir(imgs_fp)]
imgs_pred_tag={fn:ImageTag_extractor(ERF_clf_fp,visual_BOW_fp,visual_feature_fp).predict(cv.imread(fn)) for fn in imgs_}
```

在GoogleEarth的街景（Street view）德国随机城市下随机的截取了6张尺寸大小不一的图像，应用图像识别器预测分类，其结果如下。


```python
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
matplotlib.rcParams['font.family'] = ['SimHei']

fig,axes=plt.subplots(2,3,sharex=True,sharey=True,figsize=(25,10))   #布局多个子图，每个子图显示一幅图像
ax=axes.flatten()  #降至1维，便于循环操作子图
i=0
for f,tag in imgs_pred_tag.items():
    img_array=Image.open(f)
    ax[i].imshow(img_array)  #显示图像
    ax[i].set_title("pred:{tag}".format(tag=tag))
    i+=1
fig.tight_layout() #自动调整子图参数，使之填充整个图像区域
fig.suptitle("images show",fontsize=14,fontweight='bold',y=1.02)
plt.show()
```


    
<a href=""><img src="./imgs/19_07.png" height='auto' width='auto' title="caDesign"></a>
    


#### 1.3.3 嵌入图像识别器到网络实验平台
将估计器部署到网络实验平台，需要将图像识别器及其相关的代码和文件（估计器、视觉词袋和图像特征）整合起来。下述将`ImageTag_extractor`和`feature_builder_BOW`类至于同一个文件中(`app/visual_perception/ImageTag_extractor.py`)，同时将`feature_builder_BOW`类中不需要的功能移除，保持代码简洁，避免干扰，并增加了`ImageTag_extractor_execution`类，方便外部调用该类，直接执行预测。而'ERF_clf.pkl'，'visual_BOW.pkl'和'visual_feature.pkl'三个必需文件置于文件夹visual_perception/vp_model中。

预测图像分类的功能是可以在页面端上传一幅图像后，用图像识别器（已经训练好的图像分类器）预测分类，因此增加了一个新的文件夹uploads用于保存上传的图像文件。图像上传的配置主要使用'flask_uploads'库实现，需要在'app/__init__.py'文件中增加相应的配置。

<a href=""><img src="./imgs/19_08.png" height='auto' width='800' title="caDesign"></a>

嵌入到网络实验平台的图像识别器


```python
#app/visual_perception/ImageTag_extractor.py
class feature_builder_BOW:
    '''
    class - (仅保留construct_feature及关联部分函数)根据所有图像关键点描述子聚类建立图像视觉词袋，获取每一图像的特征（码本）映射的频数统计
    '''

    def __init__(self, num_cluster=32):
        self.num_clusters = num_cluster

    def extract_features(self, img):
        import cv2 as cv
        '''
        function - 提取图像特征

        Paras:
        img - 读取的图像
        '''
        star_detector = cv.xfeatures2d.StarDetector_create()
        key_points = star_detector.detect(img)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kp, des = cv.xfeatures2d.SIFT_create().compute(img_gray, key_points)  # SIFT特征提取器提取特征
        return des

    def normalize(self, input_data):
        import numpy as np
        '''
        fuction - 归一化数据

        input_data - 待归一化的数组
        '''
        sum_input = np.sum(input_data)
        if sum_input > 0:
            return input_data / sum_input  # 单一数值/总体数值之和，最终数值范围[0,1]
        else:
            return input_data

    def construct_feature(self, img, kmeans):
        import numpy as np
        '''
        function - 使用聚类的视觉词袋构建图像特征（构造码本）

        Paras:
        img - 读取的单张图像
        kmeans - 已训练的聚类模型
        '''
        des = self.extract_features(img)
        labels = kmeans.predict(des.astype(np.float))  # 对特征执行聚类预测类标
        feature_vector = np.zeros(self.num_clusters)
        for i, item in enumerate(feature_vector):  # 计算特征聚类出现的频数/直方图
            feature_vector[labels[i]] += 1
        feature_vector_ = np.reshape(feature_vector, ((1, feature_vector.shape[0])))
        return self.normalize(feature_vector_)

class ImageTag_extractor:
    '''
    class - 图像识别器，基于图像分类模型，视觉词袋以及图像特征
    '''

    def __init__(self, ERF_clf_fp, visual_BOW_fp, visual_feature_fp):
        from sklearn import preprocessing
        import pickle
        with open(ERF_clf_fp, 'rb') as f:  # 读取存储的图像分类器模型
            self.clf = pickle.load(f)

        with open(visual_BOW_fp, 'rb') as f:  # 读取存储的聚类模型和聚类中心点
            self.kmeans = pickle.load(f)

        '''对标签编码'''
        with open(visual_feature_fp, 'rb') as f:
            self.feature_map = pickle.load(f)
        self.label_words = [x['object_class'] for x in self.feature_map]
        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.label_words)

    def predict(self, img):
        import numpy as np
        feature_vector=feature_builder_BOW().construct_feature(img, self.kmeans)  # 提取图像特征，之前定义的feature_builder_BOW()类，可放置于util.py文件中，方便调用
        label_nums = self.clf.predict(np.asarray(feature_vector))  # 进行图像识别/分类
        image_tag = self.le.inverse_transform([int(x) for x in label_nums])[0]  # 获取图像分类标签
        return image_tag


class ImageTag_extractor_execution:
    def __init__(self,img_url):
        self.img_url=img_url
        self.ERF_clf_fp='app/visual_perception/vp_model/ERF_clf.pkl'
        self.visual_BOW_fp = 'app/visual_perception/vp_model/visual_BOW.pkl'
        self.visual_feature_fp = 'app/visual_perception/vp_model/visual_feature.pkl'

    def execution(self):
        import cv2 as cv
        print("*"*50)
        print(self.img_url)
        imgs_pred_tag=ImageTag_extractor(self.ERF_clf_fp, self.visual_BOW_fp, self.visual_feature_fp).predict(cv.imread(self.img_url))
        return  imgs_pred_tag
```

配置'flask_uploads'


```python
#app/__init__.py
#...
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
import os
#...
photos=UploadSet('photos', IMAGES)

def create_app(config_name):
    #...
    basedir=os.path.abspath(os.path.dirname(__file__))
    app.config['UPLOADED_PHOTOS_DEST']=os.path.join(basedir,'uploads')  # you'll need to create a folder named uploads
    configure_uploads(app, photos)
    patch_request_class(app)
    #...
```

配置图像文件上传Web表达


```python
#app/visual_perception/forms.py
#...
from .. import photos
#...
class upload_img(FlaskForm):
    photo=FileField(validators=[FileAllowed(photos, 'Image only!'), FileRequired('File was empty!')])
    submit=SubmitField('Upload')
```

配置路由和视图函数，在视图函数中调用图像识别器。


```python
#app/visual_perception/views.py
#...
from .. import photos
from .ImageTag_extractor import ImageTag_extractor_execution
#...
@visual_perception.route("/img_prediction",methods=['GET','POST'])
def img_prediction():
    import os
    form=upload_img()
    if form.validate_on_submit():
        img_name=photos.save(form.photo.data)
        img_url=photos.url(img_name)
        img_fp=os.path.join(current_app.config['UPLOADED_PHOTOS_DEST'],img_name)
        imgs_pred_tag=ImageTag_extractor_execution(img_fp).execution()
        return render_template('vp/img_prediction.html', form=form, img_url=img_url, imgs_pred_tag=imgs_pred_tag)
    else:
        img_url=None
        return render_template('vp/img_prediction.html', form=form, img_url=img_url)
```

配置'预测图像分类'的模板页面


```python
<!--templates/vp/img_prediction.html --> 
{% extends "base.html" %}
{% import "bootstrap/wtf.html" as wtf %}
{% import "_macros.html" as macros %}

{% block title %}caDesign - visual perception{% endblock %}

{% block page_content %}
    <div class="jumbotron">
        <h3>预测图像分类</h3>

        <p>
            通过'参与图像分类'获取训练数据集（798幅图像）;--->应用Start特征检测器和SIFT尺度不变特征变换提取图像关键点描述子;--->聚类图像描述子建立视觉词袋(BOW);
            --->提取图像特征映射，建立训练数据集特征向量;--->极端随机森林(extremely randomized trees, Extra-Tress)训练分类估计器，建立图像分类器;--->应用估计器构建图像识别器。
        </p>

        <p>
            <a class="btn btn-secondary btn-lg" href="{{ url_for('visual_perception.vp') }}" role="button">实验主页</a>
            &nbsp<a class="btn btn-primary btn-lg" href="{{ url_for('visual_perception.imgs_classification') }}" role="button">参与图像分类</a>
            &nbsp<a class="btn btn-primary btn-lg" href="{{ url_for('visual_perception.vp') }}" role="button">空间类型分布/待</a>
        </p>
    </div>

    <form method="POST" enctype="multipart/form-data">
         {{ form.hidden_tag() }}
         {{ form.photo }}
         {% for error in form.photo.errors %}
             <span style="color: red;">{{ error }}</span>
         {% endfor %}
         {{ form.submit }}
    </form>

    {% if img_url %}
        <br>
        <div class="thumbnail">
            <img src="{{ img_url }}" >
            <div class="caption">
                <h4>预测结果：{{ imgs_pred_tag }}</h4>
                <p>[林荫,窄建,窄木,宽建,宽木,阔建,阔木,开阔] </p>
            </div>
        </div>
    {% endif %}

{% endblock %}
```

#### 1.3.4 部署
在服务器(Linux系统)下部署基于Flask构建的网络实验平台，其基本流程是：1.建立python虚拟环境，并根据生成的requirements.txt文件安装库配置环境；2.安装nginx，uwsgi，flask；3.建立一个简单的App应用，测试程序；4.本地项目上传至服务器虚拟环境目录下；5.建立uWSGI入口点(entry points)；6.配置uWSGI；6.建立Upstart Script；7. 配置Nginx；8.supervisor进程守护程序。

虽然网络上可以搜索到大量在Linux系统服务器部署Flask的教程，但是因为系统可能会存在不同，说明文字上的不清晰，Flask文件结构的差异，python环境的变化，尤其库版本的变化，通常在部署时很难一帆风顺。即使给出的配置步骤没有问题，部署的过程也会容易出现不经意的错误，因此需要耐心的跟随步骤操作，同时最好用markdown工具将步骤，所用到的命令记录下来，方便查看。具体可以尝试参考[How To Serve Flask Applications with uWSGI and Nginx on Ubuntu 14.04](https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-uwsgi-and-nginx-on-ubuntu-14-04)或者自行搜索。

### 1.5 要点
#### 1.5.1 数据处理技术

* 应用计算机视觉，提取图像关键点描述子用于构建视觉词袋（BOW）的方法

* 决策树算法，及应用极端随机森林建立分类预测估计器（模型）

* 应用pickle库存储模型及数据

* 独热编码(one-hot encoding)与整型编码(LabelEncoder)

* 用classification_report查看分类精度报告

#### 1.5.2 新建立的函数

* function - 按照分类提取图像路径与规律, `load_training_data(imgs_df,classi_field,classi_list,path_field)`

* class - 根据所有图像关键点描述子聚类建立图像视觉词袋，获取每一图像的特征（码本）映射的频数统计, `feature_builder_BOW`

包括：

* function - 提取图像特征, `extract_features(self,img)`

* function - 聚类所有图像的特征（描述子/SIFT），建立视觉词袋, `visual_BOW(self,des_all)`

* function - 提取图像特征，返回所有图像关键点聚类视觉词袋, `get_visual_BOW(self,training_data)`

* uction - 归一化数据, ` normalize(self,input_data)`

* function - 使用聚类的视觉词袋构建图像特征（构造码本）, `construct_feature(self,img,kmeans)`

* function - 返回每个图像的特征映射（码本映射）, `get_feature_map(self,training_data,kmeans)`

---

* function - 根据指定的（多个）列，将分类转换为整数表示，区间为[0,分类数-1], ``df_multiColumns_LabelEncoder(df,columns=None)

* function - 计算信息熵分量, `entropy_compomnent(numerator,denominator)`

* function - 计算信息增量(IG), `IG(df_dummies)`

* function - 使用决策树分类，并打印决策树流程图表。迁移于Sklearn的'Understanding the decision tree structure', https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py, `decisionTree_structure(X,y,criterion='entropy',cv=None,figsize=(6, 6))`

* class - 用极端随机森林训练图像分类器, `ERF_trainer`

* class - 图像识别器，基于图像分类模型，视觉词袋以及图像特征, `ImageTag_extractor`

#### 1.5.3 所调用的库


```python
import sqlite3
import pandas as pd
import util
import os
import cv2 as cv
from tqdm import tqdm
import numpy as np
import pickle
import math
from matplotlib import pyplot as plt
import matplotlib
from PIL import Image

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
```

#### 1.5.4 参考文献
1. Cavin Hackeling. Mastering Machine Learning with scikit-learn[M].Packt Publishing Ltd.July 2017.Second published. 中文版为：Cavin Hackeling.张浩然译.scikit-learning 机器学习[M].人民邮电出版社.2019.2
