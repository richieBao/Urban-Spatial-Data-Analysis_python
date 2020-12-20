> Created on  Jan 14 2019 @author: Richie Bao-caDesign设计(cadesign.cn)__+updated on Sun Nov 29 23\34\53 2020 by Richie Bao

## 1. [SQLite]数据库，[Flask] 构建实验用网络应用平台，逆向工程

### 1.1 [SQLite]数据库-基础
当数据量开始膨胀，常规数据存储方式的简单文件形式，虽然逻辑简单，但可扩展性差，不能解决数据的完整性、一致性以及安全性等一系列问题，由此产生了数据管理系统(Database Management System,DBMS)，即数据库(database)。数据库是按照一定规则保存数据，能予多个用户共享，具有尽可能小的冗余度，与应用程序彼此独立，并能通过查询取回程序所需数据的数据几何。数据库有多种类型，例如与分布处理技术结合产生的分布式数据库，与并行处理技术结合产生的并行数据库，特定领域的地理数据库、空间数据库等。Web最常使用基于关系模型的数据库，即关系数据库，或称为SQL(Structured Query Language)数据库，使用结构化查询语言操作。与之相反的是今年来流行的文档数据库和键-值对数据库，即NoSQL数据库。其中关系型数据库把数据存储在表中，表的列colunm为字段field，每一字段为“样本”的属性，行row为每一“样本”数据。常用的关系型数据库有MySQL(其替代品包括MariaDB等)，以及SQLite。

[SQLite](https://www.sqlite.org/index.html)是一个C语言库（SQL数据库引擎），小型、快速、自包含(self-contained)、高高可靠性，功能齐全，已有超过1万亿(1e12)SQLite数据库在活跃的使用。其文档格式稳定、跨平台，向后兼容，同时其开发人员保证在2050年一直保持这种格式。因此在空间数据分析方法研究中，选择SQLite数据库。

对于SQLite关系型数据库的操作，包含通过SQLite命令执行(SQL语句)，通过python等语言执行(大多数数据库引擎都有对应的python包)。对于python，使用两个库，一个是[sqlite3](https://docs.python.org/3/library/sqlite3.html)操作SQLite数据库的库，另一是[SQLAlchemy(flask_sqlalchemy)库](https://www.sqlalchemy.org/)(数据库抽象层代码包，可以直接处理高等级的python对象，而不用关注表、文档或查询语言等数据库实体)。当然pandas等库也提供了其对应格式直接读写数据库的方法，进一步简化了对数据库的操作。

对SQLite数据库，引用*漫画数据库*中的数据，结合代码实现阐释。同时使用[DB Browser for SQLite(DB4S)](https://sqlitebrowser.org/)辅助查看、管理SQLite数据库。


> 参考文献
> 1.  高桥麻奈著,崔建锋译.株式会社TREND-PRO漫画制作.漫画数据库[M].科学出版社.北京,2010.5.
2. Miguel Grinberg.Flask Web Development: Developing Web Applications with Python[M]. O'Reilly Media; 2nd edition.April 3, 2018. 中文版：Miguel Grinberg.安道译.Flask Web开发：基于Python的Web应用开发实战[M].人民邮电出版社,2018.


```python
%%cmd
sqlite3 version
```

    Microsoft Windows [Version 10.0.18363.1256]
    (c) 2019 Microsoft Corporation. All rights reserved.
    
    (openCVpytorch) C:\Users\richi\omen-richiebao_s\omen_github\Urban-Spatial-Data-Analysis_python\notebook\BaiduMapPOIcollection_ipynb>sqlite3 version
    
    (openCVpytorch) C:\Users\richi\omen-richiebao_s\omen_github\Urban-Spatial-Data-Analysis_python\notebook\BaiduMapPOIcollection_ipynb>


```python
import sqlalchemy
sqlalchemy.__version__
```




    '1.3.19'



* 根据*漫画数据库*中的销售数据集录入数据


```python
import pandas as pd
from datetime import datetime
#定义假设的数据字典
sales_dic={'idx':[1101,1102,1103,1104,1105],'date':[datetime(2020,3,5),datetime(2020,3,7),datetime(2020,3,8),datetime(2020,3,10),datetime(2020,3,12)],"exporting_country_ID":[12,23,25,12,25]}
exporting_country_dic={"exporting_country_ID":[12,23,25],'exporting_country_name':['kenya','brazil','peru']}
sale_details_dic={'idx':[1101,1101,1102,1103,1104,1105,1105],'commodity_code':[101,102,103,104,101,103,104],'number':[1100,300,1700,500,2500,2000,700]}
commodity_dic={'commodity_code':[101,102,103,104],'commodity_name':['muskmelon','strawberry','apple','lemon']}

#为方便数据管理，将字典格式数据转换为DataFrame格式
sales_table=pd.DataFrame.from_dict(sales_dic)
exporting_country_table=pd.DataFrame.from_dict(exporting_country_dic)
sale_details_table=pd.DataFrame.from_dict(sale_details_dic)
commodity_table=pd.DataFrame.from_dict(commodity_dic)
```

* 创建数据库（链接）

在当前目录下创建数据库，使用`engine=create_engine('sqlite:///x.sqlite')`语句；相对或绝对路径创建数据库，例如`engine=create_engine('sqlite:///./data/fruits.sqlite'）`或`engine=create_engine('sqlite:///absolute/data/fruits.sqlite'）`；如果创建内存数据库，格式如下`engine=create_engine('sqlite://')`或`engine=create_engine('sqlite:///:memory:', echo=True)`。Unix、Max及Window系统的文件路径分隔符可能不同，如果出现异常，可以尝试在/或\切换，同时注意\也是转义符号，因此可能需要写成\\\\。


```python
from sqlalchemy import create_engine

db_fp=r'./data/fruits.sqlite'
engine=create_engine('sqlite:///'+'\\\\'.join(db_fp.split('\\')),echo=True) 
```

执行`create_engine`语句，只是建立了数据库链接，只有向其写入表数据(或者对数据库执行任务，例如`engine.connect()`)，才会在硬盘指定路径下找到该文件。如果存在同名数据库，重复执行此语句，只是实现数据库链接


```python
connection=engine.connect()
connection.close()
```

    2020-12-18 13:40:59,378 INFO sqlalchemy.engine.base.Engine SELECT CAST('test plain returns' AS VARCHAR(60)) AS anon_1
    2020-12-18 13:40:59,379 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 13:40:59,380 INFO sqlalchemy.engine.base.Engine SELECT CAST('test unicode returns' AS VARCHAR(60)) AS anon_1
    2020-12-18 13:40:59,381 INFO sqlalchemy.engine.base.Engine ()
    

* 向数据库中写入表及数据

1. 写入方法-pandas.DataFrame.to_sql 

其中参数'if_exists'，可以选择'fail'-为默认值，如果表存在，则返回异常；'replace'-先删除已经存在的表，再重新插入新表；'append'-向存在的表中追加行。

pandas方法，不需要建立模型（表结构），数据库模型的表述方法通常易于与机器学习的模型，或者算法模型的说法混淆，因此为了便于区分，这里用表结构代替模型的说法。pandas可以根据DataFrame格式数据信息，尤其包含有自动生成的数据类型，直接在数据库中建立对应数据格式的表，不需要自行预先定义表结构。但是在应用程序中调入表中数据时，又往往需要调用表结构来读取数据库信息，例如Flask Web框架（参看Flask部分阐述）等。因此可以用DB4S来查看刚刚建立的SQLite数据库及写入的表，可以看到表结构，根据表结构的数据类型信息，再手工建立表结构。表结构通常以类(class)的形式定义。但是手工定义相对比较繁琐，尤其字段比较多，不容易确定数据类型时，可以使用数据库逆向工程的方法，例如使用sqlacodegen库生成数据库表结构。


```python
def df2SQLite(engine,df,table,method='fail'):
    '''
    function - pandas方法把DataFrame格式数据写入数据库（同时创建表）
    
    Paras:
    engine - 数据库链接
    df - DataFrame格式数据，待写入数据库
    table - 表名称
    method - 写入方法，'fail'，'replace'或'append'
    '''
    try:    
        df.to_sql(table,con=engine,if_exists="%s"%method)
        if method=='replace':            
            print("_"*10,'the %s table has been overwritten...'%table)                  
        elif method=='append':
            print("_"*10,'the %s table has been appended...'%table)
        else:
            print("_"*10,'the %s table has been written......'%table)
    except:
        print("_"*10,'the %s table has been existed......'%table)
method='fail'  
table='sales'
df=sales_table
df2SQLite(engine,df,table,method)
```

    2020-12-18 21:05:02,636 INFO sqlalchemy.engine.base.Engine PRAGMA main.table_info("sales")
    2020-12-18 21:05:02,638 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 21:05:02,639 INFO sqlalchemy.engine.base.Engine PRAGMA temp.table_info("sales")
    2020-12-18 21:05:02,639 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 21:05:02,642 INFO sqlalchemy.engine.base.Engine 
    CREATE TABLE sales (
    	"index" BIGINT, 
    	idx BIGINT, 
    	date DATETIME, 
    	"exporting_country_ID" BIGINT
    )
    
    
    2020-12-18 21:05:02,642 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 21:05:02,648 INFO sqlalchemy.engine.base.Engine COMMIT
    2020-12-18 21:05:02,649 INFO sqlalchemy.engine.base.Engine CREATE INDEX ix_sales_index ON sales ("index")
    2020-12-18 21:05:02,650 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 21:05:02,656 INFO sqlalchemy.engine.base.Engine COMMIT
    2020-12-18 21:05:02,657 INFO sqlalchemy.engine.base.Engine BEGIN (implicit)
    2020-12-18 21:05:02,658 INFO sqlalchemy.engine.base.Engine INSERT INTO sales ("index", idx, date, "exporting_country_ID") VALUES (?, ?, ?, ?)
    2020-12-18 21:05:02,659 INFO sqlalchemy.engine.base.Engine ((0, 1101, '2020-03-05 00:00:00.000000', 12), (1, 1102, '2020-03-07 00:00:00.000000', 23), (2, 1103, '2020-03-08 00:00:00.000000', 25), (3, 1104, '2020-03-10 00:00:00.000000', 12), (4, 1105, '2020-03-12 00:00:00.000000', 25))
    2020-12-18 21:05:02,661 INFO sqlalchemy.engine.base.Engine COMMIT
    __________ the sales table has been written......
    


```python
df2SQLite(engine,df,table,method='fail')
```

    2020-12-18 13:43:18,408 INFO sqlalchemy.engine.base.Engine PRAGMA main.table_info("sales")
    2020-12-18 13:43:18,408 INFO sqlalchemy.engine.base.Engine ()
    __________ the sales table has been existed......
    


```python
%%cmd
sqlacodegen sqlite:///./data/fruits.sqlite --tables sales --outfile sales_table_structure.py
```

    Microsoft Windows [Version 10.0.18363.1256]
    (c) 2019 Microsoft Corporation. All rights reserved.
    
    (openCVpytorch) C:\Users\richi\omen-richiebao_s\omen_github\Urban-Spatial-Data-Analysis_python\notebook\BaiduMapPOIcollection_ipynb>sqlacodegen sqlite:///./data/fruits.sqlite --tables sales --outfile sales_table_structure.py
    
    (openCVpytorch) C:\Users\richi\omen-richiebao_s\omen_github\Urban-Spatial-Data-Analysis_python\notebook\BaiduMapPOIcollection_ipynb>

由sqlacodegen库生成SQLite数据库中'sales'表结构。对于sqlacodegen方法可以在命令行中输入`sqlacodegen --help`查看。生成的表结构写入到指定的文件中。

```python
# coding: utf-8
from sqlalchemy import BigInteger, Column, DateTime, MetaData, Table
metadata = MetaData()
t_sales = Table(
    'sales', metadata,
    Column('index', BigInteger, index=True),
    Column('idx', BigInteger),
    Column('date', DateTime),
    Column('exporting_country_ID', BigInteger)
)
```


2. 写入方法-sqlalchemy创建表结构，及写入表

定义的表结构需要继承d`eclarative_base()`映射类。完成表结构的定义后，执行`BASE.metadata.create_all(engine, checkfirst=True)`写入表结构，注意此时并未写入数据。


```python
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy as db

BASE=declarative_base() #基本映射类，需要自定义的表结构继承

class exporting_country(BASE):
    __tablename__='exporting_country'     
    
    index=db.Column(db.Integer, primary_key=True, autoincrement=True) #自动生成的索引列
    exporting_country_ID=db.Column(db.Integer)
    exporting_country_name=db.Column(db.Text)
    
    def __repr__(self): #用于表结构打印时输出的字符串，亦可以不用写。
        return '<exporting_country %r>'%self.exporting_country_ID 
exporting_country.__table__ #查看表结构
```




    Table('exporting_country', MetaData(bind=None), Column('index', Integer(), table=<exporting_country>, primary_key=True, nullable=False), Column('exporting_country_ID', Integer(), table=<exporting_country>), Column('exporting_country_name', Text(), table=<exporting_country>), schema=None)




```python
BASE.metadata.create_all(engine, checkfirst=True) #checkfirst=True，检查该表是否存在，如果存在则不建立，默认为True。可以增加tables=[Base.metadata.tables['exporting_country']]参数指定创建哪些表，或者直接使用exporting_country.__table__.create(engine, checkfirst=True)方法
```

    2020-12-18 15:12:54,227 INFO sqlalchemy.engine.base.Engine PRAGMA main.table_info("exporting_country")
    2020-12-18 15:12:54,228 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 15:12:54,229 INFO sqlalchemy.engine.base.Engine PRAGMA temp.table_info("exporting_country")
    2020-12-18 15:12:54,230 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 15:12:54,232 INFO sqlalchemy.engine.base.Engine 
    CREATE TABLE exporting_country (
    	"index" INTEGER NOT NULL, 
    	"exporting_country_ID" INTEGER, 
    	exporting_country_name TEXT, 
    	PRIMARY KEY ("index")
    )
    
    
    2020-12-18 15:12:54,232 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 15:12:54,238 INFO sqlalchemy.engine.base.Engine COMMIT
    

将数据写入到新定义的表中。使用`session.add_all`方法可以一次性写入多组数据，但是需要将其转换为对应的格式。


```python
from sqlalchemy.orm import sessionmaker
SESSION=sessionmaker(bind=engine) #建立会话链接
session=SESSION() #实例化

def zip_dic_tableSQLite(dic,table_model):
    '''
    function - 按字典的键，成对匹配，返回用于写入SQLite数据库的列表
    
    Paras:
    dic - 字典格式数据
    table_model - 表结构（模型）。数据将写入到该表中
    '''
    keys=list(dic.keys())
    vals=dic.values()
    vals_zip=list(zip(*list(vals)))
    #[{k:i for k,i in zip(keys, v)} for v in vals_zip]     
    return [table_model(**{k:i for k,i in zip(keys, v)}) for v in vals_zip]

exporting_country_table_model=zip_dic_tableSQLite(exporting_country_dic,exporting_country)
session.add_all(exporting_country_table_model)
session.commit()
```

    2020-12-18 17:09:00,532 INFO sqlalchemy.engine.base.Engine BEGIN (implicit)
    2020-12-18 17:09:00,532 INFO sqlalchemy.engine.base.Engine INSERT INTO exporting_country ("exporting_country_ID", exporting_country_name) VALUES (?, ?)
    2020-12-18 17:09:00,533 INFO sqlalchemy.engine.base.Engine (12, 'kenya')
    2020-12-18 17:09:00,541 INFO sqlalchemy.engine.base.Engine INSERT INTO exporting_country ("exporting_country_ID", exporting_country_name) VALUES (?, ?)
    2020-12-18 17:09:00,541 INFO sqlalchemy.engine.base.Engine (23, 'brazil')
    2020-12-18 17:09:00,542 INFO sqlalchemy.engine.base.Engine INSERT INTO exporting_country ("exporting_country_ID", exporting_country_name) VALUES (?, ?)
    2020-12-18 17:09:00,543 INFO sqlalchemy.engine.base.Engine (25, 'peru')
    2020-12-18 17:09:00,544 INFO sqlalchemy.engine.base.Engine COMMIT
    

用pandas写入SQLite数据库的方法，将剩下的两组数据写入。同时应用sqlacodegen库生成对应的数据库表结构。


```python
df2SQLite(engine,sale_details_table,table='sale_details')
df2SQLite(engine,commodity_table,table='commodity')
```

    2020-12-18 21:04:39,931 INFO sqlalchemy.engine.base.Engine PRAGMA main.table_info("sale_details")
    2020-12-18 21:04:39,931 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 21:04:39,931 INFO sqlalchemy.engine.base.Engine PRAGMA temp.table_info("sale_details")
    2020-12-18 21:04:39,932 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 21:04:39,934 INFO sqlalchemy.engine.base.Engine 
    CREATE TABLE sale_details (
    	"index" BIGINT, 
    	idx BIGINT, 
    	commodity_code BIGINT, 
    	number BIGINT
    )
    
    
    2020-12-18 21:04:39,935 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 21:04:39,941 INFO sqlalchemy.engine.base.Engine COMMIT
    2020-12-18 21:04:39,941 INFO sqlalchemy.engine.base.Engine CREATE INDEX ix_sale_details_index ON sale_details ("index")
    2020-12-18 21:04:39,942 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 21:04:39,947 INFO sqlalchemy.engine.base.Engine COMMIT
    2020-12-18 21:04:39,950 INFO sqlalchemy.engine.base.Engine BEGIN (implicit)
    2020-12-18 21:04:39,950 INFO sqlalchemy.engine.base.Engine INSERT INTO sale_details ("index", idx, commodity_code, number) VALUES (?, ?, ?, ?)
    2020-12-18 21:04:39,951 INFO sqlalchemy.engine.base.Engine ((0, 1101, 101, 1100), (1, 1101, 102, 300), (2, 1102, 103, 1700), (3, 1103, 104, 500), (4, 1104, 101, 2500), (5, 1105, 103, 2000), (6, 1105, 104, 700))
    2020-12-18 21:04:39,953 INFO sqlalchemy.engine.base.Engine COMMIT
    __________ the sale_details table has been written......
    2020-12-18 21:04:39,960 INFO sqlalchemy.engine.base.Engine PRAGMA main.table_info("commodity")
    2020-12-18 21:04:39,961 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 21:04:39,963 INFO sqlalchemy.engine.base.Engine PRAGMA temp.table_info("commodity")
    2020-12-18 21:04:39,964 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 21:04:39,966 INFO sqlalchemy.engine.base.Engine 
    CREATE TABLE commodity (
    	"index" BIGINT, 
    	commodity_code BIGINT, 
    	commodity_name TEXT
    )
    
    
    2020-12-18 21:04:39,967 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 21:04:39,975 INFO sqlalchemy.engine.base.Engine COMMIT
    2020-12-18 21:04:39,976 INFO sqlalchemy.engine.base.Engine CREATE INDEX ix_commodity_index ON commodity ("index")
    2020-12-18 21:04:39,977 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 21:04:39,983 INFO sqlalchemy.engine.base.Engine COMMIT
    2020-12-18 21:04:39,985 INFO sqlalchemy.engine.base.Engine BEGIN (implicit)
    2020-12-18 21:04:39,986 INFO sqlalchemy.engine.base.Engine INSERT INTO commodity ("index", commodity_code, commodity_name) VALUES (?, ?, ?)
    2020-12-18 21:04:39,987 INFO sqlalchemy.engine.base.Engine ((0, 101, 'muskmelon'), (1, 102, 'strawberry'), (2, 103, 'apple'), (3, 104, 'lemon'))
    2020-12-18 21:04:39,989 INFO sqlalchemy.engine.base.Engine COMMIT
    __________ the commodity table has been written......
    

使用sqlacodegen库分别生成对应的数据库表结构。


```python
%%cmd
sqlacodegen sqlite:///./data/fruits.sqlite --tables sale_details --outfile sale_details_table_structure.py
sqlacodegen sqlite:///./data/fruits.sqlite --tables commodity --outfile commodity_table_structure.py
```

    Microsoft Windows [Version 10.0.18363.1256]
    (c) 2019 Microsoft Corporation. All rights reserved.
    
    (openCVpytorch) C:\Users\richi\omen-richiebao_s\omen_github\Urban-Spatial-Data-Analysis_python\notebook\BaiduMapPOIcollection_ipynb>sqlacodegen sqlite:///./data/fruits.sqlite --tables sale_details --outfile sale_details_table_structure.py
    
    (openCVpytorch) C:\Users\richi\omen-richiebao_s\omen_github\Urban-Spatial-Data-Analysis_python\notebook\BaiduMapPOIcollection_ipynb>sqlacodegen sqlite:///./data/fruits.sqlite --tables commodity --outfile commodity_table_structure.py
    
    (openCVpytorch) C:\Users\richi\omen-richiebao_s\omen_github\Urban-Spatial-Data-Analysis_python\notebook\BaiduMapPOIcollection_ipynb>


```python
# coding: utf-8
from sqlalchemy import BigInteger, Column, MetaData, Table, Text
metadata = MetaData()

t_sale_details = Table(
    'sale_details', metadata,
    Column('index', BigInteger, index=True),
    Column('idx', BigInteger),
    Column('commodity_code', BigInteger),
    Column('number', BigInteger)
)

t_commodity = Table(
    'commodity', metadata,
    Column('index', BigInteger, index=True),
    Column('commodity_code', BigInteger),
    Column('commodity_name', Text)
)
```

* 查询数据库

使用类定义的表结构，在应用`session.query()`读取数据库时返回的是一个对象'<exporting_country 12>'，需要给定字段读取具体的值。

读取的方法有多种，可以自行搜索查询。


```python
exporting_country_query=session.query(exporting_country).filter_by(exporting_country_ID=12).first() #.all()将读取所有匹配，.first()仅返回首个匹配对象
print("_"*50)
print(exporting_country_query)
print(exporting_country_query.exporting_country_name,exporting_country_query.exporting_country_ID)
```

    2020-12-18 19:07:47,662 INFO sqlalchemy.engine.base.Engine SELECT exporting_country."index" AS exporting_country_index, exporting_country."exporting_country_ID" AS "exporting_country_exporting_country_ID", exporting_country.exporting_country_name AS exporting_country_exporting_country_name 
    FROM exporting_country 
    WHERE exporting_country."exporting_country_ID" = ?
     LIMIT ? OFFSET ?
    2020-12-18 19:07:47,663 INFO sqlalchemy.engine.base.Engine (12, 1, 0)
    __________________________________________________
    <exporting_country 12>
    kenya 12
    

使用sqlacodegen库生成数据库表结构，是使用`sqlalchemy.Table`定义。在应用`session.query()`读取数据库时返回的是一个元组，顺序包含所有字段的值。


```python
# coding: utf-8
from sqlalchemy import BigInteger, Column, DateTime, MetaData, Table
t_sales = Table(
    'sales', metadata,
    Column('index', BigInteger, index=True),
    Column('idx', BigInteger),
    Column('date', DateTime),
    Column('exporting_country_ID', BigInteger)
)

sales_query=session.query(t_sales).filter_by(idx=1101).first()
print("_"*50)
print(sales_query)
```

    2020-12-18 19:48:47,459 INFO sqlalchemy.engine.base.Engine SELECT sales."index" AS sales_index, sales.idx AS sales_idx, sales.date AS sales_date, sales."exporting_country_ID" AS "sales_exporting_country_ID" 
    FROM sales 
    WHERE sales.idx = ?
     LIMIT ? OFFSET ?
    2020-12-18 19:48:47,460 INFO sqlalchemy.engine.base.Engine (1101, 1, 0)
    __________________________________________________
    (0, 1101, datetime.datetime(2020, 3, 5, 0, 0), 12)
    

应用pandas读取数据库相对sqlite3和SQLAlchemy库而言，也较为简单，不需要配置表结构，能直接读取。


```python
import sqlite3
import pandas as pd
db_fp=r'./data/fruits.sqlite'
conn=sqlite3.connect(db_fp)
df_sqlite=pd.read_sql('select * from sqlite_master',con=conn) #pd.read_sql将读取数据库结构(database structure)信息
df_sqlite
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>name</th>
      <th>tbl_name</th>
      <th>rootpage</th>
      <th>sql</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>table</td>
      <td>exporting_country</td>
      <td>exporting_country</td>
      <td>4</td>
      <td>CREATE TABLE exporting_country (\n\t"index" IN...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>table</td>
      <td>sale_details</td>
      <td>sale_details</td>
      <td>5</td>
      <td>CREATE TABLE sale_details (\n\t"index" BIGINT,...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>index</td>
      <td>ix_sale_details_index</td>
      <td>sale_details</td>
      <td>7</td>
      <td>CREATE INDEX ix_sale_details_index ON sale_det...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>table</td>
      <td>commodity</td>
      <td>commodity</td>
      <td>8</td>
      <td>CREATE TABLE commodity (\n\t"index" BIGINT, \n...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>index</td>
      <td>ix_commodity_index</td>
      <td>commodity</td>
      <td>6</td>
      <td>CREATE INDEX ix_commodity_index ON commodity (...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>table</td>
      <td>sales</td>
      <td>sales</td>
      <td>2</td>
      <td>CREATE TABLE sales (\n\t"index" BIGINT, \n\tid...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>index</td>
      <td>ix_sales_index</td>
      <td>sales</td>
      <td>3</td>
      <td>CREATE INDEX ix_sales_index ON sales ("index")</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sales=pd.read_sql_table('sales', 'sqlite:///./data/fruits.sqlite') #pd.read_sql_table从数据库中读取指定的表
df_sales
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>idx</th>
      <th>date</th>
      <th>exporting_country_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1101</td>
      <td>2020-03-05</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1102</td>
      <td>2020-03-07</td>
      <td>23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1103</td>
      <td>2020-03-08</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1104</td>
      <td>2020-03-10</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1105</td>
      <td>2020-03-12</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sales_query=pd.read_sql_query('select idx,exporting_country_ID from sales', con=conn) #pd.read_sql_query将根据SQL query 或 SQLAlchemy Selectable查询语句读取特定的值 
df_sales_query
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>idx</th>
      <th>exporting_country_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1101</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1102</td>
      <td>23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1103</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1104</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1105</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>



* 增-删数据库

在sqlacodegen库生成数据库表结构，并运行，'sales'表则被存储于metadata元数据中。如果再定义一个类，同样指向这个表，则需要配置'extend_existing': True，表示在已有列基础上进行扩展，即sqlalchemy允许类是表的子集（一个表可以指向多个表结构的类）。


```python
metadata.tables
```




    immutabledict({'sale_details': Table('sale_details', MetaData(bind=None), Column('index', BigInteger(), table=<sale_details>), Column('idx', BigInteger(), table=<sale_details>), Column('commodity_code', BigInteger(), table=<sale_details>), Column('number', BigInteger(), table=<sale_details>), schema=None), 'commodity': Table('commodity', MetaData(bind=None), Column('index', BigInteger(), table=<commodity>), Column('commodity_code', BigInteger(), table=<commodity>), Column('commodity_name', Text(), table=<commodity>), schema=None), 'sales': Table('sales', MetaData(bind=None), Column('index', BigInteger(), table=<sales>), Column('idx', BigInteger(), table=<sales>), Column('date', DateTime(), table=<sales>), Column('exporting_country_ID', BigInteger(), table=<sales>), schema=None)})




```python
class sales(BASE):
    __tablename__='sales'     
    __table_args__ = {'extend_existing': True} 
    
    index=db.Column(db.Integer, primary_key=True, autoincrement=True) #因为该sales类是在执行t_sales之后定义，只能是在原有表上扩展，无法修改原表结构属性，因此index字段并不会实现自动增加的属性。需要手动增加index字段值
    idx=db.Column(db.Integer)
    date=db.Column(db.DateTime)
    exporting_country_ID=db.Column(db.Integer)
    
from sqlalchemy.orm import sessionmaker
SESSION=sessionmaker(bind=engine) #建立会话链接
session=SESSION() #实例化    
    
new_sale=sales(index=5,idx=1106,date=datetime(2020,12,18),exporting_country_ID=25)
session.add(new_sale)
session.commit()
```

    2020-12-18 21:05:29,980 INFO sqlalchemy.engine.base.Engine BEGIN (implicit)
    2020-12-18 21:05:29,982 INFO sqlalchemy.engine.base.Engine INSERT INTO sales ("index", idx, date, "exporting_country_ID") VALUES (?, ?, ?, ?)
    2020-12-18 21:05:29,983 INFO sqlalchemy.engine.base.Engine (5, 1106, '2020-12-18 00:00:00.000000', 25)
    2020-12-18 21:05:29,989 INFO sqlalchemy.engine.base.Engine COMMIT
    


```python
del_sale=session.query(sales).filter_by(idx=1106).first() #如果该行中有值为空，例如在增加该行数据时未定义写入index=5字段，该语句返回值会未空。如允许出现空值，在定义表结构时需要配置nullabley=True
```

    2020-12-18 21:05:37,533 INFO sqlalchemy.engine.base.Engine BEGIN (implicit)
    2020-12-18 21:05:37,534 INFO sqlalchemy.engine.base.Engine SELECT sales."index" AS sales_index, sales.idx AS sales_idx, sales.date AS sales_date, sales."exporting_country_ID" AS "sales_exporting_country_ID" 
    FROM sales 
    WHERE sales.idx = ?
     LIMIT ? OFFSET ?
    2020-12-18 21:05:37,534 INFO sqlalchemy.engine.base.Engine (1106, 1, 0)
    


```python
session.delete(del_sale)
session.commit()
```

    2020-12-18 21:05:42,770 INFO sqlalchemy.engine.base.Engine DELETE FROM sales WHERE sales."index" = ?
    2020-12-18 21:05:42,771 INFO sqlalchemy.engine.base.Engine (5,)
    2020-12-18 21:05:42,780 INFO sqlalchemy.engine.base.Engine COMMIT
    

* 修改数据库




```python
mod_sale=session.query(sales).filter_by(idx=1105).first()
mod_sale.exporting_country_ID=23 #修改字段值
session.commit()
```

    2020-12-18 21:09:54,886 INFO sqlalchemy.engine.base.Engine BEGIN (implicit)
    2020-12-18 21:09:54,888 INFO sqlalchemy.engine.base.Engine SELECT sales."index" AS sales_index, sales.idx AS sales_idx, sales.date AS sales_date, sales."exporting_country_ID" AS "sales_exporting_country_ID" 
    FROM sales 
    WHERE sales.idx = ?
     LIMIT ? OFFSET ?
    2020-12-18 21:09:54,888 INFO sqlalchemy.engine.base.Engine (1105, 1, 0)
    2020-12-18 21:09:54,891 INFO sqlalchemy.engine.base.Engine UPDATE sales SET "exporting_country_ID"=? WHERE sales."index" = ?
    2020-12-18 21:09:54,891 INFO sqlalchemy.engine.base.Engine (23, 4)
    2020-12-18 21:09:54,896 INFO sqlalchemy.engine.base.Engine COMMIT
    

### 1.2 [SQLite]数据库-表间关系(多表关联)

在第1部分建立表结构时，并未配置表间关联，各个表是独立的，如果想通过一个表的数据字段查询另一个表的字段内容就比较困难，例如想根据销售数量，查询对应的商品名称时，是无法直接在'commodity'商品表中直接查询商品名称的，需要先根据待查询的销售数量例如300，在'sale_details'销售明细表里找到对应的commodity_code商品编码为102，根据这个商品编码，再在'commodity'商品表找到对应的商品名称为'strawberry'。因为这个过程很繁琐，尤其数据库结构和表结构进一步复杂，这个问题会更凸显，因此需要建立表间的联系。

SQLite的表间关系配置，可以包括1对多，多对1，1对1和多对多，SQLAlchemy给出表结构配置的方法[Relationship Configuration](https://docs.sqlalchemy.org/en/14/orm/relationships.html)，可以根据其阐述进行配置。在配置时，参数'back_populates'定义反向引用，用于建立双向关系，例如销售明细表->商品表，均包括`relationship()`语句，显示的定义关系属性。如果使用参数'backref'添加反向引用，会自动在另一侧建立关系属性，为'back_populates'的简化形式。参数`uselist=True`（默认值）时，为1对多关系，如果配置1对1时，需要将其配置为`uselist=False`。在配置表关系时，为了能够清晰易读，通常以表明作为变量名，例如销售明细表->商品表，销售明细表为父表(parent)，商品表为子表(child)，父表中语句`commodity=relationship('commodity',uselist=False, back_populates="sale_details") `以子表为变量名，而子表中语句`sale_details=relationship('sale_details',back_populates="commodity")`以父表为变量名，这样可以更清晰的表述表之间的关系。

可以使用sqlacodegen库生成数据库表结构，往往应用于类似pandas写入数据库而没有定义表结构的情况下，这是一种逆向工程。下述已经定义4个表结构，那么则可以使用逆向工程反馈表的内容和表之间的关系，例如使用[Visual Paradigm](https://www.visual-paradigm.com/)反馈有下表关系，即统一建模语言(Unified Modeling Language,UML)。可以清晰直观的读出表结构和表间关系，其中'sales'是'exporting_country'的父表，连接的关键字段（ForeignKey）是'exporting_country_ID'；同时'sales'是'sale_details'的子表，联系的关键字段是'idx'，其它的关系以此类推，一目了然。

<a href=""><img src="./imgs/18_01.png" height="auto" width="auto" title="caDesign"></a>

```python
from sqlalchemy import create_engine

db_fp=r'./data/fruits_relational.sqlite'
engine=create_engine('sqlite:///'+'\\\\'.join(db_fp.split('\\')),echo=True) 
```


```python
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy as db
from sqlalchemy.orm import relationship
from sqlalchemy.schema import ForeignKey

BASE=declarative_base() #基本映射类，需要自定义的表结构继承

#销售明细表
class sale_details(BASE):
    __tablename__='sale_details'
    index=db.Column(db.Integer, primary_key=True, autoincrement=True) 
    commodity_code=db.Column(db.Integer)
    number=db.Column(db.Integer)    
    idx=db.Column(db.Integer)
    
    sales=relationship('sales',uselist=False, back_populates="sale_details")
    
    commodity=relationship('commodity',uselist=False, back_populates="sale_details")  

#销售表
class sales(BASE):
    __tablename__='sales'     
    
    index=db.Column(db.Integer, primary_key=True, autoincrement=True) #自动生成的索引列    
    date=db.Column(db.DateTime)
    exporting_country_ID=db.Column(db.Integer)
    
    exporting_country=relationship('exporting_country',uselist=False,back_populates="sales")
    
    idx=db.Column(db.Integer,ForeignKey('sale_details.idx'))
    sale_details=relationship('sale_details',back_populates="sales")    

#出口国表    
class exporting_country(BASE):
    __tablename__='exporting_country'     
    
    index=db.Column(db.Integer, primary_key=True, autoincrement=True) 
    exporting_country_name=db.Column(db.Text)    
    
    exporting_country_ID=db.Column(db.Integer,db.ForeignKey('sales.exporting_country_ID'))
    sales=relationship('sales',back_populates="exporting_country")  

#商品表
class commodity(BASE):
    __tablename__='commodity'
    index=db.Column(db.Integer, primary_key=True, autoincrement=True)     
    commodity_name=db.Column(db.Text)
    
    commodity_code=db.Column(db.Integer,ForeignKey('sale_details.commodity_code'))
    sale_details=relationship('sale_details',back_populates="commodity")

    
BASE.metadata.create_all(engine, checkfirst=True)#将所有表结构写入数据库
```

    2020-12-18 23:44:47,638 INFO sqlalchemy.engine.base.Engine SELECT CAST('test plain returns' AS VARCHAR(60)) AS anon_1
    2020-12-18 23:44:47,641 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 23:44:47,642 INFO sqlalchemy.engine.base.Engine SELECT CAST('test unicode returns' AS VARCHAR(60)) AS anon_1
    2020-12-18 23:44:47,642 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 23:44:47,643 INFO sqlalchemy.engine.base.Engine PRAGMA main.table_info("sale_details")
    2020-12-18 23:44:47,644 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 23:44:47,645 INFO sqlalchemy.engine.base.Engine PRAGMA temp.table_info("sale_details")
    2020-12-18 23:44:47,645 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 23:44:47,645 INFO sqlalchemy.engine.base.Engine PRAGMA main.table_info("sales")
    2020-12-18 23:44:47,646 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 23:44:47,646 INFO sqlalchemy.engine.base.Engine PRAGMA temp.table_info("sales")
    2020-12-18 23:44:47,646 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 23:44:47,647 INFO sqlalchemy.engine.base.Engine PRAGMA main.table_info("exporting_country")
    2020-12-18 23:44:47,647 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 23:44:47,648 INFO sqlalchemy.engine.base.Engine PRAGMA temp.table_info("exporting_country")
    2020-12-18 23:44:47,648 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 23:44:47,648 INFO sqlalchemy.engine.base.Engine PRAGMA main.table_info("commodity")
    2020-12-18 23:44:47,649 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 23:44:47,650 INFO sqlalchemy.engine.base.Engine PRAGMA temp.table_info("commodity")
    2020-12-18 23:44:47,651 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 23:44:47,652 INFO sqlalchemy.engine.base.Engine 
    CREATE TABLE sale_details (
    	"index" INTEGER NOT NULL, 
    	commodity_code INTEGER, 
    	number INTEGER, 
    	idx INTEGER, 
    	PRIMARY KEY ("index")
    )
    
    
    2020-12-18 23:44:47,652 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 23:44:47,661 INFO sqlalchemy.engine.base.Engine COMMIT
    2020-12-18 23:44:47,662 INFO sqlalchemy.engine.base.Engine 
    CREATE TABLE sales (
    	"index" INTEGER NOT NULL, 
    	date DATETIME, 
    	"exporting_country_ID" INTEGER, 
    	idx INTEGER, 
    	PRIMARY KEY ("index"), 
    	FOREIGN KEY(idx) REFERENCES sale_details (idx)
    )
    
    
    2020-12-18 23:44:47,663 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 23:44:47,670 INFO sqlalchemy.engine.base.Engine COMMIT
    2020-12-18 23:44:47,671 INFO sqlalchemy.engine.base.Engine 
    CREATE TABLE commodity (
    	"index" INTEGER NOT NULL, 
    	commodity_name TEXT, 
    	commodity_code INTEGER, 
    	PRIMARY KEY ("index"), 
    	FOREIGN KEY(commodity_code) REFERENCES sale_details (commodity_code)
    )
    
    
    2020-12-18 23:44:47,672 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 23:44:47,677 INFO sqlalchemy.engine.base.Engine COMMIT
    2020-12-18 23:44:47,679 INFO sqlalchemy.engine.base.Engine 
    CREATE TABLE exporting_country (
    	"index" INTEGER NOT NULL, 
    	exporting_country_name TEXT, 
    	"exporting_country_ID" INTEGER, 
    	PRIMARY KEY ("index"), 
    	FOREIGN KEY("exporting_country_ID") REFERENCES sales ("exporting_country_ID")
    )
    
    
    2020-12-18 23:44:47,679 INFO sqlalchemy.engine.base.Engine ()
    2020-12-18 23:44:47,686 INFO sqlalchemy.engine.base.Engine COMMIT
    

将数据写入各个表。


```python
from sqlalchemy.orm import sessionmaker
SESSION=sessionmaker(bind=engine) #建立会话链接
session=SESSION() #实例化

sales_=zip_dic_tableSQLite(sales_dic,sales)
exporting_country_=zip_dic_tableSQLite(exporting_country_dic,exporting_country)
sale_details_=zip_dic_tableSQLite(sale_details_dic,sale_details)
commodity_=zip_dic_tableSQLite(commodity_dic,commodity)

session.add_all(sales_)
session.add_all(exporting_country_)
session.add_all(sale_details_)
session.add_all(commodity_)
session.commit()
```

    2020-12-18 23:46:39,663 INFO sqlalchemy.engine.base.Engine BEGIN (implicit)
    2020-12-18 23:46:39,664 INFO sqlalchemy.engine.base.Engine INSERT INTO sale_details (commodity_code, number, idx) VALUES (?, ?, ?)
    2020-12-18 23:46:39,664 INFO sqlalchemy.engine.base.Engine (101, 1100, 1101)
    2020-12-18 23:46:39,675 INFO sqlalchemy.engine.base.Engine INSERT INTO sale_details (commodity_code, number, idx) VALUES (?, ?, ?)
    2020-12-18 23:46:39,675 INFO sqlalchemy.engine.base.Engine (102, 300, 1101)
    2020-12-18 23:46:39,677 INFO sqlalchemy.engine.base.Engine INSERT INTO sale_details (commodity_code, number, idx) VALUES (?, ?, ?)
    2020-12-18 23:46:39,678 INFO sqlalchemy.engine.base.Engine (103, 1700, 1102)
    2020-12-18 23:46:39,679 INFO sqlalchemy.engine.base.Engine INSERT INTO sale_details (commodity_code, number, idx) VALUES (?, ?, ?)
    2020-12-18 23:46:39,680 INFO sqlalchemy.engine.base.Engine (104, 500, 1103)
    2020-12-18 23:46:39,680 INFO sqlalchemy.engine.base.Engine INSERT INTO sale_details (commodity_code, number, idx) VALUES (?, ?, ?)
    2020-12-18 23:46:39,681 INFO sqlalchemy.engine.base.Engine (101, 2500, 1104)
    2020-12-18 23:46:39,682 INFO sqlalchemy.engine.base.Engine INSERT INTO sale_details (commodity_code, number, idx) VALUES (?, ?, ?)
    2020-12-18 23:46:39,684 INFO sqlalchemy.engine.base.Engine (103, 2000, 1105)
    2020-12-18 23:46:39,685 INFO sqlalchemy.engine.base.Engine INSERT INTO sale_details (commodity_code, number, idx) VALUES (?, ?, ?)
    2020-12-18 23:46:39,686 INFO sqlalchemy.engine.base.Engine (104, 700, 1105)
    2020-12-18 23:46:39,687 INFO sqlalchemy.engine.base.Engine INSERT INTO commodity (commodity_name, commodity_code) VALUES (?, ?)
    2020-12-18 23:46:39,688 INFO sqlalchemy.engine.base.Engine ('muskmelon', 101)
    2020-12-18 23:46:39,689 INFO sqlalchemy.engine.base.Engine INSERT INTO commodity (commodity_name, commodity_code) VALUES (?, ?)
    2020-12-18 23:46:39,690 INFO sqlalchemy.engine.base.Engine ('strawberry', 102)
    2020-12-18 23:46:39,691 INFO sqlalchemy.engine.base.Engine INSERT INTO commodity (commodity_name, commodity_code) VALUES (?, ?)
    2020-12-18 23:46:39,691 INFO sqlalchemy.engine.base.Engine ('apple', 103)
    2020-12-18 23:46:39,692 INFO sqlalchemy.engine.base.Engine INSERT INTO commodity (commodity_name, commodity_code) VALUES (?, ?)
    2020-12-18 23:46:39,692 INFO sqlalchemy.engine.base.Engine ('lemon', 104)
    2020-12-18 23:46:39,694 INFO sqlalchemy.engine.base.Engine INSERT INTO sales (date, "exporting_country_ID", idx) VALUES (?, ?, ?)
    2020-12-18 23:46:39,694 INFO sqlalchemy.engine.base.Engine ('2020-03-05 00:00:00.000000', 12, 1101)
    2020-12-18 23:46:39,695 INFO sqlalchemy.engine.base.Engine INSERT INTO sales (date, "exporting_country_ID", idx) VALUES (?, ?, ?)
    2020-12-18 23:46:39,695 INFO sqlalchemy.engine.base.Engine ('2020-03-07 00:00:00.000000', 23, 1102)
    2020-12-18 23:46:39,695 INFO sqlalchemy.engine.base.Engine INSERT INTO sales (date, "exporting_country_ID", idx) VALUES (?, ?, ?)
    2020-12-18 23:46:39,696 INFO sqlalchemy.engine.base.Engine ('2020-03-08 00:00:00.000000', 25, 1103)
    2020-12-18 23:46:39,697 INFO sqlalchemy.engine.base.Engine INSERT INTO sales (date, "exporting_country_ID", idx) VALUES (?, ?, ?)
    2020-12-18 23:46:39,697 INFO sqlalchemy.engine.base.Engine ('2020-03-10 00:00:00.000000', 12, 1104)
    2020-12-18 23:46:39,698 INFO sqlalchemy.engine.base.Engine INSERT INTO sales (date, "exporting_country_ID", idx) VALUES (?, ?, ?)
    2020-12-18 23:46:39,699 INFO sqlalchemy.engine.base.Engine ('2020-03-12 00:00:00.000000', 25, 1105)
    2020-12-18 23:46:39,700 INFO sqlalchemy.engine.base.Engine INSERT INTO exporting_country (exporting_country_name, "exporting_country_ID") VALUES (?, ?)
    2020-12-18 23:46:39,700 INFO sqlalchemy.engine.base.Engine ('kenya', 12)
    2020-12-18 23:46:39,701 INFO sqlalchemy.engine.base.Engine INSERT INTO exporting_country (exporting_country_name, "exporting_country_ID") VALUES (?, ?)
    2020-12-18 23:46:39,701 INFO sqlalchemy.engine.base.Engine ('brazil', 23)
    2020-12-18 23:46:39,701 INFO sqlalchemy.engine.base.Engine INSERT INTO exporting_country (exporting_country_name, "exporting_country_ID") VALUES (?, ?)
    2020-12-18 23:46:39,702 INFO sqlalchemy.engine.base.Engine ('peru', 25)
    2020-12-18 23:46:39,703 INFO sqlalchemy.engine.base.Engine COMMIT
    

通过正向引用或者反向引用轻松的获取关联表中对应的数据。例如由商品销售数量找到对应的商品名称。


```python
sale_details_info=session.query(sale_details).filter_by(number=300).first()
commodity_info=sale_details_info.commodity
commodity_name=commodity_info.commodity_name
print("_"*50)
print("销量number=300的商品名为:%s"%commodity_name)
```

    2020-12-19 00:56:45,046 INFO sqlalchemy.engine.base.Engine SELECT sale_details."index" AS sale_details_index, sale_details.commodity_code AS sale_details_commodity_code, sale_details.number AS sale_details_number, sale_details.idx AS sale_details_idx 
    FROM sale_details 
    WHERE sale_details.number = ?
     LIMIT ? OFFSET ?
    2020-12-19 00:56:45,047 INFO sqlalchemy.engine.base.Engine (300, 1, 0)
    __________________________________________________
    销量number=300的商品名为:strawberry
    

### 1.3 [Flask] 构建实验用网络应用平台
在很多场景中我们需要借助网络完成相关任务，例如展示研究内容，开展问卷调查收集数据，提供服务（例如机器学习或深度学习中已训练好的模型在线预测等）。因为这些任务需要更多的自由性，能够‘放任’的读写和处理数据，也需要根据不同的任务调整网页内容，因此使用类似[WIX](https://www.wix.com/)快速网页构建服务提供的方式很难满足数据分析、可视化、模型预测服务及布局调研信息等内容。那么[Flask英文](https://flask.palletsprojects.com/en/1.1.x/)，([中文](https://dormousehole.readthedocs.io/en/latest/))成为空间数据分析研究最好的网络实验构建平台。一方面Flask是应用python语言编写，是数据分析、大数据分析、机器学习和深度学习广泛使用的语言，亦是本书使用的语言；同时，Flask是一个轻量级的可定制框架，灵活、轻便、安全，容易上手，从而能够快速的学习并实现自行搭建网络平台完成相关网络实验部署；Flask也并不限制使用何种数据库，何种模板样式，具有强劲的自由拓展性，实现自由的定制需求。

学习Flask推荐阅读其[官方文档](https://flask.palletsprojects.com/en/1.1.x/)，以及教材*Flask Web Development: Developing Web Applications with Python(FWD)*（有中文版）。城市空间数据分析网络实验平台的建设，是以上述教材提供的案例为基础（包含社交博客搭建），在此之上扩展不同的实验任务，位于'Experiment'标签之下，网络实验的内容以及架构代码同样开源，位于GitHub上[caDesign_ExperimentPlatform](https://github.com/richieBao/caDesign_ExperimentPlatform)代码仓库中。

Flask目前已经完全的整合进了[PyCharm](https://www.jetbrains.com/pycharm/)，因此推荐使用Pycharm构建Flask网络实验平台，无需自行搭建Flask的python环境，同时网页的开发Pycharm提供了非常友好的写代码的环境，能够节约研究者搭建的时间。FWD教材提供的案例电子邮件传输协议(SMTP)基于Google Gmail，如果在中国可以使用QQ邮箱提供的服务。FWD教材的写作方式是按照学习的过程递进的讲述，后续代码要基于前述的代码，因此对于初学者跳跃式的阅读并不是很好的选择。虽然递进的讲述符合学习的规律，但是作者并未给出整体的结构框架，丰富的代码类和函数不断的更新，变量之间关系的复杂性，容易让读者失去方向，因此给出代码工程的统一建模语言(Unified Modeling Language,UML)很必要。

借助代码逆向工程(Reverse Engineering)，使用[Visual Paradigm(VP)](https://www.visual-paradigm.com/)完成部分代码的逆向工程实现，目前除了VP，还有[Pyreverse](https://www.logilab.org/blogentry/6883)等大量逆向工程工具自动生成UML图表。但是因为代码的复杂性和结构的丰富性，目前很难找到自动生成全部关联的工具，因此下图FWD教材案例的UML图表，大部分却是手工添加。对Flask结构的把握，主要包括A-配置+初始化（主程序）；B-模板和页面；C-路由与视图函数；D-Web表单；E-数据库模型(基于SQLAlchemy)；F-数据库(SQLite)，等7大部分。Flask的编写也是抓住这几个部分的关系，完成不同功能实现。因为代码书写的关键是不断的调试来验证已有代码是否顺利运行，并达到了书写的目的，而Flask是网页的开发，因此对于代码的验证，模板页面部分需要查看页面是否达显示正常，而数据处理部分仍然可以不断的用pring()方法，打印的结果会显示在运行窗口下。建议代码书写前执行`set FLASK_DEBUG=1`，这样运行窗口下的错误提示，也将在页面中显示。对于简单的Flask开发，因为实现功能简单，因此实现函数通常位于同一文件之下，但是如果项目工程比较大，众多实现功能如果不加以明确区分，往往造成代码的混乱，因此Flask引入了蓝本(blueprint)的概念，简单讲就是将不同功能实现放置于不同的文件夹（包）下，并构建子文件夹下代码与主程序代码的关联，可以互相调用方法、函数和属性。网络实验部署是采用蓝图的大型应用工厂，使用一个应用进程得到多个应用实例，易于操作。对于Flask大型应用的把握，需要一开始查看文件夹的结构，这反映了当前应用是如果用Flask架构的。

对Flask的理解需要把握应用包(文件夹)的结构，蓝本实现的方法，配置与初始化的关系，路由与视图函数的关系，视图函数与模板页面的关系，Web表单与视图函数的关系，应用数据库模型(表单)读写数据库(SQLite)及与视图函数的关系，以及显示和隐式的代码关系。对隐式代码关系的理解尤为重要，因为突然冒出来的属性变量往往找不到源头，这因为Flask已经帮助完成了相关的任务，只给了输出。同时，因为涉及到页面模板，需要动态的读写视图函数的数据，同时也需要在页面模板内处理数据，Flask应用的[Jinjia](https://jinja.palletsprojects.com/en/2.11.x/)模板引擎实现这些功能，其语法也遵循大部分程序语言的结构。


<a href=""><img src="./imgs/18_02.jpg" height='auto' width='auto' title="caDesign"></a>

网络实验平台的工程名为'caDesign_ExperimentPlatform'（即根目录），配置(config.py)和主程序(caDesign_Experiment.py)，以及SQLite数据库(data-dev.sqlite)位于根目录下，migrations为数据库迁移文件夹（可以管理工程版本），test文件夹为测试内容，venv是应用PyCharm建立Flask工程时自动创建的python环境。所有应用位于app文件包下，static文件夹（系统生成）放置图片,.css等文件，templates文件夹（系统生成）放置.html的页面模板，main,auth为FWD教材社交博客的功能应用，data文件夹用于放置相关数据，visual_perception文件夹为视觉感知-基于图像空间分类实验的网络实现。

<a href=""><img src="./imgs/18_03.png" height='auto' width=800 title="caDesign"></a>

下述仅列出了主要的模板页面，类似博客等页面是嵌套在主页等模板页面中，为了便于将其部署于不同模板页面下，通常将此类模板设计成单独的子模板，其模板名为'_post.html'，方便迁移。此外还有403、404、500等错误页面模板。

<a href=""><img src="./imgs/18_04.jpg" height='auto' width='auto' title="caDesign"></a>


### 1.4 视觉感知-基于图像的空间分类:问卷调研
计算机视觉的发展应用邻域日益广泛，例如机器人、无人驾驶、文档分析、医疗诊断等智能自主系统。其在规划设计领域的作用也日益凸显， 尤其百度、Google的街景图像，以及无人驾驶项目带来的大量序列图像和社交网络的图像，都推动着计算机视觉在规划领域潜在的应用前景。 视觉感知部分包括系列实验，例如基于图像的空间分类与城市空间类别分布、图像分割下空间分类识别、视觉评价、绿量研究，以及遥感影像用地类型 解译及依据标准的空间生成等内容。

基于图像的空间分类，方法一是应用Star、SIFT提取特征点（关键点）和描述子，聚类(K-Means)图像特征，进而建立视觉词袋（bag-of-words,BOW）。 BOW作为特征向量，输入到图像分类器（例如应用Extremely randomized trees, Extra-Trees/ET）进行训练。训练好的模型作为图像识别器 预测新的图像，并应用到更广泛的城市区域内，通过预测的空间分类研究城市类别分布。这个基于图像分类的空间类型可以根据不同的目的进行分类，例如 研究城市空间地面视野的郁密度，空间的开合程度，可以分类有林荫道、窄巷（步行为主）多建筑、窄巷有林木、宽道（1-2条）多建筑、宽道多林木、 干道（大于3条，4条居多）多建筑、干道多林木、干道开阔等。方法二是，并不计算图像特征，而是直接应用深度学习()的方法训练模型。

如果将多项视觉感知的子项研究综合起来，以及结合非视觉感知类的分析技术，能够进一步拓展城市空间类型或感知的研究范畴。视觉感知-基于图像的空间分类：问卷调研部分，是使用KITTI数据集中的图像作为城市空间识别的素材，并以FWD教材案例为网络实验平台的基础，在次基础上扩展实验部分内容。其下代码是迁移了指定路径下返回所有文件夹及其下文件结构的代码，列出了'caDesign_ExperimentPlatform'网络实验平台下app应用文件夹的文件结构，在FWD教材案例基础上增加了文件夹（蓝本）visual_perception，其下该阶段包括'__init__.py','forms.py'和'views.py'3个文件，其中'forms.py'中基于'wtforms'库定义了问卷调研表格，但因为其单选按钮为纵向排列，并没有使用，而是之间在.html模板中直接自行定义。模板文件夹'templates'下新增了'vp'文件夹，该阶段包括'vp.html','imgs_classification.html'两个文件，分别为'视觉感知-基于图像的空间分类'的说明导航首页，以及’参与图像分类‘的问卷调研页。


```python
from pathlib import Path

class DisplayablePath(object):
    '''
    class - 返回指定路径下所有文件夹及其下文件的结构。代码未改动，迁移于'https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python'
    '''
    
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(list(path
                               for path in root.iterdir()
                               if criteria(path)),
                          key=lambda s: str(s).lower())
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(path,
                                         parent=displayable_root,
                                         is_last=is_last,
                                         criteria=criteria)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (self.display_filename_prefix_last
                            if self.is_last
                            else self.display_filename_prefix_middle)

        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle
                         if parent.is_last
                         else self.display_parent_prefix_last)
            parent = parent.parent

        return ''.join(reversed(parts))
    
app_root=r'C:\Users\richi\omen-richiebao_s\omen_github\caDesign_ExperimentPlatform\app'
paths = DisplayablePath.make_tree(Path(app_root))
for path in paths:
    print(path.displayable())
```

    app/
    ├── __init__.py
    ├── __pycache__/
    │   ├── __init__.cpython-39.pyc
    │   ├── decorators.cpython-39.pyc
    │   ├── email.cpython-39.pyc
    │   ├── exceptions.cpython-39.pyc
    │   ├── fake.cpython-39.pyc
    │   └── models.cpython-39.pyc
    ├── auth/
    │   ├── __init__.py
    │   ├── __pycache__/
    │   │   ├── __init__.cpython-39.pyc
    │   │   ├── forms.cpython-39.pyc
    │   │   └── views.cpython-39.pyc
    │   ├── forms.py
    │   └── views.py
    ├── data/
    │   └── info_2011_09_26_drive_0009_sync.pkl
    ├── decorators.py
    ├── email.py
    ├── exceptions.py
    ├── fake.py
    ├── main/
    │   ├── __init__.py
    │   ├── __pycache__/
    │   │   ├── __init__.cpython-39.pyc
    │   │   ├── errors.cpython-39.pyc
    │   │   ├── forms.cpython-39.pyc
    │   │   └── views.cpython-39.pyc
    │   ├── errors.py
    │   ├── forms.py
    │   └── views.py
    ├── models.py
    ├── static/
    │   ├── favicon.ico
    │   ├── KITTI/
    │   │   └── imgs_2011_09_26_drive_0009_sync/
    │   │       ├── 0000000000.png
    │   │       ├── 0000000010.png
    │   │       ├── 0000000020.png
    │   │       ├── 0000000030.png
    │   │       ├── 0000000040.png
    │   │       ├── 0000000050.png
    │   │       ├── 0000000060.png
    │   │       ├── 0000000070.png
    │   │       ├── 0000000080.png
    │   │       ├── 0000000090.png
    │   │       ├── 0000000100.png
    │   │       ├── 0000000110.png
    │   │       ├── 0000000120.png
    │   │       ├── 0000000130.png
    │   │       ├── 0000000140.png
    │   │       ├── 0000000150.png
    │   │       ├── 0000000160.png
    │   │       ├── 0000000170.png
    │   │       ├── 0000000180.png
    │   │       ├── 0000000190.png
    │   │       ├── 0000000200.png
    │   │       ├── 0000000210.png
    │   │       ├── 0000000220.png
    │   │       ├── 0000000230.png
    │   │       ├── 0000000240.png
    │   │       ├── 0000000250.png
    │   │       ├── 0000000260.png
    │   │       ├── 0000000270.png
    │   │       ├── 0000000280.png
    │   │       ├── 0000000290.png
    │   │       ├── 0000000300.png
    │   │       ├── 0000000310.png
    │   │       ├── 0000000320.png
    │   │       ├── 0000000330.png
    │   │       ├── 0000000340.png
    │   │       ├── 0000000350.png
    │   │       ├── 0000000360.png
    │   │       ├── 0000000370.png
    │   │       ├── 0000000380.png
    │   │       ├── 0000000390.png
    │   │       ├── 0000000400.png
    │   │       ├── 0000000410.png
    │   │       ├── 0000000420.png
    │   │       ├── 0000000430.png
    │   │       └── 0000000440.png
    │   └── styles.css
    ├── templates/
    │   ├── 403.html
    │   ├── 404.html
    │   ├── 500.html
    │   ├── _comments.html
    │   ├── _macros.html
    │   ├── _posts.html
    │   ├── auth/
    │   │   ├── change_email.html
    │   │   ├── change_password.html
    │   │   ├── email/
    │   │   │   ├── change_email.html
    │   │   │   ├── change_email.txt
    │   │   │   ├── confirm.html
    │   │   │   ├── confirm.txt
    │   │   │   ├── reset_password.html
    │   │   │   └── reset_password.txt
    │   │   ├── login.html
    │   │   ├── register.html
    │   │   ├── reset_password.html
    │   │   └── unconfirmed.html
    │   ├── base.html
    │   ├── edit_post.html
    │   ├── edit_profile.html
    │   ├── followers.html
    │   ├── index.html
    │   ├── mail/
    │   │   ├── new_user.html
    │   │   └── new_user.txt
    │   ├── moderate.html
    │   ├── post.html
    │   ├── user.html
    │   └── vp/
    │       ├── imgs_classification.html
    │       └── vp.html
    └── visual_perception/
        ├── __init__.py
        ├── __pycache__/
        │   ├── __init__.cpython-39.pyc
        │   ├── forms.cpython-39.pyc
        │   └── views.cpython-39.pyc
        ├── forms.py
        └── views.py
    

#### 1.4.1 配置*视觉感知-基于图像的空间分类*蓝本
当建立一个新的子项目时，是在app下建立一个独属的文件夹（蓝本/子包），此处视觉感知实验的蓝本文件夹名为'visual_perception'。并在子包中新建__init__.py文件，调入'Blueprint'，实例化一个蓝本类对象，并调入views(.py)，把路由与蓝本关联起来，即在全局作用域下可以使用该蓝本下的view.py下视图函数及相关方法。同时，需要在app下的__init__.py文件内对应增加配置。


```python
#app/visual_perception/__init__.py
from flask import Blueprint

visual_perception=Blueprint('visual_perception', __name__) #参数：蓝本名称，和蓝本所在的包或模块，默认使用__name__。
from . import views
```


```python
#app/__init__.py
from .visual_perception import visual_perception as vp_blueprint
app.register_blueprint(vp_blueprint, url_prefix='/vp')
```

#### 1.4.2 定义SQLite表结构，写入图像信息

* 定义SQLite表结构

定义两个表结构（模型），`class vp_imgs(db.Model)`用于存储原始图像的信息，包括图像路径（位于static文件夹下），经纬度和高程信息。`class vp_classification(db.Model)`表结构，用于存储分类的结果，定义有8个分类，对应c1-c8，同时也引入图像路径字段'imgs_fp'，以及分类结果'classification'字段。一开始时，只有'vp_imgs'表中有数据，是处理好的图像信息，直接写入到该表中。而'vp_classification'表没有数据，只有在模板页面中点击了单选按钮，选择分类提交后，将该信息写入到该表中。同时这两个表之间建立了1对1的关系，指定的外键为`db.ForeignKey('vp_imgs.index')`，并建立了双向关系('back_populates="vp_imgs"'，和'back_populates="vp_classification"')，这样可以之间通过一个表的信息读取另一个表的字段信息。例如在'imgs_classification.html'模板中，`img_info.vp_classification.classification`的Jinja语句下通过"vp_imgs"表的一个行信息'img_info'，链接到"vp_classification"表中的分类信息'classification'，从而可以在页面中显示当前图像的分类信息。


定义好表结构后，可以在Pycharm的Terminal终端下敲入：`flask shell`启动shell会话，然后执行`from caDesign_Experiment import db`调入数据库实例化对象db，'caDesign_Experiment'为对应的app名，再执行`db.create_all()`，将新建的表结构写入到SQLite数据库（属于库已有的表保持不变）。


```python
#app/models.py
class vp_imgs(db.Model):
    __tablename__ = 'vp_imgs'
    index = db.Column(db.Integer, primary_key=True, autoincrement=True)
    imgs_fp=db.Column(db.Text,unique=True,nullable=False)
    lat=db.Column(db.Float)
    lon=db.Column(db.Float)
    alt=db.Column(db.Float)
    vp_classification=db.relationship('vp_classification',uselist=False, back_populates="vp_imgs")

class vp_classification(db.Model):
    __tablename__ = 'vp_classification'
    id=db.Column(db.Integer,primary_key=True,autoincrement=True)
    imgs_fp=db.Column(db.Text,unique=True,nullable=False)
    c_1 = db.Column(db.Integer)
    c_2 = db.Column(db.Integer)
    c_3 = db.Column(db.Integer)
    c_4 = db.Column(db.Integer)
    c_5 = db.Column(db.Integer)
    c_6 = db.Column(db.Integer)
    c_7 = db.Column(db.Integer)
    c_8 = db.Column(db.Integer)
    classification=db.Column(db.Text)
    timestamp=db.Column(db.DateTime,default=datetime.now)
    #index = db.Column(db.Integer)
    index=db.Column(db.Integer,db.ForeignKey('vp_imgs.index'))
    vp_imgs=db.relationship('vp_imgs',back_populates="vp_classification")
```

* 写入图像信息

先定义表结构，并写入数据库后再向表中写入数据，而不是直接应用pandas的方法直接默认表结构写入到数据库，是因为要建立表之间的关系，默认的方式则无法建立表关系。定义函数`imgs_compression_cv`实现图像的压缩，因为图像要在网络页面中显示，较大的图像加载速度慢，影响体验。函数`KITTI_info_gap`是针对KITTI数据集的操作，因为该数据集是用于无人驾驶场景下计算机视觉算法评测数据集，图像连续，因此通过该函数可以处理KITTI提供的.txt（包含经纬度，高程等信息）文件，保持数据与图像压缩函数给定的参数`gap`保持一致，即隔一段距离提取一张图像。`KITTI_info2sqlite`函数，则是将提取的图像信息写入到数据库表中，因为表以及存在，因此使用`method="append"`方法。


```python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:11:02 2020

@author: Richie Bao-caDesign设计(cadesign.cn).Chicago
"""
def imgs_compression_cv(imgs_root,imwrite_root,imgsPath_fp,gap=1,png_compression=9,jpg_quality=100):
    from pathlib import Path
    import cv2 as cv
    import numpy as np
    from tqdm import tqdm
    import os
    import pandas as pd
    '''
    function - 使用OpenCV的方法压缩保存图像    
    
    Paras:
    imgs_root - 待处理的图像文件根目录
    imwrite_root - 图像保存根目录
    gap - 无人驾驶场景下的图像通常是紧密连续的，可以剔除部分图像避免干扰， 默认值为1
    png_compression - png格式压缩值，默认为9,
    jpg_quality - jpg格式压缩至，默认为100    
    
    jpg_quality: for jpeg only. 0 - 100 (higher means better). Default is 95.
    png_compression: For png only. 0 - 9 (higher means a smaller size and longer compression time).
    '''
    if not os.path.exists(imwrite_root):
        os.makedirs(imwrite_root)
    
    imgs_root=Path(imgs_root)
    imgs_fp=[p for p in imgs_root.iterdir()][::gap]
    imgs_save_fp=[]
    for img_fp in tqdm(imgs_fp):
        img_save_fp=str(Path(imwrite_root).joinpath(img_fp.name))
        img=cv.imread(str(img_fp ))
        if img_fp.suffix=='.png':
            cv.imwrite(img_save_fp,img,[int(cv.IMWRITE_PNG_COMPRESSION), png_compression])
            imgs_save_fp.append(img_save_fp)
        elif img_fp.suffix=='.jpg':
            cv.imwrite(img_save_fp,img,[int(cv.IMWRITE_JPEG_QUALITY), jpg_quality])
            imgs_save_fp.append(strimg_save_fp)
        else:
            print("Only .jpg and .png format files are supported.")
   
    pd.DataFrame(imgs_save_fp,columns=['imgs_fp']).to_pickle(imgsPath_fp)
    return imgs_save_fp
 
def KITTI_info_gap(KITTI_info_fp,save_fp,gap=1):
    import pandas as pd
    from pathlib import Path
    '''
    function - 读取KITTI文件信息，1-包括经纬度，惯性导航系统信息等的.txt文件。只返回经纬度、海拔信息
    '''

    txt_root=Path(KITTI_info_fp)
    txt_fp=[str(p) for p in txt_root.iterdir()][::gap]
    # print(txt_fp)
    columns=["lat","lon","alt","roll","pitch","yaw","vn","ve","vf","vl","vu","ax","ay","ay","af","al","au","wx","wy","wz","wf","wl","wu","pos_accuracy","vel_accuracy","navstat","numsats","posmode","velmode","orimode"]
    drive_info=pd.concat([pd.read_csv(item,delimiter=' ',header=None) for item in txt_fp],axis=0)
    drive_info.columns=columns
    drive_info=drive_info.reset_index()    
    
    drive_info_coordi=drive_info[["lat","lon","alt"]]
    drive_info_coordi.to_pickle(save_fp)

    return drive_info_coordi

def KITTI_info2sqlite(imgsPath_fp,info_fp,replace_path,db_fp,field,method='fail'):
    from pathlib import Path
    import pandas as pd
    from sqlalchemy import create_engine  
    '''
    function - 将KITTI图像路径与经纬度信息对应起来，并存入SQLite数据库
    
    Paras:
    imgsPath_fp,
    info_fp,
    replace_path,
    db_fp,field,
    method='fail'    
    
    if_exists{‘fail’, ‘replace’, ‘append’}, default ‘fail’
    '''
    imgsPath=pd.read_pickle(imgsPath_fp)
    #flask Jinja的url_for仅支持'/,因此需要替换'\\'
    imgsPath_replace=imgsPath.imgs_fp.apply(lambda row:str(Path(replace_path).joinpath(Path(row).name)).replace('\\','/'))
    print(imgsPath_replace)
    info=pd.read_pickle(info_fp)
    imgs_df=pd.concat([imgsPath_replace,info],axis=1)
    # print(imgs_df)
    
    engine=create_engine('sqlite:///'+'\\\\'.join(db_fp.split('\\')),echo=True)     
    # print(engine)
    # print("_"*50)
    try:
        imgs_df.to_sql('%s'%field,con=engine,if_exists="%s"%method)
        print("if_exists=%s:------Data has been written to the database!"%method)
    except:
        print("_"*15,'the %s table has been existed...'%field)


if __name__=="__main__":
    #A - 使用OpenCV的方法压缩保存图像  
    # imgs_root=r'D:\dataset\KITTI\2011_09_26_drive_0009_sync\image_03\data'
    # imwrite_root=r'D:\dataset\KITTI\imgs_compression\imgs_2011_09_26_drive_0009_sync'
    # imgsPath_fp=r'D:\dataset\KITTI\imgs_compression\imgsPath_2011_09_26_drive_0009_sync.pkl'
    # imgs_save_fp=imgs_compression_cv(imgs_root,imwrite_root,imgsPath_fp,gap=10)
    
    #B - 读取KITTI经纬度信息，可以指定间隔提取距离
    # KITTI_info_fp=r'D:\dataset\KITTI\2011_09_26_drive_0009_sync\oxts\data'
    # save_fp=r'D:\dataset\KITTI\imgs_compression\info_2011_09_26_drive_0009_sync.pkl'
    # drive_info=KITTI_info_gap(KITTI_info_fp,save_fp,gap=10)
    
    #C - 将文件路径信息写入数据库
    imgsPath_fp=r'D:\dataset\KITTI\imgs_compression\imgsPath_2011_09_26_drive_0009_sync.pkl'
    info_fp=r'D:\dataset\KITTI\imgs_compression\info_2011_09_26_drive_0009_sync.pkl'
    replace_path=r'static\KITTI\imgs_2011_09_26_drive_0009_sync'
    db_fp=r'C:\Users\richi\omen-richiebao_s\omen_github\caDesign_ExperimentPlatform\data-dev.sqlite'
    KITTI_info2sqlite(imgsPath_fp,info_fp,replace_path,db_fp,field='vp_imgs',method="append")  
```


#### 1.4.3 定义路由和视图函数
配置完蓝本之后，在'app/visual_perception/views.py'文件下定义视图函数，指定路由（对应的模板页面）。首先可以定义一个简单的视图函数，例如Flask官方那个最简单的代码来测试是否蓝本配置成功：

```python
@app.route('/hello')
def hello_world():
    return 'Hello, World!'
```

只是需要重新分配一个路由，例如上述修改为`@app.route('/hello')`，这样可以在`http://127.0.0.1:5000/hello`统一资源定位系统(Uniform Resource Locater,URL)，即网页地址下打开，如果返回显示'Hello, World!'，则可以说明蓝本配置无误。目前包括两个模板页面'vp.html','imgs_classification.html'，分别对应视图函数`vp()`和`imgs_classification()`。vp视图函数网页地址（路由）指向`"/vp"`,对应模板'vp.html'下的内容比较简单，只是发布该实验项目的说明，并向模板中传入了`current_time=datetime.datetime.utcnow()`参数，可以在模板页面下显示本地时间。imgs_classification视图函数页面指向"/imgs_classification"，对应模板页面'imgs_classification.html'，因为对图像分类，需要读写数据库，以及显示图像，表单提交等动作，要稍显复杂。

首先确定分类的标准，将街道空间划分为：1-林荫道、2-窄巷（步行为主）多建筑、3-窄巷有林木、4-宽道（1-2条）多建筑、5-宽道多林木、 6-干（阔）道（大于3条，4条居多）多建筑、7-干（阔）道多林木、8-干（阔）道开阔。 分别标识为：林荫、窄建、窄木、宽建、宽木、阔建、阔木，及开阔，总共8类。根据分类信息可以在模板中定义表单，以及定义对应的数据库表结构。视图函数则需要思考'问卷调研'的动作，1-首先是读取图像路径信息，传入模板后可以显示图像；2-每一图像下对应表单，有8个单选按钮，当选择其中之一后，点击提交按钮，表单信息将返回到视图函数中(POST)；3-在视图函数中读取表单信息，将其写入到数据库中（这涉及到数据表结构的设计），不同人点击的分类可能不同，当单击一个分类，则对应写入该分类的数据库表字段中，计数方式为累加。确定图像的最终分类是对应哪个分类的累加数最多；4-因为对图像给了分类，分类信息也被写入到对应表中，则可以将分类结果显示在页面中。


```python
#app/visual_perception/views.py
from . import visual_perception
from flask import render_template,url_for, request, session,current_app,redirect
import datetime
from .. import db
from ..models import vp_imgs,vp_classification
import pandas as pd
from .forms import imgs_classi

@visual_perception.route("/vp",methods=['GET','POST'])
def vp():

    return render_template('vp/vp.html',current_time=datetime.datetime.utcnow())

@visual_perception.route("/imgs_classification",methods=['GET','POST'])
def imgs_classification():
    #form=imgs_classi()

    page=request.args.get('page', 1, type=int)
    query=vp_imgs.query
    pagination=query.order_by(vp_imgs.index).paginate(page, per_page=current_app.config['FLASKY_POSTS_PER_PAGE'],error_out=False)
    imgs_info=pagination.items

    vp_classi=vp_classification.query.all()
    #print("_"*50)
    #exist=db.session.query(db.exists().where(vp_classification.index == 0)).scalar()
    #print(exist)

    if request.method == 'GET':
        return render_template('vp/imgs_classification.html', imgs_info=imgs_info,vp_classi=vp_classi,pagination=pagination) #form=form,
    else:
        img_index=request.form.get('img_index')
        img_fp=request.form.get('img_fp')
        classi=int(request.form.get('classi'))
        img_current=vp_classification.query.filter(vp_classification.imgs_fp==img_fp).first()
        classi_dic_value={1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}
        classi_dic_name={1:u'林荫',2:u'窄建',3:u'窄木',4:u'宽建',5:u'宽木',6:u'阔建',7:u'阔木',8:u'开阔'} #中文一定要加u，即unicode( str_name )，否则服务器段如果是py2.7，会提示错误。
        classi_dic_value.update({classi:1})
        img_classification=classi_dic_name[classi]
        if not img_current:
            img_classi_info=vp_classification(imgs_fp=img_fp,
                                              c_1=classi_dic_value[1],
                                              c_2=classi_dic_value[2],
                                              c_3=classi_dic_value[3],
                                              c_4=classi_dic_value[4],
                                              c_5=classi_dic_value[5],
                                              c_6=classi_dic_value[6],
                                              c_7=classi_dic_value[7],
                                              c_8=classi_dic_value[8],
                                              classification=img_classification,
                                              index=img_index)
            db.session.add(img_classi_info)
            db.session.commit()
        else:
            query_results=[{1:c_1,2:c_2,3:c_3,4:c_4,5:c_5,6:c_6,7:c_7,8:c_8} for c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8 in db.session.query(vp_classification.c_1,vp_classification.c_2,vp_classification.c_3,vp_classification.c_4,vp_classification.c_5,vp_classification.c_6,vp_classification.c_7,vp_classification.c_8).filter(vp_classification.imgs_fp==img_fp)][0]
            query_results_update=pd.DataFrame.from_dict(query_results,orient='index').add(pd.DataFrame.from_dict(classi_dic_value,orient='index'))
            img_classification_=classi_dic_name[query_results_update.idxmax()[0]]
            query_results_update_dic=query_results_update.squeeze('columns').to_dict()
            query_results_update_dic.update({'classification':img_classification_})
            query_results_update_dic_=dict(zip(['c_1','c_2','c_3','c_4','c_5','c_6','c_7','c_8','classification'],query_results_update_dic.values()))
            vp_classification.query.filter_by(imgs_fp=img_fp).update(query_results_update_dic_)
            db.session.commit()

    return render_template('vp/imgs_classification.html',imgs_info=imgs_info,vp_classi=vp_classi,pagination=pagination) #,form=form 
```


#### 1.4.4 定义模板

模板的定义中需要注意对Jinja(2)的使用，[Jinja](https://jinja.palletsprojects.com/en/2.11.x/)是现代而又设计友好用于python的模板语言，起源于'Django'。


```python
<!--templates/vp/vp.html --> 
{% extends "base.html" %}
{% import "bootstrap/wtf.html" as wtf %}

{% block title %}caDesign - visual perception{% endblock %}

{% block page_content %}
    <div class="jumbotron">
        <h1>视觉感知-基于图像的空间分类</h1>
        <p>
            计算机视觉的发展应用邻域日益广泛，例如机器人、无人驾驶、文档分析、医疗诊断等智能自主系统。其在规划设计领域的作用也日益凸显，
            尤其百度、Google的街景图像，以及无人驾驶项目带来的大量序列图像和社交网络的图像，都推动着计算机视觉在规划领域潜在的应用前景。
            视觉感知部分包括系列实验，例如基于图像的空间分类与城市空间类别分布、图像分割下空间分类识别、视觉评价、绿量研究，以及遥感影像用地类型
            解译及依据标准的空间生成等内容。
        </p>
        <p>
            基于图像的空间分类，方法一是应用Star、SIFT提取特征点（关键点）和描述子，聚类(K-Means)图像特征，进而建立视觉词袋（bag-of-words,BOW）。
            BOW作为特征向量，输入到图像分类器（例如应用Extremely randomized trees, Extra-Trees/ET）进行训练。训练好的模型作为图像识别器
            预测新的图像，并应用到更广泛的城市区域内，通过预测的空间分类研究城市类别分布。这个基于图像分类的空间类型可以根据不同的目的进行分类，例如
            研究城市空间地面视野的郁密度，空间的开合程度，可以分类有林荫道、窄巷（步行为主）多建筑、窄巷有林木、宽道（1-2条）多建筑、宽道多林木、
            干道（大于3条，4条居多）多建筑、干道多林木、干道开阔等。方法二是，并不计算图像特征，而是直接应用深度学习()的方法训练模型。
        </p>
        <p>
            如果将多项视觉感知的子项研究综合起来，以及结合非视觉感知类的分析技术，能够进一步拓展城市空间类型或感知的研究范畴。
        </p>

        <p>
            <a class="btn btn-primary btn-lg" href="{{ url_for('visual_perception.imgs_classification') }}" role="button">参与图像分类</a>
            &nbsp<a class="btn btn-primary btn-lg" href="{{ url_for('visual_perception.vp') }}" role="button">预测图像分类</a>
            &nbsp<a class="btn btn-primary btn-lg" href="{{ url_for('visual_perception.vp') }}" role="button">空间类型分布</a>
        </p>
        <p>本地时间：{{ moment(current_time).format('LLL') }}</p>
        <p>{{ moment(current_time).fromNow(refresh=True) }}</p>
    </div>
{% endblock %}
```


```python
<!--templates/vp/imgs_classification.html --> 
{% extends "base.html" %}
{% import "bootstrap/wtf.html" as wtf %}
{% import "_macros.html" as macros %}

{% block title %}caDesign - visual perception{% endblock %}

{% block page_content %}

    <div class="jumbotron">
        <h3>参与图像分类</h3>
        <p>
            将街道空间划分为：1-林荫道、2-窄巷（步行为主）多建筑、3-窄巷有林木、4-宽道（1-2条）多建筑、5-宽道多林木、 6-干（阔）道（大于3条，4条居多）多建筑、7-干（阔）道多林木、8-干（阔）道开阔。
            分别标识为：林荫、窄建、窄木、宽建、宽木、阔建、阔木，及开阔，总共8类。
        </p>

        <h6>@ARTICLE{Geiger2013IJRR, author = {Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun}, title = {Vision meets Robotics: The KITTI Dataset}, journal = {International Journal of Robotics Research (IJRR)}, year = {2013} }</h6>

        <p>
            <a class="btn btn-secondary btn-lg" href="{{ url_for('visual_perception.vp') }}" role="button">实验主页</a>
            &nbsp<a class="btn btn-primary btn-lg" href="{{ url_for('visual_perception.vp') }}" role="button">预测图像分类</a>
            &nbsp<a class="btn btn-primary btn-lg" href="{{ url_for('visual_perception.vp') }}" role="button">空间类型分布</a>
        </p>
    </div>

    <ul class="question-list-group">
        {% for img_info in imgs_info %}
            <li style="float:left">
                <div class="row">
                    <div class=" col-md-10">
                        <div class="thumbnail">
                            <img src="{{ url_for('static',filename=img_info.imgs_fp[7:]) }}" alt="">
                            <div class="caption">
                                <h4>ID：{{ img_info.index }}
                                    {{ img_info.vp_classification.classification}}
                                </h4>
                                        <iframe name="formDestination" class="iframe", style="display:none;"></iframe>
                                        <form action="" method="post" target="formDestination">
                                        <input type="radio" name="classi" value="1"/>林荫&nbsp
                                        <input type="radio" name="classi" value="2"/>窄建&nbsp
                                        <input type="radio" name="classi" value="3"/>窄木&nbsp
                                        <input type="radio" name="classi" value="4"/>宽建&nbsp
                                        <input type="radio" name="classi" value="5"/>宽木&nbsp
                                        <input type="radio" name="classi" value="6"/>阔建&nbsp
                                        <input type="radio" name="classi" value="7"/>阔木&nbsp
                                        <input type="radio" name="classi" value="8"/>开阔&nbsp

                                        <input type="hidden" name="img_index" value="{{img_info.index }}">
                                        <input type="hidden" name="img_fp" value="{{ img_info.imgs_fp}}">

                                        <input type="submit" value="提交" class="btn btn-secondary btn-sm" onClick="this.form.submit(); this.disabled=true; this.value='已提交'; ">
                                   </form>
                            </div>
                        </div>
                    </div>

                </div>
            </li>
        {% endfor %}
    </ul>

{% if pagination %}
<div class="pagination">
    {{ macros.pagination_widget(pagination, 'visual_perception.imgs_classification') }}
</div>
{% endif %}

{% endblock %}
```

<a href=""><img src="./imgs/18_05.png" height='auto' width=800 title="caDesign"></a>







### 1.5 要点
#### 1.5.1 数据处理技术

* [SQLite]数据库-建、读、写、增、改

* SQLAlchemy管理SQLite数据库

* %%cmd 命令行

* 表间关系(多表关联)

* [Flask]Web 框架

* Jinja2

#### 1.5.2 新建立的函数

* function - pandas方法把DataFrame格式数据写入数据库（同时创建表）, `df2SQLite(engine,df,table,method='fail')`

* function - 按字典的键，成对匹配，返回用于写入SQLite数据库的列表, `zip_dic_tableSQLite(dic,table_model)`

* class - 返回指定路径下所有文件夹及其下文件的结构。代码未改动，迁移于'https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python', `DisplayablePath(object)`

* function - 使用OpenCV的方法压缩保存图像  , `imgs_compression_cv(imgs_root,imwrite_root,imgsPath_fp,gap=1,png_compression=9,jpg_quality=100)`

* function - 读取KITTI文件信息，1-包括经纬度，惯性导航系统信息等的.txt文件。只返回经纬度、海拔信息, `KITTI_info_gap(KITTI_info_fp,save_fp,gap=1)`

* function - 将KITTI图像路径与经纬度信息对应起来，并存入SQLite数据库, `KITTI_info2sqlite(imgsPath_fp,info_fp,replace_path,db_fp,field,method='fail')`

#### 1.5.3 所调用的库


```python
import sqlalchemy
import pandas as pd
import sqlite3
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy import BigInteger, Column, DateTime, MetaData, Table,Text
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy as db
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import relationship
from sqlalchemy.schema import ForeignKey

from pathlib import Path

from flask import Blueprint
from flask import render_template,url_for, request, session,current_app,redirect
import datetime
```

#### 1.5.4 参考文献
1. 高桥麻奈著,崔建锋译.株式会社TREND-PRO漫画制作.漫画数据库[M].科学出版社.北京,2010.5.
2. Miguel Grinberg.Flask Web Development: Developing Web Applications with Python[M]. O'Reilly Media; 2nd edition.April 3, 2018. 中文版：Miguel Grinberg.安道译.Flask Web开发：基于Python的Web应用开发实战[M].人民邮电出版社,2018.
