# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:54:49 2020

@author:Richie Bao-caDesign设计(cadesign.cn)
ref: https://stackoverflow.com/questions/41656176/tkinter-canvas-zoom-move-pan   27
"""
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
    
    import pickle as pkl
    with open(os.path.join(workspace,r'sampling_position.pkl'),'wb') as handle:
        pkl.dump(app.MW.sample_coordi_recover,handle)
    