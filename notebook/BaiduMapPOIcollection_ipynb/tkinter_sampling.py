# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:01:24 2020

@author:Richie Bao-caDesign设计(cadesign.cn)
"""
import tkinter as tk 
from PIL import Image, ImageTk
import numpy as np
import os
import matplotlib.colors as mpc
from skimage.util import img_as_ubyte

workspace=r"C:\Users\richi\omen-richiebao\omen_github\Urban-Spatial-Data-Analysis_python\notebook\BaiduMapPOIcollection_ipynb\data"
img_fp=os.path.join(workspace,'a_191018_exposure_rescaled.npy')
img_array=np.load(img_fp)

window=tk.Tk() #实例化对象，建立窗口window
window.title('sampling') #窗口名称
#window.geometry('800x800') #配置窗口大小

r,g,b=3,2,1
img_rgb=np.dstack((img_array[r],img_array[g],img_array[b]))

def rescale_array(array,domain=[0,255]):
    import numpy as np
    '''
    function - 指定区间，比例缩放数组
    
    Paras:
        array - 待缩放的数组
        domain - 缩放区间
    '''
    array_min,array_max=np.min(array),np.max(array)
    scale_ratio=(domain[1]-domain[0])/(array_max-array_min)
    diff_min=domain[0]-scale_ratio*array_min
    return scale_ratio*array+diff_min

#img_rgb_255=rescale_array(img_rgb).astype(np.uint8)  #使用自定义方法
img_rgb_255=img_as_ubyte(img_rgb) #使用skimage提供的方法
image=Image.fromarray(img_rgb_255)
w_img=ImageTk.PhotoImage(image)

canvas=tk.Canvas(window,width=490,height=460) #配置画布
canvas.pack()
canvas.create_image(0,0,anchor='nw',image=w_img)

print("+"*50+'start')
window.mainloop() #主窗口循环显示
print("_"*50+'end')