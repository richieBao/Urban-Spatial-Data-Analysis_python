# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 16:24:53 2020

Richie Bao-caDesign设计(cadesign.cn)
"""
attri_f="attri_F"

def func_A():
    print("importlib_func_A/func_A")

class cls_A:
    attri_g="attri_G"
    
    def func_B():
        print("importlib_func_A/cls_A/func_B")
        
if __name__=="__main__":
    func_A()