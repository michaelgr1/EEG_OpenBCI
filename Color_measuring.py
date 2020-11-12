# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 22:57:58 2019

@author: digor
"""
import time

from EV3_direct import EV3

Rob1=EV3("00:16:53:4f:bd:54",1)
Rob1.connect()
print(time.time())
for i in range(50):
    data=Rob1.measure_reflected_light(2)
    #print(data)


print(time.time())
Rob1.disconnect()



