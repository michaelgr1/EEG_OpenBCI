# -*- coding: utf-8 -*-
"""
@author: digor
"""
import time

from EV3_direct import EV3

Rob1=EV3("00:16:53:4f:bd:54",1)
Rob1.connect()
Rob1.big_motor_move(2,-50)
Rob1.start_motor(2)
time.sleep(0.5)
Rob1.stop_motor(2)
time.sleep(0.5)
Rob1.big_motor_move(3,-50)
Rob1.start_motor(3)
time.sleep(0.5)
Rob1.stop_motor(3)
Rob1.disconnect()






