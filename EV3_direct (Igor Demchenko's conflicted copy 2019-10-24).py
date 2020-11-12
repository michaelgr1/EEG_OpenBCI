# -*- coding: utf-8 -*-
"""
@author: digor
"""
import struct

import bluetooth


class EV3:
    def __init__(self, BDADDR, PORT):
        self.BDADDR = BDADDR
        self.PORT = PORT
    
    
    def connect(self):
        self.sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        self.sock.connect((self.BDADDR, self.PORT))
    def disconnect(self):
        self.sock.close()
        
    def big_motor_move(self,motors,vel):
        #print('hello')
        b=bytearray()
        b.append(0x0a)
        b.append(0x00)
        b.append(0x00)
        b.append(0x00)
        b.append(0x80)
        b.append(0x00)
        b.append(0x00)
        b.append(0xa4)
        b.append(0x00)
        
        # MOTORS BYTES
        b.append(motors)
        
        # Velocity
        b.append(0x81)
        if vel<=0:
            b.append(int(255+120*vel/100))
        else:
            b.append(int(120*vel/100))
        self.sock.send(bytes(b))        
        
        #b.append(0x12 & 0xff)
        #print(b)
        #Motor_code=b'\x12\x00\x00\x00\x80\x00\x00\xae\x00\x06\x81\x32\x00\x82\x84\x01\x82\xb4\x00\x01'
        #b=b'\x0a\x00\x00\x00\x80\x00\x00\xa4\x00\x06\x81\x32'
 
        
    def start_motor(self,motors):
        b=b'\x08\x00\x00\x00\x80\x00\x00\xa6\x00\x06'
        b=bytearray()
        b.append(0x08)
        b.append(0x00)
        b.append(0x00)
        b.append(0x00)
        b.append(0x80)
        b.append(0x00)
        b.append(0x00)
        b.append(0xa6)
        b.append(0x00)
        b.append(motors)
        self.sock.send(bytes(b))
        
        
    def stop_motor(self):
        b=b'\x09\x00\x00\x00\x80\x00\x00\xa3\x00\x06\x00'
        self.sock.send(b)
        
    def measure_reflected_light(self,port):
        b=b'\x0D\x00\x00\x00\x00\x04\x00\x99\x1D\x00\x02\x00\x00\x01\x60'
        self.sock.send(b)
        data=(self.sock.recv(10))       
        first=(bytes(data)[7])
        second=(bytes(data)[8])
        binary=bytearray()
        binary.append(0x00)
        binary.append(0x00)
        binary.append(first)
        binary.append(second)
        result=(struct.unpack('<f',binary))
        return result[0]
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
