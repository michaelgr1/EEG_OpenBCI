import binascii
import bluetooth
import time

import ev3
import numerical

MAC_ADDRESS = "00:16:53:4f:bd:54"

# socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
#
# socket.connect((MAC_ADDRESS, 1))
#
# print("Connected")
#
# command = bytearray()
#
#
# # Message size = 6
# command.append(0x06)
# command.append(0x00)
#
# # Message counter
# command.append(0x2A)
# command.append(0x00)
#
# # Message type = Direct command reply
# command.append(0x00)
#
# # Memory size - Empty (Header)
# command.append(0x00)
# command.append(0x00)
#
# # Operation do nothing
# command.append(0x01)
#
# print(binascii.hexlify(command))
# print(numerical.hex_str(command))
#
# socket.send(bytes(command))
#
# result = socket.recv(5)
#
# print(binascii.hexlify(result))
# print(numerical.hex_str(result))
#
# socket.close()
#
# print("Disconnected")

robot = ev3.EV3(MAC_ADDRESS)
robot.connect()

# builder = ev3.CommandBuilder()
# sound = ev3.OperationSound()
# sound.tone(50, 500, 5000)
# builder.append_command(sound)
#
# stop_cmd = builder.build(message_counter=42)
#
# print(binascii.hexlify(stop_cmd))
#
# robot.send_direct_command(stop_cmd)
#
# time.sleep(2)
#
# builder = ev3.CommandBuilder()
# sound.stop_sound()
# builder.append_command(sound)
#
# stop_cmd = builder.build(message_counter=42)
#
# print("Stopping sound")
# robot.send_direct_command(stop_cmd)

builder = ev3.CommandBuilder()
draw = ev3.OperationDraw()

draw.fill_window(False, 0, 0)
builder.append_command(draw)

draw.draw_line(True, 0, 0, 177, 127)
builder.append_command(draw)

draw.draw_circle(True, 50, 50, 10)
builder.append_command(draw)

draw.draw_circle(True, 100, 100, 20)
builder.append_command(draw)

draw.fill_rect(True, 0, 60, 100, 20)
builder.append_command(draw)

draw.fill_circle(True, 130, 50, 20)
builder.append_command(draw)

draw.update()
builder.append_command(draw)

draw_cmd = builder.build(message_counter=0)

print(binascii.hexlify(draw_cmd))

robot.send_direct_command(draw_cmd)

robot.disconnect()
