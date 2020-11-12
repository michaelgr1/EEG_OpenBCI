import time

import bluetooth


def append_to_byte_array(arr: bytearray, val: int):
    byte_count = 0

    while val > (pow(2, (byte_count * 8)) - 1):
        byte_count += 1

    val_in_bytes = val.to_bytes(byte_count, "big")

    for b in val_in_bytes:
        arr.append(b)


MAC_ADDRESS = "00:16:53:4f:bd:54"

OP_SOUND = 0x94

CMD_SOUND_TONE = 0x01

volume = 50

freq = 255

duration = 1000

socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)

socket.connect((MAC_ADDRESS, 1))

print("Connected")

command = bytearray()
command.append(0x0F)
command.append(0x00)
command.append(0x00)
command.append(0x00)
command.append(0x80)
command.append(0x00)
command.append(0x00)
command.append(0x94)
command.append(0x01)
command.append(0x81)
command.append(0x02)
command.append(0x82)
command.append(0xE8)
command.append(0x03)
command.append(0x82)
command.append(0xB8)
command.append(0x0B)

# append_to_byte_array(command, OP_SOUND)
# append_to_byte_array(command, CMD_SOUND_TONE)
# append_to_byte_array(command, 0x32)
# append_to_byte_array(command, 0x01)
# append_to_byte_array(command, 0xF4)
# append_to_byte_array(command, 0x03)
# append_to_byte_array(command, 0xe8)

print(command)

#command.reverse()

print(command)

socket.send(bytes(command))

time.sleep(1)

socket.close()

print("Disconnected")


# append_to_byte_array(None, 1)
#
# append_to_byte_array(None, 255)
#
# append_to_byte_array(None, 256)
