import math

import bluetooth

import numerical


#
# https://www.lego.com/cdn/cs/set/assets/blt77bd61c3ac436ea3/LEGO_MINDSTORMS_EV3_Firmware_Developer_Kit.pdf
#


class EV3:

    def __init__(self, mac_address: str):
        self.socket = bluetooth.BluetoothSocket()
        self.mac_address = mac_address

    def connect(self):
        self.socket.connect((self.mac_address, 1))

    def disconnect(self):
        self.socket.close()

    def send_direct_command(self, command: bytearray):
        self.socket.send(bytes(command))

    def receive_reply(self, size: int):
        return self.socket.recv(size)

class Constants:
    DIRECT_COMMAND_REPLY = 0x00
    DIRECT_COMMAND_NO_REPLY = 0x80

    LC1 = 0x81
    LC2 = 0x82
    LC3 = 0x83
    LC4 = 0x84


def lc0(value: int) -> bytearray:
    hex = numerical.to_hex(value, 2)
    if len(hex) > 2:
        print("Value too big for LC0")
        return bytearray()
    result = bytearray()
    result.append(int(hex, 16))
    return result


def lc1(value: int, label: bool = True) -> bytearray:
    hex = numerical.to_hex(value, 2)
    if len(hex) > 2:
        print("Value too big for LC1")
        return bytearray()
    result = bytearray()
    if label:
        result.append(Constants.LC1)
    result.append(int(hex, 16))
    return result


def lc2(value: int, label: bool = True) -> bytearray:
    hex = numerical.to_hex(value, length=4)

    if len(hex) > 4:
        print("Value to big for LC2")
        return bytearray()

    # Split into byte strings
    byte_strings = []

    for i in range(0, len(hex), 2):
        byte_strings.append(hex[i:i+2])

    byte_strings.reverse()

    result = bytearray()
    if label:
        result.append(Constants.LC2)

    # Add reversed bytes (little endian)
    for byte_string in byte_strings:
        result.append(int(byte_string, 16))

    return result


def lc4(value: int, label: bool = True) -> bytearray:
    hex = numerical.to_hex(value, length=8)

    if len(hex) > 8:
        print("Value to big for LC4")
        return bytearray()

    # Split into byte strings
    byte_strings = []

    for i in range(0, len(hex), 2):
        byte_strings.append(hex[i:i+2])

    byte_strings.reverse()

    result = bytearray()
    if label:
        result.append(Constants.LC4)

    # Add reversed bytes (little endian)
    for byte_string in byte_strings:
        result.append(int(byte_string, 16))

    return result


def append_all(arr1: bytearray, arr2: bytearray):
    """
    Add the contents of arr2 to arr1
    :param arr1:
    :param arr2:
    :return:
    """
    for b in arr2:
        arr1.append(b)


class Command:
    def get_bytearray(self):
        return bytearray()


class OperationSound(Command):
    OP_CODE = 0x94

    CMD_BREAK = 0x00
    CMD_TONE = 0x01
    CMD_PLAY = 0x02
    CMD_REPEAT = 0x03

    def __init__(self):
        self.bytes = bytearray()

    def stop_sound(self):
        self.bytes.clear()
        self.bytes.append(OperationSound.OP_CODE)
        self.bytes.append(OperationSound.CMD_BREAK)

    def tone(self, volume, frequency, duration):
        self.bytes.clear()
        self.bytes.append(OperationSound.OP_CODE)
        self.bytes.append(OperationSound.CMD_TONE)

        # Volume as LC1
        if not (0 <= volume <= 100):
            print("Illegal value for volume")
            return

        volume_argument = lc1(volume)

        append_all(self.bytes, volume_argument)

        # Frequency as LC2
        if not (250 <= frequency <= 10000):
            print("Illegal value for frequency")
            return

        frequency_argument = lc2(frequency)

        append_all(self.bytes, frequency_argument)

        # Duration as LC2
        duration_argument = lc2(duration)

        append_all(self.bytes, duration_argument)

    def get_bytearray(self):
        return self.bytes


class OperationDraw(Command):
    OP_CODE = 0x84

    CMD_UPDATE = 0x00
    CMD_PIXEL = 0x02
    CMD_LINE = 0x03
    CMD_CIRCLE = 0x04
    CMD_TEXT = 0x05
    CMD_ICON = 0x06
    CMD_PICTURE = 0x07
    CMD_FLOAT_VALUE = 0x08
    CMD_FILL_RECT = 0x09
    CMD_RECT = 0x0A
    CMD_NOTIFICATION = 0x0B
    CMD_QUESTION = 0X0C
    CMD_KEYBOARD = 0x0D
    CMD_BROWSE = 0x0E
    CMD_VERT_BAR = 0x0F
    CMD_INVERSE_RECT = 0x10
    CMD_SELECT_FONT = 0x11
    CMD_TOP_LINE = 0x12
    CMD_FILL_WINDOW = 0x13
    CMD_DOT_LINE = 0x15
    CMD_VIEW_VALUE = 0x16
    CMD_VIEW_UNIT = 0x17
    CMD_FILL_CIRCLE = 0x18
    CMD_STORE = 0x19
    CMD_RESTORE = 0x1A
    CMD_ICON_QUESTION = 0x1B
    CMD_BMP_FILE = 0x1C
    CMD_GRAPH_SETUP = 0x1E
    CMD_GRAPH_DRAW = 0x0F
    CMD_TEXT_BOX = 0x20

    def __init__(self):
        self.bytes = bytearray()

    def update(self):
        self.bytes.clear()
        self.bytes.append(self.OP_CODE)
        self.bytes.append(self.CMD_UPDATE)

    def draw_pixel(self, color: bool, x: int, y: int):
        if not (0 <= x <= 177) or not (0 <= y <= 127):
            print("Illegal values")
            return
        self.bytes.clear()
        self.bytes.append(self.OP_CODE)
        self.bytes.append(self.CMD_PIXEL)

        # Color as 1 byte
        if color:  # Black
            self.bytes.append(0x01)
        else:  # White
            self.bytes.append(0x00)

        # x as LC2
        append_all(self.bytes, lc2(x))

        # y as LC2
        append_all(self.bytes, lc2(y))

    def draw_line(self, color: bool, x0: int, y0: int, x1: int, y1: int):
        self.bytes.clear()
        self.bytes.append(self.OP_CODE)
        self.bytes.append(self.CMD_LINE)

        # Color as 1 byte
        if color:  # Black
            self.bytes.append(0x01)
        else:  # White
            self.bytes.append(0x00)

        # Coordinates as Data16
        append_all(self.bytes, lc2(x0))
        append_all(self.bytes, lc2(y0))

        append_all(self.bytes, lc2(x1))
        append_all(self.bytes, lc2(y1))

    def draw_circle(self, color: bool, x0: int, y0: int, radius: int):
        self.bytes.clear()
        self.bytes.append(self.OP_CODE)
        self.bytes.append(self.CMD_CIRCLE)

        # Color as 1 byte
        if color:  # Black
            self.bytes.append(0x01)
        else:  # White
            self.bytes.append(0x00)

        append_all(self.bytes, lc2(x0))
        append_all(self.bytes, lc2(y0))
        append_all(self.bytes, lc2(radius))

    def fill_rect(self, color: bool, x: int, y: int, width: int, height: int):
        self.bytes.clear()
        self.bytes.append(self.OP_CODE)
        self.bytes.append(self.CMD_FILL_RECT)

        # Color as 1 byte
        if color:  # Black
            self.bytes.append(0x01)
        else:  # White
            self.bytes.append(0x00)

        append_all(self.bytes, lc2(x))
        append_all(self.bytes, lc2(y))
        append_all(self.bytes, lc2(width))
        append_all(self.bytes, lc2(height))

    def fill_circle(self, color: bool, x, y, radius):
        self.bytes.clear()
        self.bytes.append(self.OP_CODE)
        self.bytes.append(self.CMD_FILL_CIRCLE)

        # Color as 1 byte
        if color:  # Black
            self.bytes.append(0x01)
        else:  # White
            self.bytes.append(0x00)

        append_all(self.bytes, lc2(x))
        append_all(self.bytes, lc2(y))
        append_all(self.bytes, lc2(radius))

    def fill_window(self, color: bool, y_start: int, height: int):
        if not (0 <= y_start <= 127):
            print("Invalid")
            return
        self.bytes.clear()
        self.bytes.append(self.OP_CODE)
        self.bytes.append(self.CMD_FILL_WINDOW)

        # Color as 1 byte
        if color:  # Black
            self.bytes.append(0x01)
        else:  # White
            self.bytes.append(0x00)

        # y start as LC2
        append_all(self.bytes, lc2(y_start))

        # height as LC2
        append_all(self.bytes, lc2(height))

    def get_bytearray(self):
        return self.bytes


class OperationOutputSetType(Command):
    OP_CODE = 0xA1

    LARGE_MOTOR = 0x07
    MEDIUM_MOTOR = 0x08

    def __init__(self):
        self.bytes = bytearray()

    def set_output_type(self, layer: int, port_number: int, output_type: int):
        # layer - Data8
        # port 0-3 Data8
        # type - Data8
        self.bytes.clear()
        self.bytes.append(self.OP_CODE)
        return

    def get_bytearray(self):
        return self.bytes


class OperationOutputPolarity(Command):
    OP_CODE = 0xA7

    def __init__(self):
        self.bytes = bytearray()

    def set_polarity(self, layer: int, output: int, polarity: int):
        # layer Data8 [0-3]
        # output - output bit field 0x00 - 0x0F Data8
        # polarity - Data8 [-1, 0, 1] -1 backward, 0 opposite, 1 forward
        self.bytes.clear()
        self.bytes.append(self.OP_CODE)
        append_all(self.bytes, lc0(layer))
        append_all(self.bytes, lc0(output))
        append_all(self.bytes, lc0(polarity))

    def get_bytearray(self):
        return self.bytes


class OperationOutputTimeSpeed(Command):

    OP_CODE = 0XAF

    def __init__(self):
        self.bytes = bytearray()

    def time_speed(self, layer: int, output: int, speed: int, step1: int, step2: int, step3: int, brake: int):
        # This function enables specifying a full motor power cycle in time. The system will
        # automatically adjust the power level to the motor to keep the specified output speed.
        # Step1 specifyes the power ramp up periode in milliseconds, Step2 specifyes the
        # constant power period in milliseconds, Step 3 specifyes the power down period in
        # milliseconds.

        # layer - Data8 [0-3]
        # output - bit field Data8 0x00 - 0x0F
        # speed - power level [-100, 100]
        # step1 - Data32 time in millis for ramp up
        # step2 - Data32 time in millis for continues run
        # step3 - Data32 time in millis for ramp down
        # brake - [0: Float, 1: Break]

        self.bytes.clear()
        self.bytes.append(self.OP_CODE)
        append_all(self.bytes, lc0(layer))
        append_all(self.bytes, lc0(output))
        append_all(self.bytes, lc1(speed))  # TODO: Negative speed
        append_all(self.bytes, lc4(step1))
        append_all(self.bytes, lc4(step2))
        append_all(self.bytes, lc4(step3))
        append_all(self.bytes, lc0(brake))

    def get_bytearray(self):
        return self.bytes


class OperationOutputStepSpeed(Command):

    OP_CODE = 0XAE

    def __init__(self):
        self.bytes = bytearray()

    def output_step_speed(self, layer: int, output: int, speed: int, step1: int, step2: int, step3: int, brake: int):
        self.bytes.clear()
        self.bytes.append(self.OP_CODE)
        append_all(self.bytes, lc0(layer))
        append_all(self.bytes, lc0(output))
        append_all(self.bytes, lc1(speed))
        append_all(self.bytes, lc2(step1))
        append_all(self.bytes, lc2(step2))
        append_all(self.bytes, lc2(step3))
        append_all(self.bytes, lc0(brake))

    def get_bytearray(self):
        return self.bytes


class OperationOutputStop(Command):

    OP_CODE = 0XA3

    def __init__(self):
        self.bytes = bytearray()

    def output_stop(self, layer: int, output: int, brake: int):
        self.bytes.clear()
        self.bytes.append(self.OP_CODE)
        append_all(self.bytes, lc0(layer))
        append_all(self.bytes, lc0(output))
        append_all(self.bytes, lc0(brake))

    def get_bytearray(self):
        return self.bytes


class OperationOutputStepSync(Command):

    OP_CODE = 0XB0

    def __init__(self):
        self.bytes = bytearray()

    def output_step_sync(self, layer: int, output: int, speed: int, turn: int, step: int, brake: int):
        self.bytes.clear()
        self.bytes.append(self.OP_CODE)
        append_all(self.bytes, lc0(layer))
        append_all(self.bytes, lc0(output))
        append_all(self.bytes, lc1(speed))
        append_all(self.bytes, lc2(turn))
        append_all(self.bytes, lc2(step))
        append_all(self.bytes, lc0(brake))

    def get_bytearray(self):
        return self.bytes


class MotorControl:

    CAR_RADIUS = 12
    CAR_MIDDLE_RADIUS = 6

    def __init__(self, left_motor: int, right_motor: int, robot: EV3):
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.output = left_motor + right_motor
        self.robot = robot

    def forward(self, speed: int):
        builder = CommandBuilder()
        command = OperationOutputStepSync()
        command.output_step_sync(0, self.output, speed, 0, 0, 1)
        builder.append_command(command)
        self.robot.send_direct_command(builder.build(False, 0, 0))

    def backward(self, speed):
        self.forward(255 - speed)

    def stop(self):
        builder = CommandBuilder()
        command = OperationOutputStop()
        command.output_stop(0, self.output, 1)
        builder.append_command(command)
        self.robot.send_direct_command(builder.build(False, 0, 0))

    def turn_left(self, degrees: int, speed: int):
        self.stop()
        turn_degrees = self.wheel_turn_degrees(degrees, self.CAR_RADIUS)

        print(turn_degrees)

        # Turning left, right motor is moving
        builder = CommandBuilder()
        command = OperationOutputStepSync()

        turn = 0
        if self.right_motor < self.left_motor:
            turn = 100
        else:
            turn = math.pow(2, 16) - 100

        turn = int(turn)

        command.output_step_sync(0, self.output, speed, turn, turn_degrees, 1)
        builder.append_command(command)
        self.robot.send_direct_command(builder.build(False, 0, 0))

    def turn_left_from_middle(self, degrees: int, speed: int):
        self.stop()
        turn_degrees = self.wheel_turn_degrees(degrees, self.CAR_MIDDLE_RADIUS)

        print(turn_degrees)

        # Turning left, right motor is moving
        builder = CommandBuilder()
        command = OperationOutputStepSync()

        turn = 0
        if self.right_motor < self.left_motor:
            turn = 200
        else:
            turn = math.pow(2, 16) - 200

        turn = int(turn)

        command.output_step_sync(0, self.output, speed, turn, turn_degrees, 1)
        builder.append_command(command)
        self.robot.send_direct_command(builder.build(False, 0, 0))

    def turn_right(self, degrees: int, speed: int):
        self.stop()
        turn_degrees = self.wheel_turn_degrees(degrees, self.CAR_RADIUS)

        print(turn_degrees)

        # Turning right, left motor is moving
        builder = CommandBuilder()
        command = OperationOutputStepSync()

        turn = 0
        if self.left_motor < self.right_motor:
            turn = 100
        else:
            turn = math.pow(2, 16) - 100

        turn = int(turn)

        command.output_step_sync(0, self.output, speed, turn, turn_degrees, 1)
        builder.append_command(command)
        self.robot.send_direct_command(builder.build(False, 0, 0))

    def turn_right_from_middle(self, degrees: int, speed: int):
        self.stop()
        turn_degrees = self.wheel_turn_degrees(degrees, self.CAR_MIDDLE_RADIUS)

        print(turn_degrees)

        # Turning left, right motor is moving
        builder = CommandBuilder()
        command = OperationOutputStepSync()

        turn = 0
        if self.left_motor < self.right_motor:
            turn = 200
        else:
            turn = math.pow(2, 16) - 200

        turn = int(turn)

        command.output_step_sync(0, self.output, speed, turn, turn_degrees, 1)
        builder.append_command(command)
        self.robot.send_direct_command(builder.build(False, 0, 0))

    @staticmethod
    def wheel_turn_degrees(rotation_degrees, radius) -> int:
        wheel_circumference = 4 * math.pi
        arc_length = radius * math.radians(rotation_degrees)
        wheel_turns = arc_length / wheel_circumference

        turn_degrees = int(wheel_turns * 360)
        return turn_degrees


class CommandBuilder:

    def __init__(self):
        self.bytes = bytearray()

    def append_command(self, command: Command):
        append_all(self.bytes, command.get_bytearray())

    def clear(self):
        self.bytes.clear()

    def build(self, reply: bool = False, message_counter: int = 0, mem_size: int = 0) -> bytearray:
        final_command = bytearray()
        command_size = len(self.bytes) + 2 + 1 + 2  # Two bytes for counter, one for type and two for header/memory size

        # Add size as LC2
        append_all(final_command, lc2(command_size, label=False))

        # Counter as LC2
        append_all(final_command, lc2(message_counter, label=False))

        # Message type
        if reply:
            final_command.append(Constants.DIRECT_COMMAND_REPLY)
        else:
            final_command.append(Constants.DIRECT_COMMAND_NO_REPLY)

        if mem_size == 0:
            append_all(final_command, lc2(0, False))
        else:
            print("Not supported yet")
            append_all(final_command, lc2(0, False))

        append_all(final_command, self.bytes)
        return final_command