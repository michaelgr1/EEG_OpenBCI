
def hex_str(arr: bytearray) -> str:

    result = ""

    for b in arr:
        result += to_hex(b)
        result += "|"

    return result


def to_hex(value: int, length: int = -1) -> str:
    """
    Convert a decimal integer to a hex string
    :param value: int to convert
    :return: converted string
    """

    power_index = -2

    result = ""

    while power_index == -2 or power_index >= 0:

        if value == 0:
            if power_index == -2:
                result = "0"
                break
            else:
                while power_index >= 0:
                    result += "0"
                    power_index -= 1
                break

        # Find largest power of 16
        power = 0
        for power in range(0, value):
            if pow(16, power) > value:
                power = power - 1
                break

        # Find multiplier
        multiplier = 0
        for multiplier in range(1, 16):
            if multiplier * pow(16, power) > value:
                multiplier = multiplier - 1
                break

        if power_index == -2:
            power_index = power

        while power_index > power:
            result += "0"
            power_index -= 1

        # Power index = power
        if multiplier < 10:
            result += str(multiplier)
        else:
            if multiplier == 10:
                result += "A"
            elif multiplier == 11:
                result += "B"
            elif multiplier == 12:
                result += "C"
            elif multiplier == 13:
                result += "D"
            elif multiplier == 14:
                result += "E"
            elif multiplier == 15:
                result += "F"
            else:
                print("Serious error, multiplier cannot be above 15 with base 16")

        value = value - multiplier * pow(16, power)
        power_index -= 1

    if length > len(result):
        while length > len(result):
            result = "0" + result

    return result
