import math
import numpy as np

def parse_data(data):
    # from niko shared in slacks. I just added "ang"
    # Important : The load calculation formula is probably incorrect! I need to change it!!
    ang = (data[1]*256 +data[0] - 0x200) * math.radians(300.0) / 1024.0
    position = (data[0] + data[1] * 256 - 512) * 5 * math.pi / 3069
    speed = (data[2] + data[3] * 256) * 5 * math.pi / 3069
    load = (data[5] & 3) * 256 + data[4]
    load *= math.pow(-1, bool(data[5] & 4))
    voltage = data[6] / 10
    temperature = data[7]

    return [ang, position, speed, load, voltage, temperature]

def is_approx_equal(a,b,degree = 1e-2):
    is_app_eq = (abs(a - b) <= max(1e-4 * max(abs(a), abs(b)), degree))
    return is_app_eq