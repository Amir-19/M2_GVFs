# on-policy question with fixed gamma for Robot Module 2
# author: amir samani
# code to "read all data" for plotting and saving at once was based on the code shared by niko on slacks ;)

from lib_robotis_hack import *
from dynamic_plotter import *
import thread
import time
import numpy as np
import signal
import utils


servo1_data = None
flag_stop = False

first_bin = 0
last_bin = 15


def gamma(state):
    # no use of state so it is fixed termination
    return 0

def cummlant(state):
    # angle is the cumulant
    return state

def feature_vector(state):
    fvector = np.zeros(last_bin+1)
    fvector[state] = 1.0
    return fvector

def read_data(servo):
    read_all = [0x02, 0x24, 0x08]
    data = servo.send_instruction(read_all, servo.servo_id)
    return utils.parse_data(data)

def policy(servo,ang,dir):
    if utils.is_approx_equal(ang,1.5):
        servo.move_angle(-1.5, blocking=False)
        dir = -1
    elif utils.is_approx_equal(ang,-1.5):
        servo.move_angle(1.5, blocking=False)
        dir = 1
    return dir

def get_angle_bin(ang,dir,bins):
    if dir == 1:
        ang_f_bin = ang + 1.5
    elif dir == -1:
        ang_f_bin = 4.5 - ang

    return np.digitize(ang_f_bin, bins)


def main():
    global servo1_data, flag_stop
    servo1_data = []

    # servo connection step
    D = USB2Dynamixel_Device(dev_name="/dev/tty.usbserial-AI03QD8V", baudrate=1000000)
    s1 = Robotis_Servo(D, 2)
    s1.move_angle(1.5,blocking=False)

    #plotting
    d1 = DynamicPlot(window_x=100, title='sensorimotor datastream servo 1', xlabel='time_step', ylabel='value')
    d1.add_line('servo 1 ang * 10')
    d1.add_line('Prediction')
    d1.add_line('Gamma*5')
    d1.add_line('Cumulant')
    dv = DynamicPlot(window_x=100, title='On-Policy Verification', xlabel='Time_Step', ylabel='Value')
    dv.add_line('Prediction')
    dv.add_line('Verifier')


    # TD lambda variables
    n_bin = last_bin
    num_state = n_bin + 1
    active_features = 1
    num_action = 2 # cw ccw
    alpha = 0.1/active_features
    lam = 0.99
    e = np.zeros(num_state)
    theta = np.zeros(num_state)

    # bin config
    bins = np.linspace(0, 6, n_bin, endpoint=False)

    # verifier variables
    cumulants = []
    gammas = []
    time_steps = []
    return_time_step = []
    time_time_step = []
    wait_num_time = 100
    prediction_steps = []

    # environemnt variables
    dir = 1
    t = 0
    last_ang = None
    last_state = None
    current_state = None
    freeze = False
    while True:
        # freeze learning to make sure we are not tracking
        if freeze and t>600:
            alpha = 0
        # reading data for servo 1
        [ang, position, speed, load, voltage, temperature] = read_data(s1)

        # bhv policy
        dir = policy(s1,ang,dir)

        current_state = get_angle_bin(ang,dir,bins)
        # TD lambda
        state = last_state
        state_prime = current_state

        if not last_state == None:

            # verifier
            reward = cummlant(last_ang)
            cumulants.append(reward)
            gammas.append(gamma(state_prime))
            time_steps.append(t)
            if len(cumulants) > wait_num_time:
                cumulants.pop(0)
                gammas.pop(0)
            if len(cumulants) == wait_num_time:
                return_time_step.append(sum([np.product(gammas[:k]) * cumulants[k] for k in range(wait_num_time)]))
                time_time_step.append(time_steps.pop(0))

            delta = reward + gamma(state_prime) * (np.dot(theta.T, feature_vector(state_prime))) - (
                np.dot(theta.T, feature_vector(state)))

            e = feature_vector(state) + gamma(state) * lam * e
            theta = theta + alpha * (np.dot(delta, e))
            prediction_steps.append(theta[state])

        # plot and save data
        d1.update(t, [ang*10, theta[state_prime], gamma(state_prime)*5,cummlant(state_prime)])

        # plot verifier
        dv_time = t
        dv_pred = 0
        dv_ver = 0
        if t >= wait_num_time:
            dv_pred = prediction_steps.pop(0)
            dv_ver = return_time_step.pop(0)
            time_time_step.pop(0)
        dv.update(dv_time, [dv_pred, dv_ver])

        # save data
        servo1_data.append([t, ang*10, theta[state_prime], gamma(state_prime)*5,cummlant(state_prime),dv_pred,dv_ver])
        # go to the next time step
        t += 1
        last_ang = ang
        last_state = current_state
        if flag_stop:
            thread.exit_thread()


# write plotting data to file before ending by ctrl+c
def signal_handler(signal, frame):
    global flag_stop, servo1_data

    # stop threads
    flag_stop = True

    # now we need to dump the sensorimotor datastream to disk
    np_servo1_data = np.asarray(servo1_data)

    np.savetxt('q1_data.txt', np_servo1_data)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()