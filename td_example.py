# first test on TD(lambda)
# states from 0 -> 17
# this is like a circular walk environment 17-> 16 -> ... -> 0 -> 17 and repeat!
# the on-policy GVF question is how many steps we should take to reach state 0 following this circular policy
# we use TD lambda to learn this on-policy estimate

import numpy as np
from dynamic_plotter import *

def next_state(state):

    if state[1] == 0:
        reward = 0
    else:
        reward = 1
    if state[1] == 9:
        new_state = 8
        action = 0
    elif state[1] == 0:
        new_state = 1
        action = 2
    elif state[1]>state[0]:
        new_state = state[1] + 1
        action = 2
    elif state[1]<state[0]:
        new_state = state[1] - 1
        action = 2
    state_p = [state[1], new_state]
    return state_p, action, reward

def gamma(state):
    if state[0] == 0:
        return 0
    return 1

def feature_vector(state):
    fvector = np.zeros(18)
    if state[1] == 9:
        fvector[state[1]] = 1.0
    elif state[1] == 0:
        fvector[state[1]] = 1.0
    elif state[1]>state[0]:
        fvector[18-state[1]] = 1.0
    elif state[1]<state[0]:
        fvector[state[1]] = 1.0

    return fvector

def plot_vec(state):
    if state[1] == 9:
        return state[1]
    elif state[1] == 0:
        return state[1]
    elif state[1]>state[0]:
        return 18-state[1]
    elif state[1]<state[0]:
        return state[1]
def experiment_on_policy():

    plotting = True
    if plotting:
        d = DynamicPlot(window_x = 100, title = 'On-Policy Predictions', xlabel = 'Time_Step', ylabel= 'Value')
        d.add_line('Prediction')
        d.add_line('State')

    # init problem
    num_state = 18
    num_action = 3


    # TODO: divide by the number of active features in the feature vector!
    alpha = 0.7/1
    lam = 0.9

    # init state, action, and time step
    state = [4,5]
    action = None
    t = 0
    e = np.zeros(num_state)

    # init the solution
    theta = np.zeros(num_state) # weights for the TD learning

    # GTD lambda algorithm main loop
    while True:

        state_prime, action, reward = next_state(state)

        # gammma state or state_prime?
        delta = reward + gamma(state_prime) * (np.dot(theta.T,feature_vector(state_prime))) - (np.dot(theta.T,feature_vector(state)))
        e =  feature_vector(state) + gamma(state) * lam * e
        theta = theta + alpha * (np.dot(delta,e))

        if t % 100 == 0:
            time_step_str = "time_step: "+str(t)+", prediction: "
            print(time_step_str+str(np.around(theta, decimals=3)))

        # go to the next step

        if plotting:
            d.update(t, [theta[plot_vec(state)],plot_vec(state)])
        state = state_prime
        t += 1

experiment_on_policy()
