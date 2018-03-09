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
        dv = DynamicPlot(window_x = 100, title = 'On-Policy Verification', xlabel = 'Time_Step', ylabel= 'Value')
        dv.add_line('Prediction')
        dv.add_line('Verifier')

    # storage for verification step
    cumulants = []
    gammas = []
    time_steps = []
    return_time_step = []
    time_time_step = []
    wait_num_time = 100
    prediction_steps = []
    # init problem
    num_state = 18
    num_action = 3


    # TODO: divide by the number of active features in the feature vector!
    alpha = 0.7/1
    lam = 0.9

    # init state, action, and time step
    state = [0,1]
    action = None
    t = 0
    e = np.zeros(num_state)

    # init the solution
    theta = np.zeros(num_state) # weights for the TD learning

    # GTD lambda algorithm main loop
    while True:

        state_prime, action, reward = next_state(state)

        # verifiction step storing
        cumulants.append(reward)
        gammas.append(gamma(state_prime))
        time_steps.append(t)
        if len(cumulants) > wait_num_time:
            cumulants.pop(0)
            gammas.pop(0)
        if len(cumulants) == wait_num_time:
            return_time_step.append(sum([np.product(gammas[:k]) * cumulants[k] for k in range(wait_num_time)]))
            time_time_step.append(time_steps.pop(0))
        # gammma state or state_prime?
        delta = reward + gamma(state_prime) * (np.dot(theta.T,feature_vector(state_prime))) - (np.dot(theta.T,feature_vector(state)))

        e =  feature_vector(state) + gamma(state) * lam * e
        theta = theta + alpha * (np.dot(delta,e))
        prediction_steps.append(theta[plot_vec(state)])

        if t % 100 == 0:
            time_step_str = "time_step: "+str(t)+", prediction: "
            print(time_step_str+str(np.around(theta, decimals=3)))


        if plotting:
            d.update(t, [theta[plot_vec(state)],plot_vec(state)])
            dv_time = t
            dv_pred = 0
            dv_ver = 0
            if t >= wait_num_time - 1:
                dv_pred = prediction_steps.pop(0)
                dv_ver = return_time_step.pop(0)
                time_time_step.pop(0)
            dv.update(t, [dv_pred,dv_ver])
        # go to the next step
        state = state_prime
        t += 1

experiment_on_policy()
