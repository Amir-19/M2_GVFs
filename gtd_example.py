# first test on GTD(lambda)
# states from 0 -> 9
# the bhv policy is choosing random action (up(2)-stay(1)-down(0)) with equalprob
# the target policy is learning how many time steps is to go to state (0) if we are always going down
# we use GTD lambda to learn this off-policy estimate

# TODO: plot the second set of weights! (magnitude of TD and correction and their direction shows many things :p)

import numpy as np
from dynamic_plotter import *

def next_state(state):

    a = np.random.choice([0,1,2])
    if state == 0:
        reward = 0
    else:
        reward = 1
    new_state = state + a - 1
    if (new_state > 9):
        new_state = 9
    elif (new_state < 0):
        new_state = 0
    return new_state, a, reward

def gamma(state):
    if state == 0:
        return 0
    return 1

def feature_vector(state):
    fvector = np.zeros(10)
    fvector[state] = 1.0
    return fvector


def experiment_off_policy():

    plotting = True
    if plotting:
        d = DynamicPlot(window_x = 100, title = 'Off-Policy Predictions', xlabel = 'Time_Step', ylabel= 'Value')
        d.add_line('Prediction')
        d.add_line('State')

    # init problem
    num_state = 10
    num_action = 3


    # TODO: divide by the number of active features in the feature vector!
    alpha = 0.1/1
    beta = 0.01/1
    lam = 0.9

    # init state, action, and time step
    state = 5
    action = None
    t = 0
    e = np.zeros(num_state)

    # init the solution
    theta = np.zeros(num_state) # weights for the TD learning
    w = np.zeros(num_state) # second set of weights for GTD

    # GTD lambda algorithm main loop
    while True:

        state_prime, action, reward = next_state(state)

        # gammma state or state_prime?
        delta = reward + gamma(state_prime) * (np.dot(theta.T,feature_vector(state_prime))) - (np.dot(theta.T,feature_vector(state)))

        if (action == 0):
            rho = 3
        else:
            rho = 0

        e = rho * (feature_vector(state) + gamma(state) * lam * e )
        theta = theta + alpha * (np.dot(delta,e) - gamma(state_prime)*(1-lam)*np.dot(e.T,w)*feature_vector(state_prime))
        w = w + beta * (np.dot(delta,e) - np.dot(feature_vector(state).T,w)*feature_vector(state))

        if t % 100 == 0:
            print(np.around(theta, decimals=3))

        # go to the next step

        if plotting:
            d.update(t, [theta[state],state])
        else:
            if t % 100 == 0:
                print(t)
        state = state_prime
        t += 1

experiment_off_policy()