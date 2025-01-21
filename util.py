"""
util.py

A selection of utilities used by other modules. 
"""

import numpy as np
import matplotlib.pyplot as plt

from models import MinimalCircuit

def C_tuning():
    """
    Visualise compass population tuning for uniform circuits. Best used from 
    python REPL.
    """
    model = MinimalCircuit(n=3)

    angles = np.linspace(0, 2*np.pi, 1000)

    compass = []
    for a in angles:
        model.update(a, 0)
        compass.append(model.C)

    compass = np.array(compass).T

    print(np.max(compass))
    print(np.min(compass))

    plt.subplot(111)
    plt.pcolormesh(compass)
    plt.show()

def G_tuning():
    """
    Visualise goal population tuning for uniform circuits. Best used from 
    python REPL.
    """
    model = MinimalCircuit(n=3)

    angles = np.linspace(0, 2*np.pi, 1000)

    goal = []
    for a in angles:
        model.update(0, a)
        goal.append(model.G)

    goal = np.array(goal).T

    print(np.max(goal))
    print(np.min(goal))

    plt.subplot(111)
    plt.pcolormesh(goal)
    plt.show()

def S_tuning():
    """
    Visualise steering population tuning for uniform circuits. Best used from 
    python REPL.
    """
    model = MinimalCircuit(n=3)
    n = 1000
    angles = np.linspace(0, 2*np.pi, n)

    S_L = []
    S_R = []
    motor = []

    for a in angles:
        steering = model.update(0, a)
        S_L.append(model.S_L)
        S_R.append(model.S_R)
        motor.append(steering)

    S_L = np.array(S_L).T
    S_R = np.array(S_R).T
    
    print(np.max(S_L))
    print(np.min(S_L))
    print(np.max(S_R))
    print(np.min(S_R))

    plt.subplot(311)
    plt.pcolormesh(S_L)
    plt.subplot(312)
    plt.pcolormesh(S_R)
    plt.subplot(313)
    plt.plot(range(n), motor)

    plt.show()    

def generate_track(length=1000, bias=0, variance=0.872, random_state=None):
    """
    Generate a von Mises random walk with a given number of steps, bias, and variance.

    :param length: The number of steps in the walk.
    :param bias: The centre of the von Mises distribution.
    :param variance: The variance of the distribution (note, NOT the concentration).
    :param random_state: A numpy RandomState object may be passed in in order to
                         use a persistent random number generator over multiple
                         functions.
    :return: An array containing each heading update in the random walk. 
    """
    if random_state == None:
        random_state = np.random.RandomState(614354)

    
    kappa = 1 / variance
    changes_per_step = random_state.vonmises(bias, kappa, size=(length - 1))

    # Always start at zero
    heading = 0
    headings = [heading]
    
    for t in range(len(changes_per_step)):
        change = changes_per_step[t]
        heading += change
        headings.append(heading)

    headings = np.array(headings) % (2*np.pi)
    
    # Return the headings
    return headings

def signed_angle(a,b):
    """
    Method to compute the signed angle between two vectors a and b using the
    perp dot product.

    If the returned angle is greater than 0, b lies to the left of a. If
    the returned angle is less than 0, then b lies to the right of a. If 
    the vectors point in opposite directions, then then plus or minus pi
    may be returned.

    :param a: The first vector.
    :param b: The second vector.
    :return: The signed angle between the two.
    """
    complex_dot = lambda x,y: np.real(x)*np.real(y) + np.imag(x)*np.imag(y)

    a_vec = np.exp(1j * a)
    b_vec = np.exp(1j * b)

    a_perp = np.imag(a_vec) - np.real(a_vec)*1j

    a_dot_b = complex_dot(a_vec, b_vec)
    ap_dot_b = complex_dot(a_perp, b_vec)

    eta = np.arctan2(ap_dot_b, a_dot_b)

    # if eta > 0, b lies left of a, and if eta < 0, b lies right of a.
    return eta
