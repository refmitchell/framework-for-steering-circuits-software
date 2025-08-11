"""
util.py

A selection of utilities used by other modules. 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from models import MinimalCircuit, UnintuitiveCircuit

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

def inner_angle(a,b):
    """
    Find the inner angular difference between two angular values.

    :param a: The first angle.
    :param b: The second angle.
    :return: The inner angle between the two.
    """
    complex_dot = lambda x,y: np.real(x)*np.real(y) + np.imag(x)*np.imag(y)

    a_vec = np.exp(1j * a)
    b_vec = np.exp(1j * b)

    return np.arccos(complex_dot(a_vec, b_vec))


def find_directed_root(f, xmin=0, xmax=(2*np.pi), tolerance=0.01, resolution=100):
    """
    Find the first root where f has a negative gradient. This primitive method
    relies on having a bounded region within which to search. Any value which
    is close enough to the root will be taken. The search on each iteration will
    have a set resolution.
    
    """
    # Sample domain
    xs = np.linspace(xmin, xmax, resolution)
    
    # Compute f within domain
    ys = [f(x) for x in xs]

    # Perform a linear search through the sampled domain
    for idx in range(len(ys) - 1):
        current = ys[idx]
        next = ys[idx + 1]

        # Skip any regions which have the wrong gradient
        if (next > current):
            continue 

        # If there is a root between the current and next results
        # return either bound if they're within tolerance, otherwise
        # iterate within the root-containing region.
        if (next < 0 and current > 0):
            if (abs(current) <= tolerance):
                # Return the xval for next
                return xs[idx]
            elif(abs(next) <= tolerance):
                # Return the xval for current
                return xs[idx + 1]
            else:
                # If neither bound is close enough, repeat the search on a smaller scale
                return find_directed_root(f, xmin=current, xmax=next, tolerance=tolerance, resolution=resolution)

    # If this point is reached then no root exists 
    return None


def compute_goal_directions_by_scanning(model, activation_value=1):
    """
    Given a model, use the scanning method to compute goal neuron directions.

    The routine will sequentially set each goal neuron to a non-zero value and
    scan over all possible compass inputs. This will generate a steering curve.
    The point at which that curve crosses zero is the goal represented by that
    neuron.
    """
    angles = np.linspace(0, 2*np.pi, 100)
    for idx in range(model.G.shape[0]):
        model.zero_neurons()
        model.G[idx] = 1
        
        root = find_directed_root(model.update_just_heading)
        steering_outputs = np.array([model.update_just_heading(a) for a in angles])
        
        plt.subplot(111)
        plt.plot(np.degrees(angles), steering_outputs)
        if (root != None):
            plt.vlines(np.degrees(root), ymin=-0.5, ymax=0.5)
        else:
            print("Root not found!")
        plt.show()
        

        
if __name__ == "__main__":
    res = np.load("DICE_result.pkl", allow_pickle=True)
    x = res.x
    model = UnintuitiveCircuit(x, print_info=True)
    compute_goal_directions_by_scanning(model)

    

    