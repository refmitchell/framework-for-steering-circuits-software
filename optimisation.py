"""
optimisation.py

This file contains all code relating to the optimisation procedure used to 
compute suitable weights for the 'unintuitive' circuit example.
"""

from scipy.optimize import differential_evolution

import numpy as np
import pickle as pkl

import models
import anti_models
import metrics

from util import recover_encoded_angle


mp2024_outputs = np.zeros((1,1))
headings = np.zeros((1,1))
goals = np.zeros((1,1))

def precompute_mp2024_outputs(samples=100):
    """
    The outputs from MP2024 are used as our target for optimisation. This function
    precomputes the steering output from that model (see models.py) and stores
    them globally so that they can be repeatedly referenced by the optimisation
    procedure without requiring re-computation.

    The form of the output is a heatmap. Linearly spaced samples from 0-360 degrees
    are computed for heading and goal inputs such that (in essence), the steering output
    for every possible goal is computed across each heading. Increasing the number
    of samples increases the resolution of the heatmap but also increases compute 
    time and comparison time during optimisation. In practice, a 10x10 grid is 
    sufficient to sample the steering output space.

    :param samples: The number of samples to use for heading and goal.
    :return: the headings and goals used, and the 2D output array.
    """
    global mp2024_outputs
    global headings
    global goals

    indices = range(samples)
    x = np.linspace(0, 2*np.pi, samples)
    headings, goals = np.meshgrid(x,x)    

    mp2024_outputs = np.zeros(headings.shape)

    model = models.MP2024()

    for x in indices:
        for y in indices:
            mp2024_outputs[x,y] = model.update(headings[x,y], goals[x,y])

    return headings, goals, mp2024_outputs

def objective(x):
    """
    The objective function used by the optimisation procedure. This function depends
    on the outputs from the MP2024 model and therefore precompute_mp2024_outputs()
    must be called before starting optimisation.

    The parameter x is a parameter vector which is used to define an instance
    of the UnintuitiveCircuit class (see models.py). The steering output from
    this instantiation is then compared to the MP2024 output using the root
    mean squared error (see metrics.py).

    :param x: The parameter vector.
    :return: The root mean squared error when comparing steering output of the
             UnintuitiveCircuit instance against the MP2024 model.
    """
    if headings.shape == (1,1):
        print("Global vars have not been initialised")
        return -1

    if headings.shape != goals.shape:
        print("Heading and goal shapes do not match.")
        return -1
    
    if headings.shape != mp2024_outputs.shape:
        print("Target output shape does not match input shapes.")
        return -1

    indices = range(headings.shape[0])
    
    model = models.UnintuitiveCircuit(x)
    outputs = np.zeros(headings.shape)

    # Steering output check
    for x in indices:
        for y in indices:
            outputs[x,y] = model.update(headings[x,y], goals[x,y])

    # decoding_weight = 0.01
    # steering_weight = (1 - decoding_weight)
    steering_error = metrics.rmse(mp2024_outputs, outputs)

    # Decoding property check
    # angles = np.linspace(0, 2*np.pi, 50)
    # decoded_angles = []
    # for a in angles:
    #     model.update(0, a)
    #     decoded_angles.append(recover_encoded_angle(model.G, model.G_prefs) % (2*np.pi))
    
    # decoded_angles = np.array(decoded_angles)
    # decoding_error = metrics.rmse(angles, decoded_angles)
    # # print("===========")
    # # print(angles)
    # # print(decoded_angles)

    # combined_error = (decoding_weight*decoding_error) + (steering_weight*steering_error)

    return steering_error

def optimisation_callback(res, convergence):
    """
    A callback function used during optimisation to print iterative output to
    stdout. This can then be streamed to a file for logging at the terminal.
    The signature is defined by the optimisation procedure used (see SciPy docs).

    :param res: The current result of the optimisation procedure.
    :param convergence: The degree of convergence of the current iteration.
    """
    print(res)
    print(convergence)
    print("RMSE: {}".format(objective(res)))


if __name__ == "__main__":
    """
    Setup and run the optimisation procedure.
    """

    """
    Change 'samples' parameter to change the resolution of the target
    for the objective function. Increasing will also increase the time
    taken to run the optimisation procedure.
    """
    precompute_mp2024_outputs(samples=50)


    """
    Bounds chosen are somewhat arbitrary. Connection weights (x[0:16]) are 
    allowed to vary from 0 to 2. The steering output weight (x[17]) is allowed
    to vary from 0 to 1. See our paper for our rationale. 
    """
    x = np.zeros(18)
    
    upper_bound = 2
    lower_bound = 0

    bounds = list(zip(np.zeros(18) + lower_bound, np.zeros(18) + upper_bound))
    bounds.append((0.0, 1.0))
    
    print(bounds)

    """
    Run the optimisation procedure (differential evolution). Setting workers to
    -1 will run the optimisation across as many threads as possible which can 
    speed up the process.
    """
    result = differential_evolution(
        objective, 
        bounds,
        callback=optimisation_callback,
        maxiter=10000,
        workers=-1
        )
    
    """
    Output the result to a python 'pickle' file. If the filename is not changed
    then old results will be overwritten.
    """
    with open("DICE_result.pkl", "wb") as f:
        pkl.dump(result, f)




    
    


