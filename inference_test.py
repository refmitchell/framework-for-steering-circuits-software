"""
inference_test.py

Test script for trialling the goal neuron pfd inference procedure.
"""

import numpy as np

from models import MinimalCircuit
from util import inner_angle

def infer_goal_pfds(model):
    """
    Given a model, this function will infer the goal neuron pfds based on the 
    compass pfds and the connection strengths between the various neurons.
    This as been separated out to allow better testing of the goal neuron 
    inference procedure.

    :param model: The neural model
    :return: An array of pfds for the goal neurons, alongside the hard-coded versions
    """

    SL_pref_vecs = model.W_csl @ model.C_pref_vecs
    SR_pref_vecs = model.W_csr @ model.C_pref_vecs
    SL_pref_vecs = np.array([np.exp(1j*np.angle(x)) for x in SL_pref_vecs])
    SR_pref_vecs = np.array([np.exp(1j*np.angle(x)) for x in SR_pref_vecs])

    # Work out left/right goal neuron preferred angles from goal-steering innervation
    # and normalise. True goal is the midpoint of the left/right goals.
    GSL_pref_vecs = model.W_gsl.T @ SL_pref_vecs
    GSR_pref_vecs = model.W_gsr.T @ SR_pref_vecs
    GSL_pref_vecs = np.array([np.exp(1j*np.angle(x)) for x in GSL_pref_vecs])
    GSR_pref_vecs = np.array([np.exp(1j*np.angle(x)) for x in GSR_pref_vecs])
    
    # Compute perp dot product of left/right goal vectors to get signed
    # angle between them. If +ve, we use the midpoint of the outer angle,
    # if -ve, use the midpoint of the inner angle (because this is how
    # the agent will actually steer)

    # Right dot left goals
    r_dot_l = [ 
        (np.real(a)*np.real(b)) + (np.imag(a)*np.imag(b)) 
        for (a,b) in zip(GSR_pref_vecs, GSL_pref_vecs)
    ]

    # Compute vectors orthogonal to those in the left goal population
    r_perp = [ -np.imag(a) + 1j*np.real(a) for a in GSR_pref_vecs ]

    # L perp dot right goals
    r_perp_dot_l = [ 
        (np.real(a)*np.real(b)) + (np.imag(a)*np.imag(b)) 
        for (a,b) in zip(r_perp, GSL_pref_vecs)
    ]

    # Project the R vector onto L and L perp to give the projection into
    # a pseudo x and y axis, using atan2 to give the signed angle.
    signed_angles = [
        np.arctan2(y,x) 
        for (x,y) in zip(r_dot_l, r_perp_dot_l)
    ]

    print(np.degrees(signed_angles))

    # A -ve sign means that R lies below the pseudo x axis given by L, and
    # R lies to the right of L. A +ve sign means that R lies above the pseudo
    # x axis given by L and R lies to the left of L, meaning we need to add
    # pi to get the true midpoint of the steering neurons (rather than just
    # the midpoint of the inner angle of the two vectors).
        
    directed_midpoint =\
        lambda p, s: p + (s*0.5) if s < 0 else p + (s*0.5) + np.pi

    # Old version
    G_prefs = [
        directed_midpoint(p,s) 
        for (p, s) 
        in zip(np.angle(GSR_pref_vecs), signed_angles)
        ]
    
    # G_prefs = [
    #     directed_midpoint(p,s) 
    #     for (p, s) 
    #     in zip(np.zeros(len(signed_angles)), signed_angles)
    #     ]
        
    
    G_prefs = [x % (2*np.pi) for x in G_prefs]
    model_G_prefs = [x % (2*np.pi) for x in model.G_prefs ]

    # Comparisons
    comp = list(zip(G_prefs, model_G_prefs))
    comp_degrees = [(np.degrees(x), np.degrees(y)) for (x,y) in comp]
    diffs = [inner_angle(x,y) for (x,y) in comp]
    diffs_degrees = [np.degrees(inner_angle(x,y)) for (x,y) in comp]

    return comp_degrees, diffs_degrees

if __name__ == "__main__":
    ns = range(3,8)
    for n in ns:
        model = MinimalCircuit(n=n, print_info=False)
        result, diffs = infer_goal_pfds(model)

        print(result)
        print(diffs)
        print("")


