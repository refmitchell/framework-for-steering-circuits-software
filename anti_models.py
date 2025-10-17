"""
anti_models.py

This file contains model definitions for models which break specific rules
in isolation, in order to demonstrate that they do not work

Only rules 1, 4, 5, and 6 are actually breakable. Rules 2 and 3 may not be 
broken.

"""

import numpy as np
from models import act

class AntiCircuit():
    def update(self, heading, goal):
        """
        Circuit update function. Given an heading and a goal, neural activities
        will be generated in the compass and goal populations and a scaled
        steering output will be returned.

        :param heading: The current heading of the agent in radians.
        :param goal: The current goal of the agent in radians.
        :return: Steering output (scaled difference between steering populations).
        """
        self.C = np.array([act(np.cos(heading - cp)) for cp in self.C_prefs])
        self.G = np.array([act(np.cos(goal - gp)) for gp in self.G_prefs])

        self.S_L = np.dot(self.W_csl, self.C) + np.dot(self.W_gs, self.G) 
        self.S_R = np.dot(self.W_csr, self.C) + np.dot(self.W_gs, self.G)
        
        self.S_R = [act(x) for x in self.S_R]
        self.S_L = [act(x) for x in self.S_L]

        output =  sum(self.S_R) - sum(self.S_L) 

        return 2*output

class AntiRuleOne(AntiCircuit):
    """
    Provides a model in which compass pfds do not form a positve basis.
    """
    def __init__(self):
        self.id = "Rule 1"
        self.C_prefs = np.radians(np.array([0, 120, 60]))        
        self.G_prefs = np.radians(np.array([60, 270, 210]))

        self.C = np.array((3,1))
        self.S_L = np.array((3,1))
        self.S_R = np.array((3,1))
        self.G = np.array((3,1))

        # Weights from C to S 
        d = 0.2
        self.W_csl = np.eye(3) * d
        self.W_csr = np.roll(np.eye(3), -1, axis=0) * d

        # Weights from G to S
        self.W_gs = np.eye(3)

class AntiRuleOneAndFive(AntiCircuit):
    """
    Provides a model in which compass pfds do not form a positve basis.
    """
    def __init__(self):
        self.id = "1 and 5"
        self.C_prefs = np.radians(np.array([0, 60, 120]))        
        self.G_prefs = np.radians(np.array([30, 90, 180]))

        self.C = np.array((3,1))
        self.S_L = np.array((3,1))
        self.S_R = np.array((3,1))
        self.G = np.array((3,1))

        # Weights from C to S 
        d = 0.2
        self.W_csl = np.eye(3) * d
        self.W_csr = np.roll(np.eye(3), -1, axis=0) * d

        # Weights from G to S
        self.W_gs = np.eye(3)        


class AntiRuleFour(AntiCircuit):
    """
    Provides a circuit in which goal neuron pfds are not coupled to their
    innervated steering neurons.
    """
    def __init__(self):
        self.id = "Rule 4"
        self.C_prefs = np.radians(np.array([240, 0, 120]))
        self.G_prefs = np.radians(np.array([120, 240, 0]))

        self.C = np.array((3,1))
        self.S_L = np.array((3,1))
        self.S_R = np.array((3,1))
        self.G = np.array((3,1))

        # Weights from C to S 
        d = 0.2
        self.W_csl = np.eye(3) * d
        self.W_csr = np.roll(np.eye(3), -1, axis=0) * d

        # Weights from G to S
        self.W_gs = np.eye(3)


class AntiRuleFive():
    """
    Provides a circuit in which goal neurons do not form a positive basis.
    """
    def __init__(self):
        self.id = "Rule 5"
        self.C_prefs = np.radians(np.array([240, 0, 120]))
        self.G_prefs = np.radians(np.array([-60, 0, 180]))

        self.C = np.array((3,1))
        self.S_L = np.array((3,1))
        self.S_R = np.array((3,1))
        self.G = np.array((3,1))

        # Weights from C to S 
        d = 0.2
        self.W_csl = np.eye(3) * d
        self.W_csr = np.roll(np.eye(3), -1, axis=0) * d

        # Weights from G to S
        self.W_gsl = np.array(
            [[1, 1, 0],
             [0, 0, 0],
             [0, 0, 1]]
        )
        self.W_gsr = np.eye(3)
    
    def update(self, heading, goal):
        """
        Circuit update function. Given an heading and a goal, neural activities
        will be generated in the compass and goal populations and a scaled
        steering output will be returned. This circuit has its own implementation
        as it has separate gsl and gsr connection matrices.

        :param heading: The current heading of the agent in radians.
        :param goal: The current goal of the agent in radians.
        :return: Steering output (scaled difference between steering populations).
        """
        self.C = np.array([act(np.cos(heading - cp)) for cp in self.C_prefs])
        self.G = np.array([act(np.cos(goal - gp)) for gp in self.G_prefs])

        self.S_L = np.dot(self.W_csl, self.C) + np.dot(self.W_gsl, self.G) 
        self.S_R = np.dot(self.W_csr, self.C) + np.dot(self.W_gsr, self.G)
        
        self.S_R = [act(x) for x in self.S_R]
        self.S_L = [act(x) for x in self.S_L]

        output =  sum(self.S_R) - sum(self.S_L) 

        return 2*output

class AntiRuleSix(AntiCircuit):
    """
    Provides a circuit in which the left/right relationship of steering pairs
    is mixed for some pairs. 
    """
    def __init__(self):
        self.id = "Rule 6"
        self.C_prefs = np.radians(np.array([-90, -45, 0, 45, 90, 135, 180, -135]))
        self.G_prefs = np.radians(np.array([-67.5, -22.5, 157.5, 67.5, 112.5, -67.5, -22.5, -112.5]))



        self.C = np.array((8,1))
        self.S_L = np.array((8,1))
        self.S_R = np.array((8,1))
        self.G = np.array((8,1))

        # Weights from C to S 
        d = 0.2
        self.W_csl = np.eye(8) * d
        self.W_csr = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
             ]
        )

        # Weights from G to S
        self.W_gs = np.eye(8)    

class TwoNeuron(AntiCircuit):
    """
    [Legacy] Experimenting with a hand-made two-neuron model 'minimal' model.
    """
    def __init__(self):
        self.id = "TwoNeuron"
        self.C_prefs = np.radians(np.array([0, 120]))        
        self.G_prefs = np.radians(np.array([60, 240]))

        self.C = np.array((2,1))
        self.S_L = np.array((2,1))
        self.S_R = np.array((2,1))
        self.G = np.array((2,1))

        # Weights from C to S 
        d = 0.2
        self.W_csl = np.eye(2) * d
        self.W_csr = np.roll(np.eye(2), -1, axis=0) * d

        # Weights from G to S
        self.W_gs = np.eye(2)


class TwoNeuronOpt(AntiCircuit):
    """
    [Legacy] Experimenting with a two-neuron model 'minimal' model.
    """
    def __init__(self, x, print_info=False):
        self.id = "TwoNeuronOpt"
        self.C_prefs = np.radians(np.array([x[0], x[1]]))        
        # self.G_prefs = np.radians(np.array([x[2], x[3]]))

        self.C = np.array((2,1))
        self.S_L = np.array((2,1))
        self.S_R = np.array((2,1))
        self.G = np.array((2,1))

        # Weights from C to S 
        # d = 0.2
        self.W_csl = np.array([[x[2], x[3]], 
                               [x[4], x[5]]])
        self.W_csr = np.array([[x[6], x[7]], 
                               [x[8], x[9]]])

        # Weights from G to S
        self.W_gsl =  np.array([[x[10], x[11]], 
                                [x[12], x[13]]])
        self.W_gsr =  np.array([[x[14], x[15]], 
                                [x[16], x[17]]])
        self.R_w = 0.5
        
        self.C_pref_vecs = np.array([np.exp(x*1j) for x in self.C_prefs])
        # Infer steering neuron preferred vectors and normalise
        self.SL_pref_vecs = self.W_csl @ self.C_pref_vecs
        self.SR_pref_vecs = self.W_csr @ self.C_pref_vecs
        self.SL_pref_vecs = np.array([np.exp(1j*np.angle(x)) for x in self.SL_pref_vecs])
        self.SR_pref_vecs = np.array([np.exp(1j*np.angle(x)) for x in self.SR_pref_vecs])

        # Work out left/right goal neuron preferred angles from goal-steering innervation
        # and normalise. True goal is the midpoint of the left/right goals.
        self.GSL_pref_vecs = self.W_gsl.T @ self.SL_pref_vecs
        self.GSR_pref_vecs = self.W_gsr.T @ self.SR_pref_vecs
        self.GSL_pref_vecs = np.array([np.exp(1j*np.angle(x)) for x in self.GSL_pref_vecs])
        self.GSR_pref_vecs = np.array([np.exp(1j*np.angle(x)) for x in self.GSR_pref_vecs])
        
        # Compute perp dot product of left/right goal vectors to get signed
        # angle between them. If +ve, we use the midpoint of the outer angle,
        # if -ve, use the midpoint of the inner angle (because this is how
        # the agent will actually steer)

        # Left dot right goals
        l_dot_r = [ 
            (np.real(a)*np.real(b)) + (np.imag(a)*np.imag(b)) 
            for (a,b) in zip(self.GSL_pref_vecs, self.GSR_pref_vecs)
        ]

        # Compute vectors orthogonal to those in the left goal population
        l_perp = [ -np.imag(a) + 1j*np.real(a) for a in self.GSL_pref_vecs ]

        # L perp dot right goals
        l_perp_dot_r = [ 
            (np.real(a)*np.real(b)) + (np.imag(a)*np.imag(b)) 
            for (a,b) in zip(l_perp, self.GSR_pref_vecs)
        ]

        # Project the R vector onto L and L perp to give the projection into
        # a pseudo x and y axis, using atan2 to give the signed angle.
        signed_angles = [
            np.arctan2(y,x) 
            for (x,y) in zip(l_dot_r, l_perp_dot_r)
        ]

        # A -ve sign means that R lies below the pseudo x axis given by L, and
        # R lies to the right of L. A +ve sign means that R lies above the pseudo
        # x axis given by L and R lies to the left of L, meaning we need to add
        # pi to get the true midpoint of the steering neurons (rather than just
        # the midpoint of the inner angle of the two vectors).

        # Correct result        
        directed_midpoint =\
            lambda p, s: p + (s*self.R_w) if s > 0 else p - (np.pi - s*(1 - self.R_w))
        

        self.G_prefs = [
            directed_midpoint(p,s) 
            for (p, s) 
            in zip(np.angle(self.GSL_pref_vecs), signed_angles)
            ]
        
        self.G_prefs = [x % (2*np.pi) for x in self.G_prefs]
        if print_info:
            # For diagnostics only
            self.SL_prefs = np.angle(self.SL_pref_vecs)
            self.SR_prefs = np.angle(self.SR_pref_vecs)

            self.GSL_prefs = np.angle(self.GSL_pref_vecs)
            self.GSR_prefs = np.angle(self.GSR_pref_vecs)

            print("Compass prefs: {}".format(np.degrees(self.C_prefs)))
            print("SL prefs: {}".format(np.degrees(self.SL_prefs)))
            print("SR prefs: {}".format(np.degrees(self.SR_prefs)))

            print("GSL prefs: {}".format(np.degrees(np.angle(self.GSL_pref_vecs))))                
            print("GSR prefs: {}".format(np.degrees(np.angle(self.GSR_pref_vecs))))                
            print("G prefs: {}".format(np.degrees(self.G_prefs)))
            print("")      
            print("R_w: {}".format(self.R_w))
            print("")

    
    def update(self, heading, goal):
        """
        Circuit update function. Given an heading and a goal, neural activities
        will be generated in the compass and goal populations and a scaled
        steering output will be returned.

        :param heading: The current heading of the agent in radians.
        :param goal: The current goal of the agent in radians.
        :return: Steering output (scaled difference between steering populations).
        """
        # compass update
        self.C = np.array([act(np.cos(heading - cp)) for cp in self.C_prefs])
        self.G = np.array([act(np.cos(goal - gp)) for gp in self.G_prefs])

        self.S_L = np.dot(self.W_csl, self.C) + np.dot(self.W_gsl, self.G) 
        self.S_R = np.dot(self.W_csr, self.C) + np.dot(self.W_gsr, self.G)
        
        self.S_R = [act(x) for x in self.S_R]
        self.S_L = [act(x) for x in self.S_L]

        # Corrected steering rule
        output = ((1-self.R_w)*sum(self.S_L)) - (self.R_w*sum(self.S_R))
        return output