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