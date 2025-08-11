"""
models.py

This file contains all code relating to model definition and construction. 
Different models are implemented as their own classes. Each model implements
the 'update' function with the same signature to provide a general interface
via which to interact with different models. 

"""

import numpy as np

def act(x):
    """
    Neural activation function
    :param x: The neural input
    :return: Firing rate
    """
    return 1 / (1 + np.exp(-2*(x - 0.6)))
    
class MinimalCircuit():
    """
    A general implementation of 'Uniformly spaced' steering circuits. That is,
    a circuit made up of N compass neurons, N goal neurons, and 2N steering neurons.
    Compass neuron preferred directions are uniformly spaced over 360 degrees. 

    This implementation can take any number N of steering neurons and generate
    a valid circuit.
    """
    def __init__(self, n=3, print_info=False):
        """
        Constructor

        :param n: The number of compass neurons in the circuit (from which other
                  population sizes are derived).
        :param print_info: Flag to print diagnostic circuit information to stdout.
        """
        self.id = "$n$ = {}".format(n)
        self.C = np.zeros((n,1))
        self.S_L = np.zeros((n,1))
        self.S_R = np.zeros((n,1))
        self.G = np.zeros((n,1))

        self.C_prefs = np.linspace(0, 2*np.pi, n, endpoint=False)
        self.C_pref_vecs = np.array([np.exp(x*1j) for x in self.C_prefs])

        # Weights from C to S 
        d = 0.2 # Need to remove dependence on 'weight' when inferring angles
        self.W_csl = np.eye(n) * d
        self.W_csr = np.roll(np.eye(n), -1, axis=0) * d

        self.W_gs = np.eye(n)
    
        self.W_gsl = self.W_gs
        self.W_gsr = self.W_gs

        # Infer goal neuron preferred directions
        self.SL_pref_vecs = self.W_csl @ self.C_pref_vecs
        self.SR_pref_vecs = self.W_csr @ self.C_pref_vecs
        self.GSL_pref_vecs = self.W_gs @ self.SL_pref_vecs
        self.GSR_pref_vecs = self.W_gs @ self.SR_pref_vecs
        self.G_pref_vecs = self.GSL_pref_vecs + self.GSR_pref_vecs

        self.G_prefs = np.angle(self.G_pref_vecs)

        if print_info:
            # For diagnostics only
            self.SL_prefs = np.angle(self.SL_pref_vecs)
            self.SR_prefs = np.angle(self.SR_pref_vecs)
    
            print("Model prefs, N = {}".format(n))
            print("Compass prefs: {}".format(np.degrees(self.C_prefs)))
            print("")
            print("SL prefs: {}".format(np.degrees(self.SL_prefs)))
            print("SR prefs: {}".format(np.degrees(self.SR_prefs)))
            print("")            
            print("G prefs: {}".format(np.degrees(self.G_prefs)))
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
        self.C = np.array([act(np.cos(heading - cp)) for cp in self.C_prefs])
        self.G = np.array([act(np.cos(goal - gp)) for gp in self.G_prefs])

        self.S_L = np.dot(self.W_csl, self.C) + np.dot(self.W_gs, self.G) 
        self.S_R = np.dot(self.W_csr, self.C) + np.dot(self.W_gs, self.G)
        
        self.S_R = [act(x) for x in self.S_R]
        self.S_L = [act(x) for x in self.S_L]

        output =  sum(self.S_R) - sum(self.S_L) 

        return 2*output
    
class UnintuitiveCircuit():
    """
    The 'unintuitive' circuit example from the paper, designed to be used with
    SciPy stochastic optimisation utilities.
    """
    def __init__(self, x, print_info=False):
        """
        Constructor. Generates a circuit with a set of connections, defined by
        the parameter vector x.

        :param x: The parameter vector which includes the various connection 
                  weights, and the steering scale parameter.
        :param print_info: Flag to print diagnostic information.
        """
        self.id = "unintuitive"

        self.C = np.zeros((4,1))
        self.S_L = np.zeros((2,1)) 
        self.S_R = np.zeros((3,1)) 
        self.G = np.zeros((3,1))


        self.C_prefs = np.radians(np.array([350, 10, 90, 200]))
        self.C_pref_vecs = np.array([np.exp(x*1j) for x in self.C_prefs])

        self.W_csl = np.array([[x[0], 0, 0, x[1]],
                               [0, 0, x[2], x[3]]])
        
        self.W_csr = np.array([
            [x[4], x[5], 0, 0],
            [0, 0, x[6], x[7]],
            [x[8], 0, 0, x[9]]]
            )
        
        self.W_gsl = np.array([[x[10], x[11], 0],
                               [x[12], 0, x[13]]])
        self.W_gsr = np.array([[x[14], 0, 0],
                               [0, x[15], 0],
                               [0, x[16], x[17]]])

        self.R_w = x[18]

        #
        # Infer goal neuron preferred directions
        #

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

        # Right dot left goals
        r_dot_l = [ 
            (np.real(a)*np.real(b)) + (np.imag(a)*np.imag(b)) 
            for (a,b) in zip(self.GSR_pref_vecs, self.GSL_pref_vecs)
        ]

        # Compute vectors orthogonal to those in the left goal population
        r_perp = [ -np.imag(a) + 1j*np.real(a) for a in self.GSR_pref_vecs ]

        # L perp dot right goals
        r_perp_dot_l = [ 
            (np.real(a)*np.real(b)) + (np.imag(a)*np.imag(b)) 
            for (a,b) in zip(r_perp, self.GSL_pref_vecs)
        ]

        # Project the R vector onto L and L perp to give the projection into
        # a pseudo x and y axis, using atan2 to give the signed angle.
        signed_angles = [
            np.arctan2(y,x) 
            for (x,y) in zip(r_dot_l, r_perp_dot_l)
        ]

        # A -ve sign means that R lies below the pseudo x axis given by L, and
        # R lies to the right of L. A +ve sign means that R lies above the pseudo
        # x axis given by L and R lies to the left of L, meaning we need to add
        # pi to get the true midpoint of the steering neurons (rather than just
        # the midpoint of the inner angle of the two vectors).
        
        # directed_midpoint =\
        #     lambda p, s: p - (np.pi + (s*self.R_w)) if s > 0 else p - (s*self.R_w)
        
        directed_midpoint =\
            lambda p, s: p + (s*self.R_w) if s > 0 else p - (np.pi + (s*self.R_w))

        self.G_prefs = [
            directed_midpoint(p,s) 
            for (p, s) 
            in zip(np.angle(self.GSR_pref_vecs), signed_angles)
            ]
        
        self.G_prefs = [x % (2*np.pi) for x in self.G_prefs]

        if print_info:
            # For diagnostics only
            self.SL_prefs = np.angle(self.SL_pref_vecs)
            self.SR_prefs = np.angle(self.SR_pref_vecs)

            self.GSL_prefs = np.angle(self.GSL_pref_vecs)
            self.GSR_prefs = np.angle(self.GSR_pref_vecs)

            print("Compass prefs: {}".format(np.degrees(self.C_prefs)))
            print("")
            print("SL prefs: {}".format(np.degrees(self.SL_prefs)))
            print("SR prefs: {}".format(np.degrees(self.SR_prefs)))
            print("")
            print("GSL prefs: {}".format(np.degrees(np.angle(self.GSL_pref_vecs))))                
            print("GSR prefs: {}".format(np.degrees(np.angle(self.GSR_pref_vecs))))
            print("")            
            print("G prefs: {}".format(np.degrees(self.G_prefs)))
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
        
        self.S_R = np.array([act(x) for x in self.S_R])
        self.S_L = np.array([act(x) for x in self.S_L])

        output = (self.R_w*sum(self.S_R)) - ((1-self.R_w)*sum(self.S_L)) 

        # Sim heading update is 
        # agent_heading -= steering
        # so S_R drives left, S_L drives right

        return output
    
    def update_just_heading(self, heading):
        """
        Update based on just a heading change, assume goal neuron activity remains
        constant

        :param heading: The current heading of the agent in radians.
        :return: Steering output (scaled difference between steering populations).
        """
        # compass update
        self.C = np.array([act(np.cos(heading - cp)) for cp in self.C_prefs])
        self.G = self.G.reshape(3)

        self.S_L = np.dot(self.W_csl, self.C) + np.dot(self.W_gsl, self.G) 
        self.S_R = np.dot(self.W_csr, self.C) + np.dot(self.W_gsr, self.G)
        
        self.S_R = np.array([act(x) for x in self.S_R])
        self.S_L = np.array([act(x) for x in self.S_L])


        output = (self.R_w*sum(self.S_R)) - ((1-self.R_w)*sum(self.S_L)) 

        # Sim heading update is 
        # agent_heading -= steering
        # so S_R drives left, S_L drives right

        return output
    
    def zero_neurons(self):
        """
        Reset all neurons to have no activity
        """
        self.C.fill(0)
        self.S_L.fill(0)
        self.S_R.fill(0)
        self.G.fill(0)

class MP2024():
    """
    An implementation of the anatomical circuit model from [1], used as a point
    of comparison for alternative model circuits.

    References:
    [1] - Peter Mussells Pires, Lingwei Zhang, Victoria Parache, LF Abbott, 
          and Gaby Maimon. Converting an allocentric goal into an egocentric 
          steering signal. Nature, pages 1–11, 2024.
    """
    def __init__(self):
        """
        Constructor. This model has a different construction to the others so
        no neural populations are instantiated.
        """
        self.id="MP2024"

    def pfl3_response(self, heading, goal, H_pref, G_pref):
        """
        PFL3 neuron model from [1]. Expression and parameters lifted directly from
        the Methods section. H_pref represents the preferred direction of the compass
        neuron which innervates this PFL3. G_pref represents the preferred direction
        of the copmass neuron which innervates this PFL3.

        :param heading: The agent's current heading.
        :param goal: The current goal.
        :param H_pref: The preferred heading direction for this specific PFL3.
        :param G_pref: The preferred goal direction for this specific PFL3.
        :return: The PFL3 response according to [1].
        """
        f = lambda x: 29.23 * np.log(1 + np.exp(2.17*(x-0.7)))
        return f(np.cos(heading - H_pref) + 0.63*np.cos(goal - G_pref))

    def update(self, heading, goal):
        """
        Model computation from [1], again taken directly from the Methods of
        that paper. All neuron preferred directions were copied by hand from 
        [1]. This implementation returns a scaled steering output such that
        it operates on the same timescale as our own models.

        :param heading: The current heading of the agent.
        :param goal: The current goal of the agent.
        :return: The (scaled) steering output of the model from [1].
        """
        goal_prefs = [15, 35, 75, 105, 135, 165, -165, -135, -105, -75, -45, -15]

        goal_prefs = np.array(goal_prefs)

        pfl3_R_prefs = [67.5, 112.5, 157.5, 157.5, -157.5, -112.5, -112.5, -67.5, -22.5, -22.5, 22.5, 67.5]
        pfl3_L_prefs = [-67.5, -22.5, 22.5, 22.5, 67.5, 112.5, 112.5, 157.5, -157.5, -157.5, -112.5, -67.5]

        # Convert to radians and multiply by -1 to use here
        goal_prefs = -1 * np.radians(goal_prefs)
        pfl3_R_prefs = -1 * np.radians(pfl3_R_prefs)
        pfl3_L_prefs = -1 * np.radians(pfl3_L_prefs)
        
        # PFL3_R responses
        R = sum([self.pfl3_response(heading, goal, p, g) for (p, g) in zip(pfl3_R_prefs, goal_prefs)])
        L = sum([self.pfl3_response(heading, goal, p, g) for (p, g) in zip(pfl3_L_prefs, goal_prefs)])

        # Return difference as steering output
        return 0.00018 * (L - R)


def compute_steering(model, heading=0, degrees=True):
    """
    Compute the steering output function for a given model and heading.
    :param model: The model under test (note: must implement the 'update()'
                  function; this is not enforced).
    :param heading: The current heading to pass to the model.
    :param degrees: Set to true if 'heading' and return values are to be in
                    degrees (default radians).
    :return: The steering outputs and the goals which generated those outputs.                    
    """
    if degrees:
        heading = np.radians(heading)

    goals = np.linspace(0, 2*np.pi, 100)
    outputs = [model.update(heading, g) for g in goals]
    return outputs, goals

# def weight_solver(a1, a2, g, degrees=True):
#     """
#     [Legacy]
#     Goal neuron direction is defined by the neurons innervated by that goal neuron.
#     This function determines the weights required to generate a specific goal
#     direction, given the preferred directions of two innervated steering neurons.

#     [Warning]
#     This function was not used and is untested.

#     """
#     if degrees:
#         a1 = np.radians(a1)
#         a2 = np.radians(a2)
#         g = np.radians(g)


#     def fun(w1):
#         # Vector sum for given weights
#         v = (w1*np.exp(1j*a1) + (1-w1)*np.exp(1j*a2))

#         # Goal and perpendicular goal vector form axes
#         g_vec = np.exp(1j*g)
#         g_perp = np.exp(1j*(g - np.pi/2))
        
#         # Project V into those goal vector axes
#         v = [np.real(v), np.imag(v)]
#         g_vec = [np.real(g_vec), np.imag(g_vec)]
#         g_perp = [np.real(g_perp), np.imag(g_perp)]
#         x = np.dot(v, g_vec)
#         y = np.dot(v, g_perp)

#         # Return signed angle
#         return np.arctan2(y,x)

#     result = root_scalar(fun, bracket=(0.0,1.0))
    
#     return result.root

# def full_steering(model):
#     """
#     Com
#     """
#     x = np.linspace(0, 2*np.pi, 100)
#     x = itertools.combinations_with_replacement(x,2)

#     [model.update(h,g) for (h,g) in x]

# def unintuitive_circuit_error(x, x_target):
#     model = UnintuitiveCircuit(x)
#     model=MP2024()
#     mp_model = MP2024()



#     ui_outputs, ui_goals = compute_steering(model)
#     mp_outputs, mp_goals = compute_steering(mp_model)

#     return metrics.rmse(ui_outputs, mp_outputs)

# def optimisation_callback(res, convergence):
#     print(res)
#     print(convergence)
#     print("RMSE: {}".format(unintuitive_circuit_error(res)))


# if __name__ == "__main__":
#     x = np.zeros(18)
    
#     upper_bound = 1
#     lower_bound = -1

#     bounds = list(zip(np.zeros(18) + lower_bound, np.zeros(18) + upper_bound))
    
#     print(bounds)

#     result = differential_evolution(
#         unintuitive_circuit_error, 
#         bounds,
#         callback=optimisation_callback,
#         maxiter=10000,
#         workers=-1
#         )

#     with open("result.pkl", "wb") as f:
#         pkl.dump(result, f)




    # x = np.array([
    #     0.09907804,
    #     0.05067848,
    #     0.09892858,
    #     0.09723734,
    #     0.00010732,
    #     0.00028387,
    #     0.00247871, 
    #     0.00913901,
    #     0.0006384,
    #     0.00374457,
    #     0.0980949,
    #     0.09934664,
    #     0.09460428,
    #     0.09961601,
    #     0.000223,
    #     0.00119506,
    #     0.00067859,
    #     0.00178532])
    # # a, b, corr = result.x
    
    # plt.subplot(111)
    # model = UnintuitiveCircuit(x)
    # outputs, goals = compute_steering(model)
    # plt.plot(np.degrees(goals), outputs)

    # model = MP2024()
    # outputs, goals = compute_steering(model)
    # plt.plot(np.degrees(goals), outputs)

    # plt.show()

    
    

    
