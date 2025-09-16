"""
test.py

Testing script used for debugging/inspecting the results of the optimisation 
procedure.

As currently configured, running this will plot the steering outputs from
the MP2024 model along with the steering output from our optimised unintuitive
circuit. These are the heatmaps shown in the figure schematising our optimisation
procedure.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["svg.fonttype"] = "none"

from models import UnintuitiveCircuit, MP2024
from anti_models import AntiRuleOne


res = np.load("DICE_result.pkl", allow_pickle=True)
x = res.x


samples = 10
umodel = UnintuitiveCircuit(x, print_info=True)
umodel = AntiRuleOne()

model = MP2024()

x = np.linspace(0, 2*np.pi, samples)
headings, goals = np.meshgrid(x,x)

model_outputs = np.zeros((samples,samples))
umodel_outputs = np.zeros((samples,samples))


indices = np.arange(0,samples,1)
for x in indices:
    for y in indices:
        model_outputs[x,y] = model.update(headings[x,y], goals[x,y])
        umodel_outputs[x,y] = umodel.update(headings[x,y], goals[x,y])
        

# G_pref_x = [np.cos(x) for x in umodel.G_prefs]
# G_pref_y = [np.sin(y) for y in umodel.G_prefs]

# gsl_pref_x = [np.real(x) for x in umodel.GSL_pref_vecs]
# gsl_pref_y = [np.imag(x) for x in umodel.GSL_pref_vecs]

# gsr_pref_x = [np.real(x) for x in umodel.GSR_pref_vecs]
# gsr_pref_y = [np.imag(x) for x in umodel.GSR_pref_vecs]

# g0_x = [gsl_pref_x[0], gsr_pref_x[0]]
# g0_y = [gsl_pref_y[0], gsr_pref_y[0]]

# g1_x = [gsl_pref_x[1], gsr_pref_x[1]]
# g1_y = [gsl_pref_y[1], gsr_pref_y[1]]

# g2_x = [gsl_pref_x[2], gsr_pref_x[2]]
# g2_y = [gsl_pref_y[2], gsr_pref_y[2]]


plt.subplot(211)
pcol = plt.pcolormesh(model_outputs)
pcol.set_edgecolor('face')

plt.xticks([0, 10], labels=["$0\degree$", "$360\degree$"])
plt.yticks([0, 10], labels=["$0\degree$", "$360\degree$"])
plt.title("MP2024 steering output")
plt.ylabel("heading")
plt.xlabel("goal")

plt.gca().set_aspect('equal')

plt.subplot(212)

pcol = plt.pcolormesh(umodel_outputs)
pcol.set_edgecolor('face')

plt.xticks([0, 10], labels=["$0\degree$", "$360\degree$"])
plt.yticks([0, 10], labels=["$0\degree$", "$360\degree$"])
plt.title("Unintuitive($\mathbf{x}$) steering output")
plt.ylabel("heading")
plt.xlabel("goal")

plt.gca().set_aspect('equal')

plt.tight_layout()
plt.show()
