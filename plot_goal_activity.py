import numpy as np
import matplotlib.pyplot as plt
from models import UnintuitiveCircuit, MinimalCircuit
from util import recover_encoded_angle
from anti_models import AntiRuleOne

if __name__ == "__main__":
    res = np.load('DICE_result.pkl', allow_pickle=True)

    control = MinimalCircuit(n=3)

    model = UnintuitiveCircuit(x=res.x, print_info=True)
    model = AntiRuleOne()

    samples = 100
    angles = np.linspace(0, 2*np.pi, samples)

    control_activities = np.zeros((samples, 3))
    model_activities = np.zeros((samples, 3))
    control_strings = []
    model_strings = []

    control_angles = []
    model_angles = []
    

    for idx in range(len(angles)):
        a = angles[idx]
        control.update(a, 0)
        model.update(a, 0)

        
        control_activities[idx, :] = control.C #.reshape(3,1)
        model_activities[idx, :] = model.C #.reshape(3,1)

        control_string = f"{control.C[0]}:{control.C[1]}:{control.C[2]}"
        model_string = f"{model.C[0]}:{model.C[1]}:{model.C[2]}"

        control_strings.append(control_string)
        model_strings.append(model_string)

        control_angles.append(recover_encoded_angle(control.C, control.C_prefs))
        model_angles.append(recover_encoded_angle(model.C, model.C_prefs))

    control_set = set(control_strings)
    model_set = set(model_strings)

    if len(control_set) == len(control_strings):
        print("CONTROLS UNIQUE")
    else:
        print("FAILURE: CONTROLS NOT UNIQUE")

    if len(model_set) == len(model_strings):
        print("MODEL OUTPUTS UNIQUE")
    else:
        print("FAILURE: MODEL OUTPUTS NOT UNIQUE")

    print("")

    print(len(model_angles) == len(set(model_angles)))

    mosaic = [['control'],
              ['c_phi'],
              ['model'],
              ['m_phi']]
    fig, axs = plt.subplot_mosaic(mosaic)

    axs['control'].pcolormesh(control_activities.T, vmax=1, vmin=0)
    axs['c_phi'].plot(angles, np.array(control_angles) % (2*np.pi))
    axs['c_phi'].plot(angles, angles)
    axs['model'].pcolormesh(model_activities.T, vmax=1, vmin=0)
    axs['m_phi'].plot(angles, model_angles)

    plt.show()
    