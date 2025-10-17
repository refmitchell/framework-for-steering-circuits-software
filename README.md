# Software repository for "A framework for constructing insect steering circuits".

This repository contains all of the model construction, simulation, and optimisation code for the manuscript "A framework for constructing insect steering circuits".


## Useful files
- `models.py` contains model definitions. 
- `anti_models.py` contains rule-breaking model definitions
- `sim.py` contains the random walk simulator.
- `optimisation.py` contains all code relating to the differential evolution process used to tune our unintuitive steering circuit example.

## Running our code
All code is written in Python. The simulations depend on NumPy and Matplotlib, 
both of which can be installed via pip. The optimisation procedure depends on 
SciPy (which can also be installed via pip). 

To reproduce our random walk simulations, simply run:

```$ python sim.py```

This will produce three files `model_tracks.png`, `antimodel_tracks.png`, and `stepped_model_tracks.png` in the plots subdirectory.
The first two are the random walk simulations for the rule-following and rule-breaking circuits respectively, the last is the continuous turn simulation.
It is also possible to modify these simulations by modifying the parameters in the code. 

## Contact
Any queries about the code or manuscript are welcome. Please use the correspondence address in the manuscript or contact me here directly.



