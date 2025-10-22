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

This will produce six files `model_tracks.svg`, `antimodel_tracks.svg`, `continuous_model_tracks.svg`, `stepped_model_tracks_30deg.svg`, `stepped_model_tracks_60deg.svg`, and `stepped_model_tracks_90deg.svg`in the plots subdirectory.
The first two are the random walk simulations for the rule-following and rule-breaking circuits respectively, the third is the continuous turn simulation.
The final three are step change simulations with step sizes of 30, 60, and 90 degrees respectively. 
These are included as supplementary information in the paper.
It is also possible to modify these simulations by modifying the parameters in the code. 

## Contact
Any queries about the code or manuscript are welcome. Please use the correspondence address in the manuscript or contact me here directly.



