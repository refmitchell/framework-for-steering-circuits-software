# Software repository for "A framework for constructing insect steering circuits".

This repository contains all of the model construction, simulation, and optimisation code for the manuscript "A framework for constructing insect steering circuits".


## Useful files
- `models.py` contains model definitions. 
- `sim.py` contains the random walk simulator.
- `optimisation.py` contains all code relating to the differential evolution process used to tune our unintuitive steering circuit example.

## Running our code
All code is written in Python. The simulations depend on NumPy and Matplotlib, 
both of which can be installed via pip. The optimisation procedure depends on 
SciPy (which can also be installed via pip). 

To reproduce our random walk simulations, simply run:

```$ python sim.py```

This will produce a file `uniform_network_tracks_0.3.png` in the plots subdirectory
which should be identical to the random walk figure in the paper. It is also possible
to modify these simulations by modifying the parameters in the code. 

## Contact
Any queries about the code are welcome. Please use the correspondence address in the manuscript or contact me here directly.



