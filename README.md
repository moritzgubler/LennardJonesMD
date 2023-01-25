With this package you can figure out the velocity needed for some particles to pass through several walls with variable thicknesses. The particles interact with the lennard jones potential.
This package consists of three programs: runSimulation, makePlot and makeMovie. Use the -h option of each program to figure out how it works. The runSimulation program needs an input file. You can find an example input file 
in the example directory of this repository.


Installation:

Dependencies: 
- Python@3.10
- ffmpeg to make the movies (can be installed with conda)

Clone this repository:

- ```git clone https://github.com/moritzgubler/LennardJonesMD.git```
- go into the LennardJones directory and execute ```pip install .```. Alternatively you can clone and install the package directly with pip: ```pip install git+https://github.com/moritzgubler/LennardJonesMD.git```

