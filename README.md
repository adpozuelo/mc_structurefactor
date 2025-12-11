# MC_StructureFactor

## Monte Carlo Structure Factor

Monte Carlo Structure Factor is software that, from a given configuration, calculates the average structure factor.
After calculating the structure factor, using the Monte Carlo method, this software optimizes the pair coefficients of the given potential to find their original values. As a cost function, the mean square error minimization with respect to the previously generated structure factor is used.

## Requisites

- [ ] [Python](https://www.python.org/) >= 3.6 version with [Numpy](https://numpy.org/) and [Matplotlib](https://matplotlib.org/) support.
- [ ] [LAMMPS](https://docs.lammps.org/Manual.html) with [GPU](https://docs.lammps.org/Speed_gpu.html) and [Python](https://docs.lammps.org/Python_install.html) support.
- [ ] [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk).

## Installation and running

Installation (e.g. Lennard Jones):

```
git clone https://git.csic.es/smcm/mc_structurefactor.git
cd mc_structurefactor/lennard_jones
make
```

Generating base structure factor:

```
python3 main_sf.py
```

Optimizing the pair coefficients of the potential:

```
python3 main_sfmc.py
```

## Visualization of results

```
python3 plot_results.py
```