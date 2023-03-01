PrESSES
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/presses/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/presses/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/PrESSES/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/PrESSES/branch/main)


projection-based embedding for selected spin states with singular value informed subsystem partitioning

### Installation
1. Download
    
        git clone https://github.com/robsmith-qcp/PRESSES.git
        cd presses/

2. create conda environment (optional)
         
        conda create --name presses python==3.9.4

3. Install

        pip install .

4. run tests
    
        pytest

### Running
This program takes a dictionary of keywords related to the embedding calculation the user would like to run.

Required keywords are the scf_method, subsystem_method, atom, active_space_atoms, and basis. 

The scf_method should either be hf (Hartree-Fock) or dft (density functional theory). If dft is chosen, the user should choose an appropriate xc keyword.

The subsystem_method can currenty be chosen from hf, dft, mp2, ccsd, ccsd(t), and eom-ccsd.

The choice of active_space_atoms should indicate the number of atoms the user wants to include in the embedded subsystem (N).

The basis can be any available in PySCF (or user defined as in PySCF's documentation) and atom is the Cartesian coordiantes of each atom in the system being studied with the active_space_atoms chosen from the first N atoms in the string.

An example input can be seen in the example directory.

### Copyright

Copyright (c) 2023, Robert L. Smith


#### Acknowledgements
Code framework is based on work done by Daniel Claudino and Nick Mayhall in the the following articles:
Journal of Chemical Theory and Computation, 15, 2, 1053-1064, (2019): 
[article](https://pubs.acs.org/doi/10.1021/acs.jctc.8b01112)
Journal of Chemical Theory and Computation, 15, 11, 6085-6096, (2019): 
[article](https://pubs.acs.org/doi/10.1021/acs.jctc.9b00682)

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.

This package requires the following:
  - numpy
  - scipy
  - pyscf
  - h5py
