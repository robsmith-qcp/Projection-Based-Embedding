PrESSES
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/presses/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/presses/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/PrESSES/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/PrESSES/branch/main)


projection-based embedding for selected spin states with singular value informed subsystem partitioning

### Copyright

Copyright (c) 2023, Robert L. Smith


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.

This package requires the following:
  - numpy
  - scipy
  - pyscf
  - h5py

### Installation
1. Download
    
        git clone https://github.com/robs-qcp
        cd presses/

2. create virtual environment (optional)
         
        virtualenv -p python3 venv
        source venv/bin/activate

3. Install

        pip install .

4. run tests
    
        pytest test/*.py
