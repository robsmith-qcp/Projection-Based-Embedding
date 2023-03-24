import presses
import pyscf
import pytest
import sys

def test():
    ref = -432.338653440287
    keywords = {}
    keywords['scf_method'] = 'dft' # If dft is chosen, the user should specify the desired exchange-correlation functional
    keywords['xc'] = 'pbe'
    keywords['subsystem_method'] = 'mp2'
    keywords['n_shells'] = 0 # must be some integer
    keywords['atom'] = '''
                       N       -0.3818881407      4.1611114168     -0.6408229443                 
                       H        0.0699951057      3.4754452670     -1.2886849830                 
                       H       -0.1194116577      5.1199130696     -0.9651863526                 
                       H       -2.1605330287      4.1581746977     -1.7845734870                 
                       H       -2.3101254082      4.8148850590     -0.1094359857                 
                       C       -1.8327830923      4.0250384094     -0.7295400346                 
                       C       -2.2826334994      2.6570994059     -0.2184951135                 
                       F       -3.6603696407      2.5672978329     -0.2949960837                 
                       F       -1.7246833805      1.6510129009     -0.9870457636                 
                       F       -1.8932387186      2.4939728064      1.0989345902 
                       '''
    keywords['active_space_atoms'] = 6 # The embedded subsystem is developed from the AOs centered on the first 6 atoms in the above string
    keywords['basis'] = 'aug-cc-pvdz'
    keywords['spin'] = 0 # in PySCF this is the number of unpaired electrons, not 2s+1
    keywords['charge'] = 0

    e, mf, ec = presses.run_embed(keywords)
    # A lower threshold is set here to match the base PBE results between Psi4 and PySCF
    # To Do: resolve the discrepancy and tighten the threshold.
    assert(abs(ref - e) < 1e-5)

if __name__== "__main__":
    test()
