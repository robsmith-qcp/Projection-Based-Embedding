import presses
import pyscf
import pytest
import sys

def test():
    keywords = {}
    keywords['scf_method'] = 'dft' # If dft is chosen, the user should specify the desired exchange-correlation functional
    keywords['xc'] = 'pbe'
    keywords['subsystem_method'] = 'mp2'
    keywords['n_shells'] = 0 # must be some integer
    keywords['split_spade'] = False
    keywords['occupied_shells'] = 1
    keywords['atom'] = '''
                       N       0.0000000000      0.0000000000     0.0000000000              
                       N       0.0000000000      1.1000000000     0.0000000000
                       '''
    keywords['active_space_atoms'] = 1 # The embedded subsystem is developed from the AOs centered on the first 6 atoms in the above string
    keywords['basis'] = 'STO-3G'
    keywords['spin'] = 0 # in PySCF this is the number of unpaired electrons, not 2s+1
    keywords['charge'] = 0

    e, mf, ec = presses.run_embed(keywords)
    ref = e
    
    keywords['occupied_shells'] = 0
    keywords['active_space_atoms'] = 2
 
    e, mf, ec = presses.run_embed(keywords)
    print(ref, e)
    
    assert(abs(ref - e) < 1e-5)

if __name__== "__main__":
    test()
