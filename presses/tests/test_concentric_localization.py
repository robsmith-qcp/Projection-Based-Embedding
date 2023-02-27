import presses
import pyscf
import pytest
import sys

def test():
    keywords = {}
    keywords['scf_method'] = 'dft' # If dft is chosen, the user should specify the desired exchange-correlation functional
    keywords['xc'] = 'b3lyp'
    keywords['subsystem_method'] = 'mp2'
    keywords['n_shells'] = 0 # must be some integer
    keywords['atom'] = '''
  O    0.2905609   -0.8428120    0.2563333
  H   -0.0385867   -1.4462899   -0.4260066
  C   -1.7496969    0.5014112   -0.0010207
  C   -0.2263491    0.4488038   -0.0357938
  H   -2.1865084   -0.1402778   -0.7870683
  H   -2.1155730    1.5282030   -0.1707984
  H   -2.1277172    0.1502959    0.9725619
  H    0.1437499    0.8207188   -1.0141501
  H    0.1942456    1.1134689    0.7367153
'''
    keywords['active_space_atoms'] = 2 # The embedded subsystem is developed from the AOs centered on the first 6 atoms in the above string
    keywords['basis'] = 'cc-pvdz'
    keywords['spin'] = 0 # in PySCF this is the number of unpaired electrons, not 2s+1
    keywords['charge'] = 0

    ref, mf, ec = presses.run_embed(keywords)
    keywords['n_shells'] = 5
    e, mf, ec = presses.run_embed(keywords)
    assert(abs(ref - e) < 1e-7)

if __name__== "__main__":
    test()
