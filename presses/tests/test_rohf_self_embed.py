import presses
import pyscf
import pytest
import sys

def test():
    keywords = {}
    keywords['scf_method'] = 'hf' # If dft is chosen, the user should specify the desired exchange-correlation functional
    keywords['subsystem_method'] = 'hf'
    keywords['concen_local_shells'] = 0 # must be some integer
    keywords['atom'] = '''
                         C   -4.1471368    0.4390617   -0.1805483
                         H   -4.8922874    1.2201878   -0.0989841
                         C   -2.6598447    0.4803558   -0.0984501
                         C   -2.0268025   -0.9111275   -0.2192962
                         H   -2.3665654    0.9484783    0.8483234
                         H   -2.2667907    1.1264296   -0.8952421
                         H   -2.3847125   -1.5397383    0.6026076
                         C   -0.4959844   -0.9163104   -0.2398395
                         H   -2.3907340   -1.3766042   -1.1395287
                         C    0.1743423   -0.4626019    1.0579782
                         H   -0.1421026   -0.2911248   -1.0681495
                         H   -0.1617826   -1.9340341   -0.4669650
                         H   -0.0909622    0.5759527    1.2755332
                         C    1.6955211   -0.5861150    1.0144075
                         H   -0.2143360   -1.0587842    1.8910562
                         H    2.1181444    0.0210856    0.2090627
                         H    2.0038447   -1.6213764    0.8448520
                         H    2.1465129   -0.2540293    1.9522987 
                       '''
    keywords['active_space_atoms'] = 1 # The embedded subsystem is developed from the AOs centered on the first 4 atoms in the above string
    keywords['basis'] = 'STO-3G'
    keywords['spin'] = 2 # in PySCF this is the number of unpaired electrons, not 2s+1
    keywords['charge'] = 0

    e, ref, ec = presses.run_embed(keywords)
    assert(abs(ref - e) < 1e-7)

if __name__== "__main__":
    test()
