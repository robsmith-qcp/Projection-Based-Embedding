import presses
import pyscf
import pytest
import sys

def test():
    keywords = {}
    keywords['scf_method'] = 'hf' # If dft is chosen, the user should specify the desired exchange-correlation functional
    keywords['subsystem_method'] = 'hf'
    keywords['n_shells'] = 0 # must be some integer
    keywords['atom'] = '''
                       C       -4.2066354000      0.3782434000     -0.1873330000                 
                       H       -4.6220764000     -0.2488589000      0.6064495000                 
                       H       -4.5320417000     -0.0402909000     -1.1437099000                 
                       H       -4.6498537000      1.3721383000     -0.0912650000                 
                       C       -2.6835636000      0.4404581000     -0.1004726000                 
                       C       -2.0237207000     -0.9334681000     -0.2284959000                 
                       H       -2.4005210000      0.9059350000      0.8477680000                 
                       H       -2.3021173000      1.0976975000     -0.8899870000                 
                       H       -2.3710724000     -1.5783613000      0.5874586000                 
                       C       -0.4924135000     -0.9205584000     -0.2450079000                 
                       H       -2.3747564000     -1.4030909000     -1.1536151000                 
                       C        0.1674308000     -0.4649642000      1.0574879000                 
                       H       -0.1450618000     -0.2842181000     -1.0676500000                 
                       H       -0.1413789000     -1.9316300000     -0.4776575000                 
                       H       -0.1156108000      0.5675465000      1.2805379000                 
                       C        1.6905024000     -0.5642664000      1.0180566000                 
                       H       -0.2140155000     -1.0726672000      1.8857352000                 
                       H        2.1059434000      0.0548467000      0.2180274000                 
                       H        2.0159077000     -1.5933522000      0.8425261000                 
                       H        2.1337217000     -0.2313363000      1.9594454000                 
                       '''
    keywords['active_space_atoms'] = 1 # The embedded subsystem is developed from the AOs centered on the first 4 atoms in the above string
    keywords['basis'] = 'STO-3G'
    keywords['spin'] = 0 # in PySCF this is the number of unpaired electrons, not 2s+1
    keywords['charge'] = 0

    e, ref, ec = presses.run_embed(keywords)
    assert(abs(ref - e) < 1e-7)

if __name__== "__main__":
    test()
