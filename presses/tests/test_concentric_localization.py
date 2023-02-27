import presses
import pyscf
import pytest
import sys

def test():
    ref = -0.83333356
    keywords = {}
    keywords['scf_method'] = 'dft' # If dft is chosen, the user should specify the desired exchange-correlation functional
    keywords['xc'] = 'b3lyp'
    keywords['subsystem_method'] = 'mp2'
    keywords['n_shells'] = 0 # must be some integer
    keywords['atom'] = '''
 C         -0.715841   -6.107749    0.000000
 C          0.715842   -6.107749    0.000000
 H         -1.245672   -7.053626    0.000000
 H          1.245673   -7.053625    0.000000
 C          1.408460   -4.934989    0.000000
 C         -1.408460   -4.934990    0.000000
 H          2.493485   -4.935051    0.000000
 H         -2.493485   -4.935052    0.000000
 C          0.726859   -3.672425    0.000000
 C         -0.726860   -3.672425    0.000000
 C          1.406111   -2.464080    0.000000
 C         -1.406112   -2.464081    0.000000
 H          2.491875   -2.465022    0.000000
 H         -2.491877   -2.465022    0.000000
 C          0.727673   -1.224439    0.000000
 C         -0.727673   -1.224439    0.000000
 C          1.406586    0.000000    0.000000
 C         -1.406587    0.000000    0.000000
 H          2.492262    0.000000    0.000000
 H         -2.492264    0.000000    0.000000
 C          0.727673    1.224439    0.000000
 C         -0.727673    1.224439    0.000000
 C          1.406111    2.464080    0.000000
 C         -1.406112    2.464081    0.000000
 H          2.491875    2.465022    0.000000
 H         -2.491876    2.465023    0.000000
 C          0.726859    3.672425    0.000000
 C         -0.726859    3.672425    0.000000
 C          1.408461    4.934989    0.000000
 C         -1.408460    4.934990    0.000000
 H          2.493485    4.935050    0.000000
 H         -2.493485    4.935052    0.000000
 C          0.715842    6.107749    0.000000
 C         -0.715841    6.107749    0.000000
 H          1.245674    7.053625    0.000000
 H         -1.245672    7.053626    0.000000
'''
    keywords['active_space_atoms'] = 10 # The embedded subsystem is developed from the AOs centered on the first 6 atoms in the above string
    keywords['basis'] = 'cc-pvdz'
    keywords['spin'] = 0 # in PySCF this is the number of unpaired electrons, not 2s+1
    keywords['charge'] = 0

    e, mf, ec = presses.run_embed(keywords)
    # A lower threshold is set here to match the base PBE results between Psi4 and PySCF
    # To Do: resolve the discrepancy and tighten the threshold.
    assert(abs(ref - e) < 1e-5)

if __name__== "__main__":
    test()
