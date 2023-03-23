import presses
import pyscf
import sys

keywords = {}
keywords['scf_method'] = 'dft' # If dft is chosen, the user should specify the desired exchange-correlation functional
keywords['subsystem_method'] = 'mp2'
keywords['conv_tol_grad'] = 1.0e-6
keywords['xc'] = 'b3lyp'
keywords['n_shells'] = 2 # must be some integer
keywords['atom'] = '''
  O   -1.2597450    3.6487177    1.4736659
  O   -0.2161919    1.5900623    3.1260603
  C   -1.7989475    2.5276382    1.5655296
  C   -1.2485092    1.4194586    2.4482454
  C   -2.9831682    2.1416774    0.8700224
  C   -1.9487762    0.1508455    2.4884098
  C   -3.1417527   -0.1504715    1.7500165
  C   -3.6201358    0.8575457    0.9620419
  C   -4.7183891    1.2605480   -0.0104161
  C   -4.0125184    2.6714702   -0.1002556
  H   -5.7299870    1.2744056    0.4090830
  H   -4.7390895    0.6942831   -0.9476899
  H   -3.6196804    2.9412203   -1.0871632
  H   -4.6008420    3.5173806    0.2729652
  H   -3.6106971   -1.1269232    1.8336260
  H   -1.5055981   -0.6007565    3.1364158 
'''
keywords['active_space_atoms'] = 2 # The embedded subsystem is developed from the AOs centered on the first 2 atoms in the above string
keywords['basis'] = 'aug-cc-pVDZ'
keywords['spin'] = 2 # in PySCF this is the number of unpaired electrons, not 2s+1
keywords['charge'] = 0

e, ref, ec = presses.run_embed(keywords)
