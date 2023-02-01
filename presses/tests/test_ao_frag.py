import presses
import pyscf
import pytest
import sys

def test_presses_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "presses" in sys.modules

def test_aos():
    """Tests the number of AOs to be included in the fragment"""
    keywords = {}
    keywords['scf_method'] = 'hf' # If dft is chosen, the user should specify the desired exchange-correlation functional
    keywords['subsystem_method'] = 'hf'
    keywords['concen_local_shells'] = 0 # must be some integer
    keywords['basis'] = 'STO-3G'
    keywords['spin'] = 0 # in PySCF this is the number of unpaired electrons, not 2s+1
    keywords['charge'] = 0
    keywords['atom'] = """
                       O 0.00000 0.00000 0.00000
                       C 0.00000 0.00000 1.12830
                       """
    keywords['active_space_atoms'] = 1
    keywords['verbose'] = 1
    keywords = presses.driver.update_keywords(keywords)
    Embed = presses.embedding.Proj_Emb(keywords)
    Orbs = presses.orbitals.Partition(Embed)
    F, J, K, C, S, P = Embed.mean_field()
    frag_list = Orbs.ao_assignment(Embed.mf, Embed.n_atoms)
    n_aos = len(frag_list)
    
    assert n_aos == 5   
