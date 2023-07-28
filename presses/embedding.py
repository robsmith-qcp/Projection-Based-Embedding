import pyscf
import numpy as np
from pyscf import mp, cc
from pyscf.mp.dfmp2_native import DFMP2

class Proj_Emb:
    '''
    A class to calculate energies and associated objects using the PySCF quantum chemistry package for projection-based embedding
    '''
    def __init__(self, keywords):
        '''
        Parameters
        ----------
        keywords   : dict
            the dictionary of terms used in the embedding calculation
        Returns
        -------
        '''
        self.keywords = keywords

    def mean_field(self):
        '''
        Parameters
        ----------
        closed_shell: bool
            dictates whether closed-shell or open-shell embedding algorithm is employed
        Returns
        -------
        F : np.array
            the Fock matrix
        V : np.array
            the potential matrix
        C : np.array
            the MO coefficients
        S : np.array
            the overlap matrix
        P : np.array
            the 1-particle reduced-density matrix
        H : np.array
            the core Hamiltonian matrix
        '''
        self.mol = pyscf.gto.Mole(
                   atom    =   self.keywords['atom'],
                   symmetry=   False, # True
                   spin    =   self.keywords['spin'],
                   charge  =   self.keywords['charge'],
                   cart    =   False, #False,  This should be set to true to match Daniel's code with the 6-31G* basis
                   basis   =   self.keywords['basis'])

        self.mol.build()
        self.xc = self.keywords['xc']    
        # Initial SCF on the whole system
        if self.mol.spin == 0 and self.keywords['scf_method'].lower() == 'hf':
            self.mf = pyscf.scf.RHF(self.mol)
            self.closed_shell = True
        elif self.mol.spin != 0 and self.keywords['scf_method'].lower() == 'hf':
            self.mf = pyscf.scf.ROHF(self.mol)
            self.closed_shell = False
        elif self.mol.spin == 0 and self.keywords['scf_method'].lower() != 'hf':
            self.mf = pyscf.dft.RKS(self.mol)
            self.mf.xc = self.xc
            self.closed_shell = True
            self.mf.grids.atom_grid = (75, 302)
            self.mf.grids.prune = pyscf.dft.treutler_prune
        elif self.mol.spin != 0 and self.keywords['scf_method'].lower() != 'hf':
            self.mf = pyscf.dft.ROKS(self.mol)
            self.mf.xc = self.xc
            self.closed_shell = False
        self.mf.verbose = self.keywords['verbose']
        self.mf.conv_tol = self.keywords['conv_tol']
        self.mf.conv_tol_grad = self.keywords['conv_tol_grad']
        self.mf.chkfile = "scf.fchk"
        self.mf.init_guess = "minao"
        self.mf.run(max_cycle=1000)

        if not self.mf.converged:
            print('Initial calculation did not converge. Please, check your chosen tolerances.')
            exit()

        self.n_atoms = self.keywords['active_space_atoms']
        # Save the Fock matrix to be used in canonicalization and concentric localization
        F = self.mf.get_fock()

        # Compute the nuclear potential
        self.Vnn = pyscf.gto.mole.energy_nuc(self.mol)

        # Compute the Coulomb and Exchange potentials
        self.J,self.K = self.mf.get_jk()

        # Compute potential
        self.V = self.mf.get_veff()

        # Define the 1-particle density matrix
        self.P = self.mf.make_rdm1()
    
        # Define the orbitals and overlap
        C = self.mf.mo_coeff
        S = self.mf.get_ovlp(self.mol)

        # Compute the core Hamiltonian
        self.H_core  = self.mf.get_hcore(self.mol)

        # Save the mean-field energy for testing
        self.E_init = self.mf.e_tot

        # Defining the nuclear-electronic potential and kinetic energy operators
        self.Vne = self.mol.intor_symmetric('int1e_nuc')
        self.T = self.mol.intor_symmetric('int1e_kin')        

        self.atom_index = self.keywords['active_space_atoms'] #- 1
        ao_list = []
        for ao_idx,ao in enumerate(self.mf.mol.ao_labels(fmt=False)):
            for i in range(self.atom_index):
                if ao[0] == i:
                    ao_list.append(ao_idx)
        #self.n_aos = self.mol.aoslice_nr_by_atom()[self.keywords['active_space_atoms']-1][3]
        print('AOs: ', ao_list)
        self.n_aos = len(ao_list)
        print(self.n_aos)
        self.S_proj = pyscf.gto.intor_cross('int1e_ovlp_sph',self.mol, self.mol)
        self.mfe = self.mf.e_tot
        self.nbas = self.mol.nao
        return F, self.V, C, S, self.P, self.H_core

    def subspace_values(self, D_A, D_B):
        '''
        Parameters
        ----------
        D_A : np.array
            the density matrix for subsystem A
        D_B : np.array
            the density matrix for subsystem B
        Returns
        -------
        Vact : np.array
            the effective 2-electron component in the embedded subsystem
        Venv : np.array
            the effective 2-electron component in the environment subsystem
        '''
        # Compute the Coulomb and Exchange potentials for each subsystem
        Jenv, Kenv = self.mf.get_jk(dm=D_B)
        Jact, Kact = self.mf.get_jk(dm=D_A)

        # Compute the potential for each subsystem
        Venv = self.mf.get_veff(dm=D_B)
        Vact = self.mf.get_veff(dm=D_A)
        V = self.mf.get_veff()

        # Compute the subsystem energies
        if self.closed_shell:
            if self.keywords['scf_method'].lower() == 'hf':
                self.E_env = np.einsum('ij,ji->', D_B, self.H_core) + 0.5 * np.einsum('ij,ji->', D_B, Venv)
                self.E_act = np.einsum('ij,ji->', D_A, self.H_core) + 0.5 * np.einsum('ij,ji->', D_A, Vact)
                self.E_cross = 0.5 * np.einsum('ij,ji->', D_B, Vact) + 0.5 * np.einsum('ij,ji->', D_A, Venv)
            else:
                self.E_env = np.einsum('ij,ji->', D_B, self.H_core) + Venv.ecoul + Venv.exc
                self.E_act = np.einsum('ij,ji->', D_A, self.H_core) + Vact.ecoul + Vact.exc
                self.E_cross = V.ecoul - Venv.ecoul - Vact.ecoul + V.exc - Venv.exc - Vact.exc
        else:
            if self.keywords['scf_method'].lower() == 'hf':
                self.E_env = np.einsum('ij,ji->', D_B[0], self.H_core) + np.einsum('ij,ji->', D_B[1], self.H_core) + 0.5 * np.einsum('ij,ji->', D_B[0], Venv[0]) + 0.5 * np.einsum('ij,ji->', D_B[1], Venv[1])
                self.E_act = np.einsum('ij,ji->', D_A[0], self.H_core) + np.einsum('ij,ji->', D_A[1], self.H_core) + 0.5 * np.einsum('ij,ji->', D_A[0], Vact[0]) + 0.5 * np.einsum('ij,ji->', D_A[1], Vact[1])
                self.E_cross = 0.5 * np.einsum('ij,ji->', D_B[0], Vact[0]) + 0.5 * np.einsum('ij,ji->', D_B[1], Vact[1]) + 0.5 * np.einsum('ij,ji->', D_A[0], Venv[0]) + 0.5 * np.einsum('ij,ji->', D_A[1], Venv[1])
            else:
                self.E_env = np.einsum('ij,ji->', D_B[0], self.H_core) + np.einsum('ij,ji->', D_B[1], self.H_core) + Venv.ecoul + Venv.exc
                self.E_act = np.einsum('ij,ji->', D_A[0], self.H_core) + np.einsum('ij,ji->', D_A[1], self.H_core) + Vact.ecoul + Vact.exc
                self.E_cross = V.ecoul - Venv.ecoul - Vact.ecoul + V.exc - Venv.exc - Vact.exc

        return Vact, Venv

    def embedded_mean_field(self, n_elec, Vemb, D_A):
        '''
        Parameters
        ----------
        n_elec : int
            the number of electrons in the active space
        Returns
        -------
        C_emb : np.array
            the MO coefficients
        S_emb : np.array
            the overlap matrix
        F_emb : np.array
            the Fock matrix
        '''
        # The embedding calculation setup
        self.mol = pyscf.gto.Mole(
                   atom    =   self.keywords['atom'],
                   symmetry=   False, # True
                   spin    =   self.keywords['spin'],
                   charge  =   self.keywords['charge'],
                   cart    =   False, # This should be set to true to match Daniel's code with the 6-31G* basis
                   basis   =   self.keywords['basis'])
        
        self.xc = self.keywords['embedded_xc']
        if self.closed_shell and self.keywords['subsystem_method'].lower() != 'dft':
            self.emb_mf = pyscf.scf.RHF(self.mol)
        elif not self.closed_shell and self.keywords['subsystem_method'].lower() != 'dft':
            self.emb_mf = pyscf.scf.ROHF(self.mol)
        elif self.closed_shell and self.keywords['subsystem_method'].lower() == 'dft':
            self.emb_mf = pyscf.scf.RKS(self.mol)
            self.emb_mf.xc = self.xc
        elif not self.closed_shell and self.keywords['subsystem_method'].lower() == 'dft':
            self.emb_mf = pyscf.scf.ROKS(self.mol)
            self.emb_mf.xc = self.xc
        self.mol.nelectron = n_elec

        self.mol.build()

        self.emb_mf.verbose = self.keywords['verbose']
        self.emb_mf.conv_tol = self.keywords['conv_tol']
        self.emb_mf.conv_tol_grad = self.keywords['conv_tol_grad']
        self.emb_mf.chkfile = "subsys_scf.chk"
        self.emb_mf.get_hcore = lambda *args: self.H_core + Vemb
        self.emb_mf.max_cycle = 200
        self.emb_mf.kernel(dm0=D_A)

        if not self.emb_mf.converged:
            print('Subsystem calculation did not converge. Please, check your chosen tolerances.')
            exit()

        self.C_emb = self.emb_mf.mo_coeff
        S_emb = self.emb_mf.get_ovlp(self.mol)
        F_emb = self.emb_mf.get_fock()
        self.V_emb = self.emb_mf.get_veff()
        self.P_emb = self.emb_mf.make_rdm1()
        self.S_pbwb = pyscf.gto.intor_cross('int1e_ovlp_sph', self.mol, self.mol)
        #self.emb_n_aos = self.mol.aoslice_nr_by_atom()[self.atom_index][3]
        return self.C_emb, S_emb, F_emb

    def embed_scf_energy(self, Vemb, D_A, Proj, mu, H):
        '''
        Parameters
        ----------
        Cocc : np.array
            the occupied MO coefficients
        Vemb : np.array
            the embedded potential
        D_A : np.array
            the density of the embedded subsystem A
        Returns
        -------
        E_emb : float
            the mean-field energy resulting from projection-based embedding
        '''
        # Computing the embedded SCF energy
        if self.closed_shell:
            if self.keywords['subsystem_method'].lower() == 'dft':
                E_emb = np.einsum('ij,ji->', self.P_emb, H) + self.V_emb.ecoul + self.V_emb.exc
            else:
                E_emb = np.einsum('ij,ji->', self.P_emb, H) + 0.5 * np.einsum('ij,ji->', self.P_emb, self.V_emb)
            correction = np.einsum('ij,ji->', Vemb, (self.P_emb-D_A))
            post = mu * np.einsum('ij,ji->', self.P_emb, Proj )
        else:
            if self.keywords['subsystem_method'].lower() == 'dft':
                E_emb = np.einsum('ij,ji->', self.P_emb[0], H) + np.einsum('ij,ji->', self.P_emb[1], H)  + self.V_emb.ecoul + self.V_emb.exc
            else:
                E_emb = np.einsum('ij,ji->', self.P_emb[0], H) + np.einsum('ij,ji->', self.P_emb[1], H) + 0.5 * np.einsum('ij,ji->', self.P_emb[0], self.V_emb[0]) + 0.5 * np.einsum('ij,ji->', self.P_emb[1], self.V_emb[1])
            correction = np.einsum('ij,ji->', Vemb, (self.P_emb[0]-D_A[0])) + np.einsum('ij,ji->', Vemb, (self.P_emb[1]-D_A[1]))
            post = mu * (np.einsum('ij,ji->', self.P_emb[0], Proj[0]) + np.einsum('ij,ji->', self.P_emb[1], Proj[1]))

        self.embed_SCF = E_emb + self.E_env + self.E_cross + correction + self.Vnn + post
        return self.embed_SCF

    def correlation_energy(self, n_effective, nact=None, Cspan=None, Ckern=None, e_orb_span=None, e_orb_kern=None):
        '''
        Parameters
        ----------
        n_effective : int
            the number of effective virtual MOs
        nact : int
            the number of active MOs
        Cspan : np.array
            the span MO coefficients
        Ckern : np.array
            the kernel MO coefficients
        e_orb_span : int
            the orbital energies in the span
        e_orb_kern : int
            the orbital energies in the kernel
        Returns
        -------
        correl_e : list
            the updated list containing the correlation energy for each shell
        '''
        mf_eff = self.emb_mf
        diff = self.mol.nao - n_effective
        if self.keywords['n_shells'] == 0:
            frozen = [i for i in range(n_effective, self.mol.nao)]
        else:
            orbs = np.hstack((self.C_emb[:,:nact],Cspan,Ckern,self.C_emb[:,n_effective:]))
            orbs_e = np.concatenate((mf_eff.mo_energy[:nact],e_orb_span,e_orb_kern,mf_eff.mo_energy[n_effective:]))
            frozen = [i for i in range(nact + Cspan.shape[1], self.mol.nao)]
            mf_eff.mo_energy = orbs_e
            mf_eff.mo_coeff = orbs

        if self.keywords['subsystem_method'].lower() == 'mp2':
            if diff > 0:
                mymp = pyscf.mp.MP2(mf_eff).set(frozen=frozen).run()
            else:
                mymp = pyscf.mp.MP2(mf_eff).run()
            correl_e = mymp.e_corr
        elif self.keywords['subsystem_method'].lower() == 'ri-mp2':
            mymp = DFMP2(mf_eff).set(frozen=frozen).run()
            correl_e = mymp.e_corr
        elif self.keywords['subsystem_method'].lower() == 'ccsd':
            mycc = pyscf.cc.CCSD(mf_eff).set(frozen=frozen).run()
            et = mycc.ccsd_t()
            correl_e = mycc.e_corr
            correl_e += et
        elif self.keywords['subsystem_method'].lower() == 'fci':
            # To Do: code FCI solver
            print('Requested correlation method has not yet been implemented.')
            print('Total mean-field energy = ', self.embed_SCF)
            pass
        elif self.keywords['subsystem_method'].lower() == 'eom-ccsd':
            mycc = pyscf.cc.CCSD(mf_eff).set(frozen=frozen,verbose=4,max_cycle=500).run()
            e_ee, c_ee = mycc.eeccsd(self.keywords['n_roots'])
            correl_e = e_ee[0]
        elif self.keywords['subsystem_method'].lower() == 'sf-eom-ccsd':
            if self.keywords['spin'] == 0:
                print('Requested correlation method is not suitable for a low-spin state.')
                pass
            if diff > 0:
                mycc = pyscf.cc.CCSD(mf_eff).set(frozen=frozen,verbose=4,max_cycle=500).run()
            else:
                mycc = pyscf.cc.CCSD(mf_eff).set(verbose=4,max_cycle=500).run()
            e_sf, c_sf = mycc.eomsf_ccsd(self.keywords['n_roots'])
            correl_e = e_sf[1] - e_sf[0]
            J = ((e_sf[0] - e_sf[1]) / (2 * self.keywords['spin'])) * 219474.63
            print('Exchange coupling constant: %f cm^-1' %J)
        elif self.keywords['subsystem_method'].lower() == 'ip-eom-ccsd':
            mycc = pyscf.cc.CCSD(mf_eff).set(frozen=frozen,verbose=4,max_cycle=500).run()
            e_ip, c_ip = mycc.ipccsd(self.keywords['n_roots'])
            correl_e = e_ip[0]
            I = (e_ip[1] - e_ip[0]) * 2625.5002
            print('Ionization potential: %f kJ/mol' %I)
        elif self.keywords['subsystem_method'].lower() == 'ea-eom-ccsd':
            mycc = pyscf.cc.CCSD(mf_eff).set(frozen=frozen,verbose=4,max_cycle=500).run()
            e_ea, c_ea = mycc.eaccsd(self.keywords['n_roots'])
            correl_e = e_ea[0]
            E = (e_ea[1] - e_ea[0]) * 2625.5002
            print('Electron affinity: %f kJ/mol' %E)
        else:
            print('Requested correlation method has not yet been implemented.')
            print('Total mean-field energy = ', self.embed_SCF)
            pass
        return correl_e

    def operator_assignment(self, operator_str, virt=True):
        '''
        Parameters
        ----------
        operator_str : str
            a string indicating the one-particle operator being assigned to seed concentric localization
        Returns
        -------
        operator : np.array
            the matrix representation of the single-particle operator
        '''
        if operator_str == 'F' and virt==True:
           operator = self.emb_mf.get_fock()
        elif operator_str == 'F' and virt==False:
           operator = self.mf.get_fock()
        elif operator_str == 'H':
            operator = self.H_core
        elif operator_str == 'S':
            operator = self.emb_mf.get_ovlp(self.mol)
        elif operator_str == 'T':
            operator = self.T
        elif operator_str == 'V':
            operator = self.Vne
        elif operator_str == 'P':
            operator = self.P
        else:
            print('Chosen operator is invalid.')
            pass
        return operator
        
def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format).

    Replace this function and doc string for your own project.

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from.

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution.
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())
