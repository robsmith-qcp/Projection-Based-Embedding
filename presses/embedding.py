import pyscf
import numpy as np
from pyscf import mp, cc

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
        J : np.array
            the Coulomb matrix
        K : np.array
            the exchange matrix
        Exc : float
            the potential energy arrising from the exchange-correlation
        Vxc : np.array
            the exchange-correlation matrix
        C : np.array
            the MO coefficients
        S : np.array
            the overlap matrix
        P : np.array
            the 1-particle reduced-density matrix
        '''
        self.mol = pyscf.gto.Mole(
                   atom    =   self.keywords['atom'],
                   symmetry=   False, # True
                   spin    =   self.keywords['spin'],
                   charge  =   self.keywords['charge'],
                   cart    =   True, # This should be set to true to match Daniel's code with the 6-31G* basis
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
            self.mf = pyscf.scf.RKS(self.mol)
            self.mf.xc = self.xc
            self.closed_shell = True
        elif self.mol.spin != 0 and self.keywords['scf_method'].lower() != 'hf':
            self.mf = pyscf.scf.ROKS(self.mol)
            self.mf.xc = self.xc
            self.closed_shell = False
        '''
        elif closed_shell and scf.lower() == 'hf' and periodic:
            self.mf = pyscf.pbc.scf.RHF(self.mol)
        elif not closed_shell and scf.lower() == 'hf' and periodic:
            self.mf = pyscf.pbc.scf.ROHF(self.mol)
        elif closed_shell and scf.lower() != 'hf' and periodic:
            self.mf = pyscf.pbc.scf.RKS(self.mol)
            self.mf.xc = self.xc
        elif not closed_shell and scf.lower() != 'hf' and periodic:
            self.mf = pyscf.pbc.scf.ROKS(self.mol)
            self.mf.xc = self.xc
        '''
        self.mf.verbose = self.keywords['verbose']
        self.mf.conv_tol = self.keywords['conv_tol']
        self.mf.conv_tol_grad = self.keywords['conv_tol_grad']
        self.mf.chkfile = "scf.fchk"
        self.mf.init_guess = "minao"
        self.mf.run(max_cycle=200)

        self.n_atoms = self.keywords['active_space_atoms']
        # Save the Fock matrix to be used in canonicalization and concentric localization
        F = self.mf.get_fock()

        # Compute the nuclear potential
        self.Vnn = pyscf.gto.mole.energy_nuc(self.mol)

        # Compute the Coulomb and Exchange potentials
        self.J,self.K = self.mf.get_jk()

        # Compute exchange-correlation potential
        if self.keywords['scf_method'].lower() == 'hf':
            self.Exc = 0.0
            self.Vxc = np.zeros([self.mol.nao,self.mol.nao])
        else:
            self.Exc = self.mf.get_veff().exc
            self.Vxc = self.mf.get_veff()

        # Define the 1-particle density matrix
        self.P = self.mf.make_rdm1()
    
        # Define the orbitals and overlap
        C = self.mf.mo_coeff
        S = self.mf.get_ovlp(self.mol)

        # Compute the core Hamiltonian
        self.H_core  = self.mf.get_hcore(self.mol)

        # Save the mean-field energy for testing
        self.E_init = self.mf.e_tot

        return F, self.J, self.K, C, S, self.P

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
        V_subsystem : np.array
            the effective 2-electron component in the embedded subsystem
        '''
        # Compute the Coulomb and Exchange potentials for each subsystem
        Jenv, Kenv = self.mf.get_jk(dm=D_B)
        Jact, Kact = self.mf.get_jk(dm=D_A)

        # Compute the exchange correlation for each subsystem
        if self.keywords['scf_method'].lower() != 'hf':
            XCenv = self.mf.get_veff(dm=D_B)
            XCact = self.mf.get_veff(dm=D_A)
            Exc_env = self.mf.get_veff(dm=D_B).exc
            Exc_act = self.mf.get_veff(dm=D_A).exc
        else:
            XCenv = self.Vxc
            XCact = self.Vxc
            Exc_env = 0.0
            Exc_act = 0.0

        # Compute the subsystem energies
        self.E_env = np.einsum('ij, ij', D_B, self.H_core) + 0.5 * np.einsum('ij, ij', D_B, self.J) - 0.25 * np.einsum('ij, ij', D_B, self.K)
        self.E_act = np.einsum('ij, ij', D_A, self.H_core) + 0.5 * np.einsum('ij, ij', D_A, self.J) - 0.25 * np.einsum('ij, ij', D_A, self.K)
        if self.keywords['scf_method'].lower() != 'hf':
            self.E_env += Exc_env
            self.E_act += Exc_act

        # Compute the non-additive component of the 2-electron integral
        '''
        Jab = 0.5*(np.einsum('ij, ij', D_A, Jenv) + np.einsum('ij, ij', D_B, Jact))
        Kab = 0.5*(np.einsum('ij, ij', D_A, Kenv) + np.einsum('ij, ij', D_B, Kact))
        if self.keywords['scf_method'].lower() == 'hf':
            XCab = 0.0
        else:
            XCab = self.mf.get_veff().exc - Exc_act - Exc_env
        self.G = Jab + Kab + XCab
        '''
        Jcross = self.J - Jact - Jenv
        Kcross = self.K - Kact - Kenv
        self.E_cross = 0.5 * np.einsum('ij, ij', D_A, Jcross) + 0.5 * np.einsum('ij, ij', D_B, Jcross) - 0.25 * np.einsum('ij, ij', D_A, Kcross) - 0.25 * np.einsum('ij, ij', D_B, Kcross) 
        V_subsystem = self.mf.get_veff(dm=D_A)

        return V_subsystem, Jact, Kact

    def embedded_mean_field(self, n_elec, Vemb):
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
                   cart    =   True, # This should be set to true to match Daniel's code with the 6-31G* basis
                   basis   =   self.keywords['basis'])

        if self.closed_shell and self.keywords['subsystem_method'].lower() == 'hf':
            self.emb_mf = pyscf.scf.RHF(self.mol)
        elif not self.closed_shell and self.keywords['subsystem_method'].lower() == 'hf':
            self.emb_mf = pyscf.scf.ROHF(self.mol)
        elif self.closed_shell and self.keywords['subsystem_method'].lower() != 'hf':
            self.emb_mf = pyscf.scf.RKS(self.mol)
            self.emb_mf.xc = self.xc
        elif not self.closed_shell and self.keywords['subsystem_method'].lower() != 'hf':
            self.emb_mf = pyscf.scf.ROKS(self.mol)
            self.emb_mf.xc = self.xc
        self.mol.nelectron = n_elec

        self.mol.build()
        print(self.mol.nelectron)

        self.emb_mf.verbose = 5
        self.emb_mf.conv_tol = 1e-8
        self.emb_mf.conv_tol_grad = 1e-6
        self.emb_mf.chkfile = "emb_scf.chk"
        self.emb_mf.init_guess = "minao"
        self.emb_mf.get_hcore = lambda *args: self.H_core + Vemb
        self.emb_mf.run(max_cycle=200)
        C_emb = self.emb_mf.mo_coeff
        S_emb = self.emb_mf.get_ovlp(self.mol)
        F_emb = self.emb_mf.get_fock()
        self.P_emb = self.emb_mf.make_rdm1()
        return C_emb, S_emb, F_emb

    def embed_scf_energy(self, Cocc, Vemb, D_A, Proj, mu):
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
        F_emb : float
            the mean-field energy resulting from projection-based embedding
        '''
        # Computing the embedded SCF energy
        E_emb = np.einsum('ij, ji', self.H_core, self.P_emb) + 0.5 * np.einsum('ij, ji', self.J, self.P_emb) - 0.25 * np.einsum('ij, ji', self.K, self.P_emb)

        correction = np.einsum('ij, ji', Vemb, (self.P_emb-D_A))

        post = np.einsum('ij, ji', self.P_emb, mu * Proj )

        embed_SCF = E_emb + self.E_env + self.E_cross + correction + self.Vnn + post
        print(E_emb, correction, post)
        return embed_SCF

    def correlation(self, nact, span_e, span_c, kenerl_e, kernel_c, Cocc, n_active_aos, shift, C, correl_e):
        '''
        Parameters
        ----------
        nact : int
            the number of active MOs
        span_e : int
            the orbital energies in the span
        span_c : np.array
            the span of the effective virtual space
        kernel_e : int
            the orbital energies in the kernel
        kernel_c : np.array
            the kernel of the effective virtual space
        Cocc : np.array
            the occupied MO coefficients
        n_active_aos : int
            the number of AOs in the active space
        shift : int
            the number to account for the subspace B orbitals being in the frozen virtual space
        C : np.array
            the MO coefficients
        correl_e : list
            the list containing the correlation energy for each shell
        Returns
        -------
        correl_e : list
            the updated list containing the correlation energy for each shell
        '''
        mf_eff = self.mf
        orbs0 = np.hstack((span_c, kernel_c))
        orbs0_e = np.concatenate((span_e, kenerl_e))
        frozen = [i for i in range(nact + span_c.shape[1], self.mol.nao)]
        orbitals = np.hstack((Cocc, span_c, C[:, shift:]))
        orbital_energies = np.concatenate((mf_eff.mo_energy[:nact], span_e, mf_eff.mo_energy[shift:]))
        mf_eff.mo_energy = orbital_energies
        mf_eff.mo_coeff = orbitals
        if correlated_method == 'mp2':
            mymp = mp.MP2(mf_eff).set(frozen=frozen).run()
            correl_e = mymp.e_corr
        elif correlated_method == 'ccsd':
            mycc = cc.CCSD(mf_eff).set(frozen=frozen).run()
            et = mycc.ccsd_t()
            correl_e = mycc.e_corr
            correl_e += et
        shell_e.append(correl_e)
        return correl_e

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
