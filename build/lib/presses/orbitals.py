import pyscf
import numpy as np
import scipy
from .embedding import *

class Partition(Proj_Emb):
    '''
    A class to hold objects associated with the orbital rotations and partitions
    '''
    def __init__(self, keywords):
        '''
        Child class of the Proj_Emb class
        '''
        Proj_Emb.__init__(self,keywords)

    def ao_assignment(self, mf, n_atoms):
        '''
        Returns
        -------
        frag : list
            the list of aos centered on the active space atoms
        '''
        frag = []
        one_s = ['H', 'He']
        two_s = ['Li', 'Be']
        two_p = ['B', 'C', 'N', 'O', 'F', 'Ne']
        three_s = ['Na', 'Mg']
        three_p = ['Al', 'Si', 'P', 'S', 'Cl', 'Ar']
        four_s = ['K', 'Ca']
        three_d = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
        four_p = ['Ga', 'Ge', 'As', 'Se', 'Br', 'Kr']
        five_s = ['Rb', 'Sr']
        four_d = ['Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
        five_p = ['In', 'Sn', 'Sb', 'Te', 'I', 'Xe']
        six_s = ['Cs', 'Ba']
        four_f = ['Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
        five_d = ['La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']
        six_p = ['Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']
        seven_s = ['Fr', 'Ra']
        five_f = ['Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
        six_d = ['Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn']
        seven_p = ['Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
        for ao_idx,ao in enumerate(mf.mol.ao_labels(fmt=False)):
            for i in range(n_atoms):
                if ao[0] == i:
                    if ao[1] in one_s:
                        if ao[2] in ('1s'):
                            frag.append(ao_idx)
                    elif ao[1] in two_s:
                        if ao[2] in ('2s'):
                            frag.append(ao_idx)
                    elif ao[1] in two_p:
                        if ao[2] in ('2s', '2p'):
                            frag.append(ao_idx)
                    elif ao[1] in three_s:
                        if ao[2] in ('3s'):
                            frag.append(ao_idx)
                    elif ao[1] in three_p:
                        if ao[2] in ('3s', '3p'):
                            frag.append(ao_idx)
                    elif ao[1] in four_s:
                        if ao[2] in ('4s'):
                            frag.append(ao_idx)
                    elif ao[1] in three_d:
                        if ao[2] in ('4s', '3d'):
                            frag.append(ao_idx)
                    elif ao[1] in four_p:
                        if ao[2] in ('4s', '3d', '4p'):
                            frag.append(ao_idx)
                    elif ao[1] in five_s:
                        if ao[2] in ('5s'):
                            frag.append(ao_idx)
                    elif ao[1] in four_d:
                        if ao[2] in ('5s', '4d'):
                            frag.append(ao_idx)
                    elif ao[1] in five_p:
                        if ao[2] in ('5s', '4d', '5p'):
                            frag.append(ao_idx)
                    elif ao[1] in six_s:
                        if ao[2] in ('6s'):
                            frag.append(ao_idx)
                    elif ao[1] in four_f:
                        if ao[2] in ('6s', '4f'):
                            frag.append(ao_idx)
                    elif ao[1] in five_d:
                        if ao[2] in ('6s', '4f', '5d'):
                            frag.append(ao_idx)
                    elif ao[1] in six_p:
                        if ao[2] in ('6s', '4f', '5d', '6p'):
                            frag.append(ao_idx)
                    elif ao[1] in seven_s:
                        if ao[2] in ('7s'):
                            frag.append(ao_idx)
                    elif ao[1] in five_f:
                        if ao[2] in ('7s', '5f'):
                            frag.append(ao_idx)
                    elif ao[1] in six_d:
                        if ao[2] in ('7s', '5f', '6d'):
                            frag.append(ao_idx)
                    elif ao[1] in seven_p:
                        if ao[2] in ('7s', '5f', '6d', '7p'):
                            frag.append(ao_idx)
        self.n_val_aos = len(frag)
        return frag

    # This function was developed using the work of Daniel Claudino's PsiEmbed
    def spade(self, S, C, n_aos):
        '''
        Parameters
        ----------
        S : np.array
            the overlap matrix
        C : np.array
            the occupied orbitals from the mean-field calculation
        n_orbs : int
            the number of AOs that define the active space to be projected into
        Returns
        -------
        Cact : np.array
            the SPADE MOs in the active space (subsystem A)
        Cenv : np.array
            the SPADE MOs in the environment space (subsystem B)
        '''
        # Using an SVD to perform the SPADE rotation
        X = scipy.linalg.sqrtm(S)
        A = (X @ C)[:n_aos, :]
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        ds = [(s[i] - s[i+1]) for i in range(len(s) - 1)]
        n_act = np.argpartition(ds, -1)[-1] + 1
        n_env = len(s) - n_act
        Cenv = C @ vh.conj().T[:, n_act:]
        Cact = C @ vh.conj().T[:, :n_act]
        return Cact, Cenv
 
    def split_spade(self, S, C, active_orbs, thresh=1.0e-6):
        '''
        Parameters
        ----------
        S : np.array
            the overlap matrix
        C : np.array
            the orbitals from the mean-field calculation used in the partition
        active_orbs : list
            the atomic orbitals to target for projection onto the active space
        thresh : float
            the threshhold used to determine which singular values will define the active space
        Returns
        -------
        Cact : np.array
            the SPADE MOs in the active space (subsystem A)
        Cenv : np.array
            the SPADE MOs in the environment space (subsystem B)
        '''
        # *** To Do: Fix poor convergence when using split-SPADE ***

        X = scipy.linalg.sqrtm(S)
        A = (X @ C)[orb_list,:]
        u,s,vh = np.linalg.svd(A, full_matrices=True)
        nkeep = 0
        for idx,si in enumerate(s):
            if si > thresh:
                nkeep += 1
            print(" Singular value: ", si)
        print(" # of orbitals to keep: ", nkeep)
    
        #Xinv = scipy.linalg.inv(X)
    
        Cact = C @ vh[:nkeep,:].conj().T
        Cenv = C @ vh[nkeep:,:].conj().T
        return Cact, Cenv

    def initial_shell(self, S, C, n_aos, S_pbwb=None):
        '''
        Parameters
        ----------
        S : np.array
            the overlap that defines how the concentric localization is built
        C_eff : np.array
            the orbitals of the virtual space, excluding the projected orbitals of subsystem B
        n_aos : int
            the number of AOs associated with the active space
        S_proj: np.array
            some projected basis # This has not been implemented yet
        Returns
        -------
        Cspan_0 : np.array
            the CL MOs connected to the active space (subsystem A) by the overlap
        Ckern_0 : np.array
            the CL MOs not connected to subsystem A by the overlap
        '''
        # To Do: implement projected bases
        C_eff = np.linalg.inv(S[:n_aos,:n_aos]) @ S_pbwb[:n_aos,:] @ C
        A = C_eff.conj().T @ S[:n_aos,:] @ C
        #nkeep = A.shape[1]
        #Aorth = A[:nkeep,:]
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        s_eff = s[:n_aos]
        self.shell = (s_eff>=1.0e-6).sum()
        Cspan_0 = C @ vh.T[:,:self.shell]
        Ckern_0 = C @ vh.T[:,self.shell:]
        return Cspan_0, Ckern_0

    def build_shell(self, Operator, Cspan_initial, Ckern_initial, shell):
        '''
        Parameters
        ----------
        Operator : np.array
            the chosen single-particle operator that defines how the concentric localization is built
        Cspan_initial : np.array
            the orbitals previously connected to subsystem A
        Ckern_initial : np.array
            the orbitals used to expand into the next shell        
        shell : int
            the number of orbitals that define the shell size
        Returns
        -------
        Cspan : np.array
            the CL MOs connected to the active space (subsystem A) by the operator
        Ckern : np.array
            the CL MOs not connected to subsystem A by the operator
        '''
        # Using an SVD to perform a shell partition in the concentric localization procedure
        A = Cspan_initial.conj().T @ Operator @ Ckern_initial
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        Ckern = Ckern_initial @ vh.conj().T[:, shell:]
        Cspan = Ckern_initial @ vh.conj().T[:, :shell]
        C = np.hstack((Cspan_initial,Ckern_initial))
        M = C.T @ Operator @ C
        return Cspan, Ckern, M

