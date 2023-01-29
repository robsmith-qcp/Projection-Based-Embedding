import pyscf
import numpy as np
import scipy
from .embedding import *

class Partition:
    '''
    A class to hold objects associated with the orbital rotations and partitions
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

    def ao_assignment(self):
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
        for i in range(self.keywords['active_space_atoms']):
            for ao_idx,ao in enumerate(self.mf.mol.ao_labels(fmt=False)):
                if ao[0] == i:
                    if ao[1] in one_s:
                        if ao[2] in ('1s'):
                            frag.append(ao_idx)
                    elif ao[1] in two_s:
                        if ao[2] in ('1s', '2s'):
                            frag.append(ao_idx)
                    elif ao[1] in two_p:
                        if ao[2] in ('1s', '2s', '2p'):
                            frag.append(ao_idx)
                    elif ao[1] in three_s:
                        if ao[2] in ('1s', '2s', '2p', '3s'):
                            frag.append(ao_idx)
                    elif ao[1] in three_p:
                        if ao[2] in ('1s', '2s', '2p', '3s', '3p'):
                            frag.append(ao_idx)
                    elif ao[1] in four_s:
                        if ao[2] in ('1s', '2s', '2p', '3s', '3p', '4s'):
                            frag.append(ao_idx)
                    elif ao[1] in three_d:
                        if ao[2] in ('1s', '2s', '2p', '3s', '3p', '4s', '3d'):
                            frag.append(ao_idx)
                    elif ao[1] in four_p:
                        if ao[2] in ('1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p'):
                            frag.append(ao_idx)
                    elif ao[1] in five_s:
                        if ao[2] in ('1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s'):
                            frag.append(ao_idx)
                    elif ao[1] in four_d:
                        if ao[2] in ('1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d'):
                            frag.append(ao_idx)
                    elif ao[1] in five_p:
                        if ao[2] in ('1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p'):
                            frag.append(ao_idx)
                    elif ao[1] in six_s:
                        if ao[2] in ('1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p', '6s'):
                            frag.append(ao_idx)
                    elif ao[1] in four_f:
                        if ao[2] in ('1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p', '6s', '4f'):
                            frag.append(ao_idx)
                    elif ao[1] in five_d:
                        if ao[2] in ('1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p', '6s', '4f', '5d'):
                            frag.append(ao_idx)
                    elif ao[1] in six_p:
                        if ao[2] in ('1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p', '6s', '4f', '5d', '6p'):
                            frag.append(ao_idx)
                    elif ao[1] in seven_s:
                        if ao[2] in ('1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p', '6s', '4f', '5d', '6p', '7s'):
                            frag.append(ao_idx)
                    elif ao[1] in five_f:
                        if ao[2] in ('1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p', '6s', '4f', '5d', '6p', '7s', '5f'):
                            frag.append(ao_idx)
                    elif ao[1] in six_d:
                        if ao[2] in ('1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p', '6s', '4f', '5d', '6p', '7s', '5f', '6d'):
                            frag.append(ao_idx)
                    elif ao[1] in seven_p:
                        if ao[2] in ('1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p', '6s', '4f', '5d', '6p', '7s', '5f', '6d', '7s'):
                            frag.append(ao_idx)
        self.n_aos = len(frag)
        return frag

    def spade(self, S, Cdocc, Csocc=None):
        '''
        Parameters
        ----------
        S : np.array
            the overlap matrix
        Cdocc : np.array
            the doubly occupied orbitals from the mean-field calculation
        Csocc : np.array
            the singly occupied orbitals from the mean-field calculation
        Returns
        -------
        Cact : np.array
            the SPADE MOs in the active space (subsystem A)
        Cenv : np.array
            the SPADE MOs in the environment space (subsystem B)
        '''
        # Using an SVD to perform the SPADE rotation
        A = scipy.linalg.sqrtm(S)
        X = (A @ Cdocc)[:self.n_aos, :]
        u, s, vh = np.linalg.svd(X, full_matrices=True)
        ds = [(s[i] - s[i+1]) for i in range(len(s) - 1)]
        n_act = np.argpartition(ds, -1)[-1] + 1
        n_env = len(s) - n_act
        Cenv = Cdocc @ vh.conj().T[:, n_act:]
        if self.closed_shell:
            Cact = Cdocc @ vh.conj().T[:, :n_act]
        else:
            X_s = (A @ Csocc)[:self.n_aos, :]
            u, s, vh_s = np.linalg.svd(X_s, full_matrices=True)
            ds = [(s[i] - s[i+1]) for i in range(len(s) - 1)]
            n_act_s = np.argpartition(ds, -1)[-1] + 1
            n_env_s = len(s) - n_act_s
            Cdocc_act = Cdocc @ vh.conj().T[:, :n_act]
            Csocc_act = Csocc @ vh_s.conj().T[:, :n_act]
            Cact = np.hstack((Cdocc_act, Csocc_act))
        '''
        A = scipy.linalg.sqrtm(S)
        Corth = A @ C
    
        u,s,vh = np.linalg.svd(Corth[orb_list,:])
        nkeep = 0
        for idx,si in enumerate(s):
            if si > thresh:
                nkeep += 1
            print(" Singular value: ", si)
        print(" # of orbitals to keep: ", nkeep)
    
        Xinv = scipy.linalg.inv(X)
    
        Cact = Xinv @ Corth @ V[0:nkeep,:].conj().T
        Cenv = Xinv @ Corth @ V[nkeep::,:].conj().T
        '''
        return Cact, Cenv

