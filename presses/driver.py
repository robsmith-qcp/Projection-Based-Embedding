import pyscf
import pyscf.tools
import numpy as np
import scipy
from functools import reduce
from .embedding import *
from .orbitals import *

def update_keywords(keywords):
    # Default keywords
    default_keywords = {}
    default_keywords['memory'] = 1000
    default_keywords['charge'] = 0
    default_keywords['spin'] = 0
    default_keywords['scf_method'] = 'hf'
    default_keywords['subsystem_method'] = 'hf'
    default_keywords['conv_tol'] = 1.0e-7
    default_keywords['conv_tol_grad'] = 1.0e-7
    default_keywords['output'] = 'output.txt'
    default_keywords['level_shift'] = 1.0e6
    default_keywords['verbose'] = 5
    default_keywords['basis'] = 'STO-3G'
    default_keywords['xc'] = 'lda,vwn'
    default_keywords['split_spade'] = False
    default_keywords['n_shells'] = 0

    # Checking if the necessary keywords have been defined
    assert 'scf_method' in keywords, '\n Choose level of theory for the initial scf'
    assert 'subsystem_method' in keywords, '\n Choose level of theory for the active region'
    assert 'active_space_atoms' in keywords, '\n Provide the number of atoms whose AOs comprise the active space, which are taken to be first atoms in you coordinate string'
    # Adding any missing keywords
    for key in default_keywords.keys():
        if key not in keywords:
            keywords[key] = default_keywords[key]
    return keywords

def semi_canonicalize(C,F):
    e,v = np.linalg.eigh(C.conj().T @ F @ C)
    C_bar = C @ v
    return C_bar, e

def idempotentcy(D, S):
    idem = (D @ S @ D) - D
    idem = idem.round()
    if not np.all(idem):
        print('\n The density matrix is idempotent!')
    else:
        print('\n Check the density matrix construction.')
        print(idem)
        exit()

def subsystem_orthogonality(Proj,D_A):
    a = mu * Proj.T @ D_A
    a = np.around(a,decimals=10)
    if a.all() != 0.0:
        print('\n Check the subsystems.')
        print(a)
        exit()

def run_embed(keywords):
    '''
    Parameters
    ----------
    options : dict
        keywords to run the embedded calculation using PySCF
    Returns
    -------
    Cact : np.array
        the SPADE MOs in the active space (subsystem A)
    Cenv : np.array
        the SPADE MOs in the environment space (subsystem B)    
    '''
    keywords = update_keywords(keywords)
    Embed = Proj_Emb(keywords)
    Orbs = Partition(Embed)

    # Start with a mean-field calculation on the whole system
    F, V, C, S, P, H = Embed.mean_field()
    
    # indexing code is courtesy of N. Mayhall
    # From the mean-field MOs, group them by occupation number
    docc_list = []
    socc_list = []
    virt_list = []

    for idx,i in enumerate(Embed.mf.mo_occ):
        if i == 0:
            virt_list.append(idx)
        elif i == 1:
            socc_list.append(idx)
        elif i == 2:
            docc_list.append(idx)

    Cdocc = C[:,docc_list]
    Csocc = C[:,socc_list]
    Cvirt = C[:,virt_list]

    # Define the 1-particle density matrices for open-shell systems
    if not Embed.closed_shell:
        Pa = P[0,:,:]
        Pb = P[1,:,:]

    # Acquire the AOs in the active space
    if Embed.keywords['split_spade']:
        frag_list = Orbs.ao_assignment(Embed.mf, Embed.n_atoms)
        print('\n AO indices used for projection: ', frag_list)

    # Use SPADE to rotate the MOs into the active space
    if Embed.keywords['split_spade']:
        Cact, Cenv = Orbs.split_spade(S, Cdocc, frag_list)
    else:
        Cact, Cenv = Orbs.spade(S, Cdocc, Embed.n_aos)
    if not Embed.closed_shell:
        Cact_d = Cact
        Cact_s = Csocc
        Cact = np.hstack((Cact_d,Cact_s))

    # Validate the fidelity of the electron occupation number in the subspace, courtesy of N. Mayhall
    if not Embed.closed_shell:
        na_act = np.trace(Cact.conj().T @ S @ Pa @ S @ Cact)
        na_env = np.trace(Cenv.conj().T @ S @ Pa @ S @ Cenv)
        na_vir = np.trace(Cvirt.conj().T @ S @ Pa @ S @ Cvirt)
        nb_act = np.trace(Cact.conj().T @ S @ Pb @ S @ Cact)
        nb_env = np.trace(Cenv.conj().T @ S @ Pb @ S @ Cenv)
        nb_vir = np.trace(Cvirt.conj().T @ S @ Pb @ S @ Cvirt)
    else:
        na_act = 0.5*np.trace(Cact.conj().T @ S @ P @ S @ Cact)
        na_env = 0.5*np.trace(Cenv.conj().T @ S @ P @ S @ Cenv)
        na_vir = 0.5*np.trace(Cvirt.conj().T @ S @ P @ S @ Cvirt)
        nb_act = 0.5*np.trace(Cact.conj().T @ S @ P @ S @ Cact)
        nb_env = 0.5*np.trace(Cenv.conj().T @ S @ P @ S @ Cenv)
        nb_vir = 0.5*np.trace(Cvirt.conj().T @ S @ P @ S @ Cvirt)

    print(" # electrons: %12s %12s" %("α", "β"))
    print("         Env: %12.8f %12.8f" %(na_env, nb_env))
    print("         Act: %12.8f %12.8f" %(na_act, nb_act))
    print("         Vir: %12.8f %12.8f" %(na_vir, nb_vir))
    na_act = round(na_act)
    nb_act = round(nb_act)
    n_elec = na_act + nb_act
    print('Number of electrons in the active space: ', n_elec)

    
    # Semicanonicalize the two subspaces and print the orbitals for viewing
    Cenv, e_orb_env = semi_canonicalize(Cenv, F)
    Cact, e_orb_act = semi_canonicalize(Cact, F)

    # Generate Molden files to visualize subsystem orbitals    
    pyscf.tools.molden.from_mo(Embed.mf.mol, "Cact.molden", Cact);
    pyscf.tools.molden.from_mo(Embed.mf.mol, "Cenv.molden", Cenv);

    # Build density matrices for each subspace
    if Embed.closed_shell:
        D_B = 2.0 * Cenv @ Cenv.conj().T
        D_A = 2.0 * Cact @ Cact.conj().T
    else:
        Denv = Cenv @ Cenv.conj().T
        D_B = np.stack((Denv,Denv))
        Dact_a = (Cact_d @ Cact_d.conj().T) + (Cact_s @ Cact_s.conj().T)
        Dact_b = Cact_d @ Cact_d.conj().T
        D_A = np.stack((Dact_a,Dact_b))
   
    # Validate the density matrices are idempotent
    if Embed.closed_shell:
        idempotentcy(D_B, S)
        idempotentcy(D_A, S)
    else:
        idempotentcy(D_B[0], S)
        idempotentcy(D_A[0], S)
        idempotentcy(D_A[1], S)

    # Compute the subsystem terms
    Vact, Venv = Embed.subspace_values(D_A, D_B)
    
    # Compute the Projector
    P_B = (S @ D_B @ S)
    mu = 1.0e6

    # Compute the embedded potential
    Vemb = V - Vact + (mu * P_B)
    if not Embed.closed_shell:
        Vemb_a = Vemb[0,:,:]
        Vemb_b = Vemb[1,:,:]
        Vemb_c = 0.5 * (Vemb[0,:,:] + Vemb[1,:,:])
        proj_cl = np.dot(Pb, S)
        proj_op = np.dot(Pa-Pb, S)
        proj_vir = np.eye(S.shape[0]) - np.dot(Pa, S)

        Vemb = 0.5 * reduce(np.dot, (proj_cl.conj().T, Vemb_c, proj_cl))
        Vemb += 0.5 * reduce(np.dot, (proj_op.conj().T, Vemb_c, proj_op))
        Vemb += 0.5 * reduce(np.dot, (proj_vir.conj().T, Vemb_c, proj_vir))
        Vemb += reduce(np.dot, (proj_op.conj().T, Vemb_b, proj_cl))
        Vemb += reduce(np.dot, (proj_op.conj().T, Vemb_a, proj_vir))
        Vemb += reduce(np.dot, (proj_vir.conj().T, Vemb_c, proj_cl))
        Vemb = Vemb + Vemb.conj().T

    H_emb = H + Vemb
        
    # Run the mean-field calculation with an embedded potential
    C_emb, S_emb, F_emb = Embed.embedded_mean_field(int(n_elec), Vemb, D_A) 
    
    P_emb = Embed.emb_mf.make_rdm1()

    # Compute the new mean-field energy
    embed_mf_e = Embed.embed_scf_energy(Vemb, D_A, P_B, mu, H)

    # If the user chooses a DFT-in-DFT approach, just print the mean-field embedding result. Otherwise, proceed with a post-HF method.
    if Embed.keywords['subsystem_method'].lower() == 'hf' or Embed.keywords['subsystem_method'].lower() == 'dft':
        e_tot = embed_mf_e
        print('Total mean-field energy = ', e_tot)
        e_c = 0
    else:
        n_act = round(na_act)
        n_effective = Embed.mol.nao - round(na_env)
        Cocc = C_emb[:,:n_act]
        Cvirt_eff = C_emb[:,n_act:n_effective]
        Cfroz = C_emb[:,n_effective:]

        # Perform concentric localization, if requested.
        if Embed.keywords['n_shells'] == 0:
            correl_e = Embed.correlation_energy(n_effective)
            e_tot = embed_mf_e + correl_e
            print('Total energy = ', e_tot)
            e_c = correl_e
        else:
            n_shells = Embed.keywords['n_shells']
            shell_e = []
            Cspan_0, Ckern_0 = Orbs.initial_shell(S_emb, Cvirt_eff, Embed.n_aos, Embed.S_pbwb)
            shell = Orbs.shell
            print('Shell size: ', shell)
            Cspan_0, e_orb_span = semi_canonicalize(Cspan_0, F_emb)
            Ckern_0, e_orb_kern = semi_canonicalize(Ckern_0, F_emb)
            correl_e = Embed.correlation_energy(n_effective, n_act, Cspan=Cspan_0, Ckern=Ckern_0, e_orb_span=e_orb_span, e_orb_kern=e_orb_kern)
            shell_e.append(correl_e)
            Cspan_old = Cspan_0
            Ckern_old = Ckern_0
            E_old = 0.0
            if n_shells > 1:
                for i in range(n_shells - 1):
                    Cspan_i, Ckern_i =  Orbs.build_shell(F_emb, Cspan_old, Ckern_old, shell)
                    Cspan_i =  np.hstack((Cspan_old,Cspan_i))
                    Cspan_i, e_orb_span_i = semi_canonicalize(Cspan_i, F_emb)
                    Ckern_i, e_orb_kern_i = semi_canonicalize(Ckern_i, F_emb)
                    E_i = Embed.correlation_energy(n_effective, n_act, Cspan=Cspan_i, Ckern=Ckern_i, e_orb_span=e_orb_span_i, e_orb_kern=e_orb_kern_i)
                    shell_e.append(E_i)
                    if E_i == E_old:
                        break
                    print('Shell ', i+1, 'Energy = ', E_i)
                    E_old = E_i
                    Cspan_old = Cspan_i
                    Ckern_old = Ckern_i

            total_e = list(map(lambda i: i+embed_mf_e, shell_e))
            print('Total energy of each shell: ', shell_e)
            e_tot = total_e[-1]
            e_c = shell_e
    return e_tot, Embed.E_init, e_c
