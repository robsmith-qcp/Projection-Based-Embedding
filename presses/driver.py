import pyscf
import pyscf.tools
import numpy as np
import scipy
import pandas as pd
import time
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
    default_keywords['conv_tol'] = 1.0e-9
    default_keywords['conv_tol_grad'] = 1.0e-6
    default_keywords['output'] = 'output.txt'
    default_keywords['level_shift'] = 1.0e6
    default_keywords['verbose'] = 4
    default_keywords['basis'] = 'STO-3G'
    default_keywords['xc'] = 'lda,vwn'
    default_keywords['split_spade'] = False
    default_keywords['n_shells'] = 0
    default_keywords['n_roots'] = 2
    default_keywords['occupied_shells'] = 0
    default_keywords['operator'] = 'F'
    default_keywords['occ_operator'] = 'P'    
    default_keywords['embedded_xc'] = 'lda,vwn'
    default_keywords['split_cutoff'] = True
    default_keywords['orthog'] = True
    default_keywords['spectrum'] = False

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

def svd_subspace_partitioning(orbitals_blocks, S, frag, orthog=True):
    """
    Find orbitals that most strongly overlap with the projector, P,  by doing rotations within each orbital block. 
    [C1, C2, C3] -> [(C1f, C2f, C3f), (C1e, C2e, C3e)]
    where C1f (C2f) and C1e (C2e) are the fragment orbitals in block 1 (2) and remainder orbitals in block 1 (2).

    Common scenarios would be 
        `orbital_blocks` = [Occ, Virt]
        or 
        `orbital_blocks` = [Occ, Sing, Virt]
    
    P[AO, frag]
    O[AO, occupied]
    U[AO, virtual]
    """
    if orthog:
        X = scipy.linalg.sqrtm(S)
    else:
        X = np.eye(orbital_blocks.shape[0]) 
    Pv = X[:,frag]
    nfrag = Pv.shape[1]
    nbas = S.shape[0]
    assert(Pv.shape[0] == nbas)
    nmo = 0
    for i in orbitals_blocks:
        assert(i.shape[0] == nbas)
        nmo += i.shape[1]


    X = scipy.linalg.sqrtm(S)

    print(" Partition %4i orbitals into a total of %4i orbitals" %(nmo, Pv.shape[1]))
    P = Pv @ np.linalg.inv(Pv.T @ S @ Pv) @ Pv.T


    s = []
    Clist = []
    spaces = []
    Cf = []
    Ce = []
    for obi, ob in enumerate(orbitals_blocks):
        print("starting block", obi)
        _,sob,Vob = np.linalg.svd(X @ P @ S @ ob, full_matrices=True)
        s.extend(sob)
        st = [str(obi),'orth','csv']
        i = '.'.join(st)
        pd.DataFrame(sob).to_csv(i)
        Clist.append(ob @ Vob.T)
        spaces.extend([obi for i in range(ob.shape[1])])
        Cf.append(np.zeros((nbas, 0)))
        Ce.append(np.zeros((nbas, 0)))

    spaces = np.array(spaces)
    s = np.array(s)

    # Sort all the singular values
    perm = np.argsort(s)[::-1]
    print(perm)
    s = s[perm]
    spaces = spaces[perm]

    Ctot = np.hstack(Clist)
    Ctot = Ctot[:,perm]    

    print(" %16s %12s %-12s" %("Index", "Sing. Val.", "Space"))
    for i in range(nfrag):
        print(" %16i %12.8f %12s*" %(i, s[i], spaces[i]))
        block = spaces[i]
        Cf[block] = np.hstack((Cf[block], Ctot[:,i:i+1]))

    for i in range(nfrag, nmo):
        if s[i] > 1e-6:
            print(" %16i %12.8f %12s" %(i, s[i], spaces[i]))
        block = spaces[i]
        Ce[block] = np.hstack((Ce[block], Ctot[:,i:i+1]))

    print("  SVD active space has the following dimensions:")
    print(" %14s %14s %14s" %("Orbital Block", "Environment", "Active"))
    for obi,ob in enumerate(orbitals_blocks):
        print(" %14i %14i %14i" %(obi, Ce[obi].shape[1], Cf[obi].shape[1]))
        assert(abs(np.linalg.det(ob.T @ S @ ob)) > 1e-12)

    return Cf, Ce 


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
    Vne = Embed.Vne
    T = Embed.T        
 
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

    if Embed.n_atoms==0:
        exit()
    # Acquire the AOs in the active space
    if Embed.keywords['split_spade']:
        frag_list = Orbs.ao_assignment(Embed.mf, Embed.n_atoms)
        print('\n AO indices used for projection: ', frag_list)

    # Use SPADE to rotate the MOs into the active space
    if Embed.keywords['split_spade']:
        Cact, Cenv, s = Orbs.split_spade(S, Cdocc, frag_list, Embed.S_proj, cutoff=Embed.keywords['split_cutoff'], orthog=Embed.keywords['orthog'])
        #svd_subspace_partitioning((Cdocc, Csocc, Cvirt), S, frag, orthog=Embed.keywords['orthog'])
        if (Embed.nbas - len(frag_list)) == 0:
            Cact = Cdocc
    else:
        Cact, Cenv, s = Orbs.spade(S, Cdocc, Embed.n_aos, Embed.S_proj, orthog=Embed.keywords['orthog'])
        if (Embed.nbas - Embed.n_aos) == 0:
            Cact = Cdocc
    user_input = Embed.keywords['output_name']
    spectrum_name = [user_input, ".csv"]
    spectrum = ''.join(spectrum_name)
    if Embed.keywords['spectrum']:
        pd.DataFrame(s).to_csv(spectrum)
    # The sizes of the pretinent spaces
    nact = Cact.shape[1]
    nenv = Cenv.shape[1]
    nvirt = Cvirt.shape[1]

    # Concentric Localization of the occupied space
    n_occshells = Embed.keywords['occupied_shells']
    if n_occshells != 0:
        O_occ = Embed.operator_assignment(Embed.keywords['occ_operator'], virt=False)
        occshell = nact
        tot_occshells = nenv//occshell + 1
        Cspan_old = Cact
        Ckern_old = Cenv
        for i in range(n_occshells):
            if i > tot_occshells:
                break
            Cspan_i, Ckern_i =  Orbs.build_shell(O_occ, Cspan_old, Ckern_old)
            Cspan_i =  np.hstack((Cspan_old,Cspan_i))
            Cspan_old = Cspan_i
            Ckern_old = Ckern_i
            #np.savez('occ_span{:04}.npz'.format(i), Cspan_i)
            #np.savez('occ_kern{:04}.npz'.format(i), Ckern_i)
            Cact = Cspan_i
            Cenv = Ckern_i
            print('Original active space', nact)
            print('New active space', Cact.shape[1])
        Orbs.visualize_operator(O_occ, Cenv, 'Occ') 
        Cenv = Ckern_i
        print('Original active space', nact)
        print('New active space', Cact.shape[1])
        nact = Cact.shape[1]

    if not Embed.closed_shell:
        print("Open-Shelled System")
        Cact_d = Cact
        Cact_s = Csocc
        Cact = np.hstack((Cact_d,Cact_s))
        nact = Cact.shape[1]
    print('Number of Active MOs: ', nact)
    print('Number of Environment MOs: ', nenv)
    print('Number of Virtual MOs: ', nvirt)
    '''
    # Concentric Localization of the virtual space
    O_vir = Embed.operator_assignment(Embed.keywords['operator'])
    nshells = Embed.keywords['n_shells']
    if nshells != 0:
        Cspan_0, Ckern_0 = Orbs.initial_shell(S, Cvirt, Embed.n_aos)
        shell = Orbs.shell
        tot_shells = nvirt//shell + 1
        Cspan_old = Cspan_0
        Ckern_old = Ckern_0
        C_list = []
        C_list.append(Cspan_old)
        if nshells > 1:
            for i in range(n_shells - 1):
                if i > tot_shells:
                    break
                Cspan_i, Ckern_i, A_i =  Orbs.build_shell(O_vir, Cspan_old, Ckern_old, shell)
                C_list.append(Cspan_i)
                Cspan_i =  np.hstack((Cspan_old,Cspan_i))
                Cspan_old = Cspan_i
                Ckern_old = Ckern_i        
                np.savez('span{:03}.npz'.format(i), Cspan_i)
                np.savez('kern{:03}.npz'.format(i), Ckern_i)
            Orbs.assemble_operator(O_vir, C_list, Ckern_old, 'Vir')
    '''

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
    #pyscf.tools.molden.from_mo(Embed.mf.mol, "Cspade.molden", Cact);
    
    # Semicanonicalize the two subspaces and print the orbitals for viewing
    Cenv, e_orb_env = semi_canonicalize(Cenv, F)
    Cact, e_orb_act = semi_canonicalize(Cact, F)

    # Generate Molden files to visualize subsystem orbitals    
    user_input = Embed.keywords['output_name']
    active = ["Cact_", user_input, ".molden"]
    environment = ["Cenv_", user_input, ".molden"]
    active_name = ''.join(active)
    environment_name = ''.join(environment)
    pyscf.tools.molden.from_mo(Embed.mf.mol, active_name, Cact);
    pyscf.tools.molden.from_mo(Embed.mf.mol, environment_name, Cenv);

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
    F = F_emb
    P_emb = Embed.emb_mf.make_rdm1()

    # Compute the new mean-field energy
    embed_mf_e = Embed.embed_scf_energy(Vemb, D_A, P_B, mu, H)
    print('Difference in energy: ', (Embed.mfe - embed_mf_e)*627.51)

    # If the user chooses a DFT-in-DFT approach, just print the mean-field embedding result. Otherwise, proceed with a post-HF method.
    if Embed.keywords['subsystem_method'].lower() == 'hf' or Embed.keywords['subsystem_method'].lower() == 'dft':
        e_tot = embed_mf_e
        print('Total mean-field energy = ', e_tot)
        e_c = 0
    else:
        nshells = Embed.keywords['n_shells']
        n_act = round(na_act)
        n_effective = Embed.mol.nao - round(na_env)
        Cocc = C_emb[:,:n_act]
        Cvirt_eff = C_emb[:,n_act:n_effective]
        Cfroz = C_emb[:,n_effective:]

        # Perform concentric localization, if requested.
        if nshells == 0:
            correl_e = Embed.correlation_energy(n_effective)
            e_tot = embed_mf_e + correl_e
            print('Total energy = ', e_tot)
            e_c = correl_e
        else:
            n_shells = Embed.keywords['n_shells']
            shell_e = []
            Cspan, Ckern = Orbs.initial_shell(S_emb, Cvirt_eff, Embed.n_aos, Embed.S_pbwb)
            shell = Orbs.shell
            nvirt = Cvirt_eff.shape[1]
            tot_shells = nvirt//shell + 1
            #np.savez('S.npz', S_emb)
            #np.savez('F.npz', F_emb)
            #np.savez('initial_span.npz', Cspan_0)
            #np.savez('initial_kern.npz', Ckern_0)
            print('Shell size: ', shell)
            print('Maximum number of shells: ',tot_shells)
            Cspan_i, e_orb_span = semi_canonicalize(Cspan, F_emb)
            Ckern_i, e_orb_kern = semi_canonicalize(Ckern, F_emb)
            correl_e = Embed.correlation_energy(n_effective, n_act, Cspan=Cspan_i, Ckern=Ckern_i, e_orb_span=e_orb_span, e_orb_kern=e_orb_kern)
            shell_e.append(correl_e)
            #E_old = 0.0
            O_vir = Embed.operator_assignment(Embed.keywords['operator'])
            if n_shells > 1:
                for i in range(n_shells - 1):
                    start = time.time()
                    Cnew, Ckern =  Orbs.build_shell(O_vir, Cspan, Ckern)
                    Cspan =  np.hstack((Cspan,Cnew))
                    print(Cspan.shape)
                    Cspan_i, e_orb_span_i = semi_canonicalize(Cspan, F_emb)
                    Ckern_i, e_orb_kern_i = semi_canonicalize(Ckern, F_emb)
                    E_i = Embed.correlation_energy(n_effective, n_act, Cspan=Cspan_i, Ckern=Ckern_i, e_orb_span=e_orb_span_i, e_orb_kern=e_orb_kern_i)
                    shell_e.append(E_i)
                    if i == tot_shells:
                        break
                    print('Shell ', i+1, 'Energy = ', E_i)
                    #E_old = E_i
                    finish = time.time() - start
                    print("time: ", finish)
                #C_list = Orbs.C_span_list
                #C_list = np.array(Orbs.C_span_list)
                #print(C_list[0].shape)
                #print(Ckern.shape)
                Orbs.visualize_operator(O_vir, Ckern, 'Vir')
            total_e = list(map(lambda i: i+embed_mf_e, shell_e))
            print('Mean-field energy: ', embed_mf_e)
            print('Total energy of each shell: ', shell_e)
            e_tot = total_e[-1]
            e_c = shell_e
            np.savetxt("results.csv", total_e)

    return e_tot, Embed.E_init, e_c
