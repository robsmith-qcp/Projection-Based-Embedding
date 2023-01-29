import pyscf
import pyscf.tools
import numpy as np
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
    default_keywords['conv_tol'] = 1.0e-8
    default_keywords['conv_tol_grad'] = 1.0e-5
    default_keywords['output'] = 'output.txt'
    default_keywords['level_shift'] = 1.0e6
    default_keywords['verbose'] = 5
    default_keywords['basis'] = 'STO-3G'
    default_keywords['xc'] = 'lda,vwn'

    # Checking if the necessary keywords have been defined
    assert 'scf_method' in keywords, '\n Choose level of theory for the initial scf'
    assert 'subsystem_method' in keywords, '\n Choose level of theory for the active region'
    assert 'active_space_atoms' in keywords, '\n Provide the number of atoms whose AOs comprise the active space, which are taken to be first atoms in you coordinate string'
    # Adding any missing keywords
    for key in default_keywords.keys():
        if key not in keywords:
            keywords[key] = default_keywords[key]
    return keywords

# This function is courtesy of N. Mayhall
def semi_canonicalize(C,F):
    e,V = np.linalg.eigh(C.conj().T @ F @ C)
    for ei in e:
        print(" Orbital Energy: ", ei)
    return C @ V

def idempotentcy(D, S):
    idem = D @ S @ D - D
    idem = idem.round()
    if not np.all(idem):
        print('\n The density matrix is idempotent!')
    else:
        print('\n Check the density matrix construction.')
        print(idem)

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
    Embed = embedding.Proj_Emb(keywords)
    Orbs = orbitals.Partition(Embed)

    # Start with a mean-field calculation on the whole system
    F, J, K, C, S, P = Embed.mean_field()
    
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

    # Acquire the list of AOs in the active space
    # ToDo: test for frag list
    frag_list = Orbs.ao_assignment()
    print('\n AO indices used for projection: ', frag_list)

    # Use SPADE to rotate the MOs into the active space
    # ToDo: test for spade orbitals
    if Embed.closed_shell:
        Cact, Cenv = Orbs.spade(S, Cdocc)
    else:
        Cact, Cenv = Orbs.spade(S, Cdocc, Csocc)

    # Validate the fidelity of the electron occupation number in the subspace, courtesy of N. Mayhall
    # ToDo: test that the subsystems do not have fractional occupations of electrons
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

    '''
    # Semicanonicalize the two subspaces and print the orbitals for viewing
    Cenv = semi_canonicalize(Cenv, F)
    Cact = semi_canonicalize(Cact, F)
    '''

    pyscf.tools.molden.from_mo(Embed.mf.mol, "Cact.molden", Cact);
    pyscf.tools.molden.from_mo(Embed.mf.mol, "Cenv.molden", Cenv);

    # Build density matrices for each subspace
    # ToDo: test idempotentcy of embedded matrices
    D_B = 2.0 * Cenv @ Cenv.conj().T
    Dact = Cact @ Cact.conj().T
    if Embed.closed_shell:
        D_A = 2.0 * Dact
    else:
        Dact_s = np.hstack((Cact, Csocc)) @ np.hstack((Cact, Csocc)).T # all singly occupied orbitals must be included the embedded subsystem
        D_A = Dact_s + Dact    
    # Validate the density matrices are idempotent
    idempotentcy(D_B, S)
    idempotentcy(D_A, S)

    # Compute the subsystem terms
    g_A = Embed.subspace_values(D_A, D_B)
    
    # Compute the Projector
    P_B = (S @ D_B @ S)
    mu = 1.0e6

    # Computing the number of orbitals in each subspace
    H_act = Cact.conj().T @ Embed.H_core @ Cact;
    nact = H_act.shape[0]
    H_env = Cenv.conj().T @ Embed.H_core @ Cenv;
    nenv = H_env.shape[0]
    print('\n # active orbitals: ', nact)
    print('\n # environment orbitals: ', nenv)

    # Compute the embedded potential
    Vemb = Embed.mf.get_veff(Embed.mol) - g_A + mu * P_B
    #Vemb = J - 0.5 * K + P - Jact + 0.5 * Kact 
    if not Embed.closed_shell:
        Vemb = 0.5 * (Vemb[0,:,:] + [1,:,:]) # TO DO: Resolve ROHF/ROKS embedded potential inclusion in the core Hamiltonian
    H_emb = H_core + Vemb
        
    # Run the mean-field calculation with an embedded potential
    n_elec = na_act + nb_act
    C_emb, S_emb, F_emb = Embed.embedded_mean_field(n_e) 
    
    emb_docc_list = []
    emb_socc_list = []
    emb_virt_list = []

    for idx,i in enumerate(Embed.emb_mf.mo_occ):
        if i == 0:
            emb_virt_list.append(idx)
        elif i == 1:
            emb_socc_list.append(idx)
        elif i == 2:
            emb_docc_list.append(idx)
        
    Cdocc_emb = C_emb[:, emb_docc_list]
    Csocc_emb = C_emb[:, emb_socc_list]
    Cvirt_emb = C_emb[:, emb_virt_list]
    P_emb = Embed.emb_mf.make_rdm1()

    # Compute the new mean-field energy
    if closed_shell:
        Cocc = Cdocc_emb
    else:
        Cocc = np.hstack((Cdocc_emb, Csocc_emb))
    embed_mf_e = Embed.embed_scf_energy(Cocc, Vemb, D_A)

    if correlated_method.lower() == 'hf' or if correlated_method.lower() == 'dft':
        e_tot = embed_mf_e
        print('Total mean-field energy = ', embed_mf_e)
    else:
    # To Do: this would probably be better reworked and called as a function in the Proj_Emb class
    # Starting concentric localization
        if not concentric_localization:
            if correlated_method.lower() == 'mp2':
                mymp = pyscf.mp.MP2(Embed.embed_mf).run()
                correl_e = mymp.e_corr
            elif correlated_method.lower() == 'ccsd':
                mycc = pyscf.cc.CCSD(Embed.embed_mf).run()
                et = mycc.ccsd_t()
                correl_e = mycc.e_corr
                correl_e += et
            elif correlated_method.lower() == 'fci':
                # To Do: code FCI solver
                print('Requested correlation method has not yet been implemented.')
                print('Total mean-field energy = ', embed_mf_e)
                break
            elif correlated_method.lower() == 'eom-ccsd':
                mycc = pyscf.cc.CCSD(Embed.embed_mf).kernel()
                e_ee, c_ee = mycc.eeccsd(nroots=2)
                correl_e = mycc.e_ee
            elif correlated_method.lower() == 'sf-eom-ccsd':
                mycc = pyscf.cc.CCSD(mf).kernel()
                e_sf, c_sf = mycc.eomsf_ccsd(nroots=2)
                correl_e = mycc.e_sf
                J = 0.5 * (e_sf[1] - e_sf[0]) * 219474.63
                print('Exchange coupling constant: %f cm^-1' %J)
            elif correlated_method.lower() == 'ip-eom-ccsd':
                mycc = pyscf.cc.CCSD(mf).kernel()
                e_ip, c_ip = mycc.ipccsd(nroots=2)
                correl_e = mycc.e_ip
                I = (e_ip[1] - e_ip[0]) * 2625.5002
                print('Ionization potential: %f kJ/mol' %I)
            elif correlated_method.lower() == 'ea-eom-ccsd':
                mycc = pyscf.cc.CCSD(mf).kernel()
                e_ea, c_ea = mycc.eaccsd(nroots=2)
                correl_e = mycc.e_ea
                E = (e_ea[1] - e_ea[0]) * 2625.5002
                print('Electron affinity: %f kJ/mol' %E)
            else:
                print('Requested correlation method has not yet been implemented.')
                print('Total mean-field energy = ', embed_mf_e)
                break
            print('Correlation energy = ', correl_e)    
            e_tot = embed_mf_e + correl_e
            print('Total energy = ', e_tot)
        else: # Starting concentric localization
            shell_e = []
            shift = Embed.mol.nao - nenv
            span_0, kernel_0, s0, shell = Embed.shell_0(shift, C_emb, nact, S_emb, n_active_aos)
            #print(s0, 0)
            span0_e, span0_c = Orbs.orbitals(span_0, F_emb)
            kenerl0_e, kernel0_c = Orbs.orbitals(kernel_0, F_emb)
            #print(span0_c.shape, kernel0_c.shape)
            E_0 = Embed.correlation(nact, span0_e, span0_c, kenerl0_e, kernel0_c, Cocc, n_active_aos, shift, C_emb)
            print('Shell ', 0, 'Energy = ', E_0)
            C_span = span0_c
            C_kernel = kernel0_c
            E_old = 0
            if n_shells > 0:
                for i in range(n_shells):
                    C_op = C_span.conj().T @ F_emb @ C_kernel
                    span_i, kernel_i, s_i = Orbs.shell_constr(C_op, shell, C_kernel, C_span)
                    span_i_e, C_span = Orbs.orbitals(span_i, F_emb)
                    kenerl_i_e, C_kernel = Orbs.orbitals(kernel_i, F_emb)
                    E_i = Embed.correlation(nact, span_i_e, C_span, kenerl_i_e, C_kernel, Cdocc, n_active_aos, shift, C_emb)
                    if E_i == E_old:
                        break
                    print('Shell ', i+1, 'Energy = ', E_i)
                    E_old = E_i

            dup = shell_e.pop()
            total_e = list(map(lambda i: i+embed_SCF, shell_e))
            print('Total energy of each shell: ', shell_e)
            e_tot = total_e[-1]
    return e_tot
