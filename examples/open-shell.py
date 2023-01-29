from driver import *

def test():
    keywords = {}
    keywords['scf_method'] = 'hf' # If dft is chosen, the user should specify the desired exchange-correlation functional
    keywords['subsystem_method'] = 'hf'
    keywords['concen_local_shells'] = 0 # must be some integer
    keywords['atom'] = '''
                       O        0.4631077991      4.5935896533     -0.3012482998                 
                       C       -0.0687996520      4.1570363336      0.9150856771                 
                       H        0.3056936341      3.1300856517      1.1164657398                 
                       H        0.3056966388      4.8215437447      1.7235467359                 
                       C       -2.1289415194      3.2278695090     -0.2237367915                 
                       C       -2.1467606806      3.6790299970      2.2469243874                 
                       C       -2.1289373058      5.5983262617      0.6270434815                 
                       C       -1.6151790368      4.1676507193      0.8855193743                 
                       H       -1.7704752042      5.9777330647     -0.3538736681                 
                       H       -3.2402492577      5.6200553067      0.6186508990                 
                       H       -1.7733698597      6.2913995243      1.4198998061                 
                       H       -1.7733771625      2.1887958189     -0.0525650581                 
                       H       -3.2402535302      3.2164401300     -0.2440302684                 
                       H       -1.7704794939      3.5588244600     -1.2220437839                 
                       H       -3.2583123087      3.6754392735      2.2569344213                 
                       H       -1.7943354448      2.6462589914      2.4585447377                 
                       H       -1.7943324313      4.3415175887      3.0669897442             
                       '''
    keywords['active_space_atoms'] = 4 # The embedded subsystem is developed from the AOs centered on the first 4 atoms in the above string
    keywords['basis'] = '6-31G*'
    keywords['spin'] = 0 # in PySCF this is the number of unpaired electrons, not 2s+1
    keywords['charge'] = -1

    run_embed(keywords)

if __name__== "__main__":
    test()
