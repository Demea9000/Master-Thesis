"""
Passive Array model returns the access voltage of each node for a RRAM crossbar array.
Can be simulated for one run, or by get_bit_map: this function goes through a simulation 
for each node and returns a map of the accessed voltage for the each node as selected. 
For array sizes > 32 get_bit_map will take a significant amount of time. 

Model based on: 

Chen, An. "A comprehensive crossbar array model with solutions 
for line resistance and nonlinear device characteristics." IEEE 
Transactions on Electron Devices 60.4 (2013): 1318-1326.

Article can be found on: https://ieeexplore.ieee.org/abstract/document/6473873
"""
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
from progress.bar import Bar
warnings.filterwarnings('ignore')
plt.style.use('seaborn-dark')


def A_i(RS_WL1, RS_WL2, R_WL, R, i, n):
    lower_diagonal = np.ones(n-1)*(-1)/R_WL
    upper_diagonal = np.ones(n-1)*(-1)/R_WL
    diagonal = (1/R[i])  + (2/R_WL)
    diagonal[0] = 1/RS_WL1[i] + 1/R[i][0] + 1/R_WL
    diagonal[-1] = 1/RS_WL2[i] + 1/R[i][n-1] + 1/R_WL
    
    Ai = np.zeros((n, n))
    u = np.diag(upper_diagonal, k=1)
    l = np.diag(lower_diagonal, k=-1)
    d = np.diag(diagonal)
    Ai += u+l+d
    
    return Ai

def A_matrix(m, n, RS_WL1, RS_WL2, R_WL, R):
    A = np.zeros((m*n, m*n))
    k = 0
    for i in range(m):
        A[k:k+n, k:k+n] = A_i(RS_WL1, RS_WL2, R_WL, R, i, n)
        k += m
    return A

def B_i(R, n, i):
    diagonal = np.ones(n)*(-1)/R[i]
    
    Bi = np.diag(diagonal)
    
    return Bi

def B_matrix(m, n, R):
    B = np.zeros([m*n, m*n])
    k = 0
    for i in range(m):
        B[k:k+n, k:k+n] = B_i(R, n, i)
        k += m
    return B

def C_j(m, n,  R, j):
    Cj = np.zeros([m, n*m])
    for i in range(0, m):
        Cj[i][n*(i) + j] = 1/R[i][j]
    
    return Cj

def C_matrix(m, n, R):

    C = C_j(m, n, R, 0)
    for j in range(1, n):
        C_old = C
        C_add = C_j(m, n, R, j)
        C = np.concatenate((C_old, C_add), axis=0)
        
    return C

def D_j(m, n, RS_BL1, RS_BL2, R_BL, R, j):
    Dj = np.zeros([m, n*m])
    for i in range(m):
        if i == 0:
            Dj[i][j] = (-1/RS_BL1[j] -1/R_BL -1/R[i][j])
            Dj[i][n + j] = 1/R_BL
        elif (1<=i<=(m-2)):
            Dj[i][n*(i-1) + j] = 1/R_BL
            Dj[i][n*(i-0) + j] = (-1/R_BL -1/R[i][j] -1/R_BL)
            Dj[i][n*(i+1) + j] = 1/R_BL
        elif (i == m-1):
            Dj[i][n*(i-1) + j] = 1/R_BL
            Dj[i][n*(i-0) + j] = (-1/RS_BL2[j] -1/R[i][j] -1/R_BL)
            
    return Dj

def D_matrix(m, n, RS_BL1, RS_BL2, R_BL, R):
    #j = 0
    D =D_j(m, n, RS_BL1, RS_BL2, R_BL, R, 0)
    for j in range(1, n):
        D_old = D
        D_add = D_j(m, n, RS_BL1, RS_BL2, R_BL, R, j)
        D = np.concatenate((D_old, D_add), axis=0)
    
    return D

def E_Wi(VAPP_WL1, RS_WL1, VAPP_WL2, RS_WL2, n, i):
    E = np.zeros(n)
    E[0] = VAPP_WL1[i]/RS_WL1[i]
    E[-1] = VAPP_WL2[i]/RS_WL2[i]
    return E

def E_Bj(VAPP_BL1, RS_BL1, VAPP_BL2, RS_BL2, m, j):
    E = np.zeros(m)
    E[0] = -VAPP_BL1[j]/RS_BL1[j]
    E[-1] = -VAPP_BL2[j]/RS_BL2[j]
    return E

def E_W(VAPP_WL1, RS_WL1, VAPP_WL2, RS_WL2, m, n):
    Ew = E_Wi(VAPP_WL1, RS_WL1, VAPP_WL2, RS_WL2, n, i = 0)
    for i in range(1, m):
        E_old = Ew
        E_add = E_Wi(VAPP_WL1, RS_WL1, VAPP_WL2, RS_WL2, n, i)
        Ew = np.concatenate((E_old, E_add), axis=0)
    
    return Ew

def E_B(VAPP_BL1, RS_BL1, VAPP_BL2, RS_BL2, m, n):
    Eb = E_Bj(VAPP_BL1, RS_BL1, VAPP_BL2, RS_BL2, m, j = 0)
    for j in range(1, n):
        E_old = Eb
        E_add = E_Bj(VAPP_BL1, RS_BL1, VAPP_BL2, RS_BL2, m, j)
        Eb = np.concatenate((E_old, E_add), axis=0)
    
    return Eb

def E_matrix(m, n, VAPP_BL1, RS_BL1, VAPP_BL2, RS_BL2, 
      VAPP_WL1, RS_WL1, VAPP_WL2, RS_WL2):
    EW = E_W(VAPP_WL1, RS_WL1, VAPP_WL2, RS_WL2, m, n)
    EB = E_B(VAPP_BL1, RS_BL1, VAPP_BL2, RS_BL2, m, n)
    
    E = np.concatenate((EW, EB), axis=0)
    
    return E

def Kirchhoff_matrix(m, n, R, R_WL, R_BL,
                    VAPP_BL1, RS_BL1, VAPP_BL2, RS_BL2, 
                    VAPP_WL1, RS_WL1, VAPP_WL2, RS_WL2):
    
    A = A_matrix(m, n, RS_WL1, RS_WL2, R_WL, R)
    B = B_matrix(m, n, R)
    C = C_matrix(m, n, R)
    D = D_matrix(m, n, RS_BL1, RS_BL2, R_BL, R)
    
    K1 = np.concatenate((A, B), axis=1)
    K2 = np.concatenate((C, D), axis=1)
    K = np.concatenate((K1, K2), axis=0)
    
    return K

def Resistance_matrix(m, n, Ron, Roff,
                      random = True, On = True,
                      variability = False, sd=1):
    '''
    Returns resistance matrix (cell resistances)
    '''
    if random==True:
        R = np.random.rand(m, n)
        if variability == True:
            for i in range(m):
                for j in range(n):
                    if R[i][j] < 0.5:
                        Roff_var = np.random.normal(Roff, sd, 1)
                        R[i][j] = Roff_var
                    else:
                        Ron_var = np.random.normal(Ron, sd, 1)
                        R[i][j] = Ron_var
        else:
            np.where(R<0.5, Roff, Ron)
    else:
        R = np.ones((m, n))
        if On == True:
            R = R*Ron
        else:
            R = R*Roff
    return R

def Resistance_1T1R_Matrix(array_size, 
                        SelWL, 
                        SelBL, 
                        Ron,
                        Roff,
                        Rp,
                        Z,
                        FEToff):
    m, n = array_size
    R = Resistance_matrix(m, n, Ron = Ron, Roff = Roff, sd=1)
    R = np.ones((m, n))*FEToff
    R[SelBL, :] = Rp
    R[SelBL, 0] = Z
    return R
    

def Vout(m, n, Rs, Ron, Roff, state="LRS"):
    rs = Rs/Ron
    k = Roff/Ron
    
    if state == "LRS":
        V_out_min = 1/(1/rs + m)
        V_out_max = k/(k/rs + (m-1)+k)
        
        return (V_out_max, V_out_min)
    
    if state == "HRS":
        V_out_min = 1/(k/rs + k*(m - 1) + 1)
        V_out_max = 1/(k/rs + m)
        
        return (V_out_max, V_out_min)
    else:
        return 0


def simulate(senseresistance_normalized = 1e-5, 
        line_resistance = 1e-2,
        random = True,
        bias_scheme = "V/2", 
        SelWL = -1, 
        SelBL = -1,
        Ron = 10e3,
        print_it = False, 
        array_size=(10, 10)):
    Vdd = 1
    V_SWL = Vdd
    if bias_scheme == "V/2":
        V_UWL = Vdd/2
        V_SBL = 0
        V_UBL = Vdd/2
    elif bias_scheme == "V/3":
        V_UWL = Vdd/3
        V_SBL = 0
        V_UBL = 2*Vdd/3
    else:
        V_UWL = 0
        V_SBL = 0
        V_UBL = 0

    Roff = 10*Ron
    R_L = line_resistance
    R_access = 1
    
    M, N = array_size
    
    Sel_WL, Sel_BL = (SelWL, SelBL)
    
    R = Resistance_matrix(M, N, Ron, Roff, random = random, On = False, 
                          variability=True, sd=1e-1*Ron)

    
    R_WL = R_L ; R_BL = R_L
    
    VAPP_WL1 = V_UWL*np.ones(M) ; VAPP_WL1[Sel_WL] = V_SWL
    VAPP_WL2 = 0*np.ones(M) ; #VAPP_WL2[Sel_WL] = V_SWL
    
    VAPP_BL1 = V_UBL*np.ones(N) ; VAPP_BL1[Sel_BL] = V_SBL
    VAPP_BL2 = 0*np.ones(N) ; #VAPP_BL2[-1] = 0
    
    RS_WL1 = R_access*np.ones(M) ; #RS_WL1[Sel_WL] = R_access
    RS_WL2 = 1e9*np.ones(M) ; #RS_WL2[-1] = 10000
    
    RS_BL1 = R_access*np.ones(N) ; #RS_BL1[Sel_BL] = R_access
    RS_BL2 = 1e9*np.ones(N) ; #RS_BL2[Sel_BL] = senseresistance_normalized*Ron
    
    start = time.time()
    K = Kirchhoff_matrix(M, N, R, R_WL, R_BL,
                        VAPP_BL1, RS_BL1, VAPP_BL2, RS_BL2, 
                        VAPP_WL1, RS_WL1, VAPP_WL2, RS_WL2)
    stop = time.time()
    E = E_matrix(M, N, VAPP_BL1, RS_BL1, VAPP_BL2, RS_BL2, 
          VAPP_WL1, RS_WL1, VAPP_WL2, RS_WL2)
    
    
    start = time.time()
    V = np.linalg.solve(K, E)

    stop = time.time()
    
    end = int(len(V)/2)
    V_WL = V[:end]/Vdd
    V_BL = V[end:]/Vdd
    
    assert(all(V_BL != V_WL))
    V_WL = np.reshape(V_WL, (M, N))
    V_BL = np.reshape(V_BL, (M, N))
    
    V_CELL = V_WL - V_BL
    
    
    plt.imshow(V_WL, interpolation='none', cmap='viridis')
    plt.title("Word Line Voltage")
    plt.grid(False)
    plt.colorbar()
    plt.clim(0, 1)
    plt.show()
    
    plt.imshow(V_BL, interpolation='none', cmap='viridis')
    plt.title("Bit Line Voltage")
    plt.grid(False)
    plt.colorbar()
    plt.clim(0, 1)
    plt.show()
    
    plt.imshow(V_CELL, interpolation='none', cmap='viridis')
    plt.title("Voltage Difference")
    plt.grid(False)
    plt.colorbar()
    plt.clim(0, 1)
    plt.show()
    
    Sense_resistance = RS_BL2[Sel_BL]
    SM = Sense_Margin(M, N, Sense_resistance, Ron, Roff)
    if print_it == True:
        print("Sensing Margin:", SM, "V")
        print("Voltage over selected cell:", V_CELL[Sel_WL][Sel_BL])

    return {"Sensing Margin" : SM,
            "Delivered Voltage" : V_CELL[Sel_WL][Sel_BL]}

def get_bit_map(m, n, bias_scheme = "V/2", plot = True, random = True, 
                    line_resistance = 1e-6, all_on=True):

    """
    Function that returns the access voltage of a passive crossbar array with each 
    node selected. 
    INPUT:
        m, n = array dimensions (should be square)
        bias_scheme: two different bias schemes: V/2 or V/3 give different results!
        plot: Boolean, returns a plot if true.
        random: Should the resistance (LRS or HRS) of each RRAM node be randomly 
                generated throughout the array?
        line_resistance: the line resistance of the array
        all_on: Boolean, if true, all the resistances are in LRS. 
    """
    
    start = time.time()

    Vdd = 1
    V_SWL = Vdd
    if bias_scheme == "V/2":
        V_UBL = Vdd/2
        V_UWL = Vdd/2
    else:
        V_UBL = 2*Vdd/3
        V_UWL = Vdd/3
    V_SBL = 0
    Ron = 10e3
    Roff = 10*Ron
    R_L = line_resistance*Ron
    R_access = 1 
    M = m ; N = n
    Map = np.ones((M, N))
    bar = Bar('Progress', max = M)
    for i in range(M):
        bar.next()
        for j in range(N):
            
            Sel_WL = i ; Sel_BL = j
    
            R = Resistance_matrix(M, N, Ron, Roff, random = random, On = all_on, 
                                  variability=True, sd=1e-4*Ron)
            
            R_WL = R_L ; R_BL = R_L
            
            VAPP_WL1 = V_UWL*np.ones(M) ; VAPP_WL1[Sel_WL] = V_SWL
            VAPP_WL2 = 0*np.ones(M) ; #VAPP_WL2[Sel_WL] = V_SWL
            
            VAPP_BL1 = V_UBL*np.ones(N) ; VAPP_BL1[Sel_BL] = V_SBL
            VAPP_BL2 = 0*np.ones(N) ; #VAPP_BL2[Sel_BL] = V_SBL
            
            RS_WL1 = R_access*np.ones(M) ; #RS_WL1[Sel_WL] = R_access
            RS_WL2 = 1e9*np.ones(M) ; #RS_WL2[-1] = 10000
            
            RS_BL1 = R_access*np.ones(N) ; #RS_BL1[Sel_BL] = R_access
            RS_BL2 = 1e9*np.ones(N) ; #RS_BL2[Sel_BL] = 0.1*Ron
            
            K = Kirchhoff_matrix(M, N, R, R_WL, R_BL,
                                VAPP_BL1, RS_BL1, VAPP_BL2, RS_BL2, 
                                VAPP_WL1, RS_WL1, VAPP_WL2, RS_WL2)
            E = E_matrix(M, N, VAPP_BL1, RS_BL1, VAPP_BL2, RS_BL2, 
                  VAPP_WL1, RS_WL1, VAPP_WL2, RS_WL2)
            
            V = np.linalg.solve(K, E)
            
            end = int(len(V)/2)
            V_WL = V[:end]/Vdd
            V_BL = V[end:]/Vdd
            
            assert(all(V_BL != V_WL))
            
            V_WL = np.reshape(V_WL, (M, N))
            V_BL = np.reshape(V_BL, (M, N))
            
            V_CELL = V_WL - V_BL
            Map[i][j] = V_CELL[i][j]
        
        time.sleep(1)

    
    bar.finish()

    if plot==True:
        plt.imshow(Map, interpolation='none', cmap='Blues')
        plt.title("BitMap")
        plt.xlabel("Bit Line")
        plt.ylabel("Word Line")
        plt.grid(False)
        plt.colorbar()
        plt.clim(0, 1)
        plt.show()
    else:
        stop = time.time()


if __name__ == "__main__":
    get_bit_map(8, 8)

