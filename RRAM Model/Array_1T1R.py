"""
Module that simulates 1T1R performance. The matrices included are A, B, C, D, Ew, Eb 
and 'Kirchoff' calculate Kirchoff's continuity equations for each node of a given 
array size. The function "simulate" simulates the array. 

"""

# Import dependencies
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

def Rpar(R1, R2):
    """
    Resistances coupled in parallel
    """
    return R1*R2/(R1+R2)
    


def simulate(RMatrix = np.ones([32, 32]),
        line_resistance = 10,
        access_resistance = 1,
        supply_voltage=1,
        array_size=(10, 10)):
    """
    Function that simulates an RRAM 1T1R crossbar array and gives the access voltage of 
    each node. The BL acts as the line connected to the gate of the transistor. 

    INPUTS:
        RMatrix: a resistance matrix representing the resistance of each node.
        line_resistance: the line resistance of the array 
        access_resistance: the access resistance for each access point 
                            (should be very small, about 1-10 Ohm)
        supply_voltage: the applied voltage on the array
        array_size: the dimensions of the crossbar array
    RETURNS:
        Array of the access voltage to each node.

    """
    Vdd = supply_voltage

    R_L = line_resistance
    R_access = access_resistance # access resistance (small)
    M, N = array_size
    
    R = RMatrix

    R_WL = R_L ; R_BL = 10 # R_BL is very small due to the integration of 1T1R subarrays. Without subarrays 
                            # R_BL and R_WL should be the same
    
    VAPP_WL1 = Vdd*np.ones(M) 
    VAPP_WL2 = 0*np.ones(M) 
    
    VAPP_BL1 = 0*np.ones(N) 
    VAPP_BL2 = 0*np.ones(N) 
    
    RS_WL1 = R_access*np.ones(M) 
    RS_WL2 = 1e10*np.ones(M) 
    
    RS_BL1 = R_access*np.ones(N) 
    RS_BL2 = R_access*np.ones(N) 
    
    # The total Matrix
    K = Kirchhoff_matrix(M, N, R, R_WL, R_BL,
                        VAPP_BL1, RS_BL1, VAPP_BL2, RS_BL2, 
                        VAPP_WL1, RS_WL1, VAPP_WL2, RS_WL2)
    
    E = E_matrix(M, N, VAPP_BL1, RS_BL1, VAPP_BL2, RS_BL2, 
          VAPP_WL1, RS_WL1, VAPP_WL2, RS_WL2)
    
    # solve the 2*n*m equations
    V = np.linalg.solve(K, E)

    
    end = int(len(V)/2)
    V_WL = V[:end]
    V_BL = V[end:]
    
    assert(all(V_BL != V_WL))
    V_WL = np.reshape(V_WL, (M, N))
    V_BL = np.reshape(V_BL, (M, N))
    
    # The access voltage is the potential difference between the WL and BL planes. 

    Vaccess = V_WL - V_BL
    return Vaccess