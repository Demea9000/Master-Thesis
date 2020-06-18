"""
This module calculates an impedance matrix based on a small signal model a 
one-transistor-one-RRAM (1T1R) nano wire cell. 
It returns the total impedance matrix as well as impedances for the 
corresponding constituents of the model. 
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn')


def Ccoax(er, t_ox, radius, tline):
    e_0 = 8.85e-12
    num = 2*np.pi*e_0*er
    den = np.log((radius+t_ox)/(radius))
    return num/den * tline

def Cpar(area, distance, er = 18*8.85e-12):
    return area*er/distance

def Zcap(C, pulse_width):
    f = pulse_width2freq(pulse_width)
    return 1/(1j*2*np.pi*f*C)

def Cu_Res(tnm):
    t = tnm*1e9
    ro_bulk = 1.68e-8
    k1 = 20
    k2 = 1+0.5*1/(1+np.exp((t-50)/100))
    ro = ((k1*(1-np.sqrt(t/100)/(1+np.exp((t-200)/100)))+t)*(k2/t))**2*ro_bulk
    return ro

def C_coax_to_coax(L, S, D):
    r = D/2
    C_area = r**2*np.pi
    SA = S*D
    Sm = (SA-C_area)/D
    
    return Cpar(L*D, Sm)

def Cfringe(W, S):
    "C_pi from paper"
    e_di = 3.9*8.85e-12
    C = e_di/(np.pi/2)*np.log(1 + 2*W/S)
    return C


def Outside_Screening(W, D):
    """
    For 35 nm node the NW diameter encloses the interconnects. This function subtracts the relevant areas
    that are not included inthe capacitive coupling bewteen the BL and WL.
    """
    # Circle Segment Screening effect
    d = W/2
    R = D/2
    c = 2*R*np.sqrt(1 - (d/R)**2) # Segment Width
    k = c/(2*R)
    As = R**2*(np.arcsin(k) - k*np.sqrt(1 - k**2)) # Segment Area
    Asr = (R**2*np.pi-2*As-W**2)/2
    for i in range(len(R)):
        if c[i] < d[i]:
            Asr[i] = As[i]
    return Asr

def Corner_Screening(W, D):
    """
    This function removes the areas on the corner of the NW
    that are not included inthe capacitive coupling bewteen the BL and WL
    """
    d = W/2
    R = D/2
    c = 2*R*np.sqrt(1 - (d/R)**2) # Segment Width
    k = c/(2*R)
    As = R**2*(np.arcsin(k) - k*np.sqrt(1 - k**2)) # Segment Area
    Acrnr = -1*(R**2*np.pi-4*As-W**2)/4
    for i in range(len(R)):
        if d[i] > c[i]:
            Acrnr[i] = 0
    return Acrnr
    
def Fringe_Cap(W, S, H, T, x_to_x):
    """
    Different Fringe capacitances between interconnects depending on their dimensions 
    INPUT:
        W: Width
        S: Spacing
        H: Height
        T: Thickness
        x_to_xo: Type of fringe capacitance: sideways (sw) and top combinations.
    """
    t = 3.7
    arg = (W + S - np.sqrt(S**2 + T**2 + 2*H*T))/(t*W)
    n = np.exp(arg)
    a = np.exp(-1*(H + T)/(S + W))
    b = np.exp((S + W)/(H + T))
    e_di = 3.9 * 8.85e-12
    
    if x_to_x == "sw-top":
        C = e_di/(np.pi/2)*np.log((H + n*T + np.sqrt(S**2 + (n*T)**2 + 
                                   2*H*n*T))/(S + H))
    elif x_to_x == "top-top":
        C = e_di*W*a*(np.log(1+2*W/S)+
                        np.exp(-1*(S+T)/(3*S)))/(W*np.pi*a+(H+T)*(np.log(1+2*W/S)+
                                                                np.exp(-1*(S+T)/(3*S))))
    elif x_to_x == "sw-sw":
        k1 = e_di*T*b*(np.log(1 + 2*T/H) + np.exp(-1*(H+W)/(3*H)))
        k2 = T*np.pi*b + (S+W)*(np.log(1+2*T/H)+np.exp(-1*(H+W)/(3*H)))
        
        C = k1/k2
    elif x_to_x == "corner":
        C = e_di/np.pi*np.sqrt((H*S)/(H**2+S**2))
    else:
        C = 1e-25
    return C
    
def pulse_width2freq(pulse_width):
    """
    Get the frequency for a corresponding sinusoidal pulse width
    """
    return 1/(2*pulse_width)

def Rpar(R1, R2):
    """
    Parallel resistance
    """
    return R1*R2/(R1+R2)

def Resistance_Matrix_1T1R(array_size,
                        Rp,
                        Z):
    """
    Function that gives the resistance matrix depending with a given array_size
    """

    m, n = array_size
    R = np.ones((m, n))*Rp
    ZA = np.ones(m)*Z
    R[:, 0] = ZA
    return R

    
def Impedance_Array(include_fringe=False, 
                plot_AV = True,
                vary_resistivity=False,
                cell = "off",
                npoints = 10,
                total_arraysize = 32,
                sub_arraysize = 128,
                subarray=False,
                pulse_width = 10, 
                node = 35,
                rho = 8e-8,
                tox=2.8,
                tg = 5,
                resistivity=1.68e-8,
                Lg = 30,
                Lgs = 80):  
    """
    The main function that gives the Impedance array using a small signal model 
    for each cell
    """

    if subarray == True:
        array_size = sub_arraysize
    elif subarray == False:
        array_size = total_arraysize*sub_arraysize

    e_0 = 8.85e-12 #Perimittivity in Vacuum
    er_ox = 18 # HfO2 relative permittivity

    
    Measurements = np.ones(1)
    Node = node*1e-9*Measurements # Node geometry
    Wp = np.array([pulse_width])*1e-9
    U = Node # Unit length
    W = U/2 # Width of interconnect
    T = W*2 # thickness of interconnect


    # Different Node Sizes

    if node==35:
        dnw = 8*1e-9*Measurements # Nanowire diameter
        Lnw = 320*1e-9*Measurements # Nanowire Length
        Lm = 50*1e-9*Measurements # Length of RRAM NW
        tshell = 2e-9*Measurements # thickness of NW diameter in RRAM 
        tox = tox*1e-9*Measurements # Oxide Thickness
        tg = tg*1e-9*Measurements# Gate thickness
        Lg = Lg*1e-9*Measurements # Gate Length
        S_gs = 80*1e-9*Measurements # Distance between gate and source in FET 
                                    
    if node==50:
        dnw = 12*1e-9*Measurements # Nanowire diameter
        Lnw = 480*1e-9*Measurements # Nanowire Length
        Lm = 75*1e-9*Measurements # Length of RRAM NW
        tshell = 3e-9*Measurements # thickness of NW diameter in RRAM 
        tox = tox*1e-9*Measurements # Oxide Thickness
        tg = 7*1e-9*Measurements# Gate thickness
        Lg = 40*1e-9*Measurements # Gate Length
        S_gs = 120*1e-9*Measurements # Distance between gate and source in FET 

    if node==100:
        dnw = 18*1e-9*Measurements # Nanowire diameter
        Lnw = 720*1e-9*Measurements # Nanowire Length
        Lm = 100*1e-9*Measurements # Length of RRAM NW
        tshell = 4e-9*Measurements # thickness of NW diameter in RRAM 
        tox = tox*1e-9*Measurements # Oxide Thickness
        tg = 10*1e-9*Measurements# Gate thickness
        Lg = 60*1e-9*Measurements # Gate Length
        S_gs = 160*1e-9*Measurements # Distance between gate and source in FET 
    
    H_Wplug = W*2 # Viaplug
    S_WLBL = Lnw-S_gs-T+H_Wplug # Distance WL to BL
    S_WLS = S_WLBL+S_gs # Distance WL till Substrate
    dDEV = dnw+2*(tshell+tox+tg) # total diameter of RRAM + shell and everything
    
    if node==35:
        Ascrn = Outside_Screening(W, dDEV) # NW larger than interconnects
        A_WLS = W**2 - 2*Ascrn
        A_BLS = Outside_Screening(W, dDEV - 2*tg)# NW larger than  interconnects
        A_BLS = W**2 - 2*A_BLS
        A_WLBL = Outside_Screening(W, dDEV)
    else:
        A_WLS = W*U - W**2
        A_BLS = W*U - np.pi*(dDEV/2)**2
        A_WLBL = W**2 - np.pi*(dDEV/2)**2

    Wscrn = A_WLS/W 
    Sscrn = W - Wscrn
    Side_fringe = W * Fringe_Cap(W = Wscrn, S = Sscrn, 
                                H = S_WLBL, T = T, x_to_x = 'sw-top')

    Overlay = Cpar(A_WLBL, S_WLBL, er_ox*e_0)
    C_BLS = Cpar(A_BLS, S_gs, er_ox*e_0)
    C_WLBL = Overlay + 2*Side_fringe

    Ascrn2 = Corner_Screening(W, dDEV)
    Wscrn2 = (W**2 - 4*Ascrn2)/W
    Sscrn2 = (W - Wscrn2)/2
    Side_fringe2 = W * Fringe_Cap(W = Wscrn2, S = Sscrn2, 
                                H = S_WLS, T = T, x_to_x = 'sw-top')
    Overlay2 = Cpar(A_WLS, S_WLS, er_ox*e_0)
    C_WLS = Overlay2 + Side_fringe2

    NW_to_NW = C_coax_to_coax(Lm, U, dDEV) # NW to NW capacitance 
    Via_to_Via = Cpar(W*H_Wplug, W, er_ox*e_0)
    WL_to_WL = Cpar(2*W*T, W) # WL to WL parallel plate capacitance

    range_param = [int(round(2e-6/x)) for x in Node]
    WL_fringeC = np.zeros((len(W), max(range_param)))
    WL_to_WL_fringe = np.zeros(len(Measurements))
    for i in range(len(W)):
        for j in range(range_param[i]):
            WL_fringeC[i][j] = 2*W[i]*Cfringe(W[i], (1+2*j)*W[i])
        WL_to_WL_fringe[i] = sum(WL_fringeC[i])
    


    C_WLWL = 2*(2*NW_to_NW + Via_to_Via + 
                WL_to_WL + 1.5*WL_to_WL_fringe) # Total WL to WL capacitance
    R_FET_off = 20*25e6 # OP transistors
    R_FET_on = 20e3 # OP transistors from nanoletters
    R_RRAM_on = 17.2*1e3 # The RRAM in LRS from experiments
    R_RRAM_off = 18.391*1e6 # The RRAM in HRS from experiments
    C_RRAM = Ccoax(18, tox, dDEV - 2*tg, Lm) # capacitance of RRAM 
    C_FET = Ccoax(18, tox, dDEV -2*(tg + tshell), Lg) # Capacitance of FET
    C_GS = 0.1*C_FET/2 # Gate-source Parasitic FET capacitance 
    C_GD = C_GS # The same for gate-drain parasitic

    Z_WLWL = Zcap(C_WLWL, Wp)
    Z_WLBL = Zcap(C_WLBL, Wp)
    Z_WLS = Zcap(C_WLS, Wp)

    ZC_RRAM = Zcap(C_RRAM, Wp) # Capacitive impedance of RRAM 
    ZR_RRAM = R_RRAM_on # Resistive impedance of RRAM
    ZR_FET = R_FET_off # Resistive impedance of FET in off mode
    ZC_GD = Zcap(C_GD, Wp) # Impedance due to parasitic gate-drain capacitance of FET

    
    if vary_resistivity == True:
        Res = resistivity
    else:
        Res = Cu_Res(W) # Resistivity of copper 

    R_WL = Res*2*W/(W*T) # Line Resistance of WL interconnect

    C_BL = 2*C_BLS + C_GS 
    R_SD = abs(ZC_GD)
    Z_BL = Zcap(C_BL, Wp) # Loss from MOSFET 
    Z_BL = Rpar(Z_BL, R_SD)
    C_WL = C_WLWL + C_WLBL*C_BLS/(C_WLBL+C_BLS) + C_WLS # Total WL capacitance
   

    Z_RRAM = ZC_RRAM*ZR_RRAM/(ZC_RRAM + ZR_RRAM) # Total RRAM impedance
    Z_FET = ZR_FET*ZC_GD/(ZC_GD+ZR_FET) # Total FET impedance
    Z_NW = Z_RRAM + Z_FET # Total nanowire impedance

    Z_RRAM_on = ZC_RRAM*R_RRAM_on/(ZC_RRAM + R_RRAM_on) 
    Z_RRAM_off = ZC_RRAM*R_RRAM_off/(ZC_RRAM + R_RRAM_off) 
    Z_FET_on = R_FET_on*ZC_GD/(ZC_GD+R_FET_on) 
    Z_NW_on = Z_FET_on + Z_RRAM_on 
    Z_NW_off = Z_FET_on + Z_RRAM_off 

    

    Z_WL = Zcap(C_WL, Wp) 
    Z_TOT = Z_NW*Z_WL/(Z_NW+Z_WL) 

    
    N = npoints

    Rvertical = abs(Z_BL)
    Rvertical_NEW = Rvertical
    Rline_new = 0
    Rij_new = 0
    Z = 0

    if plot_AV == True:
        X = np.logspace(0, 4.5, N, base=10.0).astype(int)
    else:
        X = np.ones(1)*array_size

    if cell == "off":
        Zcell = abs(Z_NW_off)
    else:
        Zcell = abs(Z_NW_on)
    u = 0
    Z0 = Zcell[u] 
    R = R_WL[u] 
    Rij = abs(Z_TOT[u]) 
    Rij_new = Rij

    R_BL = np.ones(array_size)
    Z = Z0
    Z2 = Z_BL
    for i in range(array_size):
        Rvertical_NEW = Rpar(Rvertical_NEW, Rvertical)
        R_BL[i] = Rvertical_NEW
        Z2 += R
        Z2 = Z2*Rvertical/(Z2+Rvertical)
    Rvertical_NEW = 0


    for i in range(array_size):
        Rline_new += R
        Rij_new = Rpar(Rij_new, Rij)
        Z = Z + R
        Z = (Z*Rij)/(Z+Rij)
    
    if subarray == True:
        R_Matrix = Resistance_Matrix_1T1R(array_size=(total_arraysize, 
        total_arraysize), Rp=Rij_new, Z=Z)
        return([R_Matrix, Rline_new, abs(Z_NW_on), 
                abs(Z_NW_off), abs(Z_TOT), abs(Z_WL), 
                abs(Z_NW), abs(Z_WLWL), abs(Z_WLBL), 
                abs(Z_WLS), abs(Z_BL), abs(R_SD)])
    

    elif subarray ==  False:
        return Z+Z2
    else:
        raise ValueError("subarray must be true or false")

    
