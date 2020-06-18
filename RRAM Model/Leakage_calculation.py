
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

def AllCaps(W, S, H, T):

    """All Fringe Capacitances per unit length"""
    t = 3.7; #fitting parameter
    n = np.exp(W + S - np.sqrt(S**2 + T**2 + 2*H*T)/(t*W))
    a = np.exp(-1*(H+T)/(S+W))
    b = np.exp((S + W)/(H + T))
    e_di = 3.9*8.85e-12
    
    C = W*e_di/(np.pi/2)*np.log((H + n*T + np.sqrt(S**2 + (n*T)**2 + 2*H*n*T))/(S + H)) \
        + 2*e_di*W*a*(np.log(1 + 2*W/S)+ np.exp(-1*(S+T)/(3*S)))/(W*np.pi*a+(H+T)*(np.log(1 + 2*W/S)+np.exp(-1*(S+T)/(3*S)))) \
        + 2*e_di*T*b*(np.log(1 + 2*T/H) + np.exp(-1*(H+W)/(3*H)))/(T*np.pi*b+(S+W)*(np.log(1 + 2*T/H) + np.exp(-1*(H+W)/(3*H)))) \
        + e_di/np.pi*np.sqrt((H*S)/(H**2+S**2))
    return C

def Inside_Screening(W, D):
    # Circle Segment Screening effect
    d = W/2
    R = D/2
    
    c = 2*R*np.sqrt(1 - (d/R)**2) # Segment Width
    k = c/(2*R)
    As = R**2*(np.arcsin(k) - k*np.sqrt(1 - k**2)) # Segment Area
    Asr = W**2 - (R**2*np.pi-2*As)
    for i in range(len(R)):
        if c[i] > d[i]:
            Asr[i] = 0
    return Asr

def Outside_Screening(W, D):
    # Circle Segment Screening effect
    d = W/2
    R = D/2
    c = 2*R*np.sqrt(1 - (d/R)**2) # Segment Width
    k = c/(2*R)
    h = R-np.sqrt(R**2-c**2/4)
    As = R**2*(np.arcsin(k) - k*np.sqrt(1 - k**2)) # Segment Area
    Asr = (R**2*np.pi-2*As-W**2)/2
    for i in range(len(R)):
        if d[i] > c[i]:
            Asr[i] = As[i]
    return Asr

def Corner_Screening(W, D):
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
    
def IL_FringeCapX(W, S, H, T, x_to_x):
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
        C = e_di*W*a*(np.log(1+2*W/S)+np.exp(-1*(S+T)/(3*S)))/(W*np.pi*a+(H+T)*(np.log(1+2*W/S)+np.exp(-1*(S+T)/(3*S))))
    elif x_to_x == "sw-sw":
        k1 = e_di*T*b*(np.log(1 + 2*T/H) + np.exp(-1*(H+W)/(3*H)))
        k2 = T*np.pi*b + (S+W)*(np.log(1+2*T/H)+np.exp(-1*(H+W)/(3*H)))
        
        C = k1/k2
    elif x_to_x == "corner":
        C = e_di/np.pi*np.sqrt((H*S)/(H**2+S**2))
    else:
        C = 1e-25
    return C

def get_RM(Vrm):
    V_LRS = np.zeros((Vrm.shape[:-1]))
    V_LRS = Vrm[0, :, :]
    V_HRS = Vrm[1, :, :]

    return V_LRS - V_HRS
    
def pulse_width2freq(pulse_width):
    return 1/(2*pulse_width)

def Rpar(R1, R2):
    
    return R1*R2/(R1+R2)

def Resistance_Matrix_1T1R(array_size,
                        Ron,
                        Roff,
                        Rp,
                        Zadded,
                        Rline,
                        Z,
                        cellmode):
    

    m, n = array_size
    R = np.ones((m, n))
    # if cellmode == 'on':
    #     R[:, -1] = Ron
    # else:
    #     R[:, -1] = Roff
    RvNEW = Zadded
    RVERT = np.ones(m)
    RVERT[0] = RvNEW
    ZA = np.ones(m)*Z
    for i in range(m):
        RvNEW += Rline
        RvNEW = Rpar(RvNEW, Zadded)
        R[i, :] += RvNEW
        RVERT[i] = RvNEW
        RvNEWpar = RvNEW
        for j in range(n):
            RvNEWpar = Rpar(RvNEWpar, RvNEW + Rp)
        ZA[i] += RvNEWpar

    R[:, 0] = ZA
    return R, RVERT

    
def Impedance_Array(include_fringe=False, 
                plot_AV = True,
                vary_pulse = True,
                cell = "off",
                npoints = 10,
                total_arraysize = 32,
                sub_arraysize = 128,
                subarray=False,
                pulse_min = 10, 
                pulse_max = 200, 
                N_meas = 2,
                node_size = 35,
                tox=2.8,
                tg = 5,
                Lg = 30,
                Lgs = 80):  

    if subarray == True:
        array_size = sub_arraysize
    elif subarray == False:
        array_size = total_arraysize*sub_arraysize

    e_0 = 8.85e-12 #Perimittivity in Vacuum
    er_ox = 18 # HfO2 relative permittivity

    
    if vary_pulse == True:
        Measurements = np.ones(N_meas)
        Node = node_size*1e-9*Measurements # Node geometry
        Wp = np.linspace(pulse_min, pulse_max, len(Measurements))*1e-9
    else:
        Measurements = np.ones(1)
        Node = node_size*1e-9*Measurements # Node geometry
        Wp = np.array([pulse_min])*1e-9
        N_meas = 1

    U = Node # Unit length
    W = U/2 # Width of interconnect
    T = W*2 # thickness of interconnect
    # S = U - W # nonoverlap spacing between interconnects

    dnw = 8*1e-9*Measurements # Nanowire diameter
    Lnw = 320*1e-9*Measurements # Nanowire Length
    Lm = 50*1e-9*Measurements # Length of RRAM NW
    tshell = 2e-9*Measurements # thickness of NW diameter in RRAM 
    tox = tox*1e-9*Measurements # Oxide Thickness
    tg = tg*1e-9*Measurements# Gate thickness
    Lg = Lg*1e-9*Measurements # Gate Length
    S_gs = 80*1e-9*Measurements # Distance between gate and source in FET (bottom electrode to substrate)

    tBL = min(30*1e-9,min(W*2)) # Bit Line Thickness
    H_Wplug = W*2 # Viaplug
    S_WLBL = Lnw-S_gs-tBL+H_Wplug # Distance WL to BL
    S_WLS = S_WLBL+S_gs # Distance WL till Substrate
    dDEV = dnw+2*(tshell+tox+tg) # total diameter of RRAM + shell and everything


    Ascrn = Outside_Screening(W, dDEV) # NW larger than interconnects
    A_WLS = W**2 - 2*Ascrn
    A_BLS = Outside_Screening(W, dDEV - 2*tg)# NW larger than  interconnects
    A_BLS = W**2 - 2*A_BLS
    A_WLBL = Inside_Screening(W, dDEV)

    Wscrn = A_WLS/W # Why devide again by W?
    Sscrn = W - Wscrn
    Side_fringe = W * IL_FringeCapX(W = Wscrn, S = Sscrn, H = S_WLBL, T = T, x_to_x = 'sw-top')

    Overlay = Cpar(A_WLBL, S_WLBL, er_ox*e_0)
    C_BL_S_side = W*IL_FringeCapX(W = Wscrn, S = Sscrn, H = S_gs, T = T, x_to_x = 'sw-top')
    C_BL_S_top = W*IL_FringeCapX(W = Wscrn, S = Sscrn, H = S_gs, T = T, x_to_x = 'top-top')
    C_BLS = Cpar(A_BLS, S_gs, er_ox*e_0)
    C_WLBL = Overlay + 2*Side_fringe

    Ascrn2 = Corner_Screening(W, dDEV)
    Wscrn2 = (W**2 - 4*Ascrn2)/W
    Sscrn2 = (W - Wscrn2)/2
    Side_fringe2 = W * IL_FringeCapX(W = Wscrn2, S = Sscrn2, H = S_WLS, T = T, x_to_x = 'sw-top')
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
    
    

    C_WLWL = 4*NW_to_NW + 2*Via_to_Via + 2*WL_to_WL + 1.5*WL_to_WL_fringe # Total WL to WL capacitanc (parallel an fringe)
    
    R_FET_off = 20*25e6 # OP transistors
    R_FET_on = 20e3 # OP transistors from nanoletters
    R_RRAM_on = 17.2*1e3 # The RRAM in LRS from experiments
    R_RRAM_off = 18.391*1e6 # The RRAM in HRS from experiments
    C_RRAM = Ccoax(18, tox, dDEV - 2*tg, Lm) # capacitance of RRAM 
    C_FET = Ccoax(18, tox, dDEV -2*(tg + tshell), Lg) # Capacitance of FET
    C_GS = 0.1*C_FET/2 # Gate-source Parasitic FET capacitance 
    C_GD = C_GS # The same for gate-drain parasitic

    Res = Cu_Res(W) # Resistivity of copper 
    R_WL = Res*2*W/(W*T) # Line Resistance of WL interconnect
    C_WL = C_WLWL + C_WLBL + C_WLS # Total WL capacitance for one cell

    
    Z_WLWL = Zcap(C_WLWL, Wp)
    Z_WLBL = Zcap(C_WLBL, Wp)
    Z_WLS = Zcap(C_WLS, Wp)

    ZC_RRAM = Zcap(C_RRAM, Wp) # Capacitive impedance of RRAM 
    ZR_RRAM = R_RRAM_on # Resistive impedance of RRAM
    ZR_FET = R_FET_off # Resistive impedance of FET in off mode
    ZC_GD = Zcap(C_GD, Wp) # Impedance due to parasitic gate-drain capacitance of FET
    #ZC_GS = Zcap(C_GS, Wp) # Impedance due to parasitic gate-source capacitance of FET


    Z_RRAM = ZC_RRAM*ZR_RRAM/(ZC_RRAM + ZR_RRAM) # Total RRAM impedance
    Z_FET = ZR_FET*ZC_GD/(ZC_GD+ZR_FET) # Total FET impedance
    Z_NW = Z_RRAM + Z_FET # Total nanowire impedance

    Z_RRAM_on = ZC_RRAM*R_RRAM_on/(ZC_RRAM + R_RRAM_on) # Again, total RRAM impedance (why 2 times?) in LRS
    Z_RRAM_off = ZC_RRAM*R_RRAM_off/(ZC_RRAM + R_RRAM_off) # Total RRAM impedance in HRS
    Z_FET_on = R_FET_on*ZC_GD/(ZC_GD+R_FET_on) # FET impedance due to gate drain capacitance and R_FET_on
    Z_NW_on = Z_FET_on + Z_RRAM_on # Nanowire impedance when RRAM in LRS and FET parasitics?
    Z_NW_off = Z_FET_on + Z_RRAM_off # Nanowire impedance when RRAM in HRS and FET parasitics

    Z_WL = Zcap(C_WL, Wp) # Wordline capacitive impedance 
    Z_TOT = Z_NW*Z_WL/(Z_NW+Z_WL) # total impedance of Nanowire (parallell coupling)
    C_BL = 2*C_BLS + C_GS# + 2*C_BL_S_side + C_BL_S_top # + Overlay2  #+ 2*WL_to_WL + WL_to_WL_fringe
    R_SD = 10e9

    Z_BL = Zcap(C_BL, Wp) # Loss from MOSFET 
    # Z_BLS = Zcap(2*C_BLS, Wp)
    Z_BL_TOT = Rpar(Z_BL, R_SD)
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

    Z0 = Zcell[u] # Impedance of worst case cell
    R = R_WL[u] # Line Resistance
    Rij = abs(Z_TOT[u]) # Resistive impedance of NW and WL
    Rij_new = Rij
   

    R_BL = np.ones(array_size)
    Z = Z0
    Z2 = Z_BL
    for i in range(array_size):
        Rvertical_NEW = Rpar(Rvertical_NEW, Rvertical)
        R_BL[i] = Rvertical_NEW
        Z2 += R
        Z2 = Z2*Rvertical/(Z2+Rvertical)


    for i in range(array_size):
        Rline_new += R
        Rij_new = Rpar(Rij_new, Rij)
        Z = Z + R
        Z = (Z*Rij)/(Z+Rij)
    
    if subarray == True:
        RMatrix = Resistance_Matrix_1T1R(array_size=(total_arraysize, total_arraysize), 
                                            Ron=Z_NW_on, Roff=Z_NW_off, Rp=Rij_new, Zadded=Rvertical_NEW,
                                            Rline=Rline_new, Z=Z, cellmode=cell)
        R_Matrix = RMatrix[0]
        Rvert = RMatrix[1]
        return([R_Matrix, abs(Z_TOT), abs(Z_WL), abs(Z_NW),
        abs(Z_WLWL), abs(Z_WLBL), 
        abs(Z_WLS), abs(Z_BL_TOT), abs(Z_BL), R_SD, Rvert])
