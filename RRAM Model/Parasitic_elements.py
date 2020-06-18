import numpy as np 
import matplotlib.pyplot as plt
import Resistance_Array_1T1R as IA
import Array_1T1R as A2

def Parallel(R1, R2):
    
    return R1*R2/(R1+R2)

def probability_map(plot=False, Subsize = 2, Vapp=1, cellmode='off', node=35, pulse_width=10):
    # pulse_width = 10 # ns

    SubArraySize = Subsize # Size of sub array (which becomes every cell in PA sim below)

    TotalArraySize = 32 # Size of PA array matrix

    VDD = Vapp #Supply voltage

    #TotalArraySize = F*SubArraySize # Total array size (number of rows/columns)
    Zsubarraysize = IA.Impedance_Array(include_fringe=True,
                plot_AV = False,
                total_arraysize=TotalArraySize,
                sub_arraysize=SubArraySize,
                subarray=True,
                npoints=1,
                pulse_width = pulse_width,
                node = node)


    [Z_TOT, Z_WL, Z_NW, Z_WLWL, Z_WLBL, Z_WLS, Z_BL, R_SD] = Zsubarraysize[4:]
    # print(Zsubarraysize)
    G_TOT = 1/Z_TOT 
    G_NW = 1/Z_NW
    G_WLWL = 1/Z_WLWL
    G_WLBL = 1/Z_WLBL
    G_WLS = 1/Z_WLS
    G_BL = 1/Z_BL
    G_SD = 1/R_SD
    G_WL = G_WLWL + G_WLBL + G_WLS
    G_WLBL_S = Parallel(G_WLBL, G_BL+G_SD) 
    G_TOT = G_NW + G_WLWL + G_WLBL_S + G_WLS
    G_TOT2 = G_NW + G_WLWL + G_WLBL + G_WLS

    print('#'*100)
    print('#'*100)
    print('#')
    print('# Leakage Percentages:')
    print('#')
    print('# Total Impedance = ', 1/G_TOT*1e-6, 'MOhm')
    print('# WL-WL leakage: ',  G_WLWL/G_TOT*100, '%' )
    print('# WL-BL-S leakage : ', G_WLBL_S/G_TOT*100, '%' )
    print('# WL-S leakage : ', G_WLS/G_TOT*100, '%')
    print('# NW leakage : ', G_NW/G_TOT*100, '%')
    print('# BL capacitive leakage percentage : ', G_BL/(G_BL + G_SD)*100, '%')
    print('# BL resistive leakage percentage : ', G_SD/(G_BL + G_SD)*100, '%')
    print('# Sum : ', (G_WLS + G_WLBL_S + G_WLWL + G_NW)/G_TOT*100, '%')
    print('#')
    print('#')
    print('#')
    print('#'*100)
    print('#'*100)



    print('#'*100)
    print('#'*100)
    print('#')
    print('# Leakage Percentages Without Added Impedance:')
    print('#')
    print('# Total Impedance = ', 1/G_TOT2*1e-6, 'MOhm')
    print('# WL-WL leakage: ',  G_WLWL/G_TOT2*100, '%' )
    print('# WL-BL leakage : ', G_WLBL/G_TOT2*100, '%' )
    print('# WL-S leakage : ', G_WLS/G_TOT2*100, '%')
    print('# NW leakage : ', G_NW/G_TOT2*100, '%')
    print('# Sum : ', (G_WLS + G_WLBL + G_WLWL + G_NW)/G_TOT2*100, '%')
    print('#')
    print('#')
    print('#')
    print('#'*100)
    print('#'*100)

    return [G_WLWL/G_TOT2*100, G_WLBL/G_TOT2*100, G_WLS/G_TOT2*100, G_NW/G_TOT2*100]





if __name__ == "__main__":
    pw = np.linspace(1, 100, 100)

    PW = pw*10

    G_WLWL = np.ones(len(pw))
    G_WLBL = np.ones(len(pw))
    G_WLS = np.ones(len(pw))
    G_NW = np.ones(len(pw))
    i = 0
    for p in PW:
        pm = probability_map(pulse_width=p)

        G_WLWL[i] = pm[0]
        G_WLBL[i] = pm[1]
        G_WLS[i] = pm[2]
        G_NW[i] = pm[3]
        i+=1

    plt.plot(PW, G_WLWL, label='BL-BL')
    plt.plot(PW, G_NW, label='NW')
    plt.legend(loc='best')
    plt.xlabel('Pulse Width (ns)')
    plt.ylabel('Leakage Portion (%)')
    plt.grid()
    plt.show()

    plt.plot(PW, G_WLBL, label='BL-WL')
    plt.plot(PW, G_WLS, label='BL-S')
    plt.xlabel('Pulse Width (ns)')
    plt.ylabel('Leakage Portion (%)')
    plt.legend(loc='best')
    plt.grid()
    plt.show()



    
