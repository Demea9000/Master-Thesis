import numpy as np 
import matplotlib.pyplot as plt
import Resistance_Array_1T1R as IA
import Array_1T1R as A2
from progress.bar import Bar

def Parallel(R1, R2):
    
    return R1*R2/(R1+R2)

def probability_map(plot=False, Subsize = 64, Vapp=1, cellmode='off', pulsewidth=10, resistivity=1.68e-8, node=35, vary_resistivity=True):
    pulse_width = pulsewidth # ns

    SubArraySize = Subsize # Size of sub array (which becomes every cell in PA sim below)

    TotalArraySize = 32 # Size of PA array matrix

    VDD = Vapp #Supply voltage

    #TotalArraySize = F*SubArraySize # Total array size (number of rows/columns)
    Zsubarraysize = IA.Impedance_Array(include_fringe=True,
                plot_AV = False,
                vary_resistivity=vary_resistivity,
                total_arraysize=TotalArraySize,
                sub_arraysize=SubArraySize,
                subarray=True,
                npoints=1,
                pulse_width = pulse_width,
                node = node,
                resistivity = resistivity)
    R = Zsubarraysize[0]
    Rline = Zsubarraysize[1]
    NWon = Zsubarraysize[2]
    NWoff = Zsubarraysize[3]


    V1 = A2.simulate(array_size=(TotalArraySize, TotalArraySize),
                    RMatrix=R,
                    line_resistance=Rline,
                    supply_voltage = VDD)
    
    return(V1[-1, -1], R[-1, -1], NWon, NWoff)

def Cu_Res(tnm):
    t = tnm*1e9
    ro_bulk = 1.68e-8
    k1 = 20
    k2 = 1+0.5*1/(1+np.exp((t-50)/100))
    ro = ((k1*(1-np.sqrt(t/100)/(1+np.exp((t-200)/100)))+t)*(k2/t))**2*ro_bulk
    return ro

def WAV_vs_arraysize():
    R = 1.68e-8
    ratios =  np.linspace(1, 10, 5)
    resistivity = ratios*R

    LogSpace = np.logspace(1, 3, base=10.0, num=20).astype(int)
    WC = np.ones(len(LogSpace))
    
    LogSpace2 = LogSpace*32
    
    for rr in resistivity:
        i = 0
        bar = Bar('Progress', max = len(LogSpace))
        print("Running for resistivity = {} Ohm meters".format(rr))
        for l in LogSpace:
            bar.next()
            worstcase = probability_map(plot=False, Subsize=l, cellmode='off', 
            pulsewidth=10, resistivity=rr, vary_resistivity=True)[0]
            WC[i] = worstcase
            i+=1
    
        plt.plot(LogSpace2, WC, label='resistivity = {}*Rbulk'.format(round(rr/R, 2)))
        bar.finish()
        
    
    # plt.legend(loc='best')
    
    plt.xscale('linear')
    plt.xlabel('Array Size (Number of Rows)')
    plt.ylabel('Access Voltage (V)')
    plt.xscale('log')
    plt.legend()
    plt.grid()
    plt.show()





if __name__ == "__main__":
    WAV_vs_arraysize()

    
