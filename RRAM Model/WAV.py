import numpy as np 
import matplotlib.pyplot as plt
import Impedance_Array as IA
import Array_1T1R as A2
from progress.bar import Bar

def Parallel(R1, R2):
    
    return R1*R2/(R1+R2)

def probability_map(plot=False, Subsize = 64, Vapp=1, cellmode='off', pulsewidth=10, node=35):
    pulse_width = pulsewidth # ns

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
    Ztotalarraysize = IA.Impedance_Array(include_fringe=True,
                plot_AV = False,
                total_arraysize=TotalArraySize,
                sub_arraysize=SubArraySize,
                subarray=False,
                npoints=1,
                pulse_width = pulse_width,
                node = node)
    R = Zsubarraysize[0]
    Rline = Zsubarraysize[1]
    NWon = Zsubarraysize[2]
    NWoff = Zsubarraysize[3]


    V1 = A2.simulate(array_size=(TotalArraySize, TotalArraySize),
                    RMatrix=R,
                    line_resistance=Rline,
                    plot_array=plot,
                    supply_voltage = VDD,
                    cellmode = cellmode)
    return(V1[-1, -1], R[-1, -1], NWon, NWoff)


def WAV_vs_arraysize():
    LogSpace = np.logspace(1, 3, base=10.0, num=20).astype(int)
    WC = np.ones(len(LogSpace))
    
    LogSpace2 = LogSpace*32
    pulsewidth = 10
    
    for n in [35, 50, 100]:
        i = 0
        print("Running for pulse width = {} nm".format(pulsewidth))
        bar = Bar('Progress', max = len(LogSpace))
        for l in LogSpace:
            bar.next()
            worstcase = probability_map(plot=False, Subsize=l, cellmode='off', pulsewidth=pulsewidth, node=n)[0]
            WC[i] = worstcase
            i+=1
    
        plt.plot(LogSpace2, WC, label='node = {} nm'.format(n))
        bar.finish()
        
    
    # plt.legend(loc='best')
    
    plt.xscale('log')
    plt.xlabel('Array Size (Number of Rows)')
    plt.ylabel('Access Voltage (V)')
    plt.legend()
    plt.grid()
    plt.show()





if __name__ == "__main__":
    WAV_vs_arraysize()

    
