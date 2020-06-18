import numpy as np 
import matplotlib.pyplot as plt
import Resistance_Array_1T1R as IA
import Array_1T1R as A2
from progress.bar import Bar

def Parallel(R1, R2):
    
    return R1*R2/(R1+R2)

def probability_map(plot=False, Subsize = 64, Vapp=1, cellmode='off', pulsewidth=10, node=50):
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
                node = node,
                cell=cellmode)
    R = Zsubarraysize[0]
    Rline = Zsubarraysize[1]
    NWon = Zsubarraysize[2]
    NWoff = Zsubarraysize[3]


    V1 = A2.simulate(array_size=(TotalArraySize, TotalArraySize),
                    RMatrix=R,
                    line_resistance=Rline,
                    supply_voltage = VDD)
    
    return(V1[-1, -1], R[-1, -1], NWon, NWoff)


def RM_vs_arraysize():
    LogSpace = np.logspace(1,3, base=10.0, num=20).astype(int)
    LogSpace2 = LogSpace*32
    i = 0

    LRS = np.ones(len(LogSpace))
    HRS = np.ones(len(LogSpace))
    Zread = 5*20*1e3
    Vread = 150e-3
    n = 35
    for pulsewidth in [10, 50, 100, 150]:
        print("Running for pulse width = {} nm".format(pulsewidth))
        
        bar = Bar('Progress', max = 2*len(LogSpace))
        for i in (0, 1):
            if i == 0:
                k = 0
                for l in LogSpace:
                    bar.next()
                    probmap = probability_map(plot=False, Subsize=l, cellmode='on', Vapp = Vread, pulsewidth = pulsewidth, node=n)
                    v, r, NWon = probmap[0], probmap[1], probmap[2][0]
                    ilrs = v/NWon
                    v = r*ilrs
                    probmap = probability_map(plot=False, Subsize=l, cellmode='on', Vapp=v, pulsewidth = pulsewidth, node=n)
                    v, r = probmap[0], probmap[1]
                    i = v/r
                    v = i*Zread
                    LRS[k] = v
                    k +=1
                    
            else:
                k = 0
                for l in LogSpace:
                    bar.next()
                    probmap = probability_map(plot=False, Subsize=l, cellmode='off', Vapp=Vread, pulsewidth = pulsewidth, node=n)
                    v, r, NWoff = probmap[0], probmap[1], probmap[3][0]
                    ihrs = v/NWoff
                    v = r*ihrs
                    probmap = probability_map(plot=False, Subsize=l, cellmode='off', Vapp=v, pulsewidth = pulsewidth, node=n)
                    v, r = probmap[0], probmap[1]
                    i = v/r
                    v = i*Zread
                    HRS[k] = v
                    k+=1
        bar.finish()

        plt.plot(LogSpace2, LRS - HRS, label='w = {} ns'.format(pulsewidth))
    plt.xscale('log')
    plt.xlabel('Array Size (Number of Rows)')
    plt.ylabel('Read Margin (V)')
    plt.grid()
    plt.legend()
    plt.show()




if __name__ == "__main__":
    # probability_map(plot=True, Subsize=32)
    RM_vs_arraysize()

    
