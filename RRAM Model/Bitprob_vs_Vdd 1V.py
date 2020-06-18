import numpy as np 
import matplotlib.pyplot as plt
import Impedance_Array as IA
import Array_1T1R as A2
from progress.bar import Bar

def Bitprob_vs_Vdd(pulse_width =10, Subsize=128, vdd = 1, node=35, cellmode = 'on'):
    pulse_width = pulse_width # ns

    SubArraySize = Subsize # Size of sub array (which becomes every cell in PA sim below)

    TotalArraySize = 32 # Size of PA array matrix

    VDD = vdd #Supply voltage

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

    V1 = A2.simulate(array_size=(TotalArraySize, TotalArraySize),
                    RMatrix=R,
                    line_resistance=Rline,
                    plot_array=False,
                    supply_voltage = VDD,
                    cellmode = cellmode)
    
    Degree = 5
    MaxSet = 0.8791946308724832
    MinSet = 0.3087248322147651
    MaxReset = 0.5167785234899329
    MinReset = 0.09395973154362416

    D = np.linspace(Degree, 0, Degree+1)

    SetFunc = [ -89.68195451,  311.27444672, -418.47804391,  270.3351811,   -81.38685227,
    9.13236572]
    ResetFunc = [ -87.9282956,   202.31215168, -179.37229496,   76.92727401,  -12.81175272,
    0.66400763]

    SetFunc = np.asarray(SetFunc)
    ResetFunc = np.asarray(ResetFunc)

    ProbSet, ProbReset = V1*0, V1*0
    s = 0
    for i in range(len(D)):
        ProbSet += SetFunc[i]*V1**D[i]
        ProbReset += ResetFunc[i]*V1**D[i]
        s += SetFunc[i]*0.036**D[i]

    # Check for values higher than highest extracted SET value gives probability 1
    ProbSet[np.where(V1 >= MaxSet)] = 1
    # Check for values lower tham lowest extracted SET value gives probability 0
    ProbSet[np.where(V1 <= MinSet)] = 0

    ProbReset[np.where(V1 >= MaxReset)] = 1
    ProbReset[np.where(V1 <= MinReset)] = 0
    ProbReset[np.where(ProbReset <= 0)] = 0

    # print(np.average(ProbSet), np.average(ProbReset))
    return (np.mean(ProbSet), np.mean(ProbReset), np.std(ProbSet), np.std(ProbReset))



if __name__ == "__main__":
    vdd = np.linspace(0.01, 1.1, 30)
    bitprobset = vdd*0
    bitprobreset = vdd*0
    stdset = vdd*0
    stdreset = vdd*0
    k = 0
    bar = Bar('Progress', max = len(vdd))
    for v in vdd:
        bar.next()
        BP = Bitprob_vs_Vdd(vdd=v)
        bitprobset[k] = BP[0]
        bitprobreset[k] = BP[1]
        stdset[k] = BP[2]
        stdreset[k] = BP[3]
        k+=1
    bar.finish()
    fig, ax,  = plt.subplots()
    ax.plot(vdd, bitprobset, '.', label="Set")
    ax.plot(vdd, bitprobreset, '.', label="Reset", color='red')
    # ax.grid()
    # ax.set_ylim(-0.05, 1.05)
    # ax.axvline(x=0.57, ymin=0, ymax=1, color='black', linestyle='--', linewidth=0.9, label="Experimental Set")
    # ax.axvline(x=0.40, ymin=0, ymax=1, color='black', linestyle='-.', linewidth=0.9, label="Experimental Reset")
    # ax.axvspan(0.57-0.23, 0.57+0.23, alpha=0.1, color='blue')
    # ax.axvspan(0.40-0.19, 0.40+0.19, alpha=0.1, color='red')
   
    # ax.legend(loc=4)
    # plt.show()

    # fig, ax = plt.subplots()
    ax.plot(vdd, stdset, '-', label="Set std")
    ax.plot(vdd, stdreset, '-', label="Reset std")
    
    # ax.axvline(x=0.57, ymin=0, ymax=1, color='black', linestyle='--', linewidth=0.9, label="Experimental Set")
    # ax.axvline(x=0.40, ymin=0, ymax=1, color='black', linestyle='-.', linewidth=0.9, label="Experimental Reset")
    # ax.axvspan(0.57-0.23, 0.57+0.23, alpha=0.1, color='blue')
    # ax.axvspan(0.40-0.19, 0.40+0.19, alpha=0.1, color='red')
    ax.set_ylabel("Average Switching Probability")
    ax.set_xlabel("Voltage Amplitude (V)")
    ax.legend(loc='best')
    ax.grid()
    ax.set_ylim(-0.05, 1.05)
    plt.show()
    
    
