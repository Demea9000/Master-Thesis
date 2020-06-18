import numpy as np 
import matplotlib.pyplot as plt
import Resistance_Array_1T1R as IA
import Array_1T1R as A2

def Parallel(R1, R2):
    
    return R1*R2/(R1+R2)

def probability_map(plot=True, Subsize = 64, Vapp=1, cellmode='off', node=35):
    pulse_width = 10 # ns

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
    R = Zsubarraysize[0]
    Rline = Zsubarraysize[1]


    V1 = A2.simulate(array_size=(TotalArraySize, TotalArraySize),
                    RMatrix=R,
                    line_resistance=Rline,
                    supply_voltage = VDD)


    if plot == True:
        plt.imshow(V1, interpolation='none', cmap='jet')
        plt.title("Access Voltage")
        plt.grid(False)
        plt.colorbar()
        plt.clim(0, 1)
        plt.xticks([0, 16, 31], labels=['0', '1024',  '2048'])
        plt.yticks([0, 16, 31], labels=['0', '2048',  '2048'])
        plt.xlabel("Wordline")
        plt.ylabel("Bitline")
        plt.show()
        
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

        plt.plot()

        plt.imshow(ProbSet, interpolation='none', cmap='jet')
        plt.title("Set Probability")
        plt.grid(False)
        plt.colorbar()
        plt.clim(0, 1)
        plt.xlabel("Wordline")
        plt.ylabel("Bitline")
        plt.xticks([0, 16, 31], labels=['0', '1024',  '2048'])
        plt.yticks([0, 16, 31], labels=['0', '1024',  '2048'])

        plt.show()
        plt.imshow(ProbReset, interpolation='none', cmap='jet')
        plt.title("Reset Probability")
        plt.grid(False)
        plt.colorbar()
        plt.clim(0, 1)
        plt.xlabel("Wordline")
        plt.ylabel("Bitline")
        plt.xticks([0, 16, 31], labels=['0', '1024',  '2048'])
        plt.yticks([0, 16, 31], labels=['0', '1024',  '2048'])
        plt.show()
    
    return(V1[-1, -1], R[-1, -1])





if __name__ == "__main__":
    probability_map()

    
