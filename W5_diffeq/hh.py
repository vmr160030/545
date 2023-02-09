import numpy as np
from tqdm import tqdm, trange

class HH(object):
    def __init__(self, gnamax=120, gkmax=36, gl=0.3, 
    Vna=-115, Vk=12, Vl=-10.613, C=1.0, TimeStep=0.01, 
    tFinal=500):
        self.gnamax = gnamax
        self.gkmax = gkmax
        self.gl = gl
        self.Vna = Vna
        self.Vk = Vk
        self.Vl = Vl
        self.C = C
        self.dt = TimeStep
        self.tFinal = tFinal
        
        self.time = np.arange(0, self.tFinal, self.dt)
        
    def initialize(self, n_init=0.6, h_init=0.3):
        self.arr_m = np.zeros(len(self.time))
        self.arr_h = np.zeros(len(self.time))
        self.arr_n = np.zeros(len(self.time))
        self.arr_V = np.zeros(len(self.time))

        self.arr_n[0] = n_init
        self.arr_h[0] = h_init

    def simulate(self, Iext):
        for i in range(len(self.time)-1):
            alpham = 0.1*(self.arr_V[i]+25)/(np.exp((self.arr_V[i]+25)/10)-1)
            betam = 4*np.exp(self.arr_V[i]/18)
            alphan = 0.01*(self.arr_V[i]+10)/(np.exp((self.arr_V[i]+10)/10)-1)
            betan = 0.125*np.exp(self.arr_V[i]/80)
            alphah = 0.07*np.exp(self.arr_V[i]/20)
            betah = 1./(np.exp((self.arr_V[i]+30)/10)+1)
            dm = self.dt * (-self.arr_m[i]*(alpham + betam) + alpham)
            self.arr_m[i+1] = self.arr_m[i] + dm
            dn = self.dt * (-self.arr_n[i]*(alphan + betan) + alphan)
            self.arr_n[i+1] = self.arr_n[i] + dn
            dh = self.dt * (-self.arr_h[i]*(alphah + betah) + alphah)
            self.arr_h[i+1] = self.arr_h[i] + dh
            Ik = self.gkmax*self.arr_n[i]**4*(self.arr_V[i] - self.Vk)
            Ina = self.gnamax*self.arr_m[i]**3*self.arr_h[i]*(self.arr_V[i]-self.Vna)
            Il = self.gl*(self.arr_V[i] - self.Vl)
            self.arr_V[i+1] = self.arr_V[i] + self.dt*(Iext[i] - Ik - Ina - Il)/self.C   # This is the voltage equation from HH 1952

    def simulate_Vclamp(self, arr_V):
        self.arr_V = arr_V
        self.arr_Ik = np.zeros(len(self.time))
        self.arr_Ina = np.zeros(len(self.time))
        self.arr_Il = np.zeros(len(self.time))
        for i in range(len(self.time)-1):
            alpham = 0.1*(self.arr_V[i]+25)/(np.exp((self.arr_V[i]+25)/10)-1)
            betam = 4*np.exp(self.arr_V[i]/18)
            alphan = 0.01*(self.arr_V[i]+10)/(np.exp((self.arr_V[i]+10)/10)-1)
            betan = 0.125*np.exp(self.arr_V[i]/80)
            alphah = 0.07*np.exp(self.arr_V[i]/20)
            betah = 1./(np.exp((self.arr_V[i]+30)/10)+1)
            
            dm = self.dt * (-self.arr_m[i]*(alpham + betam) + alpham)
            self.arr_m[i+1] = self.arr_m[i] + dm
            dn = self.dt * (-self.arr_n[i]*(alphan + betan) + alphan)
            self.arr_n[i+1] = self.arr_n[i] + dn
            dh = self.dt * (-self.arr_h[i]*(alphah + betah) + alphah)
            self.arr_h[i+1] = self.arr_h[i] + dh
            
            self.arr_Ik[i] = self.gkmax*self.arr_n[i]**4*(self.arr_V[i] - self.Vk)
            self.arr_Ina[i] = self.gnamax*self.arr_m[i]**3*self.arr_h[i]*(self.arr_V[i]-self.Vna)
            self.arr_Il[i] = self.gl*(self.arr_V[i] - self.Vl)
            
            # self.arr_V[i+1] = self.arr_V[i] + self.dt*(Iext[i] - Ik - Ina - Il)/self.C   # This is the voltage equation from HH 1952

        