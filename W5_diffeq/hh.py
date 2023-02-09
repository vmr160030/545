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
            Ik, Ina, Il = self.update_currents(i)
            self.arr_V[i+1] = self.update_voltage(Ik, Ina, Il, Iext[i], i)

    def simulate_Vclamp(self, arr_V):
        self.arr_V = arr_V
        self.arr_Ik = np.zeros(len(self.time))
        self.arr_Ina = np.zeros(len(self.time))
        self.arr_Il = np.zeros(len(self.time))
        for i in range(len(self.time)-1):
            self.arr_Ik[i], self.arr_Ina[i], self.arr_Il[i] = self.update_currents(i)

    def update_alpham(self, Vprev):
        return 0.1*(Vprev+25)/(np.exp((Vprev+25)/10)-1)

    def update_betam(self, Vprev):
        return 4*np.exp(Vprev/18)

    def update_alphan(self, Vprev):
        return 0.01*(Vprev+10)/(np.exp((Vprev+10)/10)-1)

    def update_betan(self, Vprev):
        return 0.125*np.exp(Vprev/80)

    def update_alphah(self, Vprev):
        return 0.07*np.exp(Vprev/20)

    def update_betah(self, Vprev):
        return 1./(np.exp((Vprev+30)/10)+1)

    def update_Ik(self, nprev, Vprev):
        return self.gkmax * nprev**4 * (Vprev - self.Vk)

    def update_Ina(self, mprev, hprev, Vprev):
        return self.gnamax * mprev**3 * hprev * (Vprev-self.Vna)

    def update_Il(self, Vprev):
        return self.gl * (Vprev - self.Vl)

    def update_currents(self, i, dt=None):
        if dt is not None:
            self.dt = dt
        alpham = self.update_alpham(self.arr_V[i])
        betam = self.update_betam(self.arr_V[i])
        alphan = self.update_alphan(self.arr_V[i])
        betan = self.update_betan(self.arr_V[i])
        alphah = self.update_alphah(self.arr_V[i])
        betah = self.update_betah(self.arr_V[i])

        dm = self.dt * (-self.arr_m[i]*(alpham + betam) + alpham)
        self.arr_m[i+1] = self.arr_m[i] + dm
        dn = self.dt * (-self.arr_n[i]*(alphan + betan) + alphan)
        self.arr_n[i+1] = self.arr_n[i] + dn
        dh = self.dt * (-self.arr_h[i]*(alphah + betah) + alphah)
        self.arr_h[i+1] = self.arr_h[i] + dh

        Ik = self.update_Ik(self.arr_n[i], self.arr_V[i])
        Ina = self.update_Ina(self.arr_m[i], self.arr_h[i], self.arr_V[i])
        Il = self.update_Il(self.arr_V[i])
        return Ik, Ina, Il

    def update_voltage(self, Ik, Ina, Il, Iext, i):
        # This is the voltage equation from HH 1952
        return self.arr_V[i] + self.dt*(Iext - Ik - Ina - Il)/self.C