from brian2 import *
import numpy as np

class Rod(object):
    def __init__(self, length=10/np.pi*umeter, diam=10*umetre):
        # Morphology params
        self.length = length
        self.diam = diam

        # Cylinder area = 2*pi*radius*length + 2*pi*radius^2
        self.cell_area = np.pi*diam*length + 2*np.pi*diam**2/4
        # print(cell_area)
        self.membrane_capacitance = 30*pfarad/self.cell_area
        self.Ra = 200*ohm*cm

        self.eqs = {}
        self.constants = {}
        self.constants['morphology'] = {'cell_area': self.cell_area}

    def define_Ih(self):
        # Define Rod_ih
        Eh_rev = -32*mV
        Vhalf_h = -82*mV
        gh_bar = 2.5*msiemens*cm**-2
        Sh = -5.33*mV
        aoh = 1*Hz

        self.constants['Ih'] = {'Eh_rev': Eh_rev, 'Vhalf_h': Vhalf_h, 'gh_bar': gh_bar, 'Sh': Sh, 'aoh': aoh}

        eqs_h = '''
        ah = aoh*exp( (V-Vhalf_h)/(2*Sh) ) : Hz
        bh = aoh*exp( -(V-Vhalf_h)/(2*Sh) ) : Hz
        tauh = 1/(ah+bh) : second
        infh = ah*tauh : 1

        dnh/dt = (infh-nh)/tauh : 1
        gh = gh_bar * nh : siemens / metre**2
        Ih = gh * (V-Eh_rev) : amp / metre**2
        '''

        self.eqs['Ih'] = eqs_h

    def define_Ip(self):
        # Photocurrent parameters
        Idark = -40*pA
        Tau1 = 50*ms
        Tau2 = 450*ms
        Tau3 = 800*ms
        RodB = 3800*ms

        self.constants['Ip/(cell_area)'] = {'Idark': Idark, 'Tau1': Tau1, 'Tau2': Tau2, 'Tau3': Tau3, 'RodB': RodB}

        eqs_input = '''
        # input photocurrent
        Ip = Idark + A*( 32*(1-exp(-t/Tau1)) - 33*(1/(1+exp(-(t-RodB)/Tau2))) + (1-exp(-t/Tau3)))/33: amp
        A : amp
        '''
        self.eqs['Ip/(cell_area)'] = eqs_input

    def define_Ileak(self):
        # Define Rod_leak
        glbar = 0.52*msiemens*cm**-2
        El_rev = -74*mV
        self.constants['Ileak'] = {'glbar': glbar, 'El_rev': El_rev}

        eqs_leak = '''
        Ileak = glbar * (V-El_rev) : amp / metre**2
        '''
        self.eqs['Ileak'] = eqs_leak

    def define_total(self):
        component_currents = list(self.eqs.keys())
        str_current_sum = ' + '.join([c for c in component_currents])
        print(str_current_sum)
        
        eqs_total = f'''
        # Transmembrane current
        Im = 0*amp/metre**2 : amp / metre**2

        dV/dt = -({str_current_sum}) / Cm : volt
        #dV/dt = -(Ih + IKx + Ileak + Ip/(cell_area)) / Cm : volt
        '''
        self.eqs['total'] = eqs_total

    def create_rod(self):
        morpho = Cylinder(diameter=self.diam, n=1, length=self.length)
        eqs = ''
        for eq in self.eqs.values():
            eqs += eq

        # Set constants
        namespace = {}
        for key, value in self.constants.items():
            for k, v in value.items():
                namespace[k] = v
        self.namespace = namespace
        neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=self.membrane_capacitance,
                               Ri=self.Ra, method='euler', namespace=namespace)
        

        self.model = neuron
        # self.net = Network(self.model)
        
        
    def run_sim(self):
        defaultclock.dt = 1*ms

        # Monitor voltage and Ip
        M = StateMonitor(self.model, ['V', 'Ip', 'Ih'], record=True, dt=10*ms)
   
        # self.model.Ca = Cainf
        # neuron.V = -65.78*mV
        self.model.A = 0*pA
        # Initialize gating variables
        # run(1*ms)
        # self.model.nKx = neuron.infKx[:][0]
        # self.model.mKv = neuron.infmKv[:][0]
        # self.model.mCa = neuron.infmCa[:][0]
        # self.model.hCa = neuron.infhCa[:][0]
        self.model.nh = self.model.infh[:][0]

        self.net.run(999*ms, namespace={})
        self.model.A = 40*pA
        self.net.run(6000*ms, namespace={})
        self.model.A = 0*pA
        self.net.run(6000*ms, namespace={})
        plot(M.t/ms, M.V[0]/mV)
        xlabel('Time (ms)')
        ylabel('Voltage (mV)')

        plt.figure()
        plot(M.t/ms, M.Ip[0]/pA)
        xlabel('Time (ms)')
        ylabel('Photocurrent (pA)')

        return M
