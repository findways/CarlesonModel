##Carlson model from Carlson et al. (2013)

from brian2 import *

################################################
# Neuron Model
################################################
print '1'
###Neuron dynamics using  Izhikevich E.M. (2004)

#Regular neuron parameters
N = 1           #Number of neurons
a = 0.02*ms
b = 0.2*ms
c = -65.0 * mV
d = 6*mV/ms
V = -70 * mV
tstep = 0.25*second
thres = 30.*mV             #Neuron threshold voltage
refac = 2*ms                #Refactory time for neuron

#Trying Alex's time constant since I kept getting unit errors
t1 = 1000*ms
t2 = 250*ms

print '2'
# Neuron dynamics equations
eqns_neurons = '''
dv/dt = ((0.04*(v**2)) + (5*v) + (140) - (u) + (I) + (5*L)) / t1 : 1 (unless refractory)
du/dt = (a * ((b*v) - u)) / t1 : 1
dL/dt = -L/t2 : 1
I:1
'''

print '3'
#After spike resetting term
resetEqs = '''
v = c
u = u + d
'''

print '4'
# Create Neuron objects
Neuron = NeuronGroup(N, model=eqns_neurons, threshold='v > 30.', reset=resetEqs, refractory=2*ms)
print '5'

################################################
#Input Model
################################################

###Generic Poisson distribution for input

#Parameters
N_input = 1       #number of articulating inputs to neuron network
freq = 15*Hz        #input firing frequency
print '6'
#Define input
input = PoissonGroup(N_input, rates=freq)
print '7'

################################################
#Synapse Model
################################################

###Synapse model from Carelson (2013)
print '8'
#parameters
alpha = 0.1
beta = 1
T = 5*second
gamma = 50
R_Target = 10*Hz     #Target firing rate for a postsynaptic excitatory neuron
Apre = 2.0e-4        # potentiation amplitude
Apost = 6.6e-5        # depression amplitude
taupre = 20*ms
taupost = 60*ms
RunTime = .1*second
tauR = 1*second

print '9'
#Synamps dynamics equations
eqns_synapse = '''
dw/dt = (alpha * w * (1 - R_avg/R_Target))*(K): 1
dApre/dt = -Apre / taupre : 1 (event-driven)
dApost/dt = -Apost / taupost : 1 (event-driven)
dR_avg/dt = -R_avg/tauR : 1
K = (R_avg) / (T * abs(1 - R_avg/R_Target)*gamma) : 1
'''

PreSyn = '''
L += w
Apre += dApre
w += beta * K * Apost
'''

PostSyn= '''
Apost += dApost
w += beta * K * Apre
R_avg += 1
'''

print '10'
#Create synapses
S = Synapses(input, Neuron, model = eqns_synapse, on_pre = PreSyn, on_post = PostSyn)

print '11'
S.connect()

print '12'
################################################
#Monitoring
################################################

M0 = SpikeMonitor(input)
M1 = StateMonitor(Neuron,['v','u'],record=True, dt=100*ms)
M2 = StateMonitor(S,['w','K','R_avg'],numpy.arange(100), dt=100*ms)
M3 = SpikeMonitor(Neuron)

print '13'

#run
run(100*second)

print '14'

print(M2.t)
print(M2.t.shape)
plt.subplot(511)
plt.plot(M0.t, M0.i, '.k')
plt.subplot(512)
plt.plot(M1.t, M1.v[0], '-b', label='Neuron 0')
plt.subplot(513)
plt.plot(M1.t, M1.u[0], '-r')
plt.subplot(514)
plt.plot(M1.t, M1.L[0], '-g')
plt.subplot(515)
plt.plot(M2.t, numpy.sum(M2.w, axis=0)) #Show cummulative weights of the synapses
plt.figure(2)
plt.subplot(311)
plt.plot(M2.t, M2.w[0], '-b')
plt.subplot(312)
plt.plot(M2.t, M2.K[0], '-r')
plt.subplot(313)
plt.plot(M2.t, M2.R_hat[0], '-g')


#Now I want to parse through the data in spike monitor, This is going to be a very simple kinda dumb thing
firingRateHack = (M3.t[1:]-M3.t[:-1])**-1.0
plt.figure(3)
plt.plot(firingRateHack)
plt.show()

