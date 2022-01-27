
import random
import math as m
from scipy.fft import fft, ifft,fftfreq
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
fs=10000
Ts=1/fs
Tsig= 50*Ts
T_pulse=Ts
T_delay = Ts
t = np.linspace(0, Tsig, fs, endpoint=False)
samplesNumber = len(t)
numOfBits = int(np.floor(Tsig/T_pulse))
samplesInBit= int(np.floor(samplesNumber/numOfBits))

print("bitnumber, samples number,bitSamples", numOfBits,samplesNumber,samplesInBit)


def dot_product_array(vect1,vect2):
	dotProduct=[]
	for i in range(int(len(vect1))):
		dotProduct.append(vect1[i]*vect2[i])
	return dotProduct

def generate_envelope_in_bit():
	x_t_envelope=[]
	t_envelope=np.linspace(0, T_pulse, int(np.floor(fs/numOfBits)), endpoint=False)
	A=2/T_pulse**2
	B=-2/T_pulse
	C=1
	for i in range(samplesInBit):
		square_x_t = A*t_envelope[i]**2+B*t_envelope[i]+C
		x_t_envelope.append(square_x_t)

	print("···········································")
	print("samplesInBit y len t envelope",samplesInBit,len(t_envelope))
	plt.figure()
	plt.plot(t_envelope,x_t_envelope)
	return(t_envelope,x_t_envelope)

def generate_envelope_sin():
	x_t_sin_envelope=[]
	t_envelope=np.linspace(0, T_pulse, int(np.floor(fs/numOfBits)), endpoint=False)
	A=1
	for i in range(samplesInBit):	#looping 200 times
		sin_x_t = A*m.sin(t_envelope[i]*fs*3*m.pi)
		x_t_sin_envelope.append(sin_x_t)

	print("···········································")
	print("samplesInBit y len t envelope",samplesInBit,len(t_envelope))
	plt.figure()
	plt.plot(t_envelope,x_t_sin_envelope)
	return(t_envelope,x_t_sin_envelope)

def generate_modulated_signal(x_t_envelope):
	x_t = []
	print(len(t))
	for i in range(numOfBits):
		bitValue=random.randint(0, 1)
		#bitValue=2.*bitValue-1
		for j in range(samplesInBit):
			if bitValue==0:
				x_t.append(x_t_envelope[j])
			else:
				x_t.append(-1 * (x_t_envelope[j]))
			#we modulate the signal pulse from 1 to -1
			
	return(x_t)


#[t_envelope,x_t_envelope]=generate_envelope_in_bit()
[t_envelope,x_t_envelope]=generate_envelope_sin()
x_t = generate_modulated_signal(x_t_envelope)

def plot_modulated_signal_and_FFT(x_t):
	x_f = fft(x_t)
	f = fftfreq(samplesNumber, Ts)[:samplesNumber//2]
	print("len(t),len(x_t) ························##",len(t),len(x_t))
	plt.figure()
	plt.plot(t,x_t)
	plt.figure()
	plt.plot(f, 2.0/samplesNumber * np.abs(x_f[0:samplesNumber//2]))
	plt.show()
		

plot_modulated_signal_and_FFT(x_t)
#autocorrelation1=np.correlate(x_t, x_t)
#autocorrelation_delay = np.correlate(x_t-T_delay)
# pwm = signal.square(2 * np.pi * 30 * t, duty=1)
# plt.figure()
# x_t_f = fft(x_t)
# x_tf = fftfreq(samplesNumber, Ts)[:samplesNumber//2]
# print("###################################",len(x_tf),len(x_t_f))
# print(" t x_t ###################################",len(t),len(x_t))

# #plt.plot(x_tf, 2.0/samplesNumber * np.abs(x_t_f[0:samplesNumber//2]))
# plt.figure()
# sig = np.sin(2 * np.pi * t)
# pwm = signal.square(2 * np.pi * 30 * t, duty=(sig + 1)/2)
# plt.subplot(2, 1, 1)
# plt.plot(t, sig)
# plt.subplot(2, 1, 2)
# plt.plot(t, pwm)
# plt.ylim(-1.5, 1.5)
# plt.figure(2)
# print(len(t),len(x_t))
# """ #plt.plot(t,x_t)
# plt.figure()
# plt.plot(x_t_f)

# plt.figure()
# plt.plot(t,x_t) """




