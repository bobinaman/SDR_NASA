import Math_Functions
from csv import QUOTE_ALL
import random
import re
from scipy.fft import fft, ifft,fftfreq
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import math
import cmath
modulation=4 #4QAM

fs=100000
fc=2e5
Ts=1/fs
Tsig= 50*Ts
T_pulse=Ts
T_delay = Ts
t = np.linspace(0, Tsig, fs, endpoint=False)
samplesNumber = len(t)
numOfBits = int(np.floor(Tsig/T_pulse))
samplesInBit= int(np.floor(samplesNumber/numOfBits))
print("bitnumber, samples number,bitSamples", numOfBits,samplesNumber,samplesInBit)
print("fs:",fs,"fc:",fc,"Ts:",Ts,"Tsignal:",Tsig,"T_pulse:",T_pulse)
#print("fs:",fs,"fc:",fc+"Ts:",Ts,"Tsignal:",Tsig,"T_pulse:",T_pulse)



	

def operating_array(vect1,vect2,operation):
	if(operation=="dot product"):
		dotProduct=[]
		for i in range(int(len(vect1))):
			dotProduct.append(vect1[i]*vect2[i])
		return dotProduct
	elif(operation=="substraction"):
		Substraction=[]
		for i in range(int(len(vect1))):
			Substraction.append(vect1[i]-vect2[i])
		return Substraction
	elif(operation=="sumation"):
		Sum=[]
		for i in range(int(len(vect1))):
			sum.append(vect1[i]+vect2[i])
		return Sum
def generate_sinusoidal_envelope_in_bit(sin_periode):
	x_t_envelope=[]
	t_envelope=np.linspace(0, T_pulse, int(np.floor(fs/numOfBits)), endpoint=False)
	Amplitude=1/2
	f_sin=1/sin_periode
	for i in range(samplesInBit):
		square_x_t = 1+ Amplitude*math.sin(2*math.pi*f_sin*t_envelope[i])
		x_t_envelope.append(square_x_t)

	print("···········································")
	print("samplesInBit y len t envelope",samplesInBit,len(t_envelope))
	plt.figure()
	plt.plot(t_envelope,x_t_envelope)
	return(t_envelope,x_t_envelope)
def generate_envelope_in_bit():
	x_t_envelope=[]
	t_envelope=np.linspace(0, T_pulse, int(np.floor(fs/numOfBits)), endpoint=False)
	A=2/T_pulse**2
	B=-2/T_pulse
	C=1
	for i in range(samplesInBit):
		square_x_t = 1#A*t_envelope[i]**2+B*t_envelope[i]+C
		x_t_envelope.append(square_x_t)

	print("···········································")
	print("samplesInBit y len t envelope",samplesInBit,len(t_envelope))
	plt.figure()
	plt.plot(t_envelope,x_t_envelope)
	return(t_envelope,x_t_envelope)
def initialize_bit_chain():

	bits=[]*numOfBits
	for i in range(numOfBits):
		bitValue=random.randint(0, 1)
		bits.append(bitValue)
	return bits

def LocalOscilator(fol,phase,signal,Tsignal1,numOfsamples):
	t=np.linspace(0,Tsignal1,numOfsamples)
	modulatedsignal= []*numOfsamples
	for i in range(len(signal)):
		modulatedsignal.append(math.cos(2*math.pi*fol*t[i]+phase)*signal[i])
	return modulatedsignal

def generate_signal_BaseBand2(x_t_envelope,data,Modulation_M):
	va=1/math.sqrt(2)
	
	M = int(math.log2(Modulation_M))
	s_t=[]*fs
	Inphases=[]*fs
	Quadratures=[]*fs
	for i in range(0,len(data),M):
		symbol=[]*M
		for j in range(i,i+M):
			symbol.append(data[j])

		symbolstr=""
		for n in range(len(symbol)):
			symbolstr=symbolstr+str(symbol[n])

		if(str(symbolstr)=="00"):
			z = complex(-va,-va)
		elif(str(symbolstr)=="10"):
			z = complex(-va,va)
		elif str(symbolstr)=="01":
			z = complex(va,-va)		
		elif(str(symbolstr)=="11"):
			z = complex(va,va)
		
		v_I = z.real
		v_Q = z.imag
		for i in range(samplesInBit):
			Inphases.append(v_I)
			Quadratures.append(v_Q)
		Tsig_Modulated=Tsig/M

	t_IQ = np.linspace(0, T_pulse/M, fs, endpoint=False)

	s_I_t = LocalOscilator(fc,0,Inphases,Tsig/M,fs)
	s_Q_t = LocalOscilator(fc,-math.pi/2,Quadratures,Tsig/M,fs)
	s_t = operating_array(s_I_t,s_Q_t,"substraction")
	
	#cos=math.cos(2*math.pi*fc*t_IQ)
	#sin=math.sin(2*math.pi*fc*t_IQ)
	#s_I_t= operating_array(v_I,cos,"dot product")
	#s_Q_t= operating_array(v_Q,sin,"dot product")
	#s_t = operating_array(v_I*cos,v_Q*sin,"substraction")
	return s_t,s_I_t,s_Q_t,Inphases,Quadratures
		#s_t = v_I*cos-v_Q*sin


def generate_signal_BaseBand(x_t_envelope,data,Modulation_M):
	modulation_bit_factor= int(math.log2(Modulation_M))
	x_t = []
	xI_t=[]
	xQ_t=[]
	isInphase = True
	for i in range(0,numOfBits,modulation_bit_factor):
		if isInphase:		
			for m in range(0,modulation_bit_factor):
				dataIndex=i
				bitValue=data[i]
				#bitValue=2.*bitValue-1
				for j in range(samplesInBit):
					if bitValue==0:
						xI_t.append(x_t_envelope[j])
					else:
						xI_t.append(-1 * (x_t_envelope[j]))
					#we modulate the signal pulse from 1 to -1

				dataIndex = dataIndex + 1
			isInphase=False
		else:

			for m in range(0,modulation_bit_factor):
				dataIndex=i
				bitValue=data[i]
				#bitValue=2.*bitValue-1
				for j in range(samplesInBit):
					if bitValue==0:
						xQ_t.append(x_t_envelope[j])
					else:
						xQ_t.append(-1 * (x_t_envelope[j]))
					#we modulate the signal pulse from 1 to -1
				dataIndex = dataIndex + 1
			isInphase=True
				
	return(xI_t,xQ_t)
def generate_modulated_signal_InphaseANDQuadrature(fc,t,xI_t,xQ_t):
	
	[xI_t,xQ_t] = Math_Functions.Set_equal_dimensions_2_IQ(xI_t,xQ_t)
	##esta mal el Tsig hay que cambairlo por el tiempo nuevo tras modular
	##ya que esto es sin modulacion
	t_IQ = t = np.linspace(0, Tsig, len(xI_t), endpoint=False)
	cos=[]
	sin=[]

	for i in range(len(t_IQ)):
		cos.append(math.cos(2*math.pi*fc*t_IQ[i]))
		sin.append(math.sin(2*math.pi*fc*t_IQ[i]))
	xIcos=operating_array(xI_t,cos,"dot product")
	xQsin=operating_array(xQ_t,sin,"dot product")
	modulated_signal = operating_array(xIcos,xQsin,"substraction")
	##modulated_signal = modulated_Inphase - modulated_Quadrature
	return modulated_signal
#def modulation_constellation():

def plot_modulated_signal_and_FFT(x_t):
	t=np.linspace(0,Tsig/math.log2(modulation),fs/2)
	x_f = fft(x_t)
	f = fftfreq(samplesNumber, Ts)[:samplesNumber//2]
	print("len(t),len(x_t) ························##",len(t),len(x_t))
	plt.figure()
	plt.plot(t,x_t)
	plt.figure()
	plt.plot(f, 2.0/samplesNumber * np.abs(x_f[0:samplesNumber//2]))
	plt.show()
		
data = initialize_bit_chain()
T_sin_envelope=T_pulse/2
 #[t_envelope,x_t_envelope]=generate_sinusoidal_envelope_in_bit(T_sin_envelope)
[t_envelope,x_t_envelope]=generate_envelope_in_bit()
[s_t_mod,sI_t,sQ_t,inphases,quadratures] = generate_signal_BaseBand2(x_t_envelope,data,modulation)

modulated_signal = generate_modulated_signal_InphaseANDQuadrature(fc,t,sI_t,sQ_t)

# plot_modulated_signal_and_FFT(xI_t)
# plot_modulated_signal_and_FFT(xQ_t)

t_IQ = t = np.linspace(0, Tsig, len(sI_t), endpoint=False)
# plt.figure(1)
# plt.plot(t_IQ,modulated_signal)


plt.figure(2)
plt.subplot(5,1,1)
plt.plot(t_IQ,inphases)
plt.subplot(5,1,2)
plt.plot(t_IQ,sI_t)
plt.subplot(5,1,3) 
plt.plot(t_IQ,quadratures)
plt.subplot(5,1,4) 
plt.plot(t_IQ,sQ_t)
plt.subplot(5,1,5) 
plt.plot(t_IQ,s_t_mod)
#plt.plot(t_IQ,xI_t)
plt.show()
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




