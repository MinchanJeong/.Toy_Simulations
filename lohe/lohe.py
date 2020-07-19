import matplotlib
from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')
from matplotlib.animation import FuncAnimation

import numpy as np
import scipy
from scipy.stats import unitary_group
import scipy.linalg
import math
import sys

proj_to_unitary = True
movierepeat = True
########################################################
#Setting
N,d = 10,3

T,dt = 6.0,0.01
fNum = int(T/dt)
#for 2*2 lohe matrix, calc speed = 32~35 frame/s

kappa = 3.0
kappa += 0*1.0j

M = 4.0

########################################################
#Variables
U = np.zeros((fNum,N,d,d),dtype = np.complex)
H = np.zeros((N,d,d),dtype = np.complex)

print('Initial Data:\nGenerate new init data and save(1) or Load saved data(0)\n')
ans = input()

if ans == '0':

	init_load = np.load('./init.npz')
	U[0,:,:,:] = init_load['init_U']
	H = init_load['H']

elif ans == '1':
	
	print('Caution: Saved initial data will be deleted\n')
	input()

	for i in range(0,N):
		U[0,i,:,:] = unitary_group.rvs(d)
		
		A = (M + 0.0j)*np.random.rand(d,d)
		phasemat = np.random.rand(d,d)*math.pi*2
		phasemat = np.cos(phasemat)+1.0j*np.sin(phasemat)
		A = np.multiply(A,phasemat)
		
		H[i,:,:] = 0.5*(A + np.conj(np.transpose(A)))
		
	H -= np.mean(H,axis=(0))

	np.savez('./init',init_U = U[0,:,:,:],H=H)

else:
	sys.exit()

########################################################
#dummies
kU = np.zeros((N,d,d,4),dtype=np.complex)
########################################################
def F(Us):
	F_U = np.zeros((N,d,d),dtype=np.complex)
	A = np.average(Us,axis=(0))


	for i in range(0,N):
		
		dum = H[i,:,:] + (kappa*0.5j)*(A.dot(np.conj(np.transpose(Us[i,:,:]))) - Us[i,:,:].dot(np.conj(np.transpose(A))))
		
		F_U[i,:,:] = -1.0j*dum.dot(Us[i,:,:])

	return F_U

########################################################
##iterate using RK4
def iterate(Us):

	kU[:,:,:,0] = F(Us)
	kU[:,:,:,1] = F(Us+0.5*dt*kU[:,:,:,0])
	kU[:,:,:,2] = F(Us+0.5*dt*kU[:,:,:,1])
	kU[:,:,:,3] = F(Us+dt*kU[:,:,:,2])

	newUs = Us + (dt/6.0) * (kU[:,:,:,0]+2.0*kU[:,:,:,1]+2.0*kU[:,:,:,2]+kU[:,:,:,3])
	
	##Projection to unitary matrix
	if proj_to_unitary:
		
		for j in range(0,N):
			x = newUs[j,:,:].reshape((d,d))
			V, _, Wh = scipy.linalg.svd(x)
			x = np.matrix(V.dot(Wh))
			angle = np.angle(scipy.linalg.det(x))
			a = (math.cos(angle/(d*1.0))-1.0j*math.sin(angle/(d*1.0)))
			newUs[j,:,:] = x*a
	
	return newUs

## Main Calculation
for i in range (1,fNum):
	U[i,:,:,:] = iterate(U[i-1,:,:,:])
	if i%1000 ==0:
		print('progress: ',i,'/',fNum)
########################################################
##Calculate diameter of {Ve^(i theta)}
##Calculate Potential ( gradient )

D_U = np.zeros(fNum)

for k in range(0,fNum):
	u = U[k,:,:,:]
	u_c = np.mean(u,axis=(0))
	
	c1 = 0
	c2 = 0
	c3 = 0
	for i in range(0,N):
		c1 += pow(np.linalg.norm(u[i,:,:]-u_c),2)

	D_U[k] = pow(c1/N,0.5)

	print(k,'/',fNum)


##############################################
#Visualization


print('Visualization:\nby graph(0) or by animation(1)\n')
ans = input()

if ans == '0':

	fig,axs = plt.subplots(nrows=2,ncols=2,figsize=(13,8))
	plt.subplots_adjust(hspace = 0.5)

	timeline = np.asarray(range(0,fNum))*dt

	axs[0][0].plot(timeline,D_U)
	axs[0][0].set_title('$\sqrt{(\sum{{\\Vert U_i - U_c \\Vert_{F}}^{2}})/N}$',fontsize=15)
	axs[0][0].set_ylim([0,4])
	axs[0][0].set_xlim([0,fNum*dt])
	axs[0][0].set_xlabel('Time(s)')
	#axs[0][0].set_ylabel('$D(e^{i\\theta}V)$')
	
	#avgthet = np.mod(np.real(np.mean(theta,axis=1)),2.0*math.pi)
	#axs[1][0].plot(timeline,D_V)
	#axs[1][0].set_title('$\sqrt{(\sum{{\\Vert V_i - V_c \\Vert_{F}}^{2}})/N}$',fontsize=15)
	#axs[1][0].set_ylim([-2*math.pi,2*math.pi])
	#axs[1][0].set_xlim([0,fNum*dt])
	#axs[1][0].set_xlabel('Time(s)')
	#axs[1][0].set_ylabel('$\\theta_c$')
	axs[1][0].set_ylim([0,4])
	axs[1][0].set_xlim([0,fNum*dt])
	axs[1][0].set_xlabel('Time(s)')
	#axs[1][0].set_ylabel('$D(e^{i\\theta}V)$')

	#axs[0][1].plot(timeline,D_V_id_dist)
	#axs[0][1].set_title('$\\mathcal{V}_{2}(\\{\\theta_i ,\\,V_i \\})/N$',fontsize=15)
	#axs[0][1].set_xlim([0,fNum*dt])
	#axs[0][1].set_xlabel('Time(s)')
	#axs[0][1].set_ylabel('$\\mathcal{V}_{2}$')
	
	#axs[1][1].plot(timeline,D_theta)
	#axs[1][1].set_title('$\\max\\,|\\theta_i - \\theta_c|$',fontsize=15)
	#axs[2].set_ylim([0)
	##axs[1][1].set_xlim([0,fNum*dt])
	#axs[1][1].set_xlabel('Time(s)')

	c = np.linalg.norm(np.mean(H,axis=(0)))
	plt.gcf().text(0.85,0.437,f'd = {d}\nN = {N}\n$\\kappa$ = {kappa}\n$\\Vert E[H_i]\\Vert$ = {c:5.3f}',fontsize=10)

	plt.show()
	plt.close()

elif ans == '1':
	
	fig = plt.figure()

	real = np.real(U[0,:,:,:].flatten())
	imag = np.imag(U[0,:,:,:].flatten())
	
	print(imag)
	input()

	scatt = plt.scatter(real,imag)
	
	name = [None]*(N*d*d)

	for i in range(0,d*d*N):
		name[i] = plt.text(real[i],imag[i],"1")
	
	def animate(n):

		real = np.real(U[n,:,:,:].flatten())
		imag = np.imag(U[n,:,:,:].flatten())
		data = np.transpose(np.vstack((real,imag)))
		scatt.set_offsets(data)
'''
		for i in range(0,d*d*N):
			name[i].set_visible(False)
			name[i] = plt.text(real[i],imag[i],"1")
'''		
		return scatt,

	anim = FuncAnimation(fig,animate,frames=fNum,interval=20,repeat=movierepeat)
	plt.show()
	plt.close()
	del anim


else:
	sys.exit()
