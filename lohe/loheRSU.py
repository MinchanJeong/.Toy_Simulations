import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import unitary_group
import scipy.linalg
import math
import sys

theta_as_mod2pi = False
proj_to_unitary = True
########################################################
#Setting
N,d = 15,8

T,dt = 50,0.01
fNum = int(T/dt)
#for 2*2 lohe matrix, calc speed = 32~35 frame/s

kappa = 9.0
kappa += 0*1.0j

M = 3.0

########################################################
#Variables
theta =  np.zeros((fNum,N),dtype = np.complex)	
V = np.zeros((fNum,N,d,d),dtype = np.complex)
H = np.zeros((N,d,d),dtype = np.complex)

 #initial settings
#theta[0,:] = np.random.rand(N)*2.0*math.pi/(d*1.0)
#t1 = np.random.rand(N)*math.pi*0.4 #0~0.5
#t2 = np.random.rand(N)*math.pi*2.0 #0~2.0
#t3 = np.random.rand(N)*math.pi*2.0 #0~2.0
#V[0,:,0,0] = np.multiply(np.cos(t1),np.cos(t2)+1.0j*np.sin(t2))
#V[0,:,1,0] = np.multiply(np.sin(t1),np.cos(t3)+1.0j*np.sin(t3))
#V[0,:,1,1] = np.conj(V[0,:,0,0])
#V[0,:,0,1] = np.conj(V[0,:,1,0])*-1.0

print('Initial Data:\nGenerate new init data and save(1) or Load saved data(0)\n')
ans = input()

if ans == '0':

	init_load = np.load('./initRSU.npz')
	V[0,:,:,:] = init_load['init_V']
	theta[0,:] = init_load['init_theta']
	H = init_load['H']

elif ans == '1':
	
	print('Caution: Saved initial data will be deleted\n')
	input()

	for i in range(0,N):
		x = unitary_group.rvs(d)
		det_angle = np.angle(scipy.linalg.det(x))
		a = (math.cos(det_angle/(d*1.0))-1.0j*math.sin(det_angle/(d*1.0)))
		x = x*a
		angle = det_angle / (d*1.0)
		V[0,i,:,:] = x
		theta[0,i] = angle
		
		A = np.zeros((d,d),dtype = np.complex)
		A = (M + 0.0j)*np.random.rand(d,d)
		phasemat = np.random.rand(d,d)*math.pi*2
		phasemat = np.cos(phasemat)+1.0j*np.sin(phasemat)
		A = np.multiply(A,phasemat)
		
		y = A - np.conj(np.transpose(A))
		H[i,:,:] = 0.5j*(y - (1.0/(d*1.0))*np.trace(y)*np.identity(d,dtype=np.complex))
	
		
	H -= np.mean(H,axis=(0))

	np.savez('./initRSU',init_V = V[0,:,:,:],init_theta = theta[0,:],H=H)

else:
	sys.exit()

########################################################
#dummies
kt = np.zeros((N,4),dtype=np.complex)
kV = np.zeros((N,d,d,4),dtype=np.complex)
########################################################
##thetas as shape (N) \\ Vs as shape (N,2,2)
def F(thetas,Vs):
	phase = np.cos(thetas)+1.0j*np.sin(thetas)
	B = np.einsum('i,ijk->ijk',phase,Vs)
	A = np.mean(B,axis=(0))

	trB = np.zeros(N,dtype=np.complex)
	trH = np.zeros(N,dtype=np.complex)

	F_V = np.zeros((N,d,d),dtype=np.complex)
	for i in range(0,N):
		B[i,:,:] = B[i,:,:].dot(np.transpose(np.conj(A)))
		B[i,:,:] = 0.5*(np.conj(np.transpose(B[i,:,:])) - B[i,:,:])
		trB[i] = np.trace(B[i,:,:].reshape((d,d)))
		trH[i] = np.trace(H[i,:,:].reshape((d,d)))
		##print(trB[i])
		F_V[i,:,:] = (-1.0j*H[i,:,:] + (1.0j/(d*1.0))*np.identity(d,dtype=np.complex)*trH[i] + kappa*(B[i,:,:] - (1.0/(d*1.0))*trB[i]*np.identity(d,dtype=np.complex))).dot(Vs[i,:,:])
		
	return ((-1.0/(d*1.0))*trH - (1.0j*kappa/(1.0*d))*trB), F_V

########################################################
##iterate using RK4
def iterate(thetas,Vs):

	kt[:,0],kV[:,:,:,0] = F(thetas,Vs)
	kt[:,1],kV[:,:,:,1] = F(thetas+0.5*dt*kt[:,0],Vs+0.5*dt*kV[:,:,:,0])
	kt[:,2],kV[:,:,:,2] = F(thetas+0.5*dt*kt[:,1],Vs+0.5*dt*kV[:,:,:,1])
	kt[:,3],kV[:,:,:,3] = F(thetas+dt*kt[:,2],Vs+dt*kV[:,:,:,2])
	
	if theta_as_mod2pi:
		newthetas = (thetas + (dt/6.0) * (kt[:,0] + 2.0*kt[:,1] + 2.0*kt[:,2] + kt[:,3]))/(2*math.pi)
		newthetas = np.real(newthetas)
		newthetas = (newthetas - np.floor(newthetas))*2.0*math.pi

	else:
		newthetas = (thetas + (dt/6.0) * (kt[:,0] + 2.0*kt[:,1] + 2.0*kt[:,2] + kt[:,3]))
		newthetas = np.real(newthetas)	

	newVs = Vs + (dt/6.0) * (kV[:,:,:,0]+2.0*kV[:,:,:,1]+2.0*kV[:,:,:,2]+kV[:,:,:,3])
	
	##Projection to unitary matrix
	if proj_to_unitary:
		
		for j in range(0,N):
			x = newVs[j,:,:].reshape((d,d))
			V, _, Wh = scipy.linalg.svd(x)
			x = np.matrix(V.dot(Wh))
			angle = np.angle(scipy.linalg.det(x))
			a = (math.cos(angle/(d*1.0))-1.0j*math.sin(angle/(d*1.0)))
			newVs[j,:,:] = x*a
	
	return newthetas, newVs

## Main Calculation
for i in range (1,fNum):
	theta[i,:],V[i,:,:,:] = iterate(theta[i-1,:],V[i-1,:,:,:])
	if i%1000 ==0:
		print('progress: ',i,'/',fNum)
########################################################
##Calculate diameter of {Ve^(i theta)}
##Calculate Potential ( gradient )

D_U = np.zeros(fNum)
D_V = np.zeros(fNum)
D_theta = np.zeros(fNum)
D_V_id_dist = np.zeros(fNum)

for k in range(0,fNum):
	phase = np.cos(theta[k,:])+1.0j*np.sin(theta[k,:])
	U = np.einsum('i,ijk->ijk',phase,V[k,:,:,:])
	U_c = np.mean(U,axis=(0))
	V_c = np.mean(V[k,:,:,:],axis=(0))
	theta_c = np.mean(theta[k,:])
	
	c1 = 0
	c2 = 0
	c3 = 0
	for i in range(0,N):
		c1 += pow(np.linalg.norm(U[i,:,:]-U_c),2)
		c2 += pow(np.linalg.norm(V[k,i,:,:]-V_c),2)
		c3 += pow(np.linalg.norm(V[k,i,:,:]-np.identity(d,dtype=np.complex)),2)

	D_theta[k] = np.max(np.abs(theta[k,:]-theta_c))
	
	D_U[k] = pow(c1/N,0.5)
	D_V[k] = pow(c2/N,0.5)
	D_V_id_dist[k] = pow(c3/N,0.5)

	print(k,'/',fNum)


##############################################
#Visualization

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
axs[1][0].plot(timeline,D_V)
axs[1][0].set_title('$\sqrt{(\sum{{\\Vert V_i - V_c \\Vert_{F}}^{2}})/N}$',fontsize=15)
#axs[1][0].set_ylim([-2*math.pi,2*math.pi])
#axs[1][0].set_xlim([0,fNum*dt])
#axs[1][0].set_xlabel('Time(s)')
#axs[1][0].set_ylabel('$\\theta_c$')
axs[1][0].set_ylim([0,4])
axs[1][0].set_xlim([0,fNum*dt])
axs[1][0].set_xlabel('Time(s)')
#axs[1][0].set_ylabel('$D(e^{i\\theta}V)$')

axs[0][1].plot(timeline,D_V_id_dist)
axs[0][1].set_title('$\\mathcal{V}_{2}(\\{\\theta_i ,\\,V_i \\})/N$',fontsize=15)
axs[0][1].set_xlim([0,fNum*dt])
axs[0][1].set_xlabel('Time(s)')
#axs[0][1].set_ylabel('$\\mathcal{V}_{2}$')

axs[1][1].plot(timeline,D_theta)
axs[1][1].set_title('$\\max\\,|\\theta_i - \\theta_c|$',fontsize=15)
#axs[2].set_ylim([0)
axs[1][1].set_xlim([0,fNum*dt])
axs[1][1].set_xlabel('Time(s)')
c = np.linalg.norm(np.mean(H,axis=(0)))
plt.gcf().text(0.85,0.437,f'd = {d}\nN = {N}\n$\\kappa$ = {kappa}\n$\\Vert E[H_i]\\Vert$ = {c:5.3f}',fontsize=10)

plt.show()
plt.close()
