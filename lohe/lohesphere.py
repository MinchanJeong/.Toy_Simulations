import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animate
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import math

eps = 1.0e-6
########################################################################

N = 100
T = 8 #(s)
dt = 0.01 # ~ 1/fps
fNum = int(T/dt)

kappa = 4
wstd = 1
wavg = 3

rot_theta = np.random.rand(N)*math.pi*0.05
rot_phi = np.random.rand(N)*2*math.pi

mat_theta = np.zeros((fNum,N))
mat_phi = np.zeros((fNum,N))
mat_theta[0,:] = np.random.rand(N)*math.pi
mat_phi[0,:] = np.random.rand(N)*2.0*math.pi

rotframe = False
movierepeat = True
avgtrace = False
#########################################################################
## Set Omega Mat _ i
rot_omega = wavg + np.random.rand(N)*2*wstd - wstd
rot_x = np.multiply(np.sin(rot_theta),np.cos(rot_phi))
rot_y = np.multiply(np.sin(rot_theta),np.sin(rot_phi))
rot_z = np.cos(rot_theta)

wx = np.sum(rot_omega * rot_x)/N
wy = np.sum(rot_omega * rot_y)/N
wz = np.sum(rot_omega * rot_z)/N

A = math.sqrt(wx*wx+wy*wy+wz*wz)
wx /= A
wy /= A
wz /= A
#########################################
## Set dummies 
r_vec = np.zeros((N,3))  #[:,0];x [:,1];y [:,2];z
theta_vec = np.zeros((N,3))
phi_vec = np.zeros((N,3))

coefmat = np.zeros((N,N))
ipmat = np.zeros((N,N))
dum_ipmat = np.zeros((N,N))

dumx = np.zeros((N,N))
dumy = np.zeros((N,N))
dumz = np.zeros((N,N))

kt = np.zeros((N,4))
kp = np.zeros((N,4))

diam = np.zeros(fNum)
#########################################

def spherevec(thetas,phies,r_vec,theta_vec,phi_vec):
	r_vec[:,0] = np.multiply(np.sin(thetas),np.cos(phies))
	r_vec[:,1] = np.multiply(np.sin(thetas),np.sin(phies))
	r_vec[:,2] = np.cos(thetas)
	theta_vec[:,0] = np.multiply(np.cos(thetas),np.cos(phies))
	theta_vec[:,1] = np.multiply(np.cos(thetas),np.sin(phies))
	theta_vec[:,2] = -1.0*np.sin(thetas)
	phi_vec[:,0] = -1.0*np.sin(phies)
	phi_vec[:,1] = np.cos(phies)
	#phi_vec[:,2] = 0
	
	return 0

def F(thetas,phies,r_vec,theta_vec,phi_vec):

	spherevec(thetas,phies,r_vec,theta_vec,phi_vec)

	coefmat = np.matmul(phies.reshape(N,1),np.ones((1,N)))-phies[np.newaxis,:]
	coefmat = np.diag(np.sin(thetas)) @ np.cos(coefmat) @ np.diag(np.sin(thetas))
	coefmat += np.matmul(np.cos(thetas).reshape(N,1),np.cos(thetas).reshape(1,N))
	coefmat = (1.0 - coefmat)/2.0
	coefmat = np.where(np.abs(coefmat)<eps,0,coefmat)
	coefmat = np.sin(2.0*np.arcsin(np.sqrt(coefmat)))

	ipmat = np.matmul(r_vec[:,0].reshape(N,1),r_vec[:,0].reshape(1,N))+np.matmul(r_vec[:,1].reshape(N,1),r_vec[:,1].reshape(1,N))+np.matmul(r_vec[:,2].reshape(N,1),r_vec[:,2].reshape(1,N))

	dum_ipmat = np.where(1 > np.power(ipmat,2),ipmat,np.sign(ipmat))
	dum_ipmat = np.sqrt(np.max(1 - np.power(dum_ipmat,2),0))
	dum_ipmat = np.where(eps > dum_ipmat, np.inf , dum_ipmat)
	dum_ipmat = 1.0/dum_ipmat
	
	coefmat = np.multiply(dum_ipmat,coefmat)

	dumx = np.multiply(ipmat,-1.0 * r_vec[:,0].reshape(N,1) @ np.ones((1,N))) + np.ones((N,1)) @ r_vec[:,0].reshape(1,N)
	dumy = np.multiply(ipmat,-1.0 * r_vec[:,1].reshape(N,1) @ np.ones((1,N))) + np.ones((N,1)) @ r_vec[:,1].reshape(1,N)
	dumz = np.multiply(ipmat,-1.0 * r_vec[:,2].reshape(N,1) @ np.ones((1,N))) + np.ones((N,1)) @ r_vec[:,2].reshape(1,N)

	theta_dot = np.multiply(np.sum(np.multiply(coefmat,dumx),axis=1),theta_vec[:,0])+np.multiply(np.sum(np.multiply(coefmat,dumy),axis=1),theta_vec[:,1])+np.multiply(np.sum(np.multiply(coefmat,dumz),axis=1),theta_vec[:,2])
	theta_dot *= kappa/N
	theta_dot += np.multiply(rot_omega,phi_vec[:,0]*rot_x+phi_vec[:,1]*rot_y+phi_vec[:,2]*rot_z)

	phi_dot = np.multiply(np.sum(np.multiply(coefmat,dumx),axis=1),phi_vec[:,0])+np.multiply(np.sum(np.multiply(coefmat,dumy),axis=1),phi_vec[:,1])+np.multiply(np.sum(np.multiply(coefmat,dumz),axis=1),phi_vec[:,2])
	phi_dot *= kappa/N
	phi_dot -= np.multiply(rot_omega,theta_vec[:,0]*rot_x+theta_vec[:,1]*rot_y+theta_vec[:,2]*rot_z)

	sininv = np.where(np.sin(thetas) < eps , 0, 1/np.sin(thetas))
	phi_dot = np.multiply(phi_dot,sininv)

	return theta_dot,phi_dot

def iterate(thetas,phies,r_vec,theta_vec,phi_vec,kt,kp):

	kt[:,0],kp[:,0] = F(thetas,phies,r_vec,theta_vec,phi_vec)
	kt[:,1],kp[:,1] = F(thetas+0.5*dt*kt[:,0],phies+0.5*dt*kp[:,0],r_vec,theta_vec,phi_vec)
	kt[:,2],kp[:,2] = F(thetas+0.5*dt*kt[:,1],phies+0.5*dt*kp[:,1],r_vec,theta_vec,phi_vec)
	kt[:,3],kp[:,3] = F(thetas+dt*kt[:,2],phies+dt*kp[:,2],r_vec,theta_vec,phi_vec)
	
	newtheta = (thetas + (dt/6.0) * (kt[:,0] + 2.0*kt[:,1] + 2.0*kt[:,2] + kt[:,3]))/(2*math.pi)
	newphi = (phies + (dt/6.0) * (kp[:,0] + 2.0*kp[:,1] + 2.0*kp[:,2] + kp[:,3]))/(2*math.pi)
	newtheta = (newtheta - np.floor(newtheta))*2.0*math.pi
	newphi = (newphi - np.rint(newphi))*2.0*math.pi

	return newtheta,newphi

## main simulation part

R_w = np.zeros((3,3))

for i in range (1,fNum):
	mat_theta[i,:],mat_phi[i,:] = iterate(mat_theta[i-1,:],mat_phi[i-1,:],r_vec,theta_vec,phi_vec,kt,kp)

## (theta,phi) to (x,y,z)

datax = np.multiply(np.sin(mat_theta),np.cos(mat_phi))
datay = np.multiply(np.sin(mat_theta),np.sin(mat_phi))
dataz = np.cos(mat_theta)

## frame to (inertial) rotating frame

if rotframe == True:

	for i in range(1,fNum):
		
		wt = -1*dt*i*A/(2.0*math.pi)
		wt = (wt - np.rint(wt))*2.0*math.pi
		
		c = math.cos(wt)
		s = math.sin(wt)
		R_w[0,:] = [c+wx*wx*(1-c),wx*wy*(1-c)-wz*s,wx*wz*(1-c)+wy*s]
		R_w[1,:] = [wy*wx*(1-c)+wz*s,c+wy*wy*(1-c),wy*wz*(1-c)-wx*s]
		R_w[2,:] = [wx*wz*(1-c)-wy*s,wz*wy*(1-c)+wx*s,c+wz*wz*(1-c)]
		
		X = R_w[0][0]*datax[i,:]+R_w[0][1]*datay[i,:]+R_w[0][2]*dataz[i,:]
		Y = R_w[1][0]*datax[i,:]+R_w[1][1]*datay[i,:]+R_w[1][2]*dataz[i,:]
		Z = R_w[2][0]*datax[i,:]+R_w[2][1]*datay[i,:]+R_w[2][2]*dataz[i,:]

		datax[i,:],datay[i,:],dataz[i,:] = X,Y,Z

## Calculate diameter

for i in range(0,fNum):


	D = np.matmul(datax[i,:].reshape(N,1),np.ones((1,N)))-datax[i,:]
	D += np.matmul(datay[i,:].reshape(N,1),np.ones((1,N)))-datay[i,:]
	D += np.matmul(dataz[i,:].reshape(N,1),np.ones((1,N)))-dataz[i,:]
	
	diam[i] = math.sqrt(np.max(D))


################################################

##fig1 = plt.figure()
#fig1.canvas.manager.window.attributes('-topmost',0)
#ax1 = Axes3D(fig1,proj_type='ortho')
fig1 = plt.figure(figsize=(13,6))
plt.subplots_adjust(hspace = 0.5, wspace = 0.4)
gridshape = (2,2)

ax1 = plt.subplot2grid(gridshape,(0,0),rowspan=2,projection='3d')
ax2 = plt.subplot2grid(gridshape,(0,1))
##ax1 = fig1.add_subplot(1,2,1,projection='3d')
##ax2 = fig1.add_subplot(1,2,2)

###################################### ax1
limit = 1.0
ax1.set_xlim3d(-1.0*limit,1.0*limit)
ax1.set_ylim3d(-1.0*limit,1.0*limit)
ax1.set_zlim3d(-1.0*limit,1.0*limit)
ticks = [-1,-0.5,0,0.5,1]
ax1.set_xticks(ticks)
ax1.set_yticks(ticks)
ax1.set_zticks(ticks)

u,v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
frame =ax1.plot_wireframe(x,y,z,color="k",linewidths=0.3)

titledict = {'fontsize':20, 'verticalalignment':'baseline'}
title = ax1.set_title('Sphere model',position=(0.5,1.0-0.05),fontdict=titledict)

graph = ax1.scatter(datax[0,:],datay[0,:],dataz[0,:],s=4,c='k')
graph2 = ax1.scatter([np.average(datax[0,:])],[np.average(datay[0,:])],[np.average(dataz[0,:])],s=1,c='r')

l = np.linspace(-1.1,1.1,200)
linex = l*wx
liney = l*wy
linez = l*wz
ax1.scatter(linex,liney,linez,s=0.3,c='k')

time_text = ax1.text2D(-0.1,-0.01,'',bbox=dict(facecolor='gray',edgecolor='black',boxstyle='round,pad=1',alpha=0.7))
######################################ax2

ax2.plot(np.asarray(range(0,fNum))*dt,diam)
ax2.set_ylim([0,np.max(diam)*1.1])
ax2.set_xlim([0,np.max(fNum)*dt])
ax2.set_xlabel('Time(s)')
ax2.set_ylabel('$D(\Theta)$')
ax2.set_title('$D(\Theta)$')

graph3 = ax2.scatter(0,diam[0],c='r')

######################################

gap = 1
wait = 30

def update_graph_show(num):
	
	if num < wait:
		time_text.set_text(f'N = {N}\nK = {kappa}\ntime = {0:5.3f}s')
	else:
		num = num - wait
		graph._offsets3d = (datax[num*gap,:],datay[num*gap,:],dataz[num*gap,:])
		
		if avgtrace == True:
			ax1.scatter([np.average(datax[num*gap,:])],[np.average(datay[num*gap,:])],[np.average(dataz[num*gap,:])],s=1,c='r')
		else:
			graph2._offsets3d = ([np.average(datax[num*gap,:])],[np.average(datay[num*gap,:])],[np.average(dataz[num*gap,:])])

		#dist = math.sqrt(pow((datax[num*gap][0]-datax[num*gap][1]),2)+pow((datay[num*gap][0]-datay[num*gap][1]),2)+pow((dataz[num*gap][0]-dataz[num*gap][1]),2))
		
		time_text.set_text(f'N = {N}\nK = {kappa}\nTime = {num*gap*dt:5.3f}s\ndiam = {diam[num*gap]:5.4f}')

		graph3.set_offsets([num*gap*dt,diam[num*gap]])


ani = animate.FuncAnimation(fig1,update_graph_show,interval=dt*1000,frames = int(fNum/gap)+wait,blit=False,repeat = movierepeat)


#######################################

c = input('\n'+'Save Data?: ')

if c=='y':

	print('Data is being saved...')
	
	ani.save(filename='motion2.mp4',writer='ffmpeg',dpi=130,fps=20)
	
	print('saving process completed')

else:
	plt.show()
	plt.close()

del ani

