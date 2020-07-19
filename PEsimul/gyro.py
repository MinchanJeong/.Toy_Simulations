import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animate
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

eps = 1.0e-6
movierepeat = True

############################################################
T, dt = 100, 0.04
O_p = 0.5
O_0 = 1.2
thet_0 = 0.0 * math.pi

y_0 = 0.0
phi_0 = 0.0 * math.pi
############################################################
fNum = int(T/dt)

thetas = np.zeros(fNum)
ys = np.zeros(fNum) ## dummy for theta_dot
Omegas = np.zeros(fNum)
phies =  np.zeros(fNum) ## phi_0 + integrate Omega 0 to t

thetas[0],ys[0],Omegas[0],phies[0] = thet_0,y_0,O_0,phi_0
############################################################
def F(y,theta):
	
	s = math.sin(theta)
	c = math.cos(theta)

	f = -1.0*((1+O_0*O_0)*s - (1+s*s)*O_0 + O_p*pow(c,4))/pow(c,3)

	return f,y

def iterate(y,theta):

	ky = np.zeros(4)
	kt = np.zeros(4)

	ky[0],kt[0] = F(y,theta)
	ky[1],kt[1] = F(y+0.5*dt*ky[0],theta+0.5*dt*kt[0])
	ky[2],kt[2] = F(y+0.5*dt*ky[1],theta+0.5*dt*kt[1])
	ky[3],kt[3] = F(y+dt*ky[2],theta+dt*kt[2])

	newy = y + (dt/6.0)*(ky[0]+2.0*ky[1]+2.0*ky[2]+ky[3])
	newtheta = theta + (dt/6.0)*(kt[0]+2.0*kt[1]+2.0*kt[2]+kt[3])

	return newy,newtheta

## main simulation
for i in range(1,fNum):
	ys[i],thetas[i] = iterate(ys[i-1],thetas[i-1]) 

Omegas = (O_0 - np.sin(thetas))/np.power(np.cos(thetas),2)

##Calculate phies by simple integration rule

for i in range(1,fNum):
	phies[i] = phies[i-1] + 0.5*dt*(Omegas[i]+Omegas[i-1])

############################################################

datax = np.multiply(np.cos(thetas),np.cos(phies))
datay = np.multiply(np.cos(thetas),np.sin(phies))
dataz = np.sin(thetas)

############################################################


fig1 = plt.figure()
fig1.canvas.manager.window.attributes('-topmost',0)
ax1 = Axes3D(fig1,proj_type='ortho')

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
title = ax1.set_title('gyro',position=(0.5,1.0-0.05),fontdict=titledict)

current = ax1.scatter([datax[0]],[datay[0]],[dataz[0]],s=100,c='orange')

n=350
current1 = ax1.scatter([datax[0:n]],[datay[0:n]],[dataz[0:n]],s=10,c='k')


def update_graph_show(num):
	current._offsets3d = ([datax[num]],[datay[num]],[dataz[num]])



ani = animate.FuncAnimation(fig1,update_graph_show,interval=dt*100,frames=fNum,blit=False,repeat = movierepeat)


plt.show()
#plt.close()

del ani


