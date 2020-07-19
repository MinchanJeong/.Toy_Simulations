import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import numpy as np
import math

eps = 1.0e-6
movierepeat = True
############################################################
T, dt = 30, 0.01
w_0 = 1.73
O_0 = 1.00 #sqrt(g/r_0)

theta_0 = 0.3 * 0.5*math.pi

phi_0 = -0.203
phidot_0 = 0.7 #rad/s

r_0 = 1.0 #
A = 0.2 # 

C = pow(math.sin(theta_0),2)*phidot_0
############################################################
fNum = int(T/dt)

thetas = np.zeros(fNum)
ys = np.zeros(fNum)

phidots = np.zeros(fNum)
phies =  np.zeros(fNum) ## phi_0 + integrate Omega 0 to t

ys[0],thetas[0],phidots[0],phies[0] = 0.0,theta_0,phidot_0,phi_0
############################################################
def F(y,theta):
	
	s = math.sin(theta)
	c = math.cos(theta)

	f = -O_0*s + pow(C,2)*c*pow(s,-3)

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

phidots = C / np.power(np.sin(thetas),2)

r = r_0 + A * np.cos(np.asarray(range(0,fNum))*w_0*dt+math.pi)

##Calculate phies by simple integration rule

for i in range(1,fNum):
	phies[i] = phies[i-1] + 0.5*dt*(phidots[i]+phidots[i-1])

############################################################

datax = r*np.multiply(np.sin(thetas),np.cos(phies))
datay = r*np.multiply(np.sin(thetas),np.sin(phies))

############################################################

fig1 = plt.figure()

plt.plot(datax,datay,c='k')
plt.scatter([datax[0]],[datay[0]],c='r',s=30)

plt.show()
plt.close()
