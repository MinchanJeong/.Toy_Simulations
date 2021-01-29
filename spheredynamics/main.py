"""
@author     : minchan
@environment: Spyder(Python 3.8.5)
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animate
from mpl_toolkits.mplot3d import Axes3D

from flow_on_sphere import set_variables,dynamics

eps = 1.0e-6
N = 300
Kappa = 10.0
T = 2.0
dt = 0.01
fNum = int(T/dt) # Total frame number
Omega_magnitude_avg  = 5
Omega_magnitude_std  = 2
set_variables.set_general_constants(eps, N, Kappa, T, dt, Omega_magnitude_avg, Omega_magnitude_std,'carte')

data_theta, data_phi = np.zeros((fNum,N)), np.zeros((fNum,N))
data_theta[0,:] = np.random.rand(N)*math.pi
data_phi[0,:] = np.random.rand(N)*2.0*math.pi

data_x, data_y, data_z = np.zeros((fNum,N)), np.zeros((fNum,N)), np.zeros((fNum,N))
data_x[0,:] = np.sin(data_theta[0,:]) * np.cos(data_phi[0,:])
data_y[0,:] = np.sin(data_theta[0,:]) * np.sin(data_phi[0,:])
data_z[0,:] = np.cos(data_theta[0,:])

iteration_option = 'xyz_then_projection' ## or 'spherical'

if iteration_option == 'spherical':

	for i in range (1,fNum):
		data_theta[i,:],data_phi[i,:] = dynamics.iterate_spherical(data_theta[i-1,:],data_phi[i-1,:])
	
	## (theta,phi) to (x,y,z)
	data_x = np.sin(data_theta) * np.cos(data_phi)
	data_y = np.sin(data_theta) * np.sin(data_phi)
	data_z = np.cos(data_theta)

elif iteration_option == 'xyz_then_projection': 
	
	for i in range (1,fNum):
		data_x[i,:],data_y[i,:],data_z[i,:] = dynamics.iterate_xyz_by_proj(data_x[i-1,:],data_y[i-1,:],data_z[i-1,:])

###########################################################################################################

movierepeat = False

fig, axs = plt.subplots(ncols=2, nrows=2,figsize=(16,8),gridspec_kw={'width_ratios': [1, 1.5]})
fig.tight_layout(pad=1.08, h_pad=0.2, w_pad=0.3, rect=None)
fig.subplots_adjust(wspace=0.15, hspace=0.15, bottom = 0.1)
gs = axs[0, 0].get_gridspec()
# remove the underlying axes
for ax in axs[:,0]:
    ax.remove()
ax_model = fig.add_subplot(gs[:,0],projection='3d')
ax1,ax2 = axs[0, 1], axs[1, 1]
###################################### ax_model initial setting
limit = 1.0
ax_model.set_xlim3d(-1.0*limit,1.0*limit)
ax_model.set_ylim3d(-1.0*limit,1.0*limit)
ax_model.set_zlim3d(-1.0*limit,1.0*limit)
ticks = [-1,-0.5,0,0.5,1]
ax_model.set_xticks(ticks)
ax_model.set_yticks(ticks)
ax_model.set_zticks(ticks)

w_x,w_y,w_z,A = dynamics.get_omega_representative()
l = np.linspace(-1.1,1.1,200)
line_x = l * w_x / A
line_y = l * w_y / A
line_z = l * w_z / A
ax_model.scatter(line_x,line_y,line_z,s=0.1,c='k')

u,v = np.mgrid[0:2*np.pi:30j, 0:np.pi:30j]
grid_theta_coord = np.cos(u)*np.sin(v)
grid_phi_coord = np.sin(u)*np.sin(v)
grid_r_coord = np.cos(v)

repr_theta   = math.acos(w_z / A)
repr_cos_theta = math.cos(repr_theta)
repr_sin_theta = math.sin(repr_theta)
repr_cos_phi = w_x / math.sqrt(w_x**2 + w_y**2)
repr_sin_phi = w_y / math.sqrt(w_x**2 + w_y**2)
repr_r_vec     = [ repr_sin_theta*repr_cos_phi , repr_sin_theta*repr_sin_phi , repr_cos_theta ]
repr_theta_vec = [ repr_cos_theta*repr_cos_phi , repr_cos_theta*repr_sin_phi , -1.0*repr_sin_theta ]
repr_phi_vec   = [ -1.0*repr_sin_phi , repr_cos_phi , 0.0 ]

grid_x = repr_r_vec[0] * grid_r_coord + repr_theta_vec[0] * grid_theta_coord + repr_phi_vec[0] * grid_phi_coord
grid_y = repr_r_vec[1] * grid_r_coord + repr_theta_vec[1] * grid_theta_coord + repr_phi_vec[1] * grid_phi_coord
grid_z = repr_r_vec[2] * grid_r_coord + repr_theta_vec[2] * grid_theta_coord + repr_phi_vec[2] * grid_phi_coord
frame =ax_model.plot_wireframe(grid_x,grid_y,grid_z,color="k",linewidths=0.2)

titledict = {'fontsize':20, 'verticalalignment':'baseline'}
title = ax_model.set_title('Lohe Sphere',position=(0.5,1.0-0.05),fontdict=titledict)

cmap = cm.jet
value_for_coloring = np.linspace(0.0,1.0,num = N)
flow = ax_model.scatter(data_x[0,:],data_y[0,:],data_z[0,:],s=4.5,c=value_for_coloring,cmap=cmap)

time_text = ax_model.text2D(-0.05,-0.12,'',fontsize=12,bbox=dict(facecolor='gray',edgecolor='black',boxstyle='round,pad=1',alpha=0.5))

###################################### ax1,2 initial setting

diam = np.zeros(fNum)
for i in range(0,fNum):
	D =  np.square(data_x[i,:] - data_x[i,:].reshape(N,1))
	D += np.square(data_y[i,:] - data_y[i,:].reshape(N,1))
	D += np.square(data_z[i,:] - data_z[i,:].reshape(N,1))
	diam[i] = math.sqrt(np.max(D))

ax1.plot(np.asarray(range(0,fNum))*dt,diam,color='black')
ax1.set_ylim([0,np.max(diam)*1.1])
ax1.set_xlim([0,np.max(fNum)*dt])
ax1.set_ylabel('$D_F (\Theta)$',fontsize=14)
marker1 = ax1.scatter(0,diam[0],c='orange')


speed = np.zeros(fNum)
for i in range(1,fNum):
	xdot = (data_x[i,:] - data_x[i-1,:])/dt
	ydot = (data_y[i,:] - data_y[i-1,:])/dt
	zdot = (data_z[i,:] - data_z[i-1,:])/dt 
	D =  np.square(xdot - xdot.reshape(N,1))
	D += np.square(ydot - ydot.reshape(N,1))
	D += np.square(zdot - zdot.reshape(N,1))
	speed[i] = math.sqrt(np.max(D))
speed[0] = speed[1]
	
ax2.plot(np.asarray(range(0,fNum))*dt,speed,color='black')
ax2.set_ylim([0,np.max(speed)*1.1])
ax2.set_xlim([0,np.max(fNum)*dt])
ax2.set_ylabel('$D_F (\dot{\Theta})$',fontsize=14)
ax2.set_xlabel('Time',fontsize=14)
marker2 = ax2.scatter(0,speed[0],c='orange')

######################################

gap = 1
wait = 30

def update_flow_show(num):
	
	if num < wait:
		time_text.set_text(f'N = {N}, κ = {Kappa}, T = {T}, time = {0:5.3f}')
	else:
		num = num - wait
		flow._offsets3d = (data_x[num*gap,:],data_y[num*gap,:],data_z[num*gap,:])

		time_text.set_text(f'N = {N}, κ = {Kappa}, T = {T}, time = {num*gap*dt:5.3f}')
		marker1.set_offsets([num*gap*dt,diam[num*gap]])
		marker2.set_offsets([num*gap*dt,speed[num*gap]])
		
ani = animate.FuncAnimation(fig,update_flow_show,interval=dt*1000,frames = int(fNum/gap)+wait,blit=False,repeat = movierepeat)


#######################################

c = input('\n'+'Save Data?: ')

if c=='y':

	print('Data is being saved...')
	ani.save(filename='./flow_sample.gif',writer='ffmpeg',dpi=80,fps=30, progress_callback = lambda i, n: print(f'Saving frame {i} of {n}'))
	print('saving process completed')

plt.show()

#############################################################################