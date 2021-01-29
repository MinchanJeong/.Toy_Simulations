"""
@author     : minchan
@environment: Spyder(Python 3.8)
"""
import numpy as np
import math

eps = 1.0e-6
N = 10
Kappa = 10000.0
T = 1   	     # Total time (s)
dt = 0.01       # Time interval between adjacent frames

Omega_magnitude_std  = 12345
Omega_magnitude_avg  = 12345

Omega_direction_theta = np.random.rand(N) * math.pi * 0.05
Omega_direction_phi = np.random.rand(N) * 2 * math.pi
Omega_magnitudes = Omega_magnitude_avg + Omega_magnitude_std * (np.random.rand(N) * 2.0 - 1.0)

Omega_xs = Omega_magnitudes * np.sin(Omega_direction_theta) * np.cos(Omega_direction_phi)
Omega_ys = Omega_magnitudes * np.sin(Omega_direction_theta) * np.sin(Omega_direction_phi)
Omega_zs = Omega_magnitudes * np.cos(Omega_direction_theta)

r_vec, theta_vec, phi_vec = np.zeros((N,3)), np.zeros((N,3)), np.zeros((N,3))  
#[:,0];x [:,1];y [:,2];z
k1, k2, k3 =  np.zeros((N,4)), np.zeros((N,4)), np.zeros((N,4))

def spherevec(thetas,phies,r_vec,theta_vec,phi_vec):
	
	r_vec[:,0] = np.multiply(np.sin(thetas),np.cos(phies))
	r_vec[:,1] = np.multiply(np.sin(thetas),np.sin(phies))
	r_vec[:,2] = np.cos(thetas)
	theta_vec[:,0] = np.multiply(np.cos(thetas),np.cos(phies))
	theta_vec[:,1] = np.multiply(np.cos(thetas),np.sin(phies))
	theta_vec[:,2] = -1.0*np.sin(thetas)
	phi_vec[:,0] = -1.0*np.sin(phies)
	phi_vec[:,1] = np.cos(phies)
	phi_vec[:,2] = 0

	return 0

def F_spherelohe(thetas,phies):

	spherevec(thetas,phies,r_vec,theta_vec,phi_vec)
	
	dumx = np.sum(r_vec[:,0] - r_vec[:,0].reshape(N,1), axis=1)
	dumy = np.sum(r_vec[:,1] - r_vec[:,1].reshape(N,1), axis=1)
	dumz = np.sum(r_vec[:,2] - r_vec[:,2].reshape(N,1), axis=1)
	
	theta_dot = Kappa/N * ( theta_vec[:,0] * dumx + theta_vec[:,1] * dumy + theta_vec[:,2] * dumz )
	theta_dot += phi_vec[:,0]*Omega_xs+phi_vec[:,1]*Omega_ys+phi_vec[:,2]*Omega_zs
	
	phi_dot  = Kappa/N * ( phi_vec[:,0] * dumx + phi_vec[:,1] * dumy + phi_vec[:,2] * dumz )
	phi_dot -= theta_vec[:,0]*Omega_xs+theta_vec[:,1]*Omega_ys+theta_vec[:,2]*Omega_zs
	sininv = np.where(np.sin(thetas) < eps , 0, 1/np.sin(thetas))
	phi_dot = np.multiply(phi_dot,sininv)
	
	return theta_dot, phi_dot

def iterate_spherical(thetas,phies):

	k1[:,0],k2[:,0] = F_spherelohe(thetas,phies)
	k1[:,1],k2[:,1] = F_spherelohe(thetas+0.5*dt*k1[:,0],phies+0.5*dt*k2[:,0])
	k1[:,2],k2[:,2] = F_spherelohe(thetas+0.5*dt*k1[:,1],phies+0.5*dt*k2[:,1])
	k1[:,3],k2[:,3] = F_spherelohe(thetas+dt*k1[:,2],phies+dt*k2[:,2])
	
	newthetas = (thetas + (dt/6.0) * (k1[:,0] + 2.0*k1[:,1] + 2.0*k1[:,2] + k1[:,3]))/(2*math.pi)
	newphies = (phies + (dt/6.0) * (k2[:,0] + 2.0*k2[:,1] + 2.0*k2[:,2] + k2[:,3]))/(2*math.pi)
	newthetas = (newthetas - np.floor(newthetas))*2.0*math.pi
	newphies = (newphies - np.rint(newphies))*2.0*math.pi

	return newthetas,newphies

def F_spherelohe_proj(xs,ys,zs):

	x_avg = np.mean(xs)
	y_avg = np.mean(ys)
	z_avg = np.mean(zs)
	inner_product_arr = np.mean( xs.reshape(N,1) * xs + ys.reshape(N,1) * ys + zs.reshape(N,1) * zs, axis = 1 )
	
	x_dot = Kappa* ( x_avg - inner_product_arr * xs) + (Omega_ys * zs  - Omega_zs * ys)
	y_dot = Kappa* ( y_avg - inner_product_arr * ys) + (Omega_zs * xs  - Omega_xs * zs)
	z_dot = Kappa* ( z_avg - inner_product_arr * zs) + (Omega_xs * ys  - Omega_ys * xs)
	
	return x_dot, y_dot, z_dot

def iterate_xyz_by_proj(xs,ys,zs):

	k1[:,0],k2[:,0],k3[:,0] = F_spherelohe_proj(xs,ys,zs)
	k1[:,1],k2[:,1],k3[:,1] = F_spherelohe_proj(xs+0.5*dt*k1[:,0],ys+0.5*dt*k2[:,0],zs+0.5*dt*k3[:,0])
	k1[:,2],k2[:,2],k3[:,2] = F_spherelohe_proj(xs+0.5*dt*k1[:,1],ys+0.5*dt*k2[:,1],zs+0.5*dt*k3[:,1])
	k1[:,3],k2[:,3],k3[:,3] = F_spherelohe_proj(xs+dt*k1[:,2],ys+dt*k2[:,2],zs+dt*k3[:,2])
	
	newxs= (xs + (dt/6.0) * (k1[:,0] + 2.0*k1[:,1] + 2.0*k1[:,2] + k1[:,3]))
	newys= (ys + (dt/6.0) * (k2[:,0] + 2.0*k2[:,1] + 2.0*k2[:,2] + k2[:,3]))
	newzs= (zs + (dt/6.0) * (k3[:,0] + 2.0*k3[:,1] + 2.0*k3[:,2] + k3[:,3]))
	
	invnorm = np.power((np.square(newxs) + np.square(newys) + np.square(newzs)),-0.5)
	newxs, newys, newzs = newxs * invnorm, newys * invnorm, newzs * invnorm
	
	return newxs, newys, newzs

def get_omega_representative():
	w_x = np.mean(Omega_xs)
	w_y = np.mean(Omega_ys)
	w_z = np.mean(Omega_zs)
	A = math.sqrt(w_x**2+w_y**2+w_z**2)
	return w_x,w_y,w_z,A

