# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 05:24:47 2021

@author: ssk98
"""
###-----------------------Basic Configuration---------------------------###

import numpy as np
import math
import flow_on_sphere.dynamics as dy

def set_general_constants(_eps,_N,_Kappa,_T,_dt,_wavg,_wstd):
	
	dy.eps = _eps
	dy.N, dy.Kappa, dy.T, dy.dt = _N, _Kappa, _T, _dt
	
	Omega_direction_theta = np.random.rand(_N)*math.pi*0.05
	Omega_direction_phi = np.random.rand(_N)*2*math.pi
	Omega_magnitudes = _wavg + _wstd*(np.random.rand(_N)*2.0 - 1.0)
	
	_Omega_xs = Omega_magnitudes * np.sin(Omega_direction_theta) * np.cos(Omega_direction_phi)
	_Omega_ys = Omega_magnitudes * np.sin(Omega_direction_theta) * np.sin(Omega_direction_phi)
	_Omega_zs = Omega_magnitudes * np.cos(Omega_direction_theta)
		
	dy.Omega_xs, dy.Omega_ys, dy.Omega_zs = _Omega_xs, _Omega_ys, _Omega_zs
	
	dy.r_vec, dy.theta_vec, dy.phi_vec = np.zeros((_N,3)), np.zeros((_N,3)), np.zeros((_N,3))  
	dy.k1, dy.k2, dy.k3 =  np.zeros((_N,4)), np.zeros((_N,4)), np.zeros((_N,4))
	
	return 0
