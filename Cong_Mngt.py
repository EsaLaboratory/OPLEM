#import modules
import os
import copy
import pandas as pd
import numpy as np
import picos as pic
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import time

from System.Network_3ph_pf import Network_3ph
import System.Assets as AS

import scipy.stats
from scipy.stats import norm

np.random.seed(1000)
#######################################
### RUN OPTIONS
#######################################
dt_raw = 1/60
T_raw = int(24/dt_raw) #Number of data time intervals
dt = 30/60 #5 minute time intervals
T = int(24/dt) #Number of intervals
dt_ems = 30/60
T_ems = int(24/dt_ems)

#######################################
### STEP 0: Load Data
#######################################
### 2) Load Data
parent_path = "C:\\Users\\cessayeh\\Anaconda3\\envs\\epymarl\\Lib\\site-packages\\gym\\envs\\OPEN_env\\Data"
Loads_data_path = os.path.join(parent_path, "Load1min.npy")  
Loads = np.load(Loads_data_path).astype(float)
N_loads_raw = Loads.shape[1]
Load_ems = Loads.transpose().reshape(-1,int(dt_ems/dt_raw)).mean(1).reshape(N_loads_raw,-1).transpose()
Load_ems = np.nan_to_num(Load_ems, nan= np.nanmean(Load_ems)) #(np.max(Load_ems)-np.min(Load_ems))*np.random.rand(1)+np.min(Load_ems)
Load_ems = np.concatenate((Load_ems, Load_ems), axis=1)
#######################################
### STEP 1: Setup the network
#######################################
network = Network_3ph() 
network.setup_network_eulv_reduced()
# set bus voltage and capacity limits
network.set_pf_limits(0.95*network.Vslack_ph, 1.05*network.Vslack_ph,
                      2000*1e3/network.Vslack_ph)
N_buses = network.N_buses
N_phases = network.N_phases

#buses that contain loads
load_buses = np.where(np.abs(network.bus_df['Pa'])+np.abs(network.bus_df['Pb'])+np.abs(network.bus_df['Pc'])>0)[0]
load_phases = []
N_load_bus_phases=0
for load_bus_idx in range(len(load_buses)):
    phase_list = []
    if np.abs(network.bus_df.iloc[load_buses[load_bus_idx]]['Pa']) > 0:
          phase_list.append(0)
    if np.abs(network.bus_df.iloc[load_buses[load_bus_idx]]['Pb']) > 0:
          phase_list.append(1)
    if np.abs(network.bus_df.iloc[load_buses[load_bus_idx]]['Pc']) > 0:
          phase_list.append(2)
    load_phases.append(np.array(phase_list))  
    N_load_bus_phases += len(phase_list)
N_loads = load_buses.size
Load_ems = Load_ems[:,:N_loads] #(365*48, 55)

#55 Homes
assets = []
for i in range(N_loads):
    Pnet = np.zeros(T)
    Qnet = np.zeros(T)
    load_i = AS.NondispatchableAsset(Pnet, Qnet, load_buses[i], dt, T, dt_ems, T_ems, phases=load_phases[i])
    load_i.Pnet_pred = load_i.Pnet
    load_i.Qnet_pred = load_i.Qnet
    assets.append(load_i)

loadw = np.concatenate((Load_ems[:90*T_ems, :],Load_ems[(183+90)*T_ems:, :]), axis=0) #(182*48,55) 
loads = Load_ems[90*T_ems: 183*T_ems, :]

meanw, means, stdw, stds = np.zeros((T_ems, N_loads)), np.zeros((T_ems, N_loads)), np.zeros((T_ems, N_loads)), np.zeros((T_ems, N_loads))
Sigmaw, Sigmas = [], []

for t in range(T_ems):
    loadwt = loadw[t::48] #(182,55)    
    meanw[t, :] =np.mean(loadwt, axis=0) #(55,)
    stdw[t, :] = np.std(loadwt, axis=0) #(55,) #(55,55)
    Sigmaw.append(np.cov(loadwt, rowvar=0))

    loadst = loads[t::48]
    means[t, :] =np.mean(loadst, axis=0)
    stds[t, :]=np.std(loadst, axis=0)
    Sigmas.append(np.cov(loadst, rowvar=0))

Cap = 18*1e3 #*dt_ems #18MVA
epsilon = [0.95, 0.9, 0.85]
fair = False

for eps in epsilon:
    print('epsilon:', eps)
    for season in ['winter', 'summer']:
        print('   season:', season)
        if season == 'summer':
            mean, std, Sigma = means, stds, Sigmas
        else:
            mean, std, Sigma = meanw, stdw, Sigmaw

        for a,asset in enumerate(assets):
            asset.Pnet = mean[:,a]
            asset.Qnet = mean[:,a]*0.05
            asset.Pnet_ems = mean[:,a]
            asset.Qnet_ems = mean[:,a]*0.05
  
        C_max, C_min = np.zeros((T_ems, N_loads)), np.zeros((T_ems, N_loads))
        C_max_det, C_min_det = np.zeros((T_ems, N_loads)), np.zeros((T_ems, N_loads))
        for t in range(T_ems):
            print('     t=',t)
            ####get linear parameters:
            A_Pslack, b_Pslack, A_vlim, b_vlim, v_abs_min_vec, v_abs_max_vec = network.get_linear_parameters(assets, t)
            #linear params shapes: (55,) () (378, 55) (378,) (378, 1) (378, 1)  378=126*3
            A_Pslack = np.expand_dims(A_Pslack, axis=0)  #(1,55)    
            A = np.concatenate((A_Pslack, -A_Pslack, A_vlim, -A_vlim), axis=0) #(1,55) + (378,55) => (758,55)
            #A = A/1e3 #linear parameters are in W (and not kW)
            B = np.concatenate( ([[Cap-(b_Pslack/1e3)]], [[(b_Pslack/1e3)-Cap]], v_abs_max_vec-np.expand_dims(b_vlim, axis=1), np.expand_dims(b_vlim, axis=1)-v_abs_min_vec), axis=0)
            # (1,1) (1,1) (378,1) (378,1) => (758,1)
            
            sig = np.dot(np.dot(A,Sigma[t]),A.T)
            #print(sig)
            sigm = np.expand_dims(abs(sig.diagonal()), axis=1)
            #print(sigm)

            ####Run stochastic optimisation
            
            p1 = pic.Problem()
            
            if not fair:
                Cmax = pic.RealVariable("Cmax", N_loads) 
                p1.add_constraint(A*Cmax <= B-np.expand_dims(np.dot(A,mean[t,:]), axis=1) -  norm.ppf(1-eps)*sigm) #abs(np.expand_dims(np.dot(A,std[t]), axis=1)))
                p1.set_objective('max', A_Pslack*Cmax) #sum(Cmax)
            else:
                Cmax = pic.RealVariable("Cmax", N_loads)
                Cmaxtot = pic.RealVariable('Cmaxtot')
                p1.add_constraint(Cmax >= Cmaxtot)
                p1.add_constraint(A*Cmax <= B-np.expand_dims(np.dot(A,mean[t,:]), axis=1) -  norm.ppf(1-eps)*sigm) #abs(np.expand_dims(np.dot(A,std[t]), axis=1)))
                p1.set_objective('max', Cmaxtot)

            p1.solve(solver='gurobi', primals=None)

            if p1.status != 'optimal':
                print('p1 not optimal for t=',t)
            C_max[t, :] = np.array(Cmax.value)[:, 0]
            pickle.dump((C_max), open( f"Cmax1_{season}_{eps}.p", "wb" ) )
            
            p2 = pic.Problem()

            if not fair:
                Cmin = pic.RealVariable("Cmin", N_loads) 
                p2.add_constraint(A*Cmin <= B-np.expand_dims(np.dot(A,mean[t,:]), axis=1) - norm.ppf(1-eps)*sigm) #abs(np.expand_dims(np.dot(A,std[t]), axis=1)))
                p2.set_objective('min', A_Pslack*Cmin) #sum(Cmin)
            else:
                Cmin = pic.RealVariable("Cmin", N_loads)
                Cmintot = pic.RealVariable('Cmintot') 
                p2.add_constraint(Cmin <= Cmintot)
                p2.add_constraint(A*Cmin <= B-np.expand_dims(np.dot(A,mean[t,:]), axis=1) - norm.ppf(1-eps)*sigm) #abs(np.expand_dims(np.dot(A,std[t]), axis=1)))
                p2.set_objective('min', Cmintot)

            p2.solve(solver='gurobi', primals=None)
            if p2.status != 'optimal':
                print('p2 not optimal for t=',t)
            C_min[t, :] = np.array(Cmin.value)[:, 0]
            pickle.dump((C_min), open( f"Cmin1_{season}_{eps}.p", "wb" ) )
            

            """
            ####Run deterministic optimisation
            p1 = pic.Problem()
            Cmax = pic.RealVariable("Cmax", N_loads) 
            #print(A*Cmax.shape)
            p1.add_constraint(Cmax>=0)
            p1.add_constraint(A*Cmax <= B-np.expand_dims(np.dot(A,mean[t,:]), axis=1) ) 
            p1.set_objective('max', sum(Cmax))
            p1.solve(solver='mosek', primals=None)
            if p1.status != 'optimal':
                print('p1 not optimal for t=',t)
            C_max_det[t, :] = np.array(Cmax.value)[:, 0]
            #pickle.dump((C_max), open( f"Cmax_det_{season}_{eps}.p", "wb" ) )

            p2 = pic.Problem()
            Cmin = pic.RealVariable("Cmin", N_loads) 
            p2.add_constraint(A*Cmin <= B-np.expand_dims(np.dot(A,mean[t,:]), axis=1))
            p2.set_objective('min', sum(Cmin))
            p2.solve(solver='mosek', primals=None)
            if p2.status != 'optimal':
                print('p2 not optimal for t=',t)
            C_min_det[t, :] = np.array(Cmin.value)[:, 0]
            #pickle.dump((C_min_det), open( f"Cmin_det_{season}_{eps}.p", "wb" ) )
            """
