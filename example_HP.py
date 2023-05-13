#import modules
import os
import copy
import pandas as pd
import numpy as np
import picos as pic
import pickle
import time

import System.Assets as AS
import System.Participant as Participant
from System.Market import ToU_market

np.random.seed(1000)
dt_raw = 1/60
T_raw = int(24/dt_raw) #Number of data time intervals
dt = 60/60 #5 minute time intervals
T = int(24/dt) #Number of intervals
dt_ems = 60/60
T_ems = int(24/dt_ems)

#######################################
### STEP 0: Load Data
#######################################
### 2) Load Data
Loads_data_path = os.path.join("Data", "Loads_1min.csv")    
Loads_raw = pd.read_csv(Loads_data_path, index_col=0, parse_dates=True).values
N_loads_raw = Loads_raw.shape[1]
Loads = Loads_raw.transpose().reshape(-1,int(dt/dt_raw)).mean(1).reshape(N_loads_raw,-1).transpose()
Load_ems = Loads.transpose().reshape(-1,int(dt_ems/dt)).mean(1).reshape(N_loads_raw,-1).transpose()

#### 3) PV Data
PV_data_path = os.path.join("Data", "PV_profiles_norm_1min.txt") 
PVpu_raw = pd.read_csv(PV_data_path, sep='\t').values 
PVpu_raw = PVpu_raw[:,:55]
PVpu_raw = np.vstack((PVpu_raw,np.zeros(55)))
N_pv_raw = PVpu_raw.shape[1]
PVpu = PVpu_raw.transpose().reshape(-1,int(dt/dt_raw)).mean(1).reshape(N_pv_raw,-1).transpose()
PVpu = PVpu[:,0]

### 3) building parameters
Tmax = 18 # degree celsius
Tmin = 16 # degree celsius
T0 = 17 # degree centigrade
#Parameters from 'Aggregate Flexibility of Thermostatically Controlled Loads'
heatmax = 5.6 #kW Max heat supplied
coolmax = 5.6 #kW Max cooling
CoP_heating = 2.5# coefficient of performance - heating
CoP_cooling = 2.5# coefficient of performance - cooling
C = 2 # kWh/ degree celsius
R = 2 #degree celsius/kW
Ta_w = np.array([280.65, 280.03,280.16, 279.77, 279.9, 279.65, 279.03, \
                 279.03, 278.52, 278.42, 277.78, 278.4, 278.54, 278.66, 278.54, \
                 278.4, 278.65, 279.27, 280.44, 281.52, 282, 282.89, 283.39, 283.02]) #column C 29/11/2017
Ta_w = np.subtract(Ta_w, 273.15)

Ta_s = np.array([300.538, 300.538, 300.225, 300.225, 297.63, 297.61, 302.63, 302.61, \
                304.07, 305.07, 306.15, 307.69, 308.86, 309.31, 308.39, 307.4, 303.15,\
                302.15, 300.584, 300.584, 300.483, 300.483, 300.483, 300.31]) # 14/07/2017, column AJ
Ta_s = np.subtract(Ta_s, 273.15)

#######################################
### STEP 3: setup the assets 
######################################
assets = []

Pnet = Loads[:,6]
Qnet = Loads[:,6]*0.05
load_i = AS.NondispatchableAsset(Pnet, Qnet, 1, dt, T, dt_ems, T_ems)
load_i.Pnet_pred = load_i.Pnet
load_i.Qnet_pred = load_i.Qnet
assets.append(load_i)

Pnet_pv_i = -PVpu*8 
pv_i = AS.NondispatchableAsset(Pnet_pv_i, np.zeros(T_ems), 1, dt, T, dt_ems, T_ems)
pv_i.Pnet_pred = pv_i.Pnet
assets.append(pv_i)

batt_i = AS.StorageAsset(8*np.ones(T_ems), np.zeros(T_ems),4*np.ones(T_ems), 4*np.ones(T_ems), 4, 4, 1, dt, T, dt_ems, T_ems, c_deg_lin = 0.05)
#assets.append(batt_i)

Tmax_bldg_i = Tmax*np.ones(T_ems)
Tmin_bldg_i = Tmin*np.ones(T_ems)
Hmax_bldg_i = heatmax
Cmax_bldg_i = coolmax
C_i = C
R_i = R
CoP_heating_i = CoP_heating
CoP_cooling_i = CoP_cooling
Ta_i = Ta_w # 22*np.ones(T_ems)
T0_i =T0
bldg_i = AS.BuildingAsset(Tmax_bldg_i, Tmin_bldg_i, Hmax_bldg_i, Cmax_bldg_i, T0_i, C_i, R_i, CoP_heating_i, CoP_cooling_i, Ta_i, 1, dt, T, dt_ems, T_ems)
assets.append(bldg_i)

#######################################
### STEP 3: Prices
######################################
## 1)setup prices
poff, pon = (33/2)/240, 33/240   ##prices from E7 tariff
TOUP1 = np.append(poff*np.ones(int(7/dt_ems)), pon*np.ones(int(17/dt_ems)))
TOUP1 = np.expand_dims(TOUP1, axis=1)
TOUP = np.repeat(TOUP1, 2, axis=1)

FiT1 = (pon/3)*np.ones(T_ems)
##### FiT <0 to assess curtailment
#FiT = np.concatenate((poff/2*np.ones(13/dt_ems), -poff/2*np.ones(6/dt_ems), poff/2*np.ones(5/dt_ems)), axis=0)
FiT1 = np.expand_dims(FiT1, axis=1)
FiT = np.repeat(FiT1, 2, axis=1)

P_demand = assets[0].Pnet_ems +assets[1].Pnet_ems
#print('Pdemand:', P_demand)

Php, P_imp, P_exp, Thp = np.zeros((T_ems,3)), np.zeros((T_ems,3)), np.zeros((T_ems,3)), np.zeros((T_ems,3))
hp = bldg_i
es=batt_i
##################################################################################################################
#################       First Method using polytope  A*P<=b for both battery and HP        ########################
###################################################################################################################
prob = pic.Problem()
Pimp = pic.RealVariable('Pimp', T_ems)
Pexp = pic.RealVariable('Pexp', T_ems)
P = pic.RealVariable('P',2*T_ems)
A, b = hp.polytope()
prob.add_constraint(A*P <= b)
### decomment if to consider battery
"""
Pb = pic.RealVariable('Pb',2*T_ems)
Ab, bb =es.polytope(0)
prob.add_constraint(Ab*Pb <= bb)
"""
prob.add_constraint( Pimp >= 0)
prob.add_constraint( Pexp >= 0)
####### 2 following lines should be replaced by the 2 that come after if considering battery
prob.add_constraint(Pimp - Pexp == P_demand + P[: T_ems] + P[T_ems:]  )
prob.set_objective('min', dt_ems*TOUP1.T*Pimp -dt_ems*FiT1.T*Pexp  )
"""
prob.add_constraint(Pimp - Pexp == P_demand + P[: T_ems] + P[T_ems:] + Pb[: T_ems] + Pb[T_ems:] )
prob.set_objective('min', dt_ems*TOUP1.T*Pimp -dt_ems*FiT1.T*Pexp +dt_ems*es.c_deg_lin*sum(Pb[t] - Pb[t+T_ems] for t in range(T_ems) ) )
"""
prob.solve(solver='gurobi')  #)
hp.update_ems(P.value[:T_ems]-P.value[T_ems:], enforce_const=False)
Thp[:, 0] = hp.Tin_ems
for t in range(T_ems):
    Php[t,0] = P[t] + P[T_ems+t]
    P_imp[t,0] = Pimp.value[t]
    P_exp[t,0] = Pexp.value[t]
##################################################################################################################
#################             Second Method using conventional equations for hp           ########################
###################################################################################################################
prob = pic.Problem()
Pimp = pic.RealVariable('Pimp', T_ems)
Pexp = pic.RealVariable('Pexp', T_ems)
P_cool = pic.RealVariable('P_cool',T_ems)
P_heat = pic.RealVariable('P_heat',T_ems)
T_in = pic.RealVariable('T_in',T_ems)
"""
Pb = pic.RealVariable('Pb',2*T_ems)
Ab, bb =es.polytope()
prob.add_constraint(Ab*Pb <= bb)
"""
prob.add_constraint(P_cool >= 0 )
prob.add_constraint(P_cool <= hp.Cmax )
prob.add_constraint(P_heat >= 0 )
prob.add_constraint(P_heat <= hp.Hmax )
#prob.add_constraint(T_in[0] == hp.T0)
prob.add_constraint(T_in[0] == hp.alpha*hp.T0 + dt_ems*hp.beta*(hp.CoP_heating*P_heat[0]-hp.CoP_cooling*P_cool[0]) + hp.gamma*hp.Ta_ems[0])
prob.add_list_of_constraints([T_in[t] == hp.alpha*T_in[t-1] + dt_ems*hp.beta*(hp.CoP_heating*P_heat[t]-hp.CoP_cooling*P_cool[t]) + hp.gamma*hp.Ta_ems[t] for t in range(1,T_ems)])
prob.add_constraint(T_in >= hp.Tmin )
prob.add_constraint(T_in <= hp.Tmax )
prob.add_constraint( Pimp >= 0)
prob.add_constraint( Pexp >= 0)
prob.add_constraint(Pimp - Pexp == P_demand + P_cool + P_heat )
prob.set_objective('min', dt_ems*TOUP1.T*Pimp -dt_ems*FiT1.T*Pexp )
"""
prob.add_constraint(Pimp - Pexp == P_demand + P_cool + P_heat + Pb[: T_ems] + Pb[T_ems:])
prob.set_objective('min', dt_ems*TOUP1.T*Pimp -dt_ems*FiT1.T*Pexp +dt_ems*es.c_deg_lin*sum(Pb[t] - Pb[t+T_ems] for t in range(T_ems) ) )
"""

prob.solve(solver='gurobi')  #)

hp.update_ems(P_heat.value-P_cool.value, enforce_const=False)
Thp[:, 1] = hp.Tin_ems
for t in range(T_ems):
    Php[t,1] = P_heat[t] + P_cool[t]
    P_imp[t,1] = Pimp.value[t]
    P_exp[t,1] = Pexp.value[t]
##################################################################################################################
#################        Third Method using EMS     (polytope for both)                   ########################
##################################################################################################################
par = Participant(1, assets)
ToU = ToU_market([par], dt_ems, T_ems, TOUP, price_exp=FiT, t_ahead_0=0)
market_clearing_outcome, schedules = ToU.market_clearing()
Thp[:,2] = hp.Tin_ems
#Php[:,2] = schedules[-1]

#######################################################
#print('Php', Php)
#print('P_imp', P_imp)
#print('P_exp', P_exp)
print('Thp', Thp)
