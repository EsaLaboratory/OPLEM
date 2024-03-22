#import modules
import os
from os.path import normpath, join
import copy
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

from oplem.Network_3ph_pf import Network_3ph
import oplem.Assets as AS
import oplem.Participant as Participant
from oplem.Market import P2P_market, ToU_market, Central_market


np.random.seed(1000)
#######################################
### RUN OPTIONS
#######################################
dt_raw = 1/60
T_raw = int(24/dt_raw) #Number of data time intervals
dt = 60/60 #5 minute time intervals
T = int(24/dt) #Number of intervals
dt_ems = 60/60
T_ems = int(24/dt_ems)

path_string = normpath('Results\\P2P\\')
if not os.path.isdir(path_string):
    os.makedirs(path_string)

#######################################
### STEP 0: Load Data
#######################################
#### 0)wholesale data
prices_path = os.path.join('Data', "half-hourly-wholesale-prices-MWh-29-06-2021.csv")
prices_wsm = pd.read_csv(prices_path, delimiter='\t')
prices_wsm = np.array(prices_wsm)  #prices £/MWh
dt_wsm = 24/len(prices_wsm)
T_wsm = len(prices_wsm)
prices_wsm_ems = np.zeros((T_ems,2)) # col0 for day ahead, col1 for intraday
if dt_ems <= dt_wsm:
    for t in range(T_wsm):
        prices_wsm_ems[t*int(dt_wsm/dt_ems) : (t+1)*int(dt_wsm/dt_ems),:] = prices_wsm[t,:]/1e3
else:
    for t in range(T_ems):
        prices_wsm_ems[t,:] = np.mean(prices_wsm[t*int(dt_ems/dt_wsm) : (t+1)*int(dt_ems/dt_wsm),:], axis=0)
        prices_wsm_ems[t,:] =prices_wsm_ems[t,:]/1e3

### 1) Load Data
Loads_data_path = os.path.join("Data", "Loads_1min.csv")    
Loads_raw = pd.read_csv(Loads_data_path, index_col=0).values
N_loads_raw = Loads_raw.shape[1]
Loads = Loads_raw.transpose().reshape(-1,int(dt/dt_raw)).mean(1).reshape(N_loads_raw,-1).transpose()
Load_ems = Loads.transpose().reshape(-1,int(dt_ems/dt)).mean(1).reshape(N_loads_raw,-1).transpose()

#### 2) PV Data
PV_data_path = os.path.join("Data", "PV_profiles_norm_1min.txt") 
PVpu_raw = pd.read_csv(PV_data_path, sep='\t').values 
PVpu_raw = PVpu_raw[:,:55]
PVpu_raw = np.vstack((PVpu_raw,np.zeros(55)))
N_pv_raw = PVpu_raw.shape[1]
PVpu = PVpu_raw.transpose().reshape(-1,int(dt/dt_raw)).mean(1).reshape(N_pv_raw,-1).transpose()
#PVpu_ems = PVpu.transpose().reshape(-1,int(dt_ems/dt)).mean(1).reshape(N_pv_raw,-1).transpose()
#PVpu = PVpu_ems[:,0]
PVpu = PVpu[:,0]#taking 1 profile, as the participants are close to each other and are subject to same weather conditions

### 3) Temperature
##winter
Ta_w = np.array([280.65, 280.03,280.16, 279.77, 279.9, 279.65, 279.03, \
                 279.03, 278.52, 278.42, 277.78, 278.4, 278.54, 278.66, 278.54, \
                 278.4, 278.65, 279.27, 280.44, 281.52, 282, 282.89, 283.39, 283.02]) #column C 29/11/2017
Ta_w = np.subtract(Ta_w, 273.15)
##summer
Ta_s = np.array([300.538, 300.538, 300.225, 300.225, 297.63, 297.61, 302.63, 302.61, \
                304.07, 305.07, 306.15, 307.69, 308.86, 309.31, 308.39, 307.4, 303.15,\
                302.15, 300.584, 300.584, 300.483, 300.483, 300.483, 300.31]) # 14/07/2017, column AJ
Ta_s = np.subtract(Ta_s, 273.15)

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

#######################################
### STEP 2: setup parameters
######################################
#### 1) Home PV parameters
N_pv = int(np.ceil(0.6*N_loads)) #nbr of homes with PV 
pv_locs = np.random.choice(N_loads, N_pv, replace=False)# [0,3,4,6] #
Ppv_home_nom = 8#power rating of the PV generation 

### 2) Home battery parameters
N_es = int(np.ceil(0.3*N_loads)) #[1,3,5,6]
es_locs = np.random.choice(N_loads, N_es, replace=False)
Pbatt_max = 4 
Ebatt_max = 8 
c1_batt_deg = 0.005 #Battery degradation costs

### 3) building parameters
N_hp = int(np.ceil(0.3*N_loads)) 
hp_locs = np.random.choice(N_loads, N_hp, replace=False) #[2,4,5,6]
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

#######################################
### STEP 3: setup the assets 
######################################
## We have one participant per bus, i.e., a participant is a home owner
assets_per_participant = [ [] for _ in range(N_loads) ]

#55 Homes
Loads_actual = Loads[:,:N_loads]
for i in range(N_loads):
    Pnet = Loads_actual[:,i]
    Qnet = Loads_actual[:,i]*0.05
    load_i = AS.NondispatchableAsset(Pnet, Qnet, load_buses[i], dt, T, dt_ems, T_ems, phases=load_phases[i])
    load_i.Pnet_pred = load_i.Pnet
    load_i.Qnet_pred = load_i.Qnet
    assets_per_participant[i].append(load_i)
    
    if i in pv_locs:
        Pnet_pv_i = -PVpu*Ppv_home_nom 
        pv_i = AS.CurtailableAsset(Pnet_pv_i, np.zeros(T_ems), load_buses[i], dt, T, dt_ems, T_ems, phases=load_phases[i], curt=True)
        pv_i.Pnet_pred = pv_i.Pnet
        assets_per_participant[i].append(pv_i)
    
    if i in es_locs:
        Emax_i = Ebatt_max*np.ones(T_ems)
        Emin_i = np.zeros(T_ems)
        ET_i = Ebatt_max*0.5
        E0_i = Ebatt_max*0.5        
        Pmax_i = Pbatt_max*np.ones(T_ems)
        Pmin_i = -Pbatt_max*np.ones(T_ems)
        batt_i = AS.StorageAsset(Emax_i, Emin_i, Pmax_i, Pmin_i, E0_i, ET_i, load_buses[i], dt, T, dt_ems, T_ems, phases=load_phases[i], c_deg_lin = c1_batt_deg)
        assets_per_participant[i].append(batt_i)
    
    if i in hp_locs:
        bldg_i = AS.BuildingAsset(Tmax*np.ones(T_ems), Tmin*np.ones(T_ems), heatmax, coolmax, T0, C, R, CoP_heating, CoP_cooling, Ta_w, load_buses[i], dt, T, dt_ems, T_ems)
        assets_per_participant[i].append(bldg_i)

##############################################
### STEP 4: Linking assets to participant object
############################################
participants = []
for i in range(N_loads):
    participant = Participant(i+1, assets_per_participant[i])
    participants.append(participant)
    
################
### STEP 5: setup the P2P Market
################
###  1)Number of separate P2P energy trading networks
N_p2p_groups = 2
price_inc=0.1
energy_trade = 1*dt_ems
t_ahead_0 = 0
N_p2p_ahead_max = 500

### 2)setup prices
price_imp = prices_wsm_ems[:,0]
price_imp = np.expand_dims(price_imp, axis=1)
price_imp = np.repeat(price_imp, load_buses.size, axis=1)

FiT = 0.06*np.ones(T_ems) 
FiT = np.expand_dims(FiT, axis=1)
FiT = np.repeat(FiT, load_buses.size, axis=1)

### 3)Setup P2P Agents
market_clearings = pd.DataFrame(columns=['time', 'seller', 'buyer', 'energy', 'price'])
all_schedules = [[] for _ in range(len(participants))]
start_time = time.time()
for n_group in range(N_p2p_groups):
    group_participants = [participant for participant in participants if participant.p_id%N_p2p_groups==n_group]
    P2P = P2P_market(group_participants, dt_ems, T_ems, price_imp = price_imp , price_exp=FiT, network=network)
    market_clearing, schedules = P2P.P2P_negotiation(energy_trade, price_inc, N_p2p_ahead_max)
    market_clearings = pd.concat([market_clearings, market_clearing], axis=0, ignore_index=True)
    for p, participant in enumerate(group_participants):
        all_schedules[participant.p_id-1] = schedules[p]
elapsed = time.time() - start_time
print('P2P time', str(elapsed) , ' sec'  )

pickle.dump((market_clearings), open( "Results\\P2P\\mc.p", "wb" ) )
pickle.dump((all_schedules), open( "Results\\P2P\\schedules.p", "wb" ) )

###3) Store assets states
Elist, Tinlist = [], []
for p, par in enumerate(participants):
    for a, asset in enumerate(par.assets):
        if isinstance(asset, AS.StorageAsset): #assets[par][a] == 'es':
            Elist.append(asset.E_ems)            
        elif isinstance(asset, AS.BuildingAsset): #assets[par][a] == 'hp':
            Tinlist.append(asset.Tin_ems)
pickle.dump((Elist), open( "Results\\P2P\\Ebatt.p", "wb" ) )
pickle.dump((Tinlist), open( "Results\\P2P\\Tin.p", "wb" ) )

###4) simulate network
network_pf = P2P.simulate_network_3ph()
voltage = np.zeros((T_ems, 2))

for t, network_sim in enumerate(network_pf):
    buses_Vpu = np.abs(network_sim.v_net_res)/network_sim.Vslack_ph  
    voltage[t, 0] = np.min(buses_Vpu)
    voltage[t, 1] = np.max(buses_Vpu)
    
    if np.max(buses_Vpu)>1.05:
        print('for t ={}, Vmax={} at index: {}'.format(t, np.max(buses_Vpu), np.argmax(buses_Vpu)))
    elif np.min(buses_Vpu)<0.95:
        print('for t ={}, Vmin={} at index: {}'.format(t, np.min(buses_Vpu), np.argmin(buses_Vpu)))

pickle.dump((voltage), open( "Results\\P2P\\voltage.p", "wb" ) )
