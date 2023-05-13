#import modules
import os
import copy
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt


from System.Network_3ph_pf import Network_3ph
import System.Assets as AS
import System.Participant as Participant
from System.Market import ToU_market

np.random.seed(1000)
#######################################
### RUN OPTIONS
#######################################
dt_raw = 1/60
T_raw = int(24/dt_raw) #Number of data time intervals
dt = 5/60 #5 minute time intervals
T = int(24/dt) #Number of intervals
dt_ems = 60/60
T_ems = int(24/dt_ems)

#######################################
### STEP 0: Load Data
#######################################
### 1) Load Data
Loads_data_path = os.path.join("Data", "Loads_1min.csv")    
Loads_raw = pd.read_csv(Loads_data_path, index_col=0, parse_dates=True).values
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
N_loads = 7 #load_buses.size

#######################################
### STEP 2: setup the assets 
######################################
#### 1) Home PV parameters
pv_locs = [0,3,4,6] #
Ppv_home_nom = 8 #power rating of the PV generation 

### 2) Home battery parameters
es_locs = [1,3,5,6]
Pbatt_max = 4 #
Ebatt_max = 8 #
c1_batt_deg = 0.005 #Battery degradation costs

### 3) building parameters
hvac_locs = [2,4,5,6]
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

## We have one participant per bus, i.e., a participant is a home owner
assets_per_participant = [ [] for _ in range(N_loads) ]


for i in range(N_loads):
    Pnet = Loads[:,i]
    Qnet = Loads[:,i]*0.05
    load_i = AS.NondispatchableAsset(Pnet, Qnet, load_buses[i], dt, T, dt_ems, T_ems, phases=load_phases[i])
    load_i.Pnet_pred = load_i.Pnet
    load_i.Qnet_pred = load_i.Qnet
    assets_per_participant[i].append(load_i)
    
    if i in pv_locs:
        Pnet_pv_i = -PVpu*Ppv_home_nom 
        pv_i = AS.NondispatchableAsset(Pnet_pv_i, np.zeros(T_ems), load_buses[i], dt, T, dt_ems, T_ems, phases=load_phases[i], curt=True)
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

    if i in hvac_locs:
        Tmax_bldg_i = Tmax*np.ones(T_ems)
        Tmin_bldg_i = Tmin*np.ones(T_ems)
        Hmax_bldg_i = heatmax
        Cmax_bldg_i = coolmax
        T0_i = T0
        C_i = C
        R_i = R
        CoP_heating_i = CoP_heating
        CoP_cooling_i = CoP_cooling
        Ta_i = Ta_w
        bus_id_bldg_i = load_buses[i]
        bldg_i = AS.BuildingAsset(Tmax_bldg_i, Tmin_bldg_i, Hmax_bldg_i, Cmax_bldg_i, T0_i, C_i, R_i, CoP_heating_i, CoP_cooling_i, Ta_i, bus_id_bldg_i, dt, T, dt_ems, T_ems)
        assets_per_participant[i].append(bldg_i)

##############################################
### STEP 4: Linking assets to participant object
############################################
participants = []
for i in range(N_loads):
    #we start id at 1, because 0 is for the slack bus/DSO/upstream
    participant = Participant(i+1, assets_per_participant[i])
    participants.append(participant)

##############################
### STEP 5: setup the Market
############################
## 1)setup prices
poff, pon = (33/2)/240, 33/240   ##prices from E7 tariff
TOUP = np.append(poff*np.ones(int(7/dt_ems)), pon*np.ones(int(17/dt_ems)))
TOUP = np.expand_dims(TOUP, axis=1)
TOUP = np.repeat(TOUP, network.N_buses, axis=1)

FiT = (pon/3)*np.ones(T_ems)
##### FiT <0 to assess curtailment
#FiT = np.concatenate((poff/2*np.ones(13/dt_ems), -poff/2*np.ones(6/dt_ems), poff/2*np.ones(5/dt_ems)), axis=0)
FiT = np.expand_dims(FiT, axis=1)
FiT = np.repeat(FiT, network.N_buses, axis=1)

### 2)initialise market
ToU = ToU_market(participants, dt_ems, T_ems, TOUP, price_exp=FiT, t_ahead_0=0)

###3) Run market clearing
start_time = time.time() 
market_clearing_outcome, schedules = ToU.market_clearing()
elapsed = time.time() - start_time
print('ToU time=' + str(elapsed) + ' sec'  )

pickle.dump((market_clearing_outcome), open( "market_clearing_TOU.p", "wb" ) )
pickle.dump((schedules), open( "schedules_TOU.p", "wb" ) )
##############################
### STEP 6: Show results
############################
#lables
assets = [['nd', 'pv'], ['nd', 'es'], ['nd', 'hp'], ['nd', 'pv', 'es'], ['nd', 'pv', 'hp'], ['nd', 'es', 'hp'], 
          ['nd', 'pv', 'es', 'hp']]
#colors for labels
colors={'nd': 'black',
        'pv': 'green',
        'hp': 'orange',
        'es': 'blue'}

schedules = pd.read_pickle('schedules_TOU.p')
market_clearing = pd.read_pickle('market_clearing_TOU.p')
    
####### schedules per participant
fig, axs = plt.subplots(4,2, figsize=(15,9))
for par in range(len(schedules)):
    schedule = schedules[par]
    for a in range(len(schedule)):
        axs[int(par/2)][par%2].plot(schedule[a], label=assets[par][a], color=colors[assets[par][a]])
        axs[int(par/2)][par%2].legend()
    axs[int(par/2)][0].set_ylabel('kW')
    axs[-1][par%2].set_xlabel('t')
plt.savefig('TOUP_schedules.jpg')

##### total import/export
imp_exp = np.zeros((len(schedules), len(schedules[0][0])))
for par in range(len(schedules)):
    for t in range(len(schedules[0][0])):
        selection =market_clearing.loc[(market_clearing['time']==t) & (market_clearing['seller']==par+1), 'energy']
        imp_exp[par, t] -= sum(selection)
        selection2 =market_clearing.loc[(market_clearing['time']==t) & (market_clearing['buyer']==par+1), 'energy']
        imp_exp[par, t] += sum(selection2)

fig, axs = plt.subplots(4,2, figsize=(15,9))
for par in range(len(schedules)):
    axs[int(par/2)][par%2].plot(imp_exp[par,:], label='imp/exp')
    axs[int(par/2)][par%2].legend()
    axs[int(par/2)][0].set_ylabel('kW')
    axs[-1][par%2].set_xlabel('t')
plt.savefig('TOUP_trades.jpg')

#### outputs battery SoC and indoor temperature
fig, axs = plt.subplots(1,2, figsize=(15,9))
for p, par in enumerate(participants):
    for a, asset in enumerate(par.assets):
        if isinstance(asset, AS.StorageAsset): #assets[par][a] == 'es':
            axs[0].plot(asset.E_ems, label='par {}'.format(p+1))
            axs[0].hlines(y=8, xmin=0, xmax=24, linewidth=2, color='black')
            axs[0].hlines(y=0, xmin=0, xmax=24, linewidth=2, color='black')
            axs[0].set_xlabel('t')
            axs[0].set_ylabel('E')
            axs[0].legend()
            
        elif isinstance(asset, AS.BuildingAsset): #assets[par][a] == 'hp':
            axs[1].plot(asset.Tin_ems, label='par {}'.format(p+1))
            axs[1].set_xlabel('t')
            axs[1].set_ylabel('T_indoor')
            axs[1].hlines(y=16, xmin=0, xmax=24, linewidth=2, color='black')
            axs[1].hlines(y=18, xmin=0, xmax=24, linewidth=2, color='black')
            axs[1].legend()  
plt.savefig('TOUP_states.jpg')      

