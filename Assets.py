#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    File name: EPG_Python_Energy_System_Framework_V1.py
    Author: Thomas Morstyn
    Date created: 12/10/2018
    Date last modified: 12/10/2018
    Python Version: 3.6.3
"""

#import modules
import copy
import pandas as pd
#import pandapower as pp
#import pandapower.networks as pn
import numpy as np
import picos as pic
#import matplotlib.pyplot as plt



__version__ = 0.1

#Base Asset Class
#An energy resource located at a particular bus in the network. 
class Asset: 
    def __init__(self, bus_id, dt, T):
        self.type = 'asset'
        self.bus_id = bus_id #id number of the bus in the network
        self.dt = dt #time interval duration
        self.T = T #number of time intervals

class Asset_3ph(Asset): 
    def __init__(self, bus_id, phases, dt, T):
        Asset.__init__(self, bus_id, dt, T)
        self.phases = np.array(phases) #Connection (wye or delta) depends on the bus
                                       #[0,1,2]  indicates 3 phase connection
                                       #Wye: [0,1] indicates an a,b connection
                                       #Delta [0] indicates a-b; [1], b-c; [2], c-a
                             
#Storage Asset Class (use for batteries, EVs etc.)
class StorageAsset(Asset):
    def __init__(self, Emax, Emin, Pmax, Pmin, E0, ET, bus_id, dt, T, dt_ems, T_ems, c_deg_lin=None):
        Asset.__init__(self, bus_id, dt, T)
        self.type = 'storage'
        self.Emax = Emax #maximum energy levels over the time series (kWh)
        self.Emin = Emin #minimum energy levels over the time series (kWh)
        self.Pmax = Pmax #maximum input powers over the time series (kW)
        self.Pmin = Pmin #minimum input powers over the time series (kW)
        self.E0 = E0  #initital energy level (kWh)
        self.ET = ET #required terminal energy level (kWh)
        self.dt_ems = dt_ems
        self.T_ems = T_ems
        self.c_deg_lin = c_deg_lin or 0
        self.E = E0*np.ones(T) #energy stored over the time series (kW)
        self.Pnet = np.zeros(T) #input powers over the time series (kW)
        self.Qnet = np.zeros(T) #reactive powers over the time series (kW)
        
    def update_control(self,Pnet):
        #update the storage system power and energy profile 
        self.Pnet = Pnet
        self.E[0] = self.E0
        for t in range(self.T-1):
            self.E[t+1] = self.E[t] + self.Pnet[t]*self.dt

    def EV_baseline(self, t_arr, T_avail, SOC_arr):
        """
        Compute the baseline consumption of an EV in the absence of flexibility
        Parameters
        ----------
        t_arr : int
            index of time slot correspnding to the plug-in of the EV
        T_avail: int
            number of time slots the EV remained plugged-in
        SOC_arr: float
            SOC of EV at arrival 
        returns
        --------
        P_ch: 1d array
            daily charging schedule of EV
        """ 

        nbr_ts = min(T_avail, np.ceil(self.Emax*(1-SOC_arr)/(self.Pmax*self.dt_ems)))

        P_ch = np.zeros(self.T_ems)
        te = int(min(self.T_ems, t_arr+nbr_ts))
        P_ch[int(t_arr): te] = self.Pmax
        #print('baseline P_ch=',P_ch)
        return P_ch

    def EV_toup_baseline(self, t_arr, T_avail, SOC_arr, toup):
        """
        Compute the baseline consumption of an EV in the absence of flexibility, response to TOUP signal
        Parameters
        ----------
        t_arr : int
            index of time slot correspnding to the plug-in of the EV
        T_avail: int
            number of time slots the EV remained plugged-in
        SOC_arr: float
            SOC of EV at arrival 
        returns
        --------
        P_ch: 1d array
            daily charging schedule of EV
        
        if T_avail+t_arr > self.T_ems:
            n = np.ceil((T_avail+t_arr)/self.T_ems)
            T_horizon = int(n*self.T_ems) 
            TOUP = np.tile(toup,(1,n))
        else: 
            T_horizon = self.T_ems
            TOUP = toup
        """ 
        n = np.ceil((T_avail+t_arr)/self.T_ems) + int(not((T_avail+t_arr)%self.T_ems))
        T_horizon = int(self.T_ems*n)
        TOUP = np.tile(toup,(int(n),))

        T_not_avail = np.array([value for value in range(T_horizon) if value not in range(t_arr, t_arr + T_avail + 1)])

        prob = pic.Problem()
        P_ch = pic.RealVariable('P_ch',T_horizon)
        P_soft = pic.RealVariable('P_soft')

        prob.add_list_of_constraints([P_ch[t]  >= self.Pmin for t in range(T_horizon)])
        prob.add_list_of_constraints([P_ch[t]  <= self.Pmax for t in range(T_horizon)])
        prob.add_list_of_constraints([P_ch[t]  == 0 for t  in T_not_avail])
        #prob.add_constraint(sum(p_ch[t]*self.dt_ems/self.Emax for t in range(T_horizon)) + SOC_arr == 1)
        prob.add_constraint(sum(P_ch[t]*self.dt_ems/self.Emax for t in range(T_horizon)) + SOC_arr >= 0.9)
        #prob.add_constraint(sum(Eff_EV*p_ch[t]*dt/Cmax for t in T_horizon) + SOC_arr + P_soft >= 0.9)

        #prob.set_objective('min', sum(P_ch[t]*TOUP[t] for t in range(T_horizon)) + P_soft*1e2)
        prob.set_objective('min', sum(P_ch[t]*TOUP[t] for t in range(T_horizon)))

        prob.solve(solver='mosek', primals=None) #, )
        if prob.status != 'optimal':
            print('non optimal solution for toup')
            P_ch_value = self.EV_baseline(t_arr, T_avail, SOC_arr)
            
        else: 
        #print('EV_toup_baseline: status {} and P_ch={}'.format(prob.status. P_ch.value))
            P_ch_value = [P_ch.value[i] for i in range(self.T_ems)]
        #print('baseline P_ch=', P_ch)
        return P_ch_value

    def EV_flexibility(self, t_arr, T_avail, SOC_arr, T_flex, flex_type='up'):
    
        n = np.ceil((T_avail+t_arr)/self.T_ems) + int(not((T_avail+t_arr)%self.T_ems))
        T_horizon = int(self.T_ems*n)

        """
        if T_avail+t_arr >= self.T_ems:
            n = np.ceil((T_avail+t_arr)/self.T_ems)
            T_horizon = int(n*self.T_ems) 
        else: 
            T_horizon = self.T_ems 
        """
        t_avail = np.arange(t_arr, T_avail+t_arr+1, dtype=int)
        #print('t_avail', t_avail)
        
        #print('T_horizon', T_horizon)
        T_not_avail = np.array([value for value in range(self.T_ems) if value not in t_avail])
        T_flex_avail =t_avail[np.isin(t_avail, T_flex)]
        #print('T_flex_avail', T_flex_avail)

        #                  T_flex_avail interset t_avail = t_avail or T_flex_avail intersect T_flex   
        if np.array_equal(T_flex_avail, t_avail) or T_flex_avail==[]: #not(np.array_equal(T_flex_avail, T_flex)):
            #print('EV not available during the flex period')
            return 0  
        
        else:
            #T = t_dep - t_arr ################pb t_dep day after!
            prob = pic.Problem()
            Flex = pic.RealVariable('flex', 1)
            p_ch = pic.RealVariable('P_ch', T_horizon)
            P_soft = pic.RealVariable('P_soft')

            prob.add_constraint(Flex >= 0)
            prob.add_list_of_constraints([p_ch[t]  >= self.Pmin for t in t_avail])
            prob.add_list_of_constraints([p_ch[t]  <= self.Pmax for t in t_avail])
            prob.add_list_of_constraints([p_ch[t]  == 0 for t  in T_not_avail])
            #prob.add_constraint(sum(Eff_EV*p_ch[t]*dt/Cmax for t in T_avail) + SOC_arr == 1)
            #prob.add_constraint(sum(p_ch[t]*self.dt_ems/self.Emax for t in t_avail) + SOC_arr +P_soft >= 0.9)
            prob.add_constraint(sum(p_ch[t]*self.dt_ems/self.Emax for t in t_avail) + SOC_arr >= 0.9)
            
            if flex_type=='up':
                prob.add_list_of_constraints([Flex <= self.Pnet[t] - p_ch[t] for t in T_flex_avail])
            else: 
                prob.add_list_of_constraints([Flex <= p_ch[t] - self.Pnet[t] for t in T_flex_avail])
            
            #prob.set_objective('max', Flex - P_soft*1e2)  
            prob.set_objective('max', Flex)

            prob.solve(solver='mosek', primals=None)
            p_ch_val = [p_ch.value[i] for i in range(T_horizon)]
            #print('new shcedule after optimisation:', p_ch_val)
            #print('EV_flexibility: status {} and Flex={}'.format(prob.status, Flex.value))    
            return Flex.value #if Flex >= min_flex else 0


#Nondispatchable Asset Class (use for inflexible loads, PV sources etc.)            
class NondispatchableAsset(Asset):
    def __init__(self, Pnet, Qnet, bus_id, dt, T, Pnet_pred = None, Qnet_pred = None):
        Asset.__init__(self, bus_id, dt, T)
        self.Pnet = Pnet #Uncontrolled real input powers over the time series (kW)
        self.Qnet = Qnet #Uncontrolled reactive input powers over the time series (kVar)
        self.Pnet_pred = Pnet_pred or Pnet #Predicted real input powers over the time series (kW)
        self.Qnet_pred = Qnet_pred or Qnet #Predicted reactive input powers over the time series (kVar)

#Storage Asset Class (use for batteries, EVs etc.)
class StorageAsset_3ph(Asset_3ph):
    def __init__(self, Emax, Emin, Pmax, Pmin, E0, ET, bus_id, phases, dt, T, dt_ems, T_ems, c_deg_lin = None):
        Asset_3ph.__init__(self, bus_id, phases, dt, T)
        self.asset_type = 'storage'
        self.Emax = Emax #maximum energy levels over the time series (kWh)
        self.Emin = Emin #minimum energy levels over the time series (kWh)
        self.Pmax = Pmax #maximum input powers over the time series (kW)
        self.Pmin = Pmin #minimum input powers over the time series (kW)
        self.E0 = E0  #initital energy level (kWh)
        self.ET = ET #required terminal energy level (kWh)
        self.E = E0*np.ones(T+1) #energy stored over the time series (kW)
        self.Pnet = np.zeros(T) #input StorageAsset_3phpowers over the time series (kW)
        self.Qnet = np.zeros(T) #reactive powers over the time series (kW)
        self.c_deg_lin = c_deg_lin or 0

    def update_control(self,Pnet):
        #update the storage system power and energy profile 
        self.Pnet = Pnet
        self.E[0] = self.E0
        for t in range(self.T -1):
            self.E[t+1] = self.E[t] + self.Pnet[t]*self.dt

    def update_control_t(self,Pnet_t,t):
        #update the storage system power and energy profile 
        self.Pnet[t] = Pnet_t
        self.E[0] = self.E0
        for t in range(self.T):
            self.E[t+1] = self.E[t] + self.Pnet[t]*self.dt


    def EV_baseline(self, t_arr, T_avail, SOC_arr):
        """
        Compute the baseline consumption of an EV in the absence of flexibility
        Parameters
        ----------
        t_arr : int
            index of time slot correspnding to the plug-in of the EV
        T_avail: int
            number of time slots the EV remained plugged-in
        SOC_arr: float
            SOC of EV at arrival 
        returns
        --------
        P_ch: 1d array
            daily charging schedule of EV
        """ 

        nbr_ts = min(T_avail, np.ceil(self.Emax*(1-SOC_arr)/(self.Pmax*self.dt_ems)))

        P_ch = np.zeros(self.T_ems)
        P_ch[int(t_arr): int(min(self.T_ems,t_arr+nbr_ts))] = self.Pmax

        return P_ch

    def EV_toup_baseline(self, t_arr, T_avail, SOC_arr, toup):
        """
        Compute the baseline consumption of an EV in the absence of flexibility, response to TOUP signal
        Parameters
        ----------
        t_arr : int
            index of time slot correspnding to the plug-in of the EV
        T_avail: int
            number of time slots the EV remained plugged-in
        SOC_arr: float
            SOC of EV at arrival 
        returns
        --------
        P_ch: 1d array
            daily charging schedule of EV
        """ 
        if T_avail+t_arr > self.T_ems:
            n = np.ceil((T_avail+t_arr)/self.T_ems)
            T_horizon = int(n*self.T_ems) 
            TOUP = np.tile(toup,(1,n))
        else: 
            T_horizon = self.T_ems
            TOUP = toup
        T_not_avail = np.array([value for value in range(T_horizon) if value not in range(t_arr, t_arr + T_avail + 1)])

        prob = pic.Problem()
        P_ch = pic.RealVariable('P_ch',T_horizon)
        P_soft = pic.RealVariable('P_soft')

        prob.add_list_of_constraints([p_ch[t]  >= self.Pmin for t in range(T_horizon)])
        prob.add_list_of_constraints([p_ch[t]  <= self.Pmax for t in range(T_horizon)])
        prob.add_list_of_constraints([p_ch[t]  == 0 for t  in T_not_avail])
        #prob.add_constraint(sum(p_ch[t]*self.dt_ems/self.Emax for t in range(T_horizon)) + SOC_arr + P_soft >= 0.9)
        prob.add_constraint(sum(p_ch[t]*self.dt_ems/self.Emax for t in range(T_horizon)) + SOC_arr >= 0.9)

        #prob.set_objective('min', sum(P_ch[t]*TOUP[t] for t in range(T_horizon)) + P_soft*1e2)
        prob.set_objective('min', sum(P_ch[t]*TOUP[t] for t in range(T_horizon)))

        prob.solve(solver='mosek', primals=None) #) 
        print('EV_toup_baseline: status {} and P_ch={}'.format(prob.status. P_ch.value))
        P_ch = [P_ch.value[i] for i in range(self.T_ems)]

        return P_th

    def EV_flexibility(self, t_arr, T_avail, SOC_arr, T_flex, flex_type='up'):
    
        if T_avail+t_arr > self.T_ems:
            n = np.ceil((T_avail+t_arr)/self.T_ems)
            T_horizon = int(n*self.T_ems) 
        else: 
            T_horizon = self.T_ems 

        t_avail = np.arange(t_arr, T_avail+t_arr+1, dtype=int)
        T_not_avail = np.array([value for value in range(self.T_ems) if value not in t_avail])
        T_flex_avail = np.arange(max(T_flex[0], t_arr), min(T_flex[-1], min(t_avail[-1], self.T_ems))+1, dtype=int)  
               
        if np.array_equal(T_flex_avail, t_avail) or not(np.array_equal(T_flex_avail, T_flex)):
            return 0  
        
        else:
            #T = t_dep - t_arr ################pb t_dep day after!
            prob = pic.Problem()
            Flex = pic.RealVariable('flex', 1)
            p_ch = pic.RealVariable('P_ch', T_horizon)
            P_soft = pic.RealVariable('P_soft')

            prob.add_constraint(Flex >= 0)
            prob.add_list_of_constraints([p_ch[t]  >= self.Pmin for t in t_avail])
            prob.add_list_of_constraints([p_ch[t]  <= self.Pmax for t in t_avail])
            prob.add_list_of_constraints([p_ch[t]  == 0 for t  in T_not_avail])
            #prob.add_constraint(sum(Eff_EV*p_ch[t]*dt/Cmax for t in T_avail) + SOC_arr == 1)
            #prob.add_constraint(sum(p_ch[t]*self.dt_ems/self.Emax for t in t_avail) + SOC_arr + P_soft >= 0.9)
            prob.add_constraint(sum(p_ch[t]*self.dt_ems/self.Emax for t in t_avail) + SOC_arr >= 0.9)
            
            if flex_type=='up':
                prob.add_list_of_constraints([Flex <= self.Pnet[t] - p_ch[t] for t in T_flex_avail])
            else: 
                prob.add_list_of_constraints([Flex <= p_ch[t] - self.Pnet[t] for t in T_flex_avail])
            
            #prob.set_objective('max', Flex - P_soft*1e2)  
            prob.set_objective('max', Flex)

            prob.solve(solver='mosek', primals=None)
            print('EV_flex: status {} and Flex={}'.format(prob.status, Flex.value))   
            return Flex.value #if Flex >= min_flex else 0
            
#Nondispatchable Asset Class (use for inflexible loads, PV sources etc.)            
class NondispatchableAsset_3ph(Asset_3ph):
    def __init__(self, Pnet, Qnet, bus_id, phases, dt, T, Pnet_pred = None, Qnet_pred = None):
        Asset_3ph.__init__(self, bus_id, phases, dt, T)
        self.Pnet = Pnet #Uncontrolled real input powers over the time series (kW)
        self.Qnet = Qnet #Uncontrolled reactive input powers over the time series (kVar)
        self.Pnet_pred = Pnet_pred or Pnet #Predicted real input powers over the time series (kW)
        self.Qnet_pred = Qnet_pred or Qnet #Predicted reactive input powers over the time series (kVar)



class PV_Controllable_3ph(Asset_3ph):
    def __init__(self, Pmax, Pmin, bus_id, phases, dt, T, dt_ems, T_ems, Pnet = None, Qnet = None):
        Asset_3ph.__init__(self, bus_id, phases, dt, T)
        self.asset_type = 'PV_controllable'
        self.Pmax = Pmax
        self.Pmin = Pmin #maximum input powers over the time series (kW)
        self.Pnet = np.zeros(T) #input powers over the time series (kW)
        self.Qnet = np.zeros(T) #reactive powers over the time series (kW)
        self.c_deg_lin = 0
        
    def update_control(self,Pnet):
        #update the storage system power and energy profile 
        self.Pnet = Pnet

    def update_control_t(self,Pnet_t,t):
        #update the storage system power and energy profile 
        self.Pnet[t] = Pnet_t


class BuildingAsset(Asset):
    """
    A building asset (use for flexibility from building HVAC)
    Parameters
    ----------
    Tmax : float
        Maximum temperature inside the building (Degree C)
    Tmin = float
        Minimum temperature inside the building (Degree C)
    Hmax : float
        Maximum power consumed by electrical heating (kW)
    Cmax : float
        Maximum power consumed by electrical cooling (kW)
    deltat: float
        Time interval after which system is allowed to change decisions (h)
    T0 : float
        Initial temperature inside the buidling (Degree C)
    C : float
        Thermal capacitance of building (kWh/Degree C)
    R : float
        Thermal resistance of building to outside environment(Degree C/kW)
    CoP_heating : float
        Coefficient of performance of the heat pump (N/A)
    CoP_cooling : float
        Coefficient of performance of the chiller (N/A)
    Ta : numpy.ndarray
        Ambient temperature (Degree C)
    alpha : float
        Coefficient of previous temperature in the temperature dynamics
        equation (N/A)
    beta : float
        Coefficient of power consumed to heat/cool the building in the
        temperature dynamics equation (Degree C/kW)
    gamma : float
        Coefficient of ambient temperature in the temperature dynamics
        equation (N/A)
    Pnet : numpy.ndarray
        Input real power (kW)
    Qnet : numpy.ndarray
        Input reactive power (kVAR)
    Returns
    -------
    Asset
    """
    def __init__(self, Tmax, Tmin, Hmax, Cmax, T0, C, R, CoP_heating, CoP_cooling, Ta, bus_id, dt,
                 T, dt_ems, T_ems):
        Asset.__init__(self, bus_id, dt, T)
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.Hmax = Hmax
        self.Cmax = Cmax
        deltat = dt_ems
        self.deltat = deltat
        self.T0 = T0
        self.C = C
        self.R = R
        self.CoP_heating = CoP_heating
        self.CoP_cooling = CoP_cooling
        self.Ta = Ta
        self.dt_ems = dt_ems
        self.T_ems = T_ems
        self.alpha = (1 - (deltat/(R*C)))
        self.beta = (deltat/C)
        self.gamma = deltat/(R*C)
        self.Pnet = np.zeros(T)   # input powers over the time series (kW)
        self.Qnet = np.zeros(T)   # reactive powers over the time series (kW)

    def update_control(self, Pnet):
        """
        Update the power consumed by the HVAC at time interval t
        Parameters
        ----------
        Pnet : numpy.ndarray
            input powers over the time series (kW)
        """
        self.Pnet = Pnet

    def maxdemand_baseline(self):
        """
        Compute the baseline consumption of a building in the absence of flexibility
        by minimizing the peak demand
        Parameters
        ----------
        c_demand : float (£/KW)
            demand charge based on the highest capacity you required during the given billing period 
        returns
        --------
        P_th: 1d array
            daily schedule
        """ 

        prob = pic.Problem()
        P_peak = pic.RealVariable('P_peak')
        #b = pic.BinaryVariable('b',self.T_ems)
        P_cool = pic.RealVariable('P_cool',self.T_ems)
        P_heat = pic.RealVariable('P_heat',self.T_ems)
        T_in = pic.RealVariable('T_in',self.T_ems)

        prob.add_list_of_constraints([P_peak >= P_heat[t] + P_cool[t] for t in range(self.T_ems)])

        prob.add_list_of_constraints([P_cool[t]  >= 0 for t in range(self.T_ems)])
        prob.add_list_of_constraints([P_cool[t]  <= self.Cmax for t in range(self.T_ems)])

        prob.add_list_of_constraints([P_heat[t]  >= 0 for t in range(self.T_ems)])
        prob.add_list_of_constraints([P_heat[t]  <= self.Hmax for t in range(self.T_ems)])

        prob.add_constraint(T_in[0] == self.T0)
        prob.add_list_of_constraints([T_in[t] == self.alpha*T_in[t-1] + self.beta*(self.CoP_heating*P_heat[t-1]-self.CoP_cooling*P_cool[t-1]) + self.gamma*self.Ta[t-1] for t in range(1,self.T_ems)])
        prob.add_list_of_constraints([T_in[t]  >= self.Tmin for t in range(self.T_ems)])
        prob.add_list_of_constraints([T_in[t]  <= self.Tmax for t in range(self.T_ems)])
        #prob.add_constraint(sum((P_heat[t] + P_cool[t]) for t in self.T_ems) <= max_demand/30)
        
        prob.set_objective('min', P_peak)
        prob.solve(solver='mosek', primals=None) #, primals=None) 
        P_th = [P_heat.value[i]+P_cool.value[i] for i in range(self.T_ems)]
        #print('HVAC_maxdemand_baseline: status {} and P_th={}'.format(prob.status, P_th))
        return P_th 


    def toup_baseline(self, toup):
        """
        Compute the baseline consumption of a building in the absence of flexibility
        Parameters
        ----------
        toup : 1d array (1, T_ems)
            time of use price
        
        Returns
        -------
        P_th: 1d array (1, T_ems)
            schedule of HVAC
        """ 
        prob = pic.Problem()
        P_cool = pic.RealVariable('P_cool',self.T_ems)
        P_heat = pic.RealVariable('P_heat',self.T_ems)
        T_in = pic.RealVariable('T_in',self.T_ems)

        prob.add_list_of_constraints([P_cool[t]  >= 0 for t in range(self.T_ems)])
        prob.add_list_of_constraints([P_cool[t]  <= self.Cmax for t in range(self.T_ems)])

        prob.add_list_of_constraints([P_heat[t]  >= 0 for t in range(self.T_ems)])
        prob.add_list_of_constraints([P_heat[t]  <= self.Hmax for t in range(self.T_ems)])

        prob.add_constraint(T_in[0] == self.T0)
        prob.add_list_of_constraints(T_in[t] == self.alpha*T_in[t-1] + self.beta*(self.CoP_heating*P_heat[t-1]-self.CoP_cooling*P_cool[t-1]) + self.gamma*self.Ta[t-1] for t in range(1,self.T_ems))
        prob.add_list_of_constraints([T_in[t]  >= self.Tmin for t in range(self.T_ems)])
        prob.add_list_of_constraints([T_in[t]  <= self.Tmax for t in range(self.T_ems)])

        prob.set_objective('min', sum((P_heat[t] + P_cool[t])*toup[t] for t in range(self.T_ems))) #

        prob.solve(solver='mosek', primals=None)  #) 

        P_th = [P_heat.value[i]+P_cool.value[i] for i in range(self.T_ems)]
        #print('HVAC_toup_baseline: status {} and P_th={}'.format(prob.status, P_th))
        return P_th 

    def flexibility(self, T_flex, flex_min=None, flex_type='up'):
        """
        Compute the flexibility that can be provided by the HVAC for the flexibility period [t_s, t_s + tau]
        Parameters
        ----------
        T0_actual : float
            Actual initial temperature inside the buidling (Degree C) (varies from asset.T0)
        T_flex: 1d array
            period of flexibility
        flex_type: 'down' or 'up'. default set to 'up'
            the type of flexibility to provide
            'up': to decrease consumption of the HVAC in flexibility period
            'down'  : to increase consumption of the HVAC in flexibility period
        Returns
        -------
        Flex: float
            the flexibility 
        """

        prob = pic.Problem()
        Flex = pic.RealVariable('flex', 1)
        P_cool = pic.RealVariable('P_cool',self.T_ems)
        P_heat = pic.RealVariable('P_heat',self.T_ems)
        T_in = pic.RealVariable('T_in',self.T_ems)
        Soft_min = pic.RealVariable('Soft_min',self.T_ems)
        Soft_max = pic.RealVariable('Soft_max',self.T_ems)

        prob.add_constraint(Flex >= 0)
        prob.add_list_of_constraints([P_cool[t]  >= 0 for t in range(self.T_ems)])
        prob.add_list_of_constraints([P_cool[t]  <= self.Cmax for t in range(self.T_ems)])

        prob.add_list_of_constraints([P_heat[t]  >= 0 for t in range(self.T_ems)])
        prob.add_list_of_constraints([P_heat[t]  <= self.Hmax for t in range(self.T_ems)])

        prob.add_constraint(T_in[0] == self.T0)
        prob.add_list_of_constraints(T_in[t] == self.alpha*T_in[t-1] + self.beta*(self.CoP_heating*P_heat[t-1]-self.CoP_cooling*P_cool[t-1]) + self.gamma*self.Ta[t-1] for t in range(1,self.T_ems))
        prob.add_list_of_constraints([T_in[t]  >= self.Tmin for t in range(self.T_ems)])
        prob.add_list_of_constraints([T_in[t]  <= self.Tmax for t in range(self.T_ems)])
        #prob.add_list_of_constraints([T_in[t] + Soft_min[t] >= Tmin for t in range(Th)])
        #prob.add_list_of_constraints([T_in[t] - Soft_max[t] <= Tmax for t in range(Th)])
        Penalty = 1e2

        if flex_type=='up':
            prob.add_list_of_constraints([Flex <= self.Pnet[t] - (P_heat[t] + P_cool[t]) for t in T_flex])           
        else:           
            prob.add_list_of_constraints([Flex <= (P_heat[t] + P_cool[t]) - self.Pnet[t] for t in T_flex]) 
        
        prob.set_objective('max', Flex)

        #prob.set_objective('max', sum((P_heat[t] + P_cool[t]) for t in T_flex))
                           #- Penalty* sum((Soft_min[t]+Soft_max[t]) for t in range(Th)))   

        prob.solve(solver='mosek', primals=None) 
        #print('HVAC flex: status {} and Flex={}'.format(prob.status, Flex.value))

        if flex_min == None:
            return Flex.value
        else:
            return Flex.value if Flex.value >= flex_min  else 0
    
    """
        if np.all(np.array(P_th) == 0):   ###optimisation has no primal solution 
            print('The building cannot provide flexibility for the period {}h-{}h'.format(t_s, t_s+tau%24))
            return flexibility, np.array(P_th)
        
        else:
            for tf in T_flex:
                if flex_type=='down':  
                    flexibility[tf] = baseline[tf] - P_th[tf]
                else:
                    flexibility[tf] = P_th[tf] - baseline[tf]
                #####is there any threshold on provided flexibility? e.g, flex < flex_max
            return flexibility, np.array(P_th)
        """
    #def rate_of_delivery(self, Tdatabase, t_s, tau, n_samples, sigma_T0=2):
    """
        Compute the rate of flexibility that can actually be provided by the HVAC  
        Parameters
        ----------
        baseline_flex: numpy.ndarray
            the planned flexibility of the HVAC
        Tdatabase: numpy.ndarray
            a database of different daily schedules of ambient temperature (Degree C)
        t_s : int
            the first hour of the call for flexibility
        tau : int 
            the duration of flexibility period (hour)
        n_samples : int
            number of scenarios used to extract the actual flexibility
        sigma_T0 : float >0
            std deviation of initial temperature inside the buidling from the planned T0
            default = 2
        Returns
        -------
        rate_of_delivery : numpy.array
            n_samples lines, each line 
                rate of delivery of the HVAC (KW) for that specific scenario

        """

    """
        #t_s, tau are in hours and converted below to the time resolution of the asset       
        t_flex_start = t_s/self.dt 
        tau_flex = tau/self.dt
        t_flex_end = t_flex_start + tau_flex
        T_flex = np.arange(t_flex_start, t_flex_end+1, dtype=int)
        actual_flexibility, rate_of_delivery = np.zeros((n_samples,self.T)), np.zeros((n_samples,self.T))

        baseline_flexibility,_ = self.flexibility(self.Pnet, self.T0, self.Ta, T_flex)

        if not(np.any(baseline_flex)):    #means optimisation of flexibility() method for the baseline schedule hasn't a primal solution 
            print('The baseline setting of the building does not allow flexibility for the period {}h-{}h'.format(t_s, t_s+tau%24))          
        else:           
            for s in range(n_samples):
                print('computing rate of delivery for scenario:',s)
                T0_actual = self.T0 + np.random.normal(0, sigma_T0) 
                if np.any(T_flex) > self.T:
                    if s+2 > len(Ta_actual):
                        break
                    Ta_actual = Tdatabase[s:s+2]
                    Ta_actual = np.append(self.time_scale_conversion(Ta_actual[0], self.T), self.time_scale_conversion(Ta_actual[1], self.T))
                else:
                    Ta_actual = Tdatabase.iloc[s]
                    Ta_actual = self.time_scale_conversion(Ta_actual.reshape(1,-1), self.T)
           
                actual_flexibility[s,:], P = self.flexibility(self.Pnet, T0_actual, Ta_actual, T_flex)

                for t in T_flex:
                    if baseline_flexibility[t]  == 0:
                        HVAC_rate_of_delivery[s,t] = np.nan
                    else:
                        rate_of_delivery[s,t] = actual_flexibility[s,t]/baseline_flexibility[t] 

        return rate_of_delivery[:,T_flex]
        
        
                flexibility : numpy.array
            flexibility that can be provided by the HVAC (KW)
        T_flex: numpy.ndarray
            the indices of flexibility period in time resolution of dt
        new_schedule : numpy.array
            the new schedule of HVAC power that accounts for the flexibility (KW)
            flexibility[t] = baseline[t] -/+ new_schedule[t] for t in T_flex (- for down flex and + for up flex)
        """
        
    """
        if np.any(T_flex) > self.T : 
            Th = 2*T
            Pbase = np.zeros(Th)
            Pbase[T:] = self.Pnet
            Pbase[:T] = self.Pnet
            Ta[T:] = Ta[:T]
        else: 
            Th = T
            Pbase = self.Pnet 
        
            

        flexibility = np.zeros(Th)
        """