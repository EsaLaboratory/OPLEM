## -*- coding: utf-8 -*-
"""
OPEN Asset module
Asset objects define distributed energy resources (DERs) and loads.
Attributes include network location, phase connection and real and reactive
output power profiles over the simulation time-series. 
Flexible Asset classes have an update control method, which is called by
EnergySystem simulation methods with control references to update the output
power profiles and state variables. The update control method also implements
constraints which limit the implementation of references. 
OPEN includes the following Asset subclasses: NondispatchableAsset for
uncontrollable loads and generation sources, StorageAsset for storage systems
and BuildingAsset for buildings with flexible heating, ventilation and air
conditioning (HVAC).
"""

# import modules
import numpy as np
import picos as pic

__version__ = "1.1.0"


class Asset:
    """
    An energy resource located at a particular bus in the network
    Parameters
    ----------
    bus_id : float
        id number of the bus in the network
    dt : float
        time interval duration
    T : int
        number of time intervals
    phases : list, optional, default [0,1,2]
        [0, 1, 2] indicates 3 phase connection \
        Wye: [0, 1] indicates an a,b connection \
        Delta: [0] indicates a-b, [1] b-c, [2] c-a
    Returns
    -------
    Asset
    """
    def __init__(self, bus_id, dt, T, phases=[0, 1, 2]):
        self.bus_id = bus_id
        self.phases = np.array(phases)
        self.dt = dt
        self.T = T


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

        self.type = 'building'

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
        
        returns
        --------
        P_th: 1d array
            daily schedule
        """ 

        prob = pic.Problem()
        P_peak = pic.RealVariable('P_peak')
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
        
        prob.set_objective('min', P_peak)
        prob.solve(solver='mosek', primals=None) #, primals=None) 
        P_th = [P_heat.value[i]+P_cool.value[i] for i in range(self.T_ems)]
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
        return P_th 

    def flexibility(self, T_flex, flex_min=None, flex_type='up'):
        """
        Compute the flexibility per time slot that can be provided by the HVAC for the flexibility period T_flex
        Parameters
        ----------
        T_flex: 1d array
            period of flexibility [t_start, t_end]
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
        #Penalty = 1e2

        if flex_type=='up':
            prob.add_list_of_constraints([Flex <= self.Pnet[t] - (P_heat[t] + P_cool[t]) for t in T_flex])           
        else:           
            prob.add_list_of_constraints([Flex <= (P_heat[t] + P_cool[t]) - self.Pnet[t] for t in T_flex]) 
        
        prob.set_objective('max', Flex)

        prob.solve(solver='mosek', primals=None) 

        if flex_min == None:
            return Flex.value
        else:
            return Flex.value if Flex.value >= flex_min  else 0

# NEEDED FOR OXEMF EV CASE
class StorageAsset(Asset):
    """
    A storage asset (use for batteries, EVs etc.)
    Parameters
    ----------
    Emax : numpy.ndarray
        maximum energy levels over the time series (kWh)
    Emin : numpy.ndarray
        minimum energy levels over the time series (kWh)
    Pmax : numpy.ndarray
        maximum input powers over the time series (kW)
    Pmin : numpy.ndarray
        minimum input powers over the time series (kW)
    E0 : float
        initial energy level (kWh)
    ET : float
        required terminal energy level (kWh)
    bus_id : float
        id number of the bus in the network
    dt : float
        time interval duration (s)
    T : int
        number of time intervals
    dt_ems : float
        time interval duration (energy management system time horizon) (s)
    T_ems : int
        number of time intervals (energy management system time horizon)
    phases : list, optional, default [0,1,2]
        [0, 1, 2] indicates 3 phase connection \
        Wye: [0, 1] indicates an a,b connection \
        Delta: [0] indicates a-b, [1] b-c, [2] c-a
    Pmax_abs : float
        max power level (kW)
    c_deg_lin : float
        battery degradation rate with energy throughput (£/kWh)
    eff : float, default 1
        charging efficiency (between 0 and 1)
    eff_opt : float, default 1
        charging efficiency to be used in optimiser (between 0 and 1).
    Pnet : numpy.ndarray
        Input real power over the simulation time series (kW)
    Qnet : numpy.ndarray
        Input reactive power over the simulation time series(kVAR)
    Returns
    -------
    Asset
    """
    def __init__(self, Emax, Emin, Pmax, Pmin, E0, ET, bus_id, dt, T, dt_ems,
                 T_ems, phases=[0, 1, 2], Pmax_abs=None, c_deg_lin=None,
                 eff=1, eff_opt=1):
        Asset.__init__(self, bus_id, dt, T, phases=phases)
        self.Emax = Emax
        self.Emin = Emin
        self.Pmax = Pmax
        self.Pmin = Pmin
        if Pmax_abs is None:
            self.Pmax_abs = max(self.Pmax)
        else:
            self.Pmax_abs = Pmax_abs
        self.E0 = E0
        self.ET = ET
        self.E = E0*np.ones(T+1)
        self.dt_ems = dt_ems
        self.T_ems = T_ems
        self.Pnet = np.zeros(T)
        self.Qnet = np.zeros(T)
        self.c_deg_lin = c_deg_lin or 0
        self.eff = eff*np.ones(100)
        self.eff_opt = eff_opt

        self.type = 'storage'

# NEEDED FOR OXEMF EV CASE
    def update_control(self, Pnet):
        """
        Update the storage system power and energy profile
        Parameters
        ----------
        Pnet : float
            input powers over the time series (kW)
        """
        self.Pnet = Pnet
        self.E[0] = self.E0
        t_ems = self.dt/self.dt_ems
        for t in range(self.T):
            P_ratio = int(100*(abs(self.Pnet[t]/self.Pmax_abs)))
            P_eff = self.eff[P_ratio-1]
            if self.Pnet[t] < 0:
                if self.E[t] <= self.Emin[int(t*t_ems)]:
                    self.E[t] = self.Emin[int(t*t_ems)]
                    self.Pnet[t] = 0
                self.E[t+1] = self.E[t] + (1/P_eff)*self.Pnet[t]*self.dt
            elif self.Pnet[t] >= 0:
                if self.E[t] >= self.Emax[int(t*t_ems)]:
                    self.E[t] = self.Emax[int(t*t_ems)]
                    self.Pnet[t] = 0
                self.E[t+1] = self.E[t] + P_eff*self.Pnet[t]*self.dt
# NEEDED FOR OXEMF EV CASE
    def update_control_t(self, Pnet_t, t):
        """
        Update the storage system power and energy at time interval t
        Parameters
        ----------
        Pnet_t : float
            input powers over the time series (kW)
        t : int
            time interval
        """
        self.Pnet[t] = Pnet_t
        self.E[0] = self.E0
        t_ems = self.dt/self.dt_ems
        P_ratio = int(100*(abs(self.Pnet[t]/self.Pmax_abs)))
        P_eff = self.eff[P_ratio-1]
        if self.Pnet[t] < 0:
            if self.E[t] <= self.Emin[int(t*t_ems)]:
                self.E[t] = self.Emin[int(t*t_ems)]
                self.Pnet[t] = 0
            self.E[t+1] = self.E[t] + (1/P_eff)*self.Pnet[t]*self.dt
        elif self.Pnet[t] >= 0:
            if self.E[t] >= self.Emax[int(t*t_ems)]:
                self.E[t] = self.Emax[int(t*t_ems)]
                self.Pnet[t] = 0
            self.E[t+1] = self.E[t] + P_eff*self.Pnet[t]*self.dt

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
        P_ch[int(t_arr): int(min(self.T_ems, t_arr+nbr_ts))] = self.Pmax
        
        return P_ch

    def EV_toup_baseline(self, t_arr, T_avail, SOC_arr, SOC_dep, toup):
        """
        Compute the baseline consumption of an EV in the absence of flexibility, response to TOUP signal
        Parameters
        ----------
        t_arr : int
            index of time slot correspnding to the plug-in of the EV
        T_avail: int
            number of time slots the EV remained plugged-in
        SOC_arr: float [0,1]
            SOC of EV at arrival 
        SOC_dep: float [0,1]
            desired SOC of EV at departure 
        toup : 1d array (1, T_ems)
            time of use price
        returns
        --------
        P_ch: 1d array
            daily charging schedule of EV
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
        prob.add_constraint(sum(P_ch[t]*self.dt_ems/self.Emax for t in range(T_horizon)) + SOC_arr >= SOC_dep)
        #prob.add_constraint(sum(Eff_EV*p_ch[t]*dt/Cmax for t in T_horizon) + SOC_arr + P_soft >= 0.9)

        #prob.set_objective('min', sum(P_ch[t]*TOUP[t] for t in range(T_horizon)) + P_soft*1e2)
        prob.set_objective('min', sum(P_ch[t]*TOUP[t] for t in range(T_horizon)))

        prob.solve(solver='mosek', primals=None) #, )
        if prob.status != 'optimal':
            print('non optimal solution for toup')
            P_ch_value = self.EV_baseline(t_arr, T_avail, SOC_arr)
            
        else: 
            P_ch_value = [P_ch.value[i] for i in range(self.T_ems)]
        
        return P_ch_value

    def EV_flexibility(self, t_arr, T_avail, SOC_arr, SOC_dep, T_flex, flex_type='up'):
        """
        Compute the flexibility per time slot that can be provided by the EV for the period T_flex       
        Parameters
        ----------
        t_arr : int
            index of time slot correspnding to the plug-in of the EV
        T_avail: int
            number of time slots the EV remained plugged-in
        SOC_arr: float [0,1]
            SOC of EV at arrival 
        SOC_dep: float [0,1]
            desired SOC of EV at departure 
        T_flex: 1d array
            period of flexibility [t_start, t_end]
        flex_type: 'down' or 'up'. default set to 'up'
            the type of flexibility to provide
            'up': to decrease consumption of the HVAC in flexibility period
            'down'  : to increase consumption of the HVAC in flexibility period
        
        returns
        --------
        Flex: float
            available flexibility
        """ 
        n = np.ceil((T_avail+t_arr)/self.T_ems) + int(not((T_avail+t_arr)%self.T_ems))
        T_horizon = int(self.T_ems*n)
        t_avail = np.arange(t_arr, T_avail+t_arr+1, dtype=int)
        T_not_avail = np.array([value for value in range(self.T_ems) if value not in t_avail])
        T_flex_avail =t_avail[np.isin(t_avail, T_flex)]

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
            prob.add_constraint(sum(p_ch[t]*self.dt_ems/self.Emax for t in t_avail) + SOC_arr >= SOC_dep)
            
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



class NondispatchableAsset(Asset):
    """
    A 3 phase nondispatchable asset class (use for inflexible loads,
    PVsources etc)
    Parameters
    ----------
    Pnet : float
        uncontrolled real input powers over the time series
    Qnet : float
        uncontrolled reactive input powers over the time series (kVar)
    bus_id : float
        id number of the bus in the network
    dt : float
        time interval duration
    T : int
        number of time intervals
    phases : list, optional, default [0,1,2]
        [0, 1, 2] indicates 3 phase connection \
        Wye: [0, 1] indicates an a,b connection \
        Delta: [0] indicates a-b, [1] b-c, [2] c-a
    Pnet_pred : float or None
        predicted real input powers over the time series (kW)
    Qnet_pred : float or None
        predicted reactive input powers over the time series (kVar)
    curt: float or None [0,1]
        percentage (%) of load/generation that can e curtailed
    Returns
    -------
    Asset
    """

    def __init__(self, Pnet, Qnet, bus_id, dt, T, phases=[0, 1, 2],
                 Pnet_pred=None, Qnet_pred=None, curt=None):
        Asset.__init__(self, bus_id, dt, T, phases=phases)
        self.Pnet = Pnet
        self.Qnet = Qnet
        if Pnet_pred is not None:
            self.Pnet_pred = Pnet_pred
        else:
            self.Pnet_pred = Pnet
        if Qnet_pred is not None:
            self.Qnet_pred = Qnet_pred
        else:
            self.Qnet_pred = Qnet

        self.type = 'ND'
