#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OPLEM Market module has two types of markets:

(1) Energy markets, and 

(2) Flexibility markets

(1) The energy market comes with subclasses of the common types of energy markets:

(i) central market

(ii) time of use market

(iii) P2P market

(iv) auction market

(2) The flexibility markets comes with one market

(i) Capacity limits market
"""


#import modules
import os
import copy
from os.path import normpath, join
import pandas as pd
import numpy as np
import pickle
import time
import picos as pic
import oplem.Participant as Participant


__version__ = "1.1.0"


class Market:
	"""
	Base Market Class
	
	Parameters
	----------
	participants : list of objects
	Containing details of each participant
	T_market : int
		Market horizon
	dt_market : float
		time interval duration (hours)
	price_imp : numpy.ndarray (``T_market``,)
		import prices from the grid (£/kWh)
	t_ahead_0: int
		starting time slot of market clearing
	P_import : numpy.ndarray (``T_market``,)
		max import from the grid (kW)	
	P_export : numpy.ndarray (``T_market``,)
		max export to the grid (kW)
	price_exp : numpy.ndarray (``T_market``,)
		export prices to the grid (£/kWh)
	network : object, default=None
		the network infrastructure of the market

		useful for market clearings that account for network constraints,

		required to run simulate_network_3ph
	
	Returns
	-------
	Market object
	"""
	
	def __init__(self, participants, dt_market, T_market, price_imp, t_ahead_0=0, P_import=None, P_export=None, price_exp=None, network=None):
		self.participants = participants
		self.dt_market = dt_market
		self.T_market = T_market
		self.t_ahead_0 = t_ahead_0
		self.P_import = P_import
		self.price_imp = price_imp
		self.P_export = P_export
		self.price_exp = price_exp
		self.network = network


		if self.P_export ==None and np.all(self.price_exp)==None:
			self.P_export=np.zeros(self.T_market)
			self.price_exp=np.zeros(self.T_market)
		if self.P_export ==None and np.all(self.price_exp)!=None:
			self.P_export=np.inf*np.ones(self.T_market)	
		if np.any(self.P_export) !=None and np.all(self.price_exp)==None:
			raise ValueError('Export power was set to be non-zero, but no values for the export prices were entered')
		if np.all(self.P_import)==None:
			self.P_import=np.inf*np.ones(self.T_market)

	def simulate_network_pp(self, single_iter=False):
		"""
		simulate the power flow of the network if the network is created using pandaspower
  
  		Parameters
		--------------
		single_iter : bool, default: False
			False: simulate network power flow for the remaining time steps of the horizon [t_ahead_0, T_market]

			True: simulate network power flow for a single optimisation time step t_ahead_0
		
		Returns
		----------
		output: dict
			the resulting network output for the simulation time steps:

			buses_Vpu : Voltage magnitude at bus (V) 

			buses_Vang : Voltage angle at bus (rad)

			buses_Pnet : Real power at bus (kW)

			buses_Qnet : Reactive power at bus (kVAR)

			Pnet_market : Real power seen by the market (kW)

			Qnet_market : Reactive power seen by the market (kVAR)
		"""

		assets_all = []
		for par in self.participants:
		    for asset in par.assets:
		        assets_all.append(asset)
		participant_all = Participant(len(self.participants)+2, assets_all)

		N_buses = self.network.bus['name'].size
		T = participant_all.assets[0].T
		dt = participant_all.assets[0].dt 
		P_demand_buses = np.zeros([T,N_buses])
		Q_demand_buses = np.zeros([T,N_buses])

		for asset in participant_all.assets:
			P_demand_buses[:,asset.bus_id] += asset.Pnet
			Q_demand_buses[:,asset.bus_id] += asset.Qnet

		t0 = self.t_ahead_0*int(self.dt_market/dt)
		if single_iter:
			T_final = (self.t_ahead_0+1)*int(self.dt_market/dt)
		else: T_final=T		

		buses_Vpu = np.zeros([T_final-t0,N_buses])
		buses_Vang = np.zeros([T_final-t0,N_buses])
		buses_Pnet = np.zeros([T_final-t0,N_buses])
		buses_Qnet = np.zeros([T_final-t0,N_buses])
		Pnet_market = np.zeros(T_final-t0)
		Qnet_market = np.zeros(T_final-t0)

		print('*** SIMULATING THE NETWORK ***')
		for t in range(t0, T_final):
            
		    network_t = copy.deepcopy(self.network)
		    for bus_id in range(N_buses):
		        P_t = P_demand_buses[t,bus_id]
		        Q_t = Q_demand_buses[t,bus_id]
		        #add P,Q loads to the network copy
		        pp.create_load(network_t,bus_id,P_t/1e3,Q_t/1e3)
		    #run the power flow simulation
		    Pnet_market[t] = network_t.res_ext_grid['p_mw'][0]*1e3
		    Qnet_market[t] = network_t.res_ext_grid['q_mvar'][0]*1e3
		    for bus_i in range(N_buses):
		        buses_Vpu[t,bus_i] = network_t.res_bus['vm_pu'][bus_i]
		        buses_Vang[t,bus_i] = network_t.res_bus['va_degree'][bus_i]
		        buses_Pnet[t,bus_i] = network_t.res_bus['p_mw'][bus_i]*1e3
		        buses_Qnet[t,bus_i] = network_t.res_bus['q_mvar'][bus_i]*1e3

		output = {'buses_Vpu':buses_Vpu,\
		        'buses_Vang':buses_Vang,\
		        'buses_Pnet':buses_Pnet,\
		        'buses_Qnet':buses_Qnet,\
		        'Pnet_market':Pnet_market,\
		        'Qnet_market':Qnet_market
                }

		return output

	def simulate_network_3ph(self, single_iter=False):
		"""
		simulate the power flow of the network if the network is created using Network_3ph() class

		Parameters
		-------------
		single_iter : bool, default False
			False: simulate network power flow for the remaining time steps of the horizon [t_ahead_0, T_market]

			True: simulate network power flow for a single optimisation time step t_ahead_0
		
		Returns
		----------
		network_pf : list
			the resulting network object for [t_ahead_0, T_market] time steps
		
		"""
		
		network_pf = []
		assets_all = []
		for par in self.participants:
		    for asset in par.assets:
		        assets_all.append(asset)

		load_buses = np.where(np.abs(self.network.bus_df['Pa'])+np.abs(self.network.bus_df['Pb'])+np.abs(self.network.bus_df['Pc'])>0)[0]
		load_phases = []
		for load_bus_idx in range(len(load_buses)):
		    phase_list = []
		    if np.abs(self.network.bus_df.iloc[load_buses[load_bus_idx]]['Pa']) > 0:
		          phase_list.append(0)
		    if np.abs(self.network.bus_df.iloc[load_buses[load_bus_idx]]['Pb']) > 0:
		          phase_list.append(1)
		    if np.abs(self.network.bus_df.iloc[load_buses[load_bus_idx]]['Pc']) > 0:
		          phase_list.append(2)
		    load_phases.append(np.array(phase_list))  

		T = assets_all[0].T
		dt = assets_all[0].dt  
		t0 = self.t_ahead_0*int(self.dt_market/dt)
		if single_iter:
			T_final = (self.t_ahead_0+1)*int(self.dt_market/dt)
		else: T_final=T	

		#for t in range(self.t_ahead_0, self.T_market):
		for t in range(t0, T_final):
		    network_t = copy.deepcopy(self.network)
		    network_t.clear_loads()

		    for b_idx in range(len(load_buses)): 
		        bus = load_buses[b_idx]
		        n_phases = len(load_phases[b_idx]) 
		        p_sum, q_sum=0, 0
		        for asset in assets_all:
		            if asset.bus_id == bus:
		                #p_sum += asset.Pnet_ems[t]
		                #q_sum += asset.Qnet_ems[t]
		                p_sum += asset.Pnet[t]
		                q_sum += asset.Qnet[t]
		        for phase in load_phases[b_idx]:                                        
		            network_t.set_load(bus, phase, p_sum/n_phases, q_sum/n_phases)
		           
		    network_t.zbus_pf()
		    network_pf.append(network_t)

		return network_pf

class Central_market(Market):
	"""
	Central Market Class

	Central market runs a central optimisation market clearing

	
	Parameters
	----------
	participants : list of objects
		Containing details of each participant
	T_market : int
		Market horizon
	dt_market : float
		time interval duration (hours)
	price_imp : numpy.ndarray
		import prices from the grid (£/kWh)
	t_ahead_0: int
		starting time slot of market clearing
	P_import : numpy.ndarray
		max import from the grid (kW)	
	P_export : numpy.ndarray
		max export to the grid (kW)
	price_exp : numpy.ndarray
		export prices to the grid (£/kWh)
	network : object, default=None
		the network infrastructure of the market

		useful for market clearings that account for network constraints,

		required to run simulate_network_3ph

	Returns
	-------
	Market object
	"""

	def __init__(self, participants, dt_market, T_market, price_imp, \
				 t_ahead_0=0, P_import=None, P_export=None, price_exp=None, network=None, nw_const=True):
		Market.__init__(self, participants, dt_market, T_market, price_imp, t_ahead_0=t_ahead_0, P_import=P_import, P_export=P_export, price_exp=price_exp, network=network)
		self.nw_const = nw_const

	def mpc_market_clearing(self, v_unconstrained_buses=[], i_unconstrained_lines=[]):
		"""
		computes an MPC version of the central optimisation

		Parameters
		-----------
		v_unconstained_buses : list, default =[]
			the list of unconstained buses in the network 
		i_unconstained_lines : list, default =[]
			the list of unconstained lines in the network

		Returns
		------------
		market_clearing_outcome: pd.dataframe
			the resulting energy exchange
	            
	     	        +----+------+--------+-------+--------+-------+
  			| id | time | seller | buyer | energy | price |
  			+====+======+========+=======+========+=======+
  			|    |	    |	     |	     |        |	      |
			+----+------+--------+-------+--------+-------+
	    	schedules: list of lists
	    		assets schedules	
       		info: dictionary
		    	'AP': list of P_slack linear coefficients, 
		    	'AV': list of voltage linear coefficients,
		    	'AL': list of lines linear coefficients,
		    	'dual_P_bus': dual vector related to power balance at each bus,
			'dual_P_slack': dual vector related to power balance at slack bus,
			'dual_vbus_max': dual vector related to max limit voltage,
			'dual_vbus_min': dual vector related to min limit voltage,
			'dual_iline_max': dual vector related to max limit thermal line,
			'dlmp': distributed locational marginal prices
		"""
		
		market_clearing = pd.DataFrame(columns=['time', 'seller', 'buyer', 'energy', 'price'])
		for t in range(self.T_market):
			self.t_ahead_0 = t
			mc, schedules, info = self.market_clearing(v_unconstrained_buses=v_unconstrained_buses, i_unconstrained_lines=i_unconstrained_lines)
			mc_t = mc[mc['time']==t]
			market_clearing = pd.concat([market_clearing, mc_t], ignore_index=True)
		return market_clearing, schedules, info
	
	def market_clearing(self, v_unconstrained_buses=[], i_unconstrained_lines=[]):
		"""
		computes assets schedules based on central optimisation

		Parameters
		-----------
		v_unconstained_buses : list, default =[]
			the list of unconstained buses in the network 
		i_unconstained_lines : list, default =[]
			the list of unconstained lines in the network

		Returns
		------------
		market_clearing_outcome: pandas Dataframe
			the resulting energy exchange
   
   			+----+------+--------+-------+--------+-------+
  			| id | time | seller | buyer | energy | price |
  			+====+======+========+=======+========+=======+
  			|    |	    |	     |	     |        |	      |
			+----+------+--------+-------+--------+-------+
		schedules: list of numpy.ndarrays
			each array contains an asset's schedule
		info: dictionary
		    	'AP': list of P_slack linear coefficients, 
		    	'AV': list of voltage linear coefficients,
		    	'AL': list of lines linear coefficients,
		    	'dual_P_bus': dual vector related to power balance at each bus,
			'dual_P_slack': dual vector related to power balance at slack bus,
			'dual_vbus_max': dual vector related to max limit voltage,
			'dual_vbus_min': dual vector related to min limit voltage,
			'dual_iline_max': dual vector related to max limit thermal line,
			'dlmp': distributed locational marginal prices

		"""

		info = {} #will be filled if nw_const=True
		buses = [] 
		assets_all= []
		for par in self.participants:
			for asset in par.assets:
				assets_all.append(asset)
				buses.append(asset.bus_id)
		buses = list(set(buses))

	    	##create a participant with all the assets
		participant_all = Participant.Participant(len(self.participants)+2, assets_all)
		P_demand = participant_all.nd_demand(self.t_ahead_0)

		### lines added for dlmps
		index_assets_load_per_bus =  [[] for _ in range(len(buses))]
		index_assets_gen_per_bus =  [[] for _ in range(len(buses))]
		for a, asset in enumerate(participant_all.assets_nd):
			b_idx=buses.index(asset.bus_id)
			if asset.LoG == 'load':
				index_assets_load_per_bus[b_idx].append(a)
			else: index_assets_gen_per_bus[b_idx].append(a)
			
		index_assets_flex_per_bus = [[] for _ in range(len(buses))] 
		for a, asset in enumerate(participant_all.assets_flex):
			b_idx=buses.index(asset.bus_id)
			index_assets_flex_per_bus[b_idx].append(a)
		#### end lines for dlmps	

		prob = pic.Problem()
		Pimp = pic.RealVariable('Pimp', self.T_market-self.t_ahead_0)
		Pexp = pic.RealVariable('Pexp', self.T_market-self.t_ahead_0)
		if len(participant_all.assets_flex):
			x = pic.RealVariable('x', (2*(self.T_market-self.t_ahead_0), len(participant_all.assets_flex)))
		if len(participant_all.assets_nd):
			p_curt = pic.RealVariable('p_curt', (self.T_market-self.t_ahead_0, len(participant_all.assets_nd)))
		#auxilary variable added to retrieve dlmps
		pbus = pic.RealVariable('pbus', (self.T_market-self.t_ahead_0, len(buses)) )
		
		if self.nw_const:	
		# account for network constraints
			A_Pslack_list, A_vlim_list, A_line_list = [], [], []
			for t_ems in range(self.T_market-self.t_ahead_0):
				#balance per bus
				prob.add_list_of_constraints([pbus[t_ems, bus] ==  sum(-p_curt[t_ems, k] for k in index_assets_load_per_bus[bus])
					                                             + sum(P_demand[t_ems, k]-p_curt[t_ems, k] for k in index_assets_gen_per_bus[bus]) #
				                                                 + sum(x[t_ems, i] + x[self.T_market-self.t_ahead_0+t_ems, i] for i in index_assets_flex_per_bus[bus]) 
				                              for bus in range(len(buses))])
				#### end lines for dlmps

			for t_ems in range(self.T_market-self.t_ahead_0):	
				A_Pslack, b_Pslack, A_vlim, b_vlim, v_abs_min_vec, v_abs_max_vec, A_lines,_, _ = self.network.get_linear_parameters(participant_all.assets, t_ems+self.t_ahead_0, self.t_ahead_0)
				A_Pslack_list.append(A_Pslack) 
				A_vlim_list.append(A_vlim)
				A_line_list.append(A_lines)

				#balance constraint
				prob.add_constraint(Pimp[t_ems]-Pexp[t_ems] == sum(A_Pslack[b]*pbus[t_ems, b] for b in range(len(buses))
 															     #+A_Pslack[b+len(buses)]*qbus[t_ems, b]
					                                              )+ b_Pslack/1e3
								   )  
				                    
				# voltage magnitude constraints:
				for bus_ph_index in range(self.network.N_phases*(self.network.N_buses-1)):
					if int(bus_ph_index/3) not in v_unconstrained_buses: 
						prob.add_constraint(sum(A_vlim[bus_ph_index,b]*1e3*pbus[t_ems, b] for b in range(len(buses)))
											+ b_vlim[bus_ph_index] <= v_abs_max_vec[bus_ph_index] )
						prob.add_constraint(sum(A_vlim[bus_ph_index,b]*1e3*pbus[t_ems, b] for b in range(len(buses)))
											+ b_vlim[bus_ph_index] >= v_abs_min_vec[bus_ph_index]) 
						
		        # Line current magnitude constraints:
				for line_ij in range(self.network.N_lines):
					if line_ij not in i_unconstrained_lines:	
						for ph in range(self.network.N_phases):
							prob.add_constraint(sum(np.nan_to_num(A_lines[3*line_ij+ph,b])*1e3*pbus[t_ems, b] for b in range(len(buses))
							                       ) 
		                                        + self.network.Jabs_I0_list[line_ij][ph] <= self.network.i_abs_max[line_ij,ph])

		else:
			P_demand = np.sum(P_demand, axis=1)
			# balance constraint
			prob.add_constraint( P_demand - sum(p_curt[:, a] for a in range(len(participant_all.assets_nd))) 
		    	               + sum(x[: self.T_market-self.t_ahead_0, a] + x[self.T_market-self.t_ahead_0:, a] for a in range(len(participant_all.assets_flex)))\
		                      == Pimp- Pexp)

		#operational constraints
		for a, asset in enumerate(participant_all.assets_flex):
			A, b = asset.polytope(self.t_ahead_0)
			#A_op.append(A)
			prob.add_constraint(A*x[:, a] <= b)
		
		#curtailment		
		for a, asset in enumerate(participant_all.assets_nd):
			if asset.LoG=='gen':
				prob.add_constraint(p_curt[:, a] <= 0) 
				prob.add_constraint(p_curt[:, a] >= np.minimum(asset.curt*asset.mpc_demand(self.t_ahead_0),0) )
			else: 
				prob.add_constraint(p_curt[:, a] >= 0) 
				prob.add_constraint(p_curt[:, a] <= np.maximum(asset.curt*asset.mpc_demand(self.t_ahead_0),0) ) 	
			
		# min/max import/export	
		Pimpmin = prob.add_constraint( Pimp >= 0)
		Pexpmin = prob.add_constraint( Pexp >= 0)
		if not(np.all(np.isinf(self.P_import))):
			prob.add_constraint( Pimp <= self.P_import[self.t_ahead_0:]) 
		if not(np.all(np.isinf(self.P_export))):
			prob.add_constraint( Pexp <=-self.P_export[self.t_ahead_0:])
		
		# set objective
		prob.set_objective('min', sum(self.dt_market*self.price_imp[self.t_ahead_0+t]*Pimp[t] - self.dt_market*self.price_exp[t+self.t_ahead_0]*Pexp[t] for t in range(self.T_market-self.t_ahead_0) )\
    	                         +sum(self.dt_market*participant_all.assets_flex[i].c_deg_lin\
    	                       	*(x[t, i] - x[t+self.T_market-self.t_ahead_0, i]) for i in range(len(participant_all.assets_flex)) for t in range(self.T_market-self.t_ahead_0) ) \
    				            #+ (vmin_soft)*1e3
    				      )

		print('Solving the program...')
		prob.solve(solver='mosek', mosek_params={'MSK_IPAR_INFEAS_REPORT_AUTO': "MSK_ON"}, verbosity=1)
		print('Optimisation status:', prob.status, prob.value) 
		
        ################################# Start DLMP computation  #######################################################################
		if self.nw_const:
			print('* Computing clearing prices ...')
			
			N_load_bus_phases = self.network.N_phases*(self.network.N_buses-1)

			#Get constraint dual variables
			dual_P_bus = np.zeros([self.T_market-self.t_ahead_0, len(buses)])
			dual_P_slack = np.zeros([self.T_market-self.t_ahead_0])
			dual_vbus_max = np.zeros([self.T_market-self.t_ahead_0,N_load_bus_phases])
			dual_vbus_min = np.zeros([self.T_market-self.t_ahead_0,N_load_bus_phases])
			dual_iline_max = np.zeros([self.T_market-self.t_ahead_0, A_line_list[0].shape[0]])

			const_index=0
			for t in range(self.T_market-self.t_ahead_0):
				for b in range(len(buses)):
					dual_P_bus[t,b]= np.squeeze(prob.get_constraint((const_index,b)).dual)
				const_index+=1

			for t in range(self.T_market-self.t_ahead_0):
				dual_P_slack[t]= np.squeeze(prob.get_constraint(const_index).dual)
				const_index += 1

				for bus_ph_index in range(N_load_bus_phases):
					if int(bus_ph_index/3) not in v_unconstrained_buses:
						dual_vbus_max[t,bus_ph_index] = np.squeeze(prob.get_constraint(const_index).dual)
						const_index += 1
						dual_vbus_min[t,bus_ph_index] = np.squeeze(prob.get_constraint(const_index).dual)
						const_index+=1

				for line_ij in range(self.network.N_lines):
					if line_ij not in i_unconstrained_lines:	
						for ph in range(self.network.N_phases):
							dual_iline_max[t, line_ij+ph] = np.squeeze(prob.get_constraint(const_index).dual)
							const_index += 1

			DLMP = np.zeros((self.T_market-self.t_ahead_0, len(buses)))
			for t in range(self.T_market-self.t_ahead_0):
				for b, bus in enumerate(buses):
					DLMP[t,b] = 1/self.dt_market*(
						        dual_P_bus[t,b]
					          + 1e-3*dual_P_slack[t]*np.nan_to_num(A_Pslack_list[t][b])
							  + np.dot( np.asarray([dual_vbus_max[t,:] - dual_vbus_min[t,:]]) ,np.nan_to_num(A_vlim_list[t][:, b][:, np.newaxis]))
							  + np.dot( np.asarray([dual_iline_max[t,:]]) , np.nan_to_num(A_line_list[t][:, 3][:, np.newaxis]) )
							  )

			info = {'AP': A_Pslack_list,
					'AV': A_vlim_list,
					'AL': A_line_list,
					'dual_P_bus': dual_P_bus,
					'dual_P_slack': dual_P_slack,
					'dual_vbus_max': dual_vbus_max,
					'dual_vbus_min': dual_vbus_min,
					'dual_iline_max': dual_iline_max,
					'dlmp': DLMP
			       }
		###################################### END DLMP computation ############################################################

		if len(participant_all.assets_flex):
			x = np.array(x.value)				
		if len(participant_all.assets_nd):
			p_curt = np.array(p_curt.value)			
		
		print('* Establishing clearing outcome & Updating resources for assets ...')
		list_clearing = []
		nd, flex = 0, 0
		schedules = [[] for _ in range(len(self.participants))]
		for p_idx, par in enumerate(self.participants):
			print('*** Updating resources for assets | participant {}...'.format(par.p_id))
			for asset in par.assets:
				bus_id = asset.bus_id
				bus = buses.index(bus_id)
				if self.nw_const:
					price = DLMP[:, bus] 

				if asset.type == 'ND':			
					asset.update_ems(p_curt[:, nd], self.t_ahead_0)	
					schedules[p_idx].append(asset.Pnet_ems[self.t_ahead_0:])
					sch = asset.Pnet_ems[self.t_ahead_0:]
					nd +=1
				else:
					if asset.type == 'storage':
						asset.update_ems(x[:self.T_market-self.t_ahead_0, flex]+ x[self.T_market-self.t_ahead_0:, flex], self.t_ahead_0, enforce_const=False)
					else:
						asset.update_ems(x[:self.T_market-self.t_ahead_0, flex]- x[self.T_market-self.t_ahead_0:, flex], self.t_ahead_0, enforce_const=False)

					schedules[p_idx].append(asset.Pnet_ems[self.t_ahead_0:])
					sch = asset.Pnet_ems[self.t_ahead_0:]
					flex +=1

				print('*** Adding asset outcome for participant {} to clearing ...'.format(par.p_id))
				for t in range(self.T_market-self.t_ahead_0):
					if  sch[t]>0:
						if not self.nw_const:
							 price = self.price_imp
						list_clearing.append([t+self.t_ahead_0, 0, par.p_id, self.dt_market*sch[t], price[t]])
					elif sch[t]<0:
						if not self.nw_const:
							price = self.price_exp
						list_clearing.append([t+self.t_ahead_0, par.p_id, 0, -self.dt_market*sch[t], price[t]])

				
		market_clearing_outcome=pd.DataFrame(list_clearing, columns=['time', 'seller', 'buyer', 'energy', 'price'])
		market_clearing_outcome = market_clearing_outcome.groupby(['time', 'seller', 'buyer', 'price'])['energy'].aggregate('sum').reset_index()

		return market_clearing_outcome, schedules, info


class ToU_market(Market):
	"""
	Time of Use Market Class

	ToU market runs a decentralised optimisation in which every participant optimises its resources
	in response to a ToU price signal
	
	Parameters
	----------
	participants : list of objects
		Containing details of each participant
	T_market : int
		Market horizon
	dt_market : float
		time interval duration (hours)
	price_imp : numpy.ndarray
		import prices from the grid (£/kWh)
	t_ahead_0: int
		starting time slot of market clearing
	P_import : numpy.ndarray
		max import from the grid (kW)
	P_export : numpy.ndarray
		max export to the grid (kW)
	price_exp : numpy.ndarray
		export prices to the grid (£/kWh)
	network : object, default=None
		the network infrastructure of the market

		useful for market clearings that account for network constraints,

		required to run simulate_network_3ph
	
	Returns
	-------
	Market object
	"""
	def __init__(self, participants, dt_market, T_market, price_imp,\
				 t_ahead_0=0, P_import=None, P_export=None, price_exp=None, network=None):
		
		Market.__init__(self, participants, dt_market, T_market, price_imp, t_ahead_0=t_ahead_0, P_import=P_import, P_export=P_export, price_exp=price_exp, network=network)
		if np.all(self.P_export) ==None:
			self.P_export = -np.inf*np.ones(self.T_market)
		if np.all(self.P_import) ==None:
			self.P_import = np.inf*np.ones(self.T_market)

	def mpc_market_clearing(self):
		"""
		computes the MPC version of the energy exchange of each participant with the grid

		Returns
		-------
		market_clearing_outcome : pandas Dataframe
            		the resulting energy exchange
            
            		+----+------+--------+-------+--------+-------+
  			| id | time | seller | buyer | energy | price |
  			+====+======+========+=======+========+=======+
  			|    |	    |	     |	     |        |	      |
			+----+------+--------+-------+--------+-------+
        	schedules : list of lists
	    		assets schedules
		"""

		market_clearing = pd.DataFrame(columns=['time', 'seller', 'buyer', 'energy', 'price'])
		for t in range(self.T_market):
			self.t_ahead_0 = t
			mc, schedules= self.market_clearing()
			mc_t = mc[mc['time']==t]
			market_clearing = pd.concat([market_clearing, mc_t], ignore_index=True)
		return market_clearing, schedules

	def market_clearing(self):
		"""
		computes the energy exchange of each participant with the grid

		Returns
		-------
		market_clearing_outcome : pandas Dataframe
			the resulting energy exchange

      			+----+------+--------+-------+--------+-------+
  			| id | time | seller | buyer | energy | price |
  			+====+======+========+=======+========+=======+
  			|    |	    |	     |	     |        |	      |
			+----+------+--------+-------+--------+-------+
		schedules : list of numpy.ndarrays
			list of assets schedules
		"""
			
		list_clearing, schedules, outputs = [], [], []
		for par in self.participants:
			print('Run EMS for Participant: ' + str(par.p_id) )
			schedule, pimp, pexp, buses = par.EMS(self.price_imp, self.P_import, self.P_export, self.price_exp, t_ahead_0=self.t_ahead_0)#, network=self.network)
			schedules.append(schedule)

			print('* Adding  participant {} trades to clearing outcome...'.format(par.p_id))
			for t in range(self.T_market-self.t_ahead_0):
				for b in range(len(buses)):
					if pimp[t, b] >0:
						list_clearing.append([t+self.t_ahead_0, 0, par.p_id, self.dt_market*pimp[t,b], self.price_imp[t+self.t_ahead_0, buses[b]] ])
					elif pexp[t,b] >0:
						list_clearing.append([t+self.t_ahead_0, par.p_id, 0, self.dt_market*pexp[t,b], self.price_exp[t+self.t_ahead_0, buses[b]] ])
			

		market_clearing_outcome=pd.DataFrame(data=list_clearing, columns=['time', 'seller', 'buyer', 'energy', 'price'])
		return market_clearing_outcome, schedules

class P2P_market(Market):
	"""
	P2P market returns the bilateral contracts between peers following the algorithm proposed by 
	Morstyn et. Al, in [2]_
	
	Parameters
	------------
	participants : list of participant instances
		Containing details of each participant
	T_market : int
		Market horizon
	dt_market : float
		time interval duration (hours)
	price_imp : numpy.ndarray
		import prices from the grid (£/kWh)
	t_ahead_0: int
		starting time slot of market clearing
	P_import : numpy.ndarray
		max import from the grid (kW)
	P_export : numpy.ndarray
		max export to the grid (kW)
	price_exp : numpy.ndarray
		export prices to the grid (£/kWh)
	network : object, default=None
		the network infrastructure of the market

		useful for market clearings that account for network constraints,

		required to run simulate_network_3ph
	fees : numpy.ndarray (``T_market-t_ahead_0``, N, N)
		the fees the peers have to pay to the DNO for using the physical infrastructure
		
		fees[t,i,j]: the fees of transferring energy between participant i and participant j at time t
	
	"""

	def __init__(self, participants, dt_market, T_market, price_imp, t_ahead_0=0, P_import=None, P_export=None, price_exp=None, network=None, fees=None):
		Market.__init__(self, participants, dt_market, T_market, price_imp, t_ahead_0=t_ahead_0, P_import=P_import, P_export=P_export, price_exp=price_exp, network=network)
		if fees == None:
			self.fees = np.zeros((self.T_market, price_imp.shape[1], price_imp.shape[1]))
		else: self.fees = fees

		################################### prepare the offers ledger here for each group###########################
		####                            ----------------------------------------------
		####                             id | time | participant | energy | price |
		####                             ----------------------------------------------
		####                            |   |      |             |        |       |
		#############################################################################################################
		list_offers = []
		for participant in self.participants:
			####shouldn't we turn Pnet, Emax,etc from T_ems scale to T_market scale?
	   		## timescale(participant.Pnet, participant.dt_ems, dt_market) 
			for t in range(self.t_ahead_0, self.T_market):
				#energy_buy = max(participant.Pnet_ems[t],0) + min(participant.Pmax[t], participant.Emax[t]-participant.E_ems[t])  #min(Pmax, Emax-E)
				#energy_sell = max(-participant.Pnet_ems[t],0) + min(participant.Pmax[t], participant.E_ems[t]-participant.Emin[t]) #min(Pmax, E-Emin)
				energy_buy = max(participant.Pnet_ems[t] + participant.Pmax[t],0) 
				energy_sell = min(participant.Pnet_ems[t] + participant.Pmin[t],0) 
				if energy_buy !=0:
					list_offers.append([t, participant.p_id, energy_buy, 0]) #self.price_exp[t+self.t_ahead_0, participant.p_id-1]
				elif energy_sell != 0:
					list_offers.append([t, participant.p_id, energy_sell, 0]) #-energy_sell?? to check!

		self.offers = pd.DataFrame(data=list_offers, columns=['time', 'participant', 'energy', 'price'])

		self.buyer_indexes, self.seller_indexes = [], [] 
		for i in range(len(self.offers)):
			if self.offers['energy'][i]> 0:
				self.buyer_indexes.append(self.offers['participant'][i])
			else: self.seller_indexes.append(self.offers['participant'][i])

		self.buyer_indexes = list(set(self.buyer_indexes))
		self.seller_indexes = list(set(self.seller_indexes))

	def mpc_P2P_negotiation(self, trade_energy, price_inc, N_p2p_ahead_max, stopping_criterion=None):
		"""
        	Runs the P2P in a receding horizon and returns the outcome of a P2P negotiation procedure

        	Parameters
        	----------
        	trade_energy : float
        		the unit amount of energy to trade (kWh)
        	price_inc: float
        		the incremental step of the peers prices
        	N_p2p_ahead_max: int
        		maximum number of contracts between 2 peers
        	stopping_criterion: int
        		number of iterations that the negociation will make before stopping

        	Returns
        	--------
        	market_clearing_outcome : pandas Dataframe
            		the resulting energy exchange
            
            		+----+------+--------+-------+--------+-------+
  			| id | time | seller | buyer | energy | price |
  			+====+======+========+=======+========+=======+
  			|    |	    |	     |	     |        |	      |
			+----+------+--------+-------+--------+-------+
        	schedules: list of lists
	    		assets schedules
        	""" 
        
		market_clearing = pd.DataFrame(columns=['time', 'seller', 'buyer', 'energy', 'price'])
		for t in range(self.T_market):
			self.t_ahead_0 = t
			mc, schedules= self.P2P_negotiation(self, trade_energy, price_inc, N_p2p_ahead_max, stopping_criterion=stopping_criterion)
			mc_t = mc[mc['time']==t]
			market_clearing = pd.concat([market_clearing, mc_t], ignore_index=True)
		return market_clearing, schedules

	def P2P_negotiation(self, trade_energy, price_inc, N_p2p_ahead_max, stopping_criterion=None): 
		"""
		Returns the outcome of a P2P negotiation procedure

		Parameters
		----------
		trade_energy : float
			the unit amount of energy to trade (kWh)
		price_inc: float
			the incremental step of the peers prices
		N_p2p_ahead_max: int
			maximum number of contracts between 2 peers
		stopping_criterion: int
			number of iterations that the negotiation will make before stopping

		Returns
		--------
		market_clearing_outcome : pandas Dataframe
		    the resulting energy exchange

	            +----+------+--------+-------+--------+-------+
  		    | id | time | seller | buyer | energy | price |
  		    +====+======+========+=======+========+=======+
  		    |    |	|	 |	 |        |       |
		    +----+------+--------+-------+--------+-------+
		schedules: list of numpy.ndarrays
			list of assets schedules
		"""

        ################
		### Setup Trades
		################
		trades_id_col = 0
		trades_seller_col = 1
		trades_buyer_col = 2
		trades_sell_price_col = 3
		trades_buy_price_col = 4
		trades_time_col = 5
		trades_energy_col = 6
		trades_tcost_col = 7
		trades_optimal_col = 8

		################
		### Day Ahead Trading
		################    
		print('**********************************************')
		print('* Setting Up Ahead Trades')
		print('**********************************************')

		trade_list_ems = np.empty((0, 9)) #X the set of potential bilateral contracts
	
		trade_index = 0
		N_trades_ahead_s2b = np.zeros((self.T_market,len(self.participants),len(self.participants)), dtype=int)
		for t in range(self.t_ahead_0,self.T_market):
			#print('Interval: ' + str(t) + ' of ' + str(self.T_market))
			for b,buyer_id in enumerate(self.buyer_indexes):
				for s, seller_id in enumerate(self.seller_indexes): 
						offer_sell = self.offers.loc[(self.offers['participant']==seller_id) & (self.offers['time']==t) & (self.offers['energy']<=0), 'energy']
						offer_buyer = self.offers.loc[(self.offers['participant']==buyer_id) & (self.offers['time']==t) & (self.offers['energy']>=0), 'energy']
						if offer_sell.size==0:
							offer_sell = pd.Series([0])
						if offer_buyer.size==0:
							offer_buyer = pd.Series([0])
						N_trades_ahead_s2b[t,s,b] = min(np.abs(np.ceil(offer_sell.values[0]/(trade_energy/self.dt_market))), \
							                                   np.ceil(offer_buyer.values[0]/(trade_energy/self.dt_market)))
						                                              
						N_trades_ahead_s2b[t,s,b] = min(N_trades_ahead_s2b[t,s,b],N_p2p_ahead_max)
						for trade_i in range(N_trades_ahead_s2b[t,s,b]):
							tcost = self.fees[t, seller_id-1, buyer_id-1]
							trade_list_ems = np.append(trade_list_ems,[[trade_index,seller_id,buyer_id,self.offers['price'][seller_id],self.offers['price'][buyer_id],t,trade_energy,tcost, 0]],axis = 0)
							trade_index +=1

		N_trades = trade_index

		par_pref_output = [None]*len(self.participants)
		done = False
		iterations =0
		while not done:
			print('*********************************************\n*\n*\n*')
			print('* Day Ahead Trading Price Adjustment: ')
			print('* Iteration: ' + str(iterations) )
			print('*\n*\n*\n*********************************************')
			done = True
			for i in range(len(self.participants)):
				print('Iter.: ' + str(iterations) + ' | Participant: ' + str(self.participants[i].p_id) )
				par_pref_output[i] = self._get_trades(self.participants[i], trade_list_ems)
		    	
			q_ds_full_list = np.zeros(N_trades)
			q_us_full_list = np.zeros(N_trades)
			q_ds_full_list[:] = np.sum(np.array(par_pref_output[i]['q_ds_full_list'])[:,0] for i in range(len(self.participants)))
			q_us_full_list[:] = np.sum(np.array(par_pref_output[i]['q_us_full_list'])[:,0] for i in range(len(self.participants)))

			for trade_idx in range(N_trades):
				if q_us_full_list[trade_idx] > q_ds_full_list[trade_idx]:
					done = False
					if trade_list_ems[trade_idx,trades_buy_price_col] > trade_list_ems[trade_idx,trades_sell_price_col]:
						trade_list_ems[trade_idx,trades_sell_price_col] += price_inc #[int(trade_list_ems[trade_idx,trades_time_col])]
					else:
						trade_list_ems[trade_idx,trades_buy_price_col] += price_inc #[int(trade_list_ems[trade_idx,trades_time_col])]
			iterations+=1

			if stopping_criterion!=None and iterations>=stopping_criterion:
				mask = np.logical_or(np.isin(trade_list_ems[:,trades_seller_col],p2p_group_agent_indexes), np.isin(trade_list_ems[:,trades_buyer_col],p2p_group_agent_indexes))
				trade_list_ems[mask, trades_tcost_col] = 1e5
		
		print('End negotiation')

		q_ds_full_ems = np.zeros(N_trades)
		q_us_full_ems = np.zeros(N_trades)
		par_pref_output_final = [None]*len(self.participants)
		list_clearing=[]
		schedules = []
		for i in range(len(self.participants)):
			print('Final setup  | Participant: ' + str(self.participants[i].p_id) )
			q_ds_full_ems[:] += np.array(par_pref_output[i]['q_ds_full_list'])[:,0]
			q_us_full_ems[:] += np.array(par_pref_output[i]['q_us_full_list'])[:,0]
			par_pref_output_final[i] = self._get_trades(self.participants[i], trade_list_ems, q_trades_req=q_us_full_ems)

			##### prepare clearing outcome
			######## add trading with main grid
			for t in range(self.T_market-self.t_ahead_0):
				if par_pref_output_final[i]['q_sup_sell'][t]: 
					list_clearing.append([self.t_ahead_0+t, self.participants[i].p_id, 0, par_pref_output_final[i]['q_sup_sell'][t], self.price_exp[t+self.t_ahead_0, self.participants[i].p_id-1]])
				elif par_pref_output_final[i]['q_sup_buy'][t]:
					list_clearing.append([self.t_ahead_0+t, 0, self.participants[i].p_id, par_pref_output_final[i]['q_sup_buy'][t], self.price_imp[t+self.t_ahead_0, self.participants[i].p_id-1]])

			#print('q sell:', par_pref_output_final[i]['q_sup_sell'])
			#print('q buy:', par_pref_output_final[i]['q_sup_buy'])
			###### update resources
			print('*********************************************\n*\n*\n*')
			print('* Updating resources... ')
			print('*\n*\n*\n*********************************************')
			nd, d = 0,0
			schedule = []
			for asset in self.participants[i].assets:
				if asset.type == 'storage':
					asset.update_ems(par_pref_output_final[i]['p_flex'][:self.T_market-self.t_ahead_0, d] + par_pref_output_final[i]['p_flex'][self.T_market-self.t_ahead_0:, d], self.t_ahead_0, enforce_const=False)
					schedule.append(asset.Pnet_ems[self.t_ahead_0:])
					d+=1
				elif asset.type == 'building':
					asset.update_ems(par_pref_output_final[i]['p_flex'][:self.T_market-self.t_ahead_0, d] - par_pref_output_final[i]['p_flex'][self.T_market-self.t_ahead_0:, d], self.t_ahead_0, enforce_const=False)
					schedule.append(asset.Pnet_ems[self.t_ahead_0:])
					d+=1
				else:
					asset.update_ems(par_pref_output_final[i]['p_curt'][:, nd], self.t_ahead_0)	
					schedule.append(asset.Pnet_ems[self.t_ahead_0:])
					#asset.Pnet_ems[self.t_ahead_0:] = asset.Pnet_ems[self.t_ahead_0:] - par_pref_output_final[i]['p_curt'][:, nd] 
					nd+=1
			schedules.append(schedule)

			### populate the last column of trade_list_ems to distinquish btw accepted and rejected trades
			trade_list_ems_b = trade_list_ems[trade_list_ems[:,trades_buyer_col]== self.participants[i].p_id]
			if len(trade_list_ems_b):
				trade_list_ems_b[:,trades_optimal_col] = np.array(par_pref_output_final[i]['q_us']).reshape(-1)
				trade_list_ems[trade_list_ems[:,trades_buyer_col]== self.participants[i].p_id] = trade_list_ems_b 

		##### continue clearing outcome
	    ####### add trading with other peers
		print('*********************************************\n*\n*\n*')
		print('* Preparing clearing outcome: ')
		print('*\n*\n*\n*********************************************')
		for trade_idx in range(len(trade_list_ems)):
			if trade_list_ems[trade_idx, trades_optimal_col]!=0:  #if trade accepted, how to verify this????
				list_clearing.append([trade_list_ems[trade_idx, trades_time_col], 
				                     trade_list_ems[trade_idx, trades_seller_col],
				                     trade_list_ems[trade_idx, trades_buyer_col],
				                     trade_list_ems[trade_idx, trades_energy_col],
				                     trade_list_ems[trade_idx, trades_sell_price_col]
				                    ])
			
		market_clearing_outcome=pd.DataFrame(data=list_clearing, columns=['time', 'seller', 'buyer', 'energy', 'price'])
			
		return market_clearing_outcome, schedules 


	def _get_trades(self,participant, trade_list, q_trades_req=None):  
		T_dec = self.T_market - self.t_ahead_0
		P_net_ems  = participant.nd_demand(self.t_ahead_0)

		#######################################
		### Get trade information and indexes
		#######################################
		trades_id_col = 0
		trades_seller_col = 1
		trades_buyer_col = 2
		trades_sell_price_col = 3
		trades_buy_price_col = 4
		trades_time_col = 5
		trades_energy_col = 6
		trades_tcost_col = 7
		trades_optimal_col = 8

		#US trades:
		trades_us = trade_list[np.where((trade_list[:,trades_buyer_col]== participant.p_id).astype(bool))]
		N_trades_us = len(trades_us)
		#ids_us = trades_us[:,trades_id_col]
		times_us = trades_us[:,trades_time_col]   
		prices_us = trades_us[:,trades_buy_price_col]
		trade_energy_us = trades_us[:,trades_energy_col]
		#if tcosts_flag:
		trade_tcost_us = trades_us[:,trades_tcost_col]
		us_t_mat = np.zeros([T_dec,N_trades_us])
		for t in range(T_dec):
			us_t_mat[t,:] = (times_us == t+self.t_ahead_0)
		#DS trades:
		trades_ds = trade_list[np.where((trade_list[:,trades_seller_col]== participant.p_id).astype(bool))]
		N_trades_ds = len(trades_ds)
		ids_ds = trades_ds[:,trades_id_col]
		times_ds = trades_ds[:,trades_time_col]
		prices_ds = trades_ds[:,trades_sell_price_col]
		trade_energy_ds = trades_ds[:,trades_energy_col]
		#if tcosts_flag:
		trade_tcost_ds = trades_ds[:,trades_tcost_col]
		#matrixes with indexes of trades for each time interval
		ds_t_mat = np.zeros([T_dec,N_trades_ds])
		for t in range(T_dec):
			ds_t_mat[t,:] = (times_ds == t+self.t_ahead_0)
		#Ensure trades us and ds are not zero            
		if  N_trades_us == 0:
			N_trades_us = 1
			ids_us = np.ones(1)
			prices_us = np.zeros(1)
			times_us = np.zeros(1)
			trade_energy_us = np.zeros(1)
			us_t_mat = np.zeros([T_dec,N_trades_us])
			trade_tcost_us = np.zeros(1)
		if  N_trades_ds == 0:
			N_trades_ds = 1
			ids_ds = np.ones(1)
			prices_ds = np.zeros(1)
			times_ds = np.zeros(1)
			trade_energy_ds = np.zeros(1)
			ds_t_mat = np.zeros([T_dec,N_trades_ds])
			trade_tcost_ds = np.zeros(1)
		#######################################
		### Set up optimisation problem and decision variables
		#######################################
		prob = pic.Problem()
		q_us = pic.RealVariable('q_us',N_trades_us)
		q_ds = pic.RealVariable('q_ds',N_trades_ds)
		q_us_bin = pic.BinaryVariable('q_us_bin',N_trades_us)
		q_ds_bin = pic.BinaryVariable('q_ds_bin',N_trades_ds)
		p_net_exp = pic.RealVariable('p_net_exp',T_dec)
		q_sup_buy = pic.RealVariable('q_sup_buy',T_dec)
		q_sup_sell = pic.RealVariable('q_sup_sell',T_dec)

		if len(participant.assets_flex):
			p_flex = pic.RealVariable('p_flex', (2*T_dec, len(participant.assets_flex)))
		else:
			p_flex = pic.new_param('p_flex', np.zeros((2*T_dec,2)))
		if len(participant.assets_nd):
			p_curt = pic.RealVariable('p_curt',(T_dec, len(participant.assets_nd)))
		else:
			p_curt = np.zeros((T_dec,2))

		#######################################
		### Add constraints
		#######################################
		
		#p_net matches trade quanitites
		prob.add_list_of_constraints([p_net_exp[t] == (1/self.dt_market)*(q_sup_sell[t] - q_sup_buy[t] + q_ds.T*ds_t_mat[t,:] - q_us.T*us_t_mat[t,:]) for t in range(T_dec)])
		# 0 kWh <= trade quantities <= trade_energy (kWh)
		
		if q_trades_req is None:
			prob.add_list_of_constraints([q_ds[trd] == q_ds_bin[trd]*trade_energy_ds[trd] for trd in range(N_trades_ds)])
			prob.add_list_of_constraints([q_us[trd] == q_us_bin[trd]*trade_energy_us[trd] for trd in range(N_trades_us)])
		else:
			q_trades_req_us = q_trades_req[np.where((trade_list[:,trades_buyer_col]== participant.p_id).astype(bool))]
			q_trades_req_ds = q_trades_req[np.where((trade_list[:,trades_seller_col]== participant.p_id).astype(bool))]
			if len(q_trades_req_ds) > 0:
				prob.add_list_of_constraints([q_ds[trd] == q_trades_req_ds[trd] for trd in range(N_trades_ds)])
			if len(q_trades_req_us) > 0:
				prob.add_list_of_constraints([q_us[trd] == q_trades_req_us[trd] for trd in range(N_trades_us)])
					
		#energy bought/sold from supplier
		prob.add_constraint(q_sup_buy  >= 0 )
		prob.add_constraint(q_sup_sell >= 0 )
			
		#flexible load constraints
		if len(participant.assets_flex):
			for a, asset in enumerate(participant.assets_flex):
				A, b = asset.polytope(self.t_ahead_0)
				prob.add_constraint(A*p_flex[:,a] <= b)
		
		#curtailing non dispatchable load
		if len(participant.assets_nd):
			for a, asset in enumerate(participant.assets_nd):
				if np.all(asset.Pnet_ems)<=0:
					prob.add_constraint(p_curt[:, a] <= 0) 
					prob.add_constraint(p_curt[:, a] >= np.minimum(asset.curt*asset.mpc_demand(self.t_ahead_0),0))

				elif np.all(asset.Pnet_ems)>=0:
					prob.add_constraint(p_curt[:, a] >= 0) 
					prob.add_constraint(p_curt[:, a] <= np.maximum(asset.curt*asset.mpc_demand(self.t_ahead_0),0) )
		
		#net power constraints
		prob.add_constraint(p_net_exp == sum(p_curt[:,a] for a in range(p_curt.shape[1]))
		                   - sum(p_flex[:self.T_market-self.t_ahead_0, i] + p_flex[self.T_market-self.t_ahead_0:, i] for i in range(p_flex.shape[1])) 
		                    - sum(P_net_ems[:, i] for i in range(P_net_ems.shape[1])) 
		                   )
		
		#######################################
		### Objective Function
		#######################################
		
		if len(participant.assets_flex):
			prob.set_objective('min',    sum(self.price_imp[t+self.t_ahead_0, participant.p_id-1]*q_sup_buy[t] 
				                          - self.price_exp[t+self.t_ahead_0, participant.p_id-1]*q_sup_sell[t]  for t in range(T_dec)
				                            )\
				                           # 
			                            + sum((prices_us[trd]+ trade_tcost_us[trd]/2)*q_us[trd]  for trd in range(N_trades_us))\
			                            + sum((-prices_ds[trd]+ trade_tcost_ds[trd]/2)*q_ds[trd]  for trd in range(N_trades_ds))\
			                            + sum(self.dt_market*participant.assets_flex[i].c_deg_lin\
	        	                       	*(p_flex[t, i] - p_flex[t+self.T_market-self.t_ahead_0, i]) for i in range(p_flex.shape[1]) for t in range(self.T_market-self.t_ahead_0))
			                    )
		else:
			prob.set_objective('min',    sum(self.price_imp[t+self.t_ahead_0, participant.p_id-1]*q_sup_buy[t] 
				                          - self.price_exp[t+self.t_ahead_0, participant.p_id-1]*q_sup_sell[t]  for t in range(T_dec)
				                            )\
										#
			                            + sum((prices_us[trd]+ trade_tcost_us[trd]/2)*q_us[trd]  for trd in range(N_trades_us))\
			                            + sum((-prices_ds[trd]+ trade_tcost_ds[trd]/2)*q_ds[trd]  for trd in range(N_trades_ds))\
			          
			                    )
		
		prob.solve(solver='mosek') #,mosek_params={'MSK_IPAR_INFEAS_REPORT_AUTO': "MSK_ON"}, verbosity=2)#, primals=None) 
		print('P2P optimisation:', prob.status)
		
		q_us_full_list = np.zeros([len(trade_list),1])
		q_ds_full_list = np.zeros([len(trade_list),1])
		q_us_full_list[np.where((trade_list[:,trades_buyer_col]== participant.p_id).astype(bool))] = q_us.value
		q_ds_full_list[np.where((trade_list[:,trades_seller_col]== participant.p_id).astype(bool))] = q_ds.value

		if len(participant.assets_nd):
			p_curt_val = np.array(p_curt.value)
		else: p_curt_val = None

		if len(participant.assets_flex):
			p_flex_val = np.array(p_flex.value)
		else: p_flex_val = None

		#print('q_us', q_us.value, 'q_ds', q_ds.value, 'p_flex', p_flex_val, 'q_sup_sell', q_sup_sell.value, 'q_sup_buy', q_sup_buy.value) #'p_curt', p_curt_val, 
		return {'opt_prob':prob,\
		        'q_us':q_us.value,\
		        'q_ds':q_ds.value,\
		        'q_us_full_list':q_us_full_list,\
		        'q_ds_full_list':q_ds_full_list,\
		        'p_net_exp':p_net_exp.value,\
		        'q_sup_sell':q_sup_sell.value,\
		        'q_sup_buy':q_sup_buy.value,\
		        'p_flex': p_flex_val,
		        'p_curt': p_curt_val
		        }


class Auction_market(Market):
	"""
	Auction Market Class

	The Auction market matches buyers and sellers 
	
	Parameters
	----------
	participants : list of objects
		Containing details of each participant
	T_market : int
		Market horizon
	dt_market : float
		time interval duration (hours)
	price_imp : numpy.ndarray
		import prices from the grid (£/kWh)
	t_ahead_0: int
		starting time slot of market clearing
	P_import : numpy.ndarray
		max import from the grid (kW)
	P_export : numpy.ndarray
		max export to the grid (kW)
	price_exp : numpy.ndarray
		export prices to the grid (£/kWh)
	network : object, default=None
		the network infrastructure of the market

		useful for market clearings that account for network constraints,

		required to run simulate_network_3ph
	offers: pandas Dataframe
		id_participant, quantity of energy, price

  		    +----------------+------+--------+-------+
  		    | id_participant | time | energy | price |
  		    +================+======+========+=======+
  		    |   	     |	    |        |       |
		    +----------------+------+-------+--------+
	
	Returns
	-------
	Market object
	"""
	def __init__(self, participants, offers, dt_market, T_market, price_imp, t_ahead_0=0, P_import=None, P_export=None, price_exp=None, network=None):
		Market.__init__(self, participants, dt_market, T_market, price_imp, t_ahead_0=t_ahead_0, P_import=P_import, P_export=P_export, price_exp=price_exp, network=network)
		self.offers

		self.buyer_indexes, self.seller_indexes = [], [] 
		for i in range(len(self.offers)):
			if self.offers['energy'][i]> 0:
				self.buyer_indexes.append(self.offers['participant'][i])
			else: self.seller_indexes.append(self.offers['participant'][i])

		self.buyer_indexes = list(set(self.buyer_indexes))
		self.seller_indexes = list(set(self.seller_indexes))

	def Auction_matching(self, priority='price'): 
		"""
		Returns the outcome of an auction matching highest bid with lowest ask matching

		Parameters
		----------
		priority: {'price', 'demand'}, default=price
			'price': the buyer with the highest bid price is matched to the seller with the lowest offer price
			
			'demand': the buyer with the highest bid demand is matched to the seller with the highest offer surplus 
		
		Returns
		-------
		market_clearing_outcome: pandas Dataframe
			the resulting energy exchange

      			+----+------+--------+-------+--------+-------+
  		    	| id | time | seller | buyer | energy | price |
  		    	+====+======+========+=======+========+=======+
  		   	|    |	    |	     |	     |        |       |
		  	+----+------+--------+-------+--------+-------+

		"""
		
		list_clearing = []
		for t in self.offers['time']: #check if t in [t0_ahead, self.T_ems]?
			#insert upstream offers in the offers df
			Bids = self.offers[(self.offers['participant'].isin(self.buyer_indexes)) & (self.offers['time']==t)]
			Bids.append([t, 0, self.P_export, self.price_exp[t]])
			Asks = self.offers[(self.offers['participant'].isin(self.seller_indexes)) & (self.offers['time']==t)]
			Asks.append([t, 0, self.P_import, self.price_imp[t]])

			#match
			while (len(Bids) > 0 and len(Asks) > 0):
				if priority == 'price':
					Bids = sorted(Bids, key=lambda x: x.price)[::-1]
					Asks = sorted(Asks, key=lambda x: x.price)
				else:
					Bids = sorted(Bids, key=lambda x: x.energy)[::-1]
					Asks = sorted(Asks, key=lambda x: x.energy) #energy<0 so assorted ascently than [::-1]

				if Bids[0].price < Asks[0].price:
					break
				else:  # Bids[0].Price >= Asks[0].Price:
					currBid = Bids.pop()
					currAsk = Asks.pop()
					if currBid.energy != np.abs(currAsk.energy):
						if currBid.energy > currAsk.energy:
							newBid = {'time': t, 'participant': currBid.participant, 'energy': currBid.energy - np.abs(currAsk.energy), 'price': currBid.price}
							Bids.insert(0, newBid)
							#(currBid.price+currAsk.price+fees[t, currAsk.participant, currBid.participant])/2?
							list_clearing.append([t, currAsk.participant, currBid.participant, self.dt_market*np.abs(currAsk.energy), (currBid.price+currAsk.price)/2])
						else:
							newAsk = {'time': t, 'participant': currAsk.participant, 'energy': np.abs(currAsk.energy) - currBid.energy, 'price': currAsk.price}
							Asks.insert(0,newAsk)
							list_clearing.append([t, currAsk.participant, currBid.participant, self.dt_market*currBid.energy, (currBid.price+currAsk.price)/2])
		
		market_clearing_outcome=pd.DataFrame(data=list_clearing ,columns=['time', 'seller', 'buyer', 'energy', 'price'])

		return market_clearing_outcome

class Capacity_limits(Market):
	"""
	Returns the maximum capacity to absorb and inject per node per time step
	following the paper [put ref here]
	"""
	def __init__(self, participants, dt_market, T_market, price_imp, t_ahead_0=0, P_import=None, P_export=None, price_exp=None, network=None):
		Market.__init__(self, participants, dt_market, T_market, price_imp, t_ahead_0=t_ahead_0, P_import=P_import, P_export=P_export, price_exp=price_exp, network=network)
		### participants should have only assets with inflexible load, and one participant per node

	def capacity_limits(epsilon, Cfirm, Sigma):
		"""
		Returns the max capacities to absorb/inject per each node for each time step

		Parameters
		----------
		epsilon: float ]0.5, 1]
			the probabilistic error, probabilistic guarantee = 1-epsilon

			for a deterministic solution (with no probabilistic guarantee, take epsilon=0.5)
		Cfirm: float generally [15, 18]*1e3 kW
			firm capacity of the transfomer upstream
		Sigma: list of numpy.ndarray
			each element of the list stores the variance of the inflexible loads at time step t 

		Returns
		-------
		Cmax: numpy.ndarray (``T_market-t_ahead_0``, Nbr of nodes)
			max capacity to absorb: 
		Cmin: numpy.ndarray (``T_market-t_ahead_0``, Nbr of nodes)
			max capacity to inject  (<=0)
		"""
		
		assets_all = []
		for par in self.participants:
			for asset in par.assets:
				assets_all.append(asset)
		participant_all = Participant(len(self.participants+2), assets_all)
		P_demand = participant_all.nd_demand(self.t_ahead_0)

		N_loads=len(self.participants)
		# t in line 887
		## include current in A, B
		
		C_max, C_min = np.zeros((self.T_market-self.t_ahead_0, N_loads)), np.zeros((self.T_market-self.t_ahead_0, N_loads))
		for t in range(self.T_market-self.t_ahead_0):
			print('t=',t+self.t_ahead_0)
			####get linear parameters:
			A_Pslack, b_Pslack, A_vlim, b_vlim, v_abs_min_vec, v_abs_max_vec, _, _, _ = network.get_linear_parameters(assets_all, t+self.t_ahead_0)#################
			A_Pslack = np.expand_dims(A_Pslack, axis=0)     
			A = np.concatenate((A_Pslack, -A_Pslack, A_vlim*1e3, -A_vlim*1e3), axis=0) 
			B = np.concatenate( ([[Cfirm-(b_Pslack/1e3)]], [[Cfirm + (b_Pslack/1e3)]], v_abs_max_vec-np.expand_dims(b_vlim, axis=1), np.expand_dims(b_vlim, axis=1)-v_abs_min_vec), axis=0)

			"""
            A_clim, b_clim, i_abs_max_vec = np.array(A_ijlim[0]), np.array(b_ijlim[0]), np.array(i_abs_max[0]) #A, b current lim
            for ij in range(1,len(A_ijlim)):
                #stack all 
                A_clim = np.concatenate((A_clim, A_ijlim[ij]), axis=0)
                b_clim = np.concatenate((b_clim, b_ijlim[ij]), axis=0)
                i_abs_max_vec = np.concatenate((i_abs_max_vec, i_abs_max[ij]), axis=0)

            A = np.concatenate((A_Pslack, -A_Pslack, A_vlim*1e3, -A_vlim*1e3, A_clim*1e3), axis=0) 
            B= np.concatenate( ([[Cap-(b_Pslack/1e3)]], [[Cap + (b_Pslack/1e3)]], \
                                 v_abs_max_vec-np.expand_dims(b_vlim, axis=1), np.expand_dims(b_vlim, axis=1)-v_abs_min_vec), \
                                 i_abs_max_vec-np.expand_dims(b_clim, axis=1),
                                 axis=0)

            """
			sig = np.dot(np.dot(A,Sigma[t+self.t_ahead_0]),A.T)
			sigm = np.expand_dims(np.sqrt(sig.diagonal()), axis=1) 

			####Run stochastic optimisation
			p1 = pic.Problem()

			Cmax = pic.RealVariable("Cmax", N_loads)
			Cmaxtot = pic.RealVariable('Cmaxtot')
			p1.add_constraint(Cmax >= Cmaxtot)
			p1.add_constraint(Cmaxtot>=0)
			p1.add_constraint(A*Cmax <= B-np.expand_dims(np.dot(A,P_demand[t,:]), axis=1) - norm.ppf(1-eps)*sigm) #sigm #abs(np.expand_dims(np.dot(A,std[t]), axis=1)))
			p1.set_objective('max', Cmaxtot)
			p1.solve(solver='mosek', primals=None)

			if p1.status != 'optimal':
				print('p1 not optimal for t=',t+self.t_ahead_0)
			else:
				C_max[t, :] = np.array(Cmax.value)[:, 0]

			p2 = pic.Problem()

			Cmin = pic.RealVariable("Cmin", N_loads)
			Cmintot = pic.RealVariable('Cmintot') 
			p2.add_constraint(Cmin <= Cmintot)
			p2.add_constraint(A*Cmin <= B-np.expand_dims(np.dot(A,P_demand[t,:]), axis=1) - norm.ppf(1-eps)*sigm) #sigm #abs(np.expand_dims(np.dot(A,std[t]), axis=1)))
			p2.set_objective('min', Cmintot)

			p2.solve(solver='mosek', primals=None)
			if p2.status != 'optimal':
				print('p2 not optimal for t=',t+self.t_ahead_0)
			else:
				C_min[t, :] = np.array(Cmin.value)[:, 0]

		return C_max, C_min
