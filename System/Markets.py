#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPEN Market Module.

The OPEN Market class has two types of markets:
(1) Energy markets, and 
(2) Flexibility markets
The energy market comes with subclasses of the common types of energy markets:
(i) wholesale market
(ii) auction market
(iii) P2P market
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
import System.Participant as Participant

#import System.P2P_Trading as P2P  #should become System.Participant
def timescale(series, t_in, t_out):
	##### check if len(series)>1 ? ##### Tems Tout
	T_in, T_out = int(24/t_in), int(24/t_out)
	series_out = np.zeros((T_out,np.asarray(series).shape[1])) 

	if t_out <= t_in:
		for t in range(T_in):
			series_out[t*int(t_in/t_out) : (t+1)*int(t_in/t_out),:] = series[t,:]
	else:
		for t in range(T_out):
			series_out[t,:] = np.mean(series[t*int(t_out/t_in) : (t+1)*int(t_out/t_in),:], axis=0)
	
	series_out = np.nan_to_num(series_out)

class Market:
	def __init__(self, participants, dt_market, T_market, price_imp, t_ahead_0=0, P_import=None, P_export=None, price_exp=None):
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
		P_import: numpy.ndarray
			max import from the grid (kW)
		price_imp: numpy.ndarray
			import prices from the grid (£/kWh)
		P_export: numpy.ndarray
			max export to the grid (kW)
		price_exp: numpy.ndarray
			export prices to the grid (£/kWh)
	    Returns
	    -------
	    Market object
	    """
		self.participants = participants
		self.dt_market = dt_market
		self.T_market = T_market
		self.t_ahead_0 = t_ahead_0
		self.P_import = P_import
		self.price_imp = price_imp
		self.P_export = P_export
		self.price_exp = price_exp

		if self.P_export ==None and np.all(self.price_exp)==None:
			self.P_export=np.zeros(self.T_market)
			self.price_exp=np.zeros(self.T_market)
		if self.P_export ==None and np.all(self.price_exp)!=None:
			self.P_export=np.inf*np.ones(self.T_market)	
		if np.any(self.P_export) !=None and np.all(self.price_exp)==None:
			raise ValueError('Export power was set to be non-zero, but no values for the export prices were entered')
		if np.all(self.P_import)==None:
			self.P_import=np.inf*np.ones(self.T_market)

class P2P_market(Market):
	def __init__(self, participants, dt_market, T_market, price_imp, t_ahead_0=0, P_import=None, P_export=None, price_exp=None, fees=None):
		Market.__init__(self, participants, dt_market, T_market, price_imp, t_ahead_0=t_ahead_0, P_import=P_import, P_export=P_export, price_exp=price_exp)
		self.fees = fees

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
					list_offers.append([t, participant.p_id, energy_buy, 0])
				elif energy_sell != 0:
					list_offers.append([t, participant.p_id, energy_sell, 0]) #-energy_sell?? to check!

		self.offers = pd.DataFrame(data=list_offers, columns=['time', 'participant', 'energy', 'price'])

		#print('offers P2P:', self.offers[self.offers['energy']<0])

		self.buyer_indexes, self.seller_indexes = [], [] 
		for i in range(len(self.offers)):
			if self.offers['energy'][i]> 0:
				self.buyer_indexes.append(self.offers['participant'][i])
			else: self.seller_indexes.append(self.offers['participant'][i])

		self.buyer_indexes = list(set(self.buyer_indexes))
		self.seller_indexes = list(set(self.seller_indexes))
		#print('buyers:', self.buyer_indexes, 'and sellers:', self.seller_indexes)

	def P2P_negotiation(self, trade_energy, price_inc, N_p2p_ahead_max, stopping_criterion=None, curt=False): 
	#add here participants preferences: e.g., 
		"""
        Returns the outcome of a P2P negotiation procedure
        Parameters
        ----------
        fees: 3 dim np.array
			the fees the peers have to pay to the DNO for using the physical infrastructure
			fees[t,i,j]: the fees of transfering energy between participant i and participant j at time t
        *args : 
            contains the different parameters specific to each P2P model needed for the negotiation
			Examples: customer preferences, max_energy to trade with a peer, etc

        returns
        --------
        market_clearing_outcome: pandas Dataframe
            the resulting energy exchange
            ----------------------------------------------
             id | time | seller | buyer | energy | price |
            ----------------------------------------------
			|	|	   |		|		|		 |	     |
            ----------------------------------------------
            (*): the slack bus (upstream market) has the index 0
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
	
		#Add 1 trade between each PV generator and each flex load owner per time interval
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
							tcost = self.fees[t, seller_id, buyer_id]
							trade_list_ems = np.append(trade_list_ems,[[trade_index,seller_id,buyer_id,self.offers['price'][seller_id],self.offers['price'][buyer_id],t,trade_energy,tcost, 0]],axis = 0)
							trade_index +=1

		N_trades = trade_index

		print('trade_list_ems:', len(trade_list_ems))

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
				#par_pref_output[i] = self._get_preferred_trades_mi(self.participants[i], trade_list_ems, curt=True)
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
			#par_pref_output_final[i] = self._get_preferred_trades_mi(self.participants[i], trade_list_ems, q_trades_req=q_us_full_ems)
			par_pref_output_final[i] = self._get_trades(self.participants[i], trade_list_ems, q_trades_req=q_us_full_ems)

			##### prepare clearing outcome
			######## add trading with main grid
			for t in range(self.T_market-self.t_ahead_0):
				if par_pref_output_final[i]['q_sup_sell'][t]: 
					list_clearing.append([self.t_ahead_0+t, self.participants[i].p_id, 0, par_pref_output_final[i]['q_sup_sell'][t], self.price_exp[t+self.t_ahead_0, self.participants[i].p_id]])
				elif par_pref_output_final[i]['q_sup_buy'][t]:
					list_clearing.append([self.t_ahead_0+t, 0, self.participants[i].p_id, par_pref_output_final[i]['q_sup_buy'][t], self.price_imp[t+self.t_ahead_0, self.participants[i].p_id]])

			###### update resources
			print('*********************************************\n*\n*\n*')
			print('* Updating resources... ')
			print('*\n*\n*\n*********************************************')
			nd, d = 0,0
			schedule = []
			for asset in self.participants[i].assets:
				if asset.type != 'ND':
					asset.update_ems(par_pref_output_final[i]['p_flex'][:self.T_market-self.t_ahead_0, d]+ par_pref_output_final[i]['p_flex'][self.T_market-self.t_ahead_0:, d], self.t_ahead_0)
					schedule.append(asset.Pnet_ems[self.t_ahead_0:])
					d+=1
				else:
					schedule.append(asset.Pnet_ems[self.t_ahead_0:])# - par_pref_output_final[i]['p_curt'][:, nd]) 
					nd+=1
			schedules.append(schedule)

			### populate the last column of trade_list_ems to distinquish btw accepted and rejected trades
			trade_list_ems_b = trade_list_ems[trade_list_ems[:,trades_buyer_col]== i]
			print('trade_list_ems_b', len(trade_list_ems_b))
			print('output final qs', par_pref_output_final[i]['q_us'])
			trade_list_ems_b[:,trades_optimal_col] = np.array(par_pref_output_final[i]['q_us']).reshape(-1)
			trade_list_ems[trade_list_ems[:,trades_buyer_col]== i] = trade_list_ems_b 

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

				"""
				market_clearing_outcome['time'][idx] = trade_list_ems[trade_idx, trades_time_col]
				market_clearing_outcome['seller'][idx] = trade_list_ems[trade_idx, trades_seller_col]
				market_clearing_outcome['buyer'][idx] = trade_list_ems[trade_idx, trades_buyer_col]
				market_clearing_outcome['energy'][idx] = trade_list_ems[trade_idx, trades_energy_col]
				market_clearing_outcome['price'][idx] = trade_list_ems[trade_idx, trades_sell_price_col]
				"""
			
		market_clearing_outcome=pd.DataFrame(data=list_clearing, columns=['time', 'seller', 'buyer', 'energy', 'price'])
			
		return market_clearing_outcome, schedules #par_pref_output_final


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
		#print('us_t_mat:', us_t_mat)
		#DS trades:
		trades_ds = trade_list[np.where((trade_list[:,trades_seller_col]== participant.p_id).astype(bool))]
		#print('trades_ds', trades_ds)
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
		#print('ds_t_mat:', ds_t_mat)
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
		#if len(participant.assets_nd):
		#	p_curt = pic.RealVariable('p_curt',(T_dec, len(participant.assets_nd)))
		#else:
		#	p_curt = np.zeros((T_dec,2))

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
		"""
		#curtailing non dispatchable load
		if len(participant.assets_nd):
			for a, asset in enumerate(participant.assets_nd):
				if np.any(asset.Pnet_ems)<=0: #generation
					prob.add_constraint(p_curt[:, a] <= 0) 
					prob.add_constraint(p_curt[:, a] >=asset.curt*asset.mpc_demand(self.t_ahead_0)) 
				else:
					prob.add_constraint(p_curt[:, a] >= 0) 
					prob.add_constraint(p_curt[:, a] <= asset.curt*asset.mpc_demand(self.t_ahead_0)) 
					#print('p_curt load:', asset.curt*asset.mpc_demand(self.t_ahead_0))
		"""
		#net power constraints
		prob.add_constraint(p_net_exp == #sum(p_curt[:,a] for a in range(p_curt.shape[1]))
		                   - sum(p_flex[:self.T_market-self.t_ahead_0, i] + p_flex[self.T_market-self.t_ahead_0:, i] for i in range(p_flex.shape[1])) 
		                    - pic.sum(P_net_ems[:, i] for i in range(P_net_ems.shape[1])) 
		                   )
		
		#######################################
		### Objective Function
		#######################################
		
		if len(participant.assets_flex):
			prob.set_objective('min',    sum(self.price_imp[t+self.t_ahead_0, participant.p_id]*q_sup_buy[t] 
				                          - self.price_exp[t+self.t_ahead_0, participant.p_id]*q_sup_sell[t]  for t in range(T_dec)
				                            )\
				                           # 
			                            + sum((prices_us[trd]+ trade_tcost_us[trd]/2)*q_us[trd]  for trd in range(N_trades_us))\
			                            + sum((-prices_ds[trd]+ trade_tcost_ds[trd]/2)*q_ds[trd]  for trd in range(N_trades_ds))\
			                            + pic.sum(self.dt_market*participant.assets_flex[i].c_deg_lin\
	        	                       	*(p_flex[t, i] - p_flex[t+self.T_market-self.t_ahead_0, i]) for i in range(p_flex.shape[1]) for t in range(self.T_market-self.t_ahead_0))
			                    )
		else:
			prob.set_objective('min',    sum(self.price_imp[t+self.t_ahead_0, participant.p_id]*q_sup_buy[t] 
				                          - self.price_exp[t+self.t_ahead_0, participant.p_id]*q_sup_sell[t]  for t in range(T_dec)
				                            )\
										#
			                            + sum((prices_us[trd]+ trade_tcost_us[trd]/2)*q_us[trd]  for trd in range(N_trades_us))\
			                            + sum((-prices_ds[trd]+ trade_tcost_ds[trd]/2)*q_ds[trd]  for trd in range(N_trades_ds))\
			          
			                    )
		
		
		prob.set_option('solver','gurobi')
		#prob.set_option('mosek_params',{'mio_max_time': 5})
		prob.solve(verbose=0, primals=None) #suppresses all outputs
		print('P2P optimisation:', prob.status)
		
		q_us_full_list = np.zeros([len(trade_list),1])
		q_ds_full_list = np.zeros([len(trade_list),1])
		q_us_full_list[np.where((trade_list[:,trades_buyer_col]== participant.p_id).astype(bool))] = q_us.value
		q_ds_full_list[np.where((trade_list[:,trades_seller_col]== participant.p_id).astype(bool))] = q_ds.value

		#if len(participant.assets_nd):
		#	p_curt_val = p_curt.value
		#else: p_curt_val = None

		if len(participant.assets_flex):
			p_flex_val = p_flex.value
		else: p_flex_val = None

		print('q_us', q_us.value, 'q_ds', q_ds.value, 'p_flex', p_flex_val, 'q_sup_sell', q_sup_sell.value, 'q_sup_buy', q_sup_buy.value) #'p_curt', p_curt_val, 
		return {'opt_prob':prob,\
		        'q_us':q_us.value,\
		        'q_ds':q_ds.value,\
		        'q_us_full_list':q_us_full_list,\
		        'q_ds_full_list':q_ds_full_list,\
		        'p_net_exp':p_net_exp.value,\
		        'q_sup_sell':q_sup_sell.value,\
		        'q_sup_buy':q_sup_buy.value,\
		        'p_flex': p_flex_val,
		        #'p_curt': p_curt_val
		        }


class Auction_market(Market):
	def __init__(self, participants, offers, dt_market, T_market, price_imp, t_ahead_0=0, P_import=None, P_export=None, price_exp=None):
		Market.__init__(self, participants, offers, dt_market, T_market, price_imp, t_ahead_0=t_ahead_0, P_import=P_import, P_export=P_export, price_exp=price_exp)

	def Auction_matching(self, type='double', fees=None): 
		"""
		Returns the outcome of an auction matching highest bid with lowest ask matching
		Parameters
		----------
		type: string
			'single':
			'double':
		fees: 2 dim np.array
			the fees the peers have to pay to the DNO for using the physical infrastructure
			fees[i,j]: the fees of transfering energy between participant i and participant j 
		returns
		-------
		market_clearing_outcome: pandas Dataframe
            the resulting energy exchange
            ----------------------------------------------
             id | time | seller | buyer | energy | price |
            ----------------------------------------------
			|	|	   |		|		|		 |	     |
            ----------------------------------------------

		"""

		###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!include offers from the upstream nw to offers before the matching!!!!!!!!!!!!!!!!!###########
		#####           add the fees/2 to the prices
		############################################################################################################################
		
		list_clearing = []
		for t in self.offers['time']: #check if t in [t0_ahead, self.T_ems]?
			#insert upstream offers in the offers df
			Bids = self.offers[(self.offers['participant'].isin(self.buyer_indexes)) & (self.offers['time']==t)]
			Bids.append([t, 0, self.P_import, self.price_imp[t]])
			Asks = self.offers[(self.offers['participant'].isin(self.seller_indexes)) & (self.offers['time']==t)]
			Asks.append([t, 0, self.P_export, self.price_exp[t]])

			#match
			while (len(Bids) > 0 and len(Asks) > 0):
				Bids = sorted(Bids, key=lambda x: x.price)[::-1]
				Asks = sorted(Asks, key=lambda x: x.price)

				if Bids[0].price < Asks[0].price:
					break
				else:  # Bids[0].Price >= Offers[0].Price:
					currBid = Bids.pop()
					currAsk = Asks.pop()
					if currBid.energy != currOffer.energy:
						if currBid.energy > currAsk.energy:
							newBid = {'time': t, 'participant': currBid.participant, 'energy': currBid.energy - currAsk.energy, 'price': currBid.price}
							Bids.insert(0, newBid)
							list_clearing.append([t, currOffer.participant, currAsk.participant, self.dt_market*currAsk.energy, (currBid.price+currAsk.price)/2])
							#row = {'time': t, 'seller': currOffer.participant, 'buyer': currAsk.participant, 'energy': currAsk.energy, 'price': (currBid.price+currAsk.price)/2}
							#market_clearing_outcome.append(row)
						else:
							newAsk = {'time': t, 'participant': currAsk.participant, 'energy': currAsk.energy - currBid.energy, 'price': currAsk.price}
							Asks.insert(0,newAsk)
							#row = {'time': t, 'seller': currOffer.participant, 'buyer': currAsk.participant, 'energy': currBid.energy, 'price': (currBid.price+currAsk.price)/2}
							#market_clearing_outcome.append(row)  
							list_clearing.append([t, currOffer.participant, currAsk.participant, self.dt_market*currBid.energy, (currBid.price+currAsk.price)/2])
		market_clearing_outcome=pd.DataFrame(data=list_clearing ,columns=['time', 'seller', 'buyer', 'energy', 'price'])

		return market_clearing_outcome

class ToU_market(Market):
	def __init__(self, participants, dt_market, T_market, price_imp,\
				 t_ahead_0=0, P_import=None, P_export=None, price_exp=None):
		
		Market.__init__(self, participants, dt_market, T_market, price_imp, t_ahead_0=t_ahead_0, P_import=P_import, P_export=P_export, price_exp=price_exp)
		if np.all(self.P_export) ==None:
			self.P_export = -np.inf*np.ones(self.T_market)
		if np.all(self.P_import) ==None:
			self.P_import = np.inf*np.ones(self.T_market)

	def market_clearing(self, use_agg=True):
		"""
		computes the energy exchange of each participant with the grid
		Parameters
		----------
		use_agg = bool, optional
		True (default) if the approximate aggregation method is used (useful for participants with a scaled number of assets)
		False if the approximate aggregation method is not used
		returns
		-------
		market_clearing_outcome: pandas Dataframe
            the resulting energy exchange
            ----------------------------------------------
             id | time | seller | buyer | energy | price |
            ----------------------------------------------
			|	|	   |		|		|		 |	     |
            ----------------------------------------------
		"""
			
		list_clearing, schedules, outputs = [], [], []
		for par in self.participants:
			print('Run EMS for Participant: ' + str(par.p_id) )
			schedule, output, pimp, pexp, buses = par.EMS(self.price_imp, self.P_import, self.P_export, self.price_exp, t_ahead_0=self.t_ahead_0, use_agg=use_agg)
			schedules.append(schedule)
			outputs.append(output)

			print('* Adding  participant {} trades to clearing outcome...'.format(par.p_id))
			for t in range(self.T_market-self.t_ahead_0):
				for b in range(len(buses)):
					if pimp[t, b] >0:
						list_clearing.append([t+self.t_ahead_0, 0, par.p_id, self.dt_market*pimp[t,b], self.price_imp[t+self.t_ahead_0, buses[b]] ])
					elif pexp[t,b] >0:
						list_clearing.append([t+self.t_ahead_0, par.p_id, 0, self.dt_market*pexp[t,b], self.price_exp[t+self.t_ahead_0, buses[b]] ])
			

		market_clearing_outcome=pd.DataFrame(data=list_clearing, columns=['time', 'seller', 'buyer', 'energy', 'price'])
		return market_clearing_outcome, schedules, outputs

class Central_market(Market):
	def __init__(self, participants, dt_market, T_market, price_imp, network, \
				 t_ahead_0=0, P_import=None, P_export=None, price_exp=None):
		Market.__init__(self, participants, dt_market, T_market, price_imp, t_ahead_0=t_ahead_0, P_import=P_import, P_export=P_export, price_exp=price_exp)
		self.network = network

	def market_clearing(self, v_unconstrained_buses=[], i_unconstrained_lines=[], use_agg=False):

		buses = [] 
		assets_all = []
		for par in self.participants:
			for asset in par.assets:
				assets_all.append(asset)
				buses.append(asset.bus_id)
		buses = list(set(buses))

		P_demand = np.zeros( (self.T_market-self.t_ahead_0, len(buses)))
		#P_curt_limits = np.zeros((self.T_ems-t_ahead_0, len(buses), 2))
		flex_assets_per_bus, inflex_assets_per_bus = [[] for _ in range(len(buses))], [[] for _ in range(len(buses))]
		flex_assets_ind_bus, nd_assets_ind_bus = [[] for _ in range(len(buses))], [[] for _ in range(len(buses))]
		flex, nd = 0, 0

		for bidx, bus in enumerate(buses):
			for asset in assets_all:
				if asset.type == 'ND'  and asset.bus_id == bus:
					P_demand[:, bidx]+= asset.mpc_demand(self.t_ahead_0)
					nd_assets_ind_bus[bidx].append(nd)
					nd+=1
				elif asset.type == 'ND' and asset.bus_id == bus:
					inflex_assets_per_bus[bidx].append(asset)
				elif asset.type != 'ND' and asset.bus_id == bus:
					flex_assets_per_bus[bidx].append(asset)
					flex_assets_ind_bus[bidx].append(flex)
					flex+=1

	    ##create a participant with all the assets
		participant_all = Participant.Participant(len(self.participants)+2, assets_all)
		#P_demand = participant_all.nd_demand(self.t_ahead_0)

		### Solve for aggregated method
		if use_agg:
			#### get Agg, bagg, c_deg_agg for each list assets_per_node:
			Agg, bagg = [], []
			c_deg = np.zeros(len(buses))
			for i in range(len(buses)):
				if len(flex_assets_per_bus[i]):
					print('{} asets in this aggregation'.format(len(flex_assets_per_bus[i])))
					A, b = self.polytope(t_ahead_0, flex_assets_per_bus[i])
					Agg.append(A)
					bagg.append(b)
					c_deg[i]= sum(flex_assets_per_bus[i][a].c_deg_lin for a in range(len(flex_assets_per_bus[i]))) / len(flex_assets_per_bus[i]) 


			prob =pic.Problem()
			Pimp = pic.RealVariable('Pimp',self.T_market-self.t_ahead_0)
			Pexp = pic.RealVariable('Pexp',self.T_market-self.t_ahead_0)
			# curt
			if len(participant_all.assets_nd):
				p_curt = pic.RealVariable('p_curt', (self.T_market-self.t_ahead_0, len(buses)))
			if len(participant_all.assets_flex):
				x = pic.RealVariable('x', (self.T_market-self.t_ahead_0, len(buses)))

			for bidx in range(len(buses)):
				#operational constraints
				if len(flex_assets_per_bus[bidx]):
					prob.add_constraint(Agg[bidx]*x[:, bidx] <= bagg[bidx])	
				#curtailment
				prob.add_constraint(p_curt[:, bidx] >= P_curt_limits[:, bidx, 0]) 
				prob.add_constraint(p_curt[:, bidx] <= P_curt_limits[:, bidx, 1]) 
				
				# balance constraint
				prob.add_constraint( P_demand[:, bidx] - p_curt[:, bidx] + x[: self.T_ems-t_ahead_0, bidx] + x[self.T_ems-t_ahead_0:, bidx] \
				                    == Pimp[:, bidx] - Pexp[:, bidx])
				# min/max import/export
				prob.add_constraint( Pimp[:, bidx] >= 0)
				prob.add_constraint( Pexp[:, bidx] >= 0)
				if not(np.all(np.isinf(P_import))):
					prob.add_constraint( Pimp[:, bidx] <= P_import[t_ahead_0:]) 
				if not(np.all(np.isinf(P_export))):
					prob.add_constraint( Pexp[:, bidx] <=-P_export[t_ahead_0:])#

			# set objective
			prob.set_objective()

			prob.set_option('solver','modek')#gurobi
			prob.solve()
			p_agg = np.array(x)

			print('* Updating resources for assets...')
			### turn the aggregated solution into assets schedules x[len(assets), 2*(T_ems-t0)]
			for bidx in range(len(buses)):
				if len(flex_assets_per_bus[bidx]):
					Par = Participant.Participant(len(self.participants)+3, flex_assets_per_bus[bidx])
					x_disagg = Par.polytope_desagregation(p_agg[:, bidx], flex_assets_per_bus[bidx], self.t_ahead_0)
					for a, asset in enumerate(flex_assets_per_bus[bidx]):
						asset.update_ems(x_disagg[:self.T_market-self.t_ahead_0, a]+ x_disagg[self.T_market-self.t_ahead_0:,a], self.t_ahead_0)
			
			nd = 0
			schedules = [[] for _ in range(len(self.participants))]
			for p_idx, par in enumerate(self.participants):
				for asset in par.assets:
					if asset.type == 'ND':
						schedules[p_idx].append(asset.mpc_demand(self.t_ahead_0)-p_curt[:, nd])
						nd +=1
					else:
						schedules[p_idx].append(asset.Pnet_ems[self.t_ahead_0:])
#######################################################################################################################################
		### non agg
		else:
			prob = pic.Problem()
			Pimp = pic.RealVariable('Pimp', self.T_market-self.t_ahead_0)
			Pexp = pic.RealVariable('Pexp', self.T_market-self.t_ahead_0)
			if len(participant_all.assets_flex):
				x = pic.RealVariable('x', (2*(self.T_market-self.t_ahead_0), len(participant_all.assets_flex)))
			if len(participant_all.assets_nd):
				p_curt = pic.RealVariable('p_curt', (self.T_market-self.t_ahead_0, len(participant_all.assets_nd)))

						
			# account for network constraints
			A_Pslack_wye_flex_list, A_Pslack_del_flex_list, A_vlim_wye_flex_list, A_vlim_del_flex_list, A_line_wye_flex_list, A_line_del_flex_list = [], [], [], [], [], []
			A_Pslack_wye_nd_list, A_Pslack_del_nd_list, A_vlim_wye_nd_list, A_vlim_del_nd_list, A_line_wye_nd_list, A_line_del_nd_list = [], [], [], [], [], []

			for t_ems in range(self.T_market-self.t_ahead_0):

				A_Pslack_flex, A_Pslack_nd, b_Pslack, A_vlim_flex, A_vlim_nd, b_vlim, v_abs_min_vec, v_abs_max_vec, A_lines_flex, A_lines_nd,\
				A_Pslack_wye_flex, A_Pslack_del_flex, A_vlim_wye_flex, A_vlim_del_flex, A_line_wye_flex, A_line_del_flex, \
				A_Pslack_wye_nd, A_Pslack_del_nd, A_vlim_wye_nd, A_vlim_del_nd, A_line_wye_nd, A_line_del_nd =\
				self.network.get_parameters(participant_all.assets_nd, participant_all.assets_flex, t_ems, self.t_ahead_0)

				#print(t_ems, np.any(np.isnan(A_Pslack)), np.any(np.isnan(b_Pslack)), np.any(np.isnan(A_vlim)), np.any(np.isnan(b_vlim)), \
                #	  np.any(np.isnan(v_abs_min_vec)), np.any(np.isnan(v_abs_max_vec)), np.any(np.isnan(A_lines)), np.any(np.isnan(A_Pslack_wye)), \
                #      np.any(np.isnan(A_Pslack_del)), np.any(np.isnan(A_vlim_wye)), np.any(np.isnan(A_vlim_del)), np.any(np.isnan(A_line_wye)), \
                #4      np.any(np.isnan(A_line_del)))

				#balance constraint
				prob.add_constraint(Pimp[t_ems]-Pexp[t_ems] == (np.sum(A_Pslack_flex[i]*
					                                            (x[t_ems, i] + x[self.T_market-self.t_ahead_0+t_ems, i])*1e3 for i in range(len(participant_all.assets_flex)))
				                                               -np.sum(A_Pslack_nd[k]*p_curt[t_ems, k]*1e3 for k in range(len(participant_all.assets_nd)))
				                                                + b_Pslack)/1e3
		                            )
				
				# voltage magnitude constraints:
				for bus_ph_index in range(self.network.N_phases*(self.network.N_buses-1)):
					if int(bus_ph_index/3) not in (np.array(v_unconstrained_buses)-1):
						prob.add_constraint(sum(A_vlim_flex[bus_ph_index,i]*
		    				                    (x[t_ems, i] + x[self.T_market-self.t_ahead_0+t_ems, i])*1e3 for i in range(len(participant_all.assets_flex)))
		    			                    -sum(A_vlim_nd[bus_ph_index,k]*p_curt[t_ems, k]*1e3 for k in range(len(participant_all.assets_nd)))
		    			                    + b_vlim[bus_ph_index] 
		    			                    <=\
		                                    v_abs_max_vec[bus_ph_index])
						prob.add_constraint(sum(A_vlim_flex[bus_ph_index,i]*
		                	                    (x[t_ems,i] + x[self.T_market-self.t_ahead_0+t_ems, i])*1e3 for i in range(len(participant_all.assets_flex)))
		                                    -sum(A_vlim_nd[bus_ph_index,k]*p_curt[t_ems, k]*1e3 for k in range(len(participant_all.assets_nd)))
		                                    + b_vlim[bus_ph_index] >=\
		                                    v_abs_min_vec[bus_ph_index])

		        # Line current magnitude constraints:
				#ij=0
				for line_ij in range(self.network.N_lines):
					if line_ij not in i_unconstrained_lines:	
						for ph in range(self.network.N_phases):
							prob.add_constraint(sum(A_lines_flex[line_ij][ph,i]*(x[t_ems, i] + x[self.T_market-self.t_ahead_0+t_ems, i])*1e3 for i in range(len(participant_all.assets_flex)))\
		                                      # -sum(A_lines[line_ij][ph,k]*p_curt[k]*1e3 for k in range(len(participant_all.assets_nd)))
		                               + self.network.Jabs_I0_list[line_ij][ph] <=\
		                               self.network.i_abs_max[line_ij,ph])

						#ij+=1
				A_Pslack_wye_flex_list.append(A_Pslack_wye_flex) 
				A_Pslack_del_flex_list.append(A_Pslack_del_flex)
				A_vlim_wye_flex_list.append(A_vlim_wye_flex)
				A_vlim_del_flex_list.append(A_vlim_del_flex)
				A_line_wye_flex_list.append(A_line_wye_flex)
				A_line_del_flex_list.append(A_line_del_flex)

				A_Pslack_wye_nd_list.append(A_Pslack_wye_nd) 
				A_Pslack_del_nd_list.append(A_Pslack_del_nd)
				A_vlim_wye_nd_list.append(A_vlim_wye_nd)
				A_vlim_del_nd_list.append(A_vlim_del_nd)
				A_line_wye_nd_list.append(A_line_wye_nd)
				A_line_del_nd_list.append(A_line_del_nd)
						# balance constraint
			#prob.add_constraint( sum(P_demand[:, b] for b in range(P_demand.shape[1]))- sum(p_curt[:, a] for a in range(len(participant_all.assets_nd))) 
			#	               + sum(x[: self.T_market-self.t_ahead_0, a] + x[self.T_market-self.t_ahead_0:, a] for a in range(len(participant_all.assets_flex)))\
			#                  == Pimp- Pexp)
			
			# min/max import/export	
			prob.add_constraint( Pimp >= 0)
			prob.add_constraint( Pexp >= 0)
			if not(np.all(np.isinf(self.P_import))):
				print('Pimport:', self.P_import[self.t_ahead_0:])
				prob.add_constraint( Pimp <= self.P_import[self.t_ahead_0:]) 
			if not(np.all(np.isinf(self.P_export))):
				print('Pexport:', self.P_export[self.t_ahead_0:])
				prob.add_constraint( Pexp <=-self.P_export[self.t_ahead_0:])
			
			#operational constraints
			for a, asset in enumerate(participant_all.assets_flex):
				A, b = asset.polytope(self.t_ahead_0)
				prob.add_constraint(A*x[:, a] <= b)
			
			#curtailment		
			for a, asset in enumerate(participant_all.assets_nd):
				if np.all(asset.Pnet_ems)<=0:
					prob.add_constraint(p_curt[:, a] <= 0) 
					prob.add_constraint(p_curt[:, a] >= asset.curt*asset.mpc_demand(self.t_ahead_0))
				else:
					prob.add_constraint(p_curt[:, a] >= 0) 
					prob.add_constraint(p_curt[:, a] <= asset.curt*asset.mpc_demand(self.t_ahead_0)) 	
			
			# set objective
			prob.set_objective('min', sum(self.dt_market*self.price_imp[self.t_ahead_0+t]*Pimp[t] - self.dt_market*self.price_exp[t+self.t_ahead_0]*Pexp[t] for t in range(self.T_market-self.t_ahead_0) )\
        	                       +  sum(self.dt_market*participant_all.assets_flex[i].c_deg_lin\
        	                       	*(x[t, i] - x[t+self.T_market-self.t_ahead_0, i]) for i in range(len(participant_all.assets_flex)) for t in range(self.T_market-self.t_ahead_0) ) \
        	                      
        				  ) #self.price_imp[self.t_ahead_0+t]*Pimp[t] - self.price_exp[t+self.t_ahead_0]*Pexp[t]
			prob.set_option('solver','mosek')#gurobi
			prob.solve()
			print('Optimisation status:', prob.status)
			
##############################################################################################################
			print('* Computing clearing prices ...')

			N_load_bus_phases = self.network.N_phases*(self.network.N_buses-1)
			
			#v_ph_const = []
			#for bus_ph_index in range(self.network.N_phases*(self.network.N_buses-1)):
			#	if int(bus_ph_index/3) not in (np.array(v_unconstrained_buses)-1):
			#		v_ph_const.append(bus_ph_index)

			#Get constraint dual variables
			dual_P_slack = np.zeros([self.T_market-self.t_ahead_0])
			dual_vbus_max = np.zeros([self.T_market-self.t_ahead_0,N_load_bus_phases])
			dual_vbus_min = np.zeros([self.T_market-self.t_ahead_0,N_load_bus_phases])
			dual_iline_max = np.zeros([self.T_market-self.t_ahead_0,self.network.N_phases*self.network.N_lines])
            
			#dual_vbus_max[:, v_ph_const] = Vmax.dual  
			#dual_vbus_min[:, v_ph_const] = Vmin.dual
			#dual_iline_max[:, ] = Line.dual
			const_index=0
			for t in range(self.T_market-self.t_ahead_0):
				#### add P_wye, P_del, 
				dual_P_slack[t]= np.squeeze(prob.get_constraint(const_index).dual)
				#if np.squeeze(prob.get_constraint(const_index).slack):
					#print('balance constraint activated at t=', t, 'slack=', np.squeeze(prob.get_constraint(const_index).slack))
				const_index += 1

				for bus_ph_index in range(N_load_bus_phases):
					if int(bus_ph_index/3) not in (np.array(v_unconstrained_buses)-1):
						dual_vbus_max[t,bus_ph_index] = np.squeeze(prob.get_constraint(const_index).dual)
						#if np.squeeze(prob.get_constraint(const_index).slack):
						#	print('max voltage constraint activated at t=', t, 'in bus', int(bus_ph_index/3), 'slack=', np.squeeze(prob.get_constraint(const_index).slack))
						const_index += 1
						dual_vbus_min[t,bus_ph_index] = np.squeeze(prob.get_constraint(const_index).dual)
						#if np.squeeze(prob.get_constraint(const_index).slack):
						#	print('min voltage constraint activated at t=', t, 'in bus', int(bus_ph_index/3), 'slack=', np.squeeze(prob.get_constraint(const_index).slack))
						const_index+=1

				for line_ij in range(self.network.N_lines):
					if line_ij not in i_unconstrained_lines:	
						for ph in range(self.network.N_phases):
							dual_iline_max[t, line_ij+ph] = np.squeeze(prob.get_constraint(const_index).dual)
							#if np.squeeze(prob.get_constraint(const_index).slack):
							#	print('thermal constraint activated at t=', t, 'in (line, phase)', line, ph, 'slack=', np.squeeze(prob.get_constraint(const_index).slack))
							const_index += 1


###############################
			LMP_P_wye_flex = np.zeros((self.T_market-self.t_ahead_0, len(participant_all.assets_flex)))
			LMP_P_del_flex = np.zeros((self.T_market-self.t_ahead_0, len(participant_all.assets_flex)))

			LMP_P_wye_nd = np.zeros((self.T_market-self.t_ahead_0, len(participant_all.assets_nd)))
			LMP_P_del_nd = np.zeros((self.T_market-self.t_ahead_0, len(participant_all.assets_nd)))
			#CHECK INDEXES 
			for t in range(self.T_market-self.t_ahead_0):
				for flex in range(len(participant_all.assets_flex)):

					LMP_P_wye_flex[t,flex] = 1/self.dt_market*(#dual_P_wye[t,ph]\
			                           dual_P_slack[t]\
			                           -sum(dual_vbus_max[t,flex]*A_vlim_wye_flex_list[t][ph,flex] for ph in range(N_load_bus_phases))\
			                           +sum(dual_vbus_min[t,flex]*A_vlim_wye_flex_list[t][ph,flex] for ph in range(N_load_bus_phases))\
			                           -sum(sum(dual_iline_max[t,flex]*A_line_wye_flex_list[t][line][line_ph,flex] for line_ph in range(self.network.N_phases)) for line in range(self.network.N_lines)))
					LMP_P_del_flex[t,flex] = 1/self.dt_market*(#dual_P_del[t,ph]\
			                           dual_P_slack[t]
			                           -sum(dual_vbus_max[t,flex]*A_vlim_del_flex_list[t][ph,flex] for ph in range(N_load_bus_phases))\
			                           +sum(dual_vbus_min[t,flex]*A_vlim_del_flex_list[t][ph,flex] for ph in range(N_load_bus_phases))\
			                           -sum(sum(dual_iline_max[t,flex]*A_line_del_flex_list[t][line][line_ph,flex] for line_ph in range(self.network.N_phases)) for line in range(self.network.N_lines)))
					
				for nd in range(len(participant_all.assets_nd)):

					LMP_P_wye_nd[t,nd] = 1/self.dt_market*(#dual_P_wye[t,ph]\
			                           dual_P_slack[t]\
			                           -sum(dual_vbus_max[t,nd]*A_vlim_wye_nd_list[t][ph,nd] for ph in range(N_load_bus_phases))\
			                           +sum(dual_vbus_min[t,nd]*A_vlim_wye_nd_list[t][ph,nd] for ph in range(N_load_bus_phases))\
			                           -sum(sum(dual_iline_max[t,nd]*A_line_wye_nd_list[t][line][line_ph,nd] for line_ph in range(self.network.N_phases)) for line in range(self.network.N_lines)))
					LMP_P_del_nd[t,nd] = 1/self.dt_market*(#dual_P_del[t,ph]\
			                           dual_P_slack[t]
			                           -sum(dual_vbus_max[t,nd]*A_vlim_del_nd_list[t][ph,nd] for ph in range(N_load_bus_phases))\
			                           +sum(dual_vbus_min[t,nd]*A_vlim_del_nd_list[t][ph,nd] for ph in range(N_load_bus_phases))\
			                           -sum(sum(dual_iline_max[t,nd]*A_line_del_nd_list[t][line][line_ph,nd] for line_ph in range(self.network.N_phases)) for line in range(self.network.N_lines)))




					"""
					LMP_P_wye[t,ph] = -1/self.dt_market*(#dual_P_wye[t,ph]\
			                           +dual_P_slack[t]*A_Pslack_wye_list[t][load_bus_phases[ph]-3]\
			                           +sum(dual_vbus_max[t,ph_2]*A_vlim_wye_list[t][ph_2,load_bus_phases[ph]-3] for ph_2 in range(N_load_bus_phases))\
			                           -sum(dual_vbus_min[t,ph_2]*A_vlim_wye_list[t][ph_2,load_bus_phases[ph]-3] for ph_2 in range(N_load_bus_phases))\
			                           +sum(sum(dual_iline_max[t,N_phases*(line-1)+line_ph]*A_line_wye_list[t][line][line_ph,load_bus_phases[ph]-3] for line_ph in range(N_phases)) for line in range(N_lines)))
					print(LMP_P_wye[t,ph])
					LMP_P_del[t,ph] = -1/self.dt_market*(#dual_P_del[t,ph]\
			                           +dual_P_slack[t]*A_Pslack_del_list[t][load_bus_phases[ph]-3]\
			                           +sum(dual_vbus_max[t,ph_2]*A_vlim_del_list[t][ph_2,load_bus_phases[ph]-3] for ph_2 in range(N_load_bus_phases))\
			                           -sum(dual_vbus_min[t,ph_2]*A_vlim_del_list[t][ph_2,load_bus_phases[ph]-3] for ph_2 in range(N_load_bus_phases))\
			                           +sum(sum(dual_iline_max[t,N_phases*(line-1)+line_ph]*A_line_del_list[t][line][line_ph,load_bus_phases[ph]-3] for line_ph in range(N_phases)) for line in range(N_lines)))
					"""
			p_curt = np.array(p_curt.value)
			x = np.array(x.value)

			print('* Establishing clearing outcome & Updating resources for assets ...')
			list_clearing = []
			nd, flex = 0, 0
			schedules = [[] for _ in range(len(self.participants))]
			for p_idx, par in enumerate(self.participants):
				print('*** Updating resources for assets | participant {}...'.format(par.p_id))
				for asset in par.assets:
					bus_id = asset.bus_id
					if asset.type == 'ND':
						if self.network.bus_df[self.network.bus_df['number']==bus_id]['connect'].values[0]=='Y':
							price = LMP_P_wye_nd[:, nd]
						else: price = LMP_P_del_nd[:, nd] 
						sch = asset.mpc_demand(self.t_ahead_0)-p_curt[:, nd]
						schedules[p_idx].append(asset.mpc_demand(self.t_ahead_0)-p_curt[:, nd])					
						nd +=1
					else:
						if self.network.bus_df[self.network.bus_df['number']==bus_id]['connect'].values[0]=='Y':
							price = LMP_P_wye_flex[:, flex]
						else: price = LMP_P_del_flex[:, flex] 
						asset.update_ems(x[:self.T_market-self.t_ahead_0, flex]+ x[self.T_market-self.t_ahead_0:, flex], self.t_ahead_0)
						schedules[p_idx].append(asset.Pnet_ems[self.t_ahead_0:])
						sch = asset.Pnet_ems[self.t_ahead_0:]
						flex +=1
					print('*** Adding asset outcome for participant {} to clearing ...'.format(par.p_id))
					for t in range(self.T_market-self.t_ahead_0):
						if  sch[t]>0:
							list_clearing.append([t+self.t_ahead_0, 0, par.p_id, self.dt_market*sch[t], price[t]])
						elif sch[t]<0:
							list_clearing.append([t+self.t_ahead_0, par.p_id, 0, -self.dt_market*sch[t], price[t]])
	
					
			#group by price? for each participant sum trades with same price?
			market_clearing_outcome=pd.DataFrame(list_clearing, columns=['time', 'seller', 'buyer', 'energy', 'price'])
			market_clearing_outcome = market_clearing_outcome.groupby(['time', 'seller', 'buyer', 'price'])['energy'].aggregate('sum').reset_index()



		return market_clearing_outcome, schedules, np.array(Pimp.value), np.array(Pexp.value)
