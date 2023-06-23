"""
This module presents a participant
A participant can be either a prosumer, an aggregator or an energy provider

"""

# import modules
import numpy as np
import pandas as pd
import picos as pic

class Participant:
	"""
	p_id: unique identifier for a participant
	assets: a list of assets managed by the participant
			assets located in the same bus => prosumer
			assets in different buses => aggregator
	"""
	def __init__(self, p_id, assets, *args):
		self.p_id = p_id
		self.assets = assets

		self.T_ems = assets[0].T_ems
		self.dt_ems = assets[0].dt_ems
		self.T = assets[0].T

		#self.Pnet = np.zeros(self.T)
		#self.Qnet = np.zeros(self.T)
		#self.Pnet_pred = np.zeros(self.T)
		#self.Qnet_pred = np.zeros(self.T)
		#
		self.Pnet_ems = np.zeros(self.T_ems)
		self.Qnet_ems = np.zeros(self.T_ems)
		self.Pnet_pred_ems = np.zeros(self.T_ems)
		self.Qnet_pred_ems = np.zeros(self.T_ems)

		#self.Pload = np.zeros(self.T)
		#self.Qload = np.zeros(self.T)
		#self.Pload_pred = np.zeros(self.T)
		#self.Qload_pred = np.zeros(self.T)
		#
		self.Pload_ems = np.zeros(self.T_ems)
		self.Qload_ems = np.zeros(self.T_ems)
		self.Pload_pred_ems = np.zeros(self.T_ems)
		self.Qload_pred_ems = np.zeros(self.T_ems)

		#self.Pgen = np.zeros(self.T)
		#self.Qgen = np.zeros(self.T)
		#self.Pgen_pred = np.zeros(self.T)
		#self.Qgen_pred = np.zeros(self.T)
		#
		self.Pgen_ems = np.zeros(self.T_ems)
		self.Qgen_ems = np.zeros(self.T_ems)
		self.Pgen_pred_ems = np.zeros(self.T_ems)
		self.Qgen_pred_ems = np.zeros(self.T_ems)

		self.Pmax = np.zeros(self.T_ems)
		self.Pmin = np.zeros(self.T_ems)
		self.Emax = np.zeros(self.T_ems)
		self.Emin = np.zeros(self.T_ems)
		self.E_ems = np.zeros(self.T_ems)
		self.c1_deg = [] 

		self.type_flex=None
		self.P_flex_ems = np.zeros(self.T_ems)


		for asset in assets:
			if asset.type == 'storage':
				self.Pmax += asset.Pmax
				self.Pmin += asset.Pmin
				#self.Emax += asset.Emax
				#self.Emin += asset.Emin
				#self.E0 += asset.E0
				#self.ET += asset.ET
				#self.E_ems += np.ones(self.T_ems)*asset.E0
				self.c1_deg.append(asset.c_deg_lin)
			elif asset.type == 'building':
				self.Pmax += max(asset.Hmax, asset.Cmax)
				#self.Pmin += -asset.Cmax
			elif asset.type == 'ND' and np.all(asset.Pnet)>=0:
				self.Pload_ems += asset.Pnet_ems
				self.Qload_ems += asset.Qnet_ems
			elif asset.type == 'ND' and np.all(asset.Pnet)<=0:
				self.Pgen_ems += asset.Pnet_ems
				self.Qgen_ems += asset.Qnet_ems
			elif asset.type == 'ND' and np.all(asset.Pnet_pred)>=0:
				self.Pload_pred_ems += asset.Pnet_pred_ems
				self.Qload_pred_ems += asset.Qnet_pred_ems
			elif asset.type == 'ND' and np.all(asset.Pnet_pred)<=0:
				self.Pgen_pred_ems += asset.Pnet_pred_ems
				self.Qgen_pred_ems += asset.Qnet_pred_ems

		self.Pnet_ems = self.Pload_ems + self.Pgen_ems
		self.Qnet_ems = self.Qload_ems + self.Qgen_ems
		self.Pnet_pred_ems = self.Pload_pred_ems + self.Pgen_pred_ems
		self.Qnet_pred_ems = self.Qload_pred_ems + self.Qgen_pred_ems
		
		self.c1_deg = np.mean(np.asarray(self.c1_deg))

		self.assets_flex, self.assets_nd = [], []
		for asset in self.assets:
			if asset.type != 'ND':
			    self.assets_flex.append(asset)
			else: self.assets_nd.append(asset)

	def polytope(self, assets, t0=0):
		"""
        Computes an outer approximation of the aggregated polytope representation of the assets operational constraints
        Ax <= b, with x=[P_in, P_out]
                 and P_in/out is the power into and out of the assets over the optimisation horizon T_ems
                 P_ch>=0 P_dis<0
        from "A concise, approximate representation of a collection of loads described by polytopes"
        Parameters:
        -----------
        assets: list 
        	list of assets objects
        t0: int, default=0
        	first time slot of aggregation in an optimisation time scale
        returns
        --------
        (A_agg, b_agg):  (2 dim numpy.ndarray, 1-dim numpy.ndarray)
        """

		list_b = [np.empty(0)]*len(assets) #self.assets_flex
		#initialise Aunique as A of asset 0
		A0, b0 = assets[0].polytope(t0) #self.assets_flex
		Aunique = A0
		list_b[0] = b0
		
		#### return the Aunique that has the aggregated unique rows, compute at the same time corresponding b for asset0
		for a, asset in enumerate(assets[1:]): #self.assets_flex[1:]
			#print('Start of the aggregated A, b calculation ...')
			A, b = asset.polytope(t0)
			for index in range(A.shape[0]):
				if not (np.any(np.all(A[index] == Aunique, axis=1))):
					#print('index:', index)
					b_new = self._find_b(A0, b0, A[index])
					Aunique = np.concatenate((Aunique, np.expand_dims(A[index], axis=0)), axis=0)
					#b0  = np.append(b0, b_new)	
					list_b[0] = np.append(list_b[0], b_new) #.append(b_new)	
		#print('end for asset 0, Aunique shape:', Aunique.shape, 'b:', list_b[0].shape)

		#################### compute new b corresponding to Aunique for the other assets #######################################
		for a, asset in enumerate(assets[1:]): #self.assets_flex[1:]
			A, b = asset.polytope(t0)  
			for index in range(Aunique.shape[0]):
				if (np.any(np.all(Aunique[index] == A, axis=1))): 
					#find index in A that corresponds to Aunique[index]
					i = np.where(np.all(Aunique[index] == A,axis=1))[0][0]
					list_b[a+1] = np.append(list_b[a+1], b[i])

				else:
					#print(Aunique.shape, A.shape, b.shape) #, Aunique[index], A, b)
					b_new = self._find_b(A, b, Aunique[index])
					#A = np.concatenate((A, np.expand_dims(Aunique[index], axis=0)), axis=0)
					#b = np.append(b, b_new)
					list_b[a+1]  = np.append(list_b[a+1], b_new)

		return Aunique, np.sum(np.asarray(list_b), axis=0)

	def _find_b(self, A1, b1, a):
		prob = pic.Problem()
		x = pic.RealVariable('x', A1.shape[1])
		prob.add_constraint(A1*x<= b1)
		prob.set_objective('max', sum(a*x.T))
		prob.solve(solver='gurobi') #
		return prob.value

	def power_desaggregation(self, p_agg, assets, t_ahead_0=0):
		"""
		produces a feasible power vector for each asset in the list from the aggregated power schedule p_agg.
		from "A concise, approximate representation of a collection of loads described by polytopes"
		Parameters
		----------
		p_agg: numpy.ndarray
			a vector containing the aggregated power injection or absoption for each time period over the optimisation horizon [t_ahead_0, T_ems]
		assets: list
			list of assets objects in the aggregation
		t_ahead_0: int, default =0
        	first time slot of aggregation in an optimisation time scale
		returns
		----------
		list_of_p: list of numpy.ndarray
			list od feasible power vector for each asset

		"""

		"""
		if len(p_agg) == self.T_ems-t_ahead_0:
			p_agg_new = []
			for i in range(len(p_agg)):
				if p_agg[i] >0:
					p_agg_new.append(p_agg[i])
					p_agg_new.append(0)
				else:
					p_agg_new.append(0)
					p_agg_new.append(p_agg[i])

			p_agg = np.array(p_agg_new)
		"""

		prob = pic.Problem()
		x = pic.RealVariable('x', (2*(self.T_ems-t_ahead_0), len(assets)))   #self.assets_flex
		x_aux = pic.RealVariable('x_aux', 2*(self.T_ems-t_ahead_0))

		for a, asset in enumerate(assets): #self.assets_flex
			A, b = asset.polytope(t_ahead_0)
			prob.add_constraint(A*x[:,a] <= b)
		prob.add_list_of_constraints([x_aux[t] == p_agg[t] - sum(x[t,:])  for t in range(2*(self.T_ems-t_ahead_0))])

		prob.set_objective('min', abs(x_aux) )
		prob.solve(solver='mosek')
		#print('x disagg:', x.value)

		opt = True
		if prob.status != 'optimal':
			print('the desaggregation could not find a feasible solution')
			opt =False
		#####!!!!!!!!!!!!!!!!!! some values are neeear zeros, convert to zero before proceeding!!!!!!!!!!!!!!!!!!
		return np.array(x.value), opt
              
	def nd_demand(self, t0):
		"""
        a power vector composed of the actual realisiation of the current time step and the predicted values for the future time steps for all
        the non dispatchale assets of the participant
        Parameters
        ---------------
        t0: int default=0
            first time slot of observation
        Returns
        ----------------
        P_demand: np.array
            power vector
        """
		#Assemble P_demand out of P actual and P predicted and convert to EMS time series scale
		P_demand = np.zeros([self.T_ems-t0,len(self.assets_nd)])
		for i in range(len(self.assets_nd)):
			"""
			for t_ems in range(t0, self.T_ems):
				if t_ems == t0:
					P_demand[t_ems-t0,i] = np.mean(self.assets_nd[i].Pnet[t_ems*int(self.dt_ems/self.dt) : (t_ems+1)*int(self.dt_ems/self.dt)])
				    #Q_demand[t_ems-t0, i] = np.mean(assets_nd[i].Qnet[t_ems*int(dt_ems/dt) : (t_ems+1)*int(dt_ems/dt)])
				else:
					P_demand[t_ems-t0, i] = np.mean(self.assets_nd[i].Pnet_pred[t_ems*int(self.dt_ems/self.dt) : (t_ems+1)*int(self.dt_ems/self.dt)])
				    #Q_demand[t_ems-t0, i] = np.mean(assets_nd[i].Qnet_pred[t_ems*int(dt_ems/dt) : (t_ems+1)*int(dt_ems/dt)])
			"""
			P_demand[:,i]= self.assets_nd[i].mpc_demand(t0)
			return P_demand

	def EMS(self, price_imp, P_import, P_export, price_exp, t_ahead_0=0, network=None):
		
		buses = []
		for asset in self.assets:
			buses.append(asset.bus_id)
		buses = list(set(buses))

		#buses_id = np.where(np.isin(network.load_buses,buses))

		P_demand = np.zeros( (self.T_ems-t_ahead_0, len(buses)))
		P_curt_limits = np.zeros((self.T_ems-t_ahead_0, len(buses), 2))
		flex_assets_per_bus = [[] for _ in range(len(buses))]
		flex_assets_ind_bus, nd_assets_ind_bus = [[] for _ in range(len(buses))], [[] for _ in range(len(buses))]
		flex, nd = 0, 0

		for bidx, bus in enumerate(buses):
			for asset in self.assets:
				if asset.type == 'ND'  and asset.bus_id == bus:
					P_demand[:, bidx]+= asset.mpc_demand(t_ahead_0)
					nd_assets_ind_bus[bidx].append(nd)
					nd+=1
				elif asset.type == 'ND' and asset.bus_id == bus and np.all(asset.Pnet_ems)>=0:
					P_curt_limits[:, bidx, 1] += asset.curt*asset.mpc_demand(t_ahead_0)
				elif asset.type == 'ND'  and asset.bus_id == bus and np.all(asset.Pnet_ems)<=0:
					P_curt_limits[:, bidx, 0] += asset.curt*asset.mpc_demand(t_ahead_0)
				elif asset.type != 'ND' and asset.bus_id == bus:
					flex_assets_per_bus[bidx].append(asset)
					flex_assets_ind_bus[bidx].append(flex)
					flex+=1

		################################################################
		# Running optimisation problem
		################################################################
		prob = pic.Problem()
		Pimp = pic.RealVariable('Pimp', (self.T_ems-t_ahead_0, len(buses)))
		Pexp = pic.RealVariable('Pexp', (self.T_ems-t_ahead_0, len(buses)))
		if len(self.assets_flex):
			x = pic.RealVariable('x', (2*(self.T_ems-t_ahead_0), len(self.assets_flex)))
		if len(self.assets_nd):
			p_curt = pic.RealVariable('p_curt', (self.T_ems-t_ahead_0, len(self.assets_nd)))
	    
		for bidx in range(len(buses)):	
			# balance constraint
			prob.add_constraint( P_demand[:, bidx] - sum(p_curt[:, a] for a in nd_assets_ind_bus[bidx]) 
				               + sum(x[: self.T_ems-t_ahead_0, a] + x[self.T_ems-t_ahead_0:, a] for a in flex_assets_ind_bus[bidx])\
			                    == Pimp[:, bidx] - Pexp[:, bidx])

			
			# min/max import/export
			prob.add_constraint( Pimp[:, bidx] >= 0)
			prob.add_constraint( Pexp[:, bidx] >= 0)
			if not(np.all(np.isinf(P_import))):
				prob.add_constraint( Pimp[:, bidx] <= P_import[t_ahead_0:]) 
			if not(np.all(np.isinf(P_export))):
				prob.add_constraint( Pexp[:, bidx] <=-P_export[t_ahead_0:])#

		#operational constraints	
		for a, asset in enumerate(self.assets_flex):
			A, b = asset.polytope(t0=t_ahead_0)
			prob.add_constraint(A*x[:, a] <= b)

		#curtailment
		for a, asset in enumerate(self.assets_nd):
			if np.all(asset.Pnet_ems)<=0:
				prob.add_constraint(p_curt[:, a] <= 0) 
				prob.add_constraint(p_curt[:, a] >= np.minimum(asset.curt*asset.mpc_demand(t_ahead_0),0) )
			else:
				prob.add_constraint(p_curt[:, a] >= 0) 
				prob.add_constraint(p_curt[:, a] <= np.maximum(asset.curt*asset.mpc_demand(t_ahead_0),0) ) 

		# set objective
		prices_import = pic.new_param('prices_import', price_imp[t_ahead_0:, buses])
		prices_export = pic.new_param('prices_export', price_exp[t_ahead_0:, buses])

		if len(self.assets_flex):
			prob.set_objective('min', self.dt_ems*(
				                      sum(prices_import[:, bidx].T*Pimp[:, bidx] - prices_export[:, bidx].T*Pexp[:, bidx] for bidx in range(len(buses)))
			                        + sum(self.assets_flex[a].c_deg_lin*(x[t, a] - x[t+self.T_ems-t_ahead_0, a]) for a in range(len(self.assets_flex)) for t in range(self.T_ems-t_ahead_0)) 
			                                      ) # 
							  )
		else:
			prob.set_objective('min', sum(self.dt_ems*prices_import[:, bidx].T*Pimp[:, bidx] - self.dt_ems*prices_export[:, bidx].T*Pexp[:, bidx] for bidx in range(len(buses)))
				                  )
		
		#prob.set_option('solver','mosek')#gurobi
		prob.solve(solver='mosek')
		
		print('* Updating resources for participant {}...'.format(self.p_id))
		if len(self.assets_flex):
			x = np.array(x.value)
		if len(self.assets_nd):
			p_curt = np.array(p_curt.value)

		#outputs = []
		for a, asset in enumerate(self.assets_flex):
			asset.update_ems(x[:self.T_ems-t_ahead_0, a] + x[self.T_ems-t_ahead_0:,a], t_ahead_0, enforce_const=False)
		for a, asset in enumerate(self.assets_nd):
			asset.update_ems(p_curt[:, a], t_ahead_0)	

		schedule = []
		nd=0
		for asset in self.assets:
			#if asset.type != 'ND':
			schedule.append(asset.Pnet_ems[t_ahead_0:])
			#else: #if asset.type == 'ND' 
			#	schedule.append(asset.mpc_demand(t_ahead_0)-p_curt[:, nd])
			#	nd+=1
		return schedule, np.array(Pimp.value), np.array(Pexp.value), buses




				



