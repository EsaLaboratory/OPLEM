#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3-phase networks module

OPLEM offers two options for network modelling. 

(1)The PandapowerNet class from the open-source python package pandapower can be used for balanced power flow analysis. 

(2)For unbalanced multi-phase power flow analysis, OPLEM offers the Network_3ph class.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import sys, os, time
import copy

__version__ = "1.0.0"

def strCpx2cpx(X):
    i=0
    for x in X:
        X[i] = complex(x)
        i+=1
    return X

class Network_3ph:
    """
    A 3-phase electric power network. Default to an unloaded IEEE 13 Bus Test 
    Feeder.

    Parameters
    ----------
    bus_df : pandas.DataFrame
        bus information, columns: ['name','number','load_type','connect',
        'Pa','Pb','Pc','Qa','Qb','Qc'], load_type: 'S' (slack),'PQ','Z' or 'I' 
        (only S and PQ are currently implemented), 
        connect: 'Y' (wye) or 'D' (delta), 'Px' in (kW), 
        'Qx' in (kVAr)
    capacitor_df : pandas.DataFrame
        capacitor information, columns ['name','number','bus','kVln',
        'connect','Qa','Qb','Qc'], connect: 'Y' (wye) or 'D' (delta), 
        'kVln': line-to-line base voltage, 'Qx' in (kVAr)
    di_iter : np.ndarray
        change in the sum of abs phase currents at each Z-Bus iteration
    dv_iter : np.ndarray
        change in the sum of abs phsae voltages at each Z-Bus iteration
    i_abs_max : numpy.ndarray
        max abs line phase currents [line, phase] (|A|)
    i_PQ_iter : numpy.ndarray
        current injected at each phase at each Z-Bus iteration [iter,phase] (A)
    i_PQ_res : numpy.ndarray
        power flow result, current injected at each phase (excl. slack) (A)
    i_net_res : numpy.ndarray
        power flow result, current injected at each phase (A)
    i_slack_res : numpy.ndarray
        power flow result, current injected at slack bus phases (A)
    Jabs_dPQdel_list : list of numpy.ndarray
        linear line abs current model, [P_delta,Q_delta] coeff. matrix list
    Jabs_dPQwye_list : list of numpy.ndarray
        linear line abs current model, [P_wye,Q_wye] coeff. matrix list
    Jabs_I0_list : list of numpy.ndarray
        linear line abs current model, constant vector list
    J_dPQdel_list : list of numpy.ndarray
        linear line current model, [P_delta,Q_delta] coeff. matrix list
    J_dPQwye_list : list of numpy.ndarray
        linear line current model, [P_wye,Q_wye] coeff. matrix list
    J_I0_list : list of numpy.ndarray
        linear line current model, constant vector list
    K_del : numpy.ndarray
        linear abs voltage model, [P_delta,Q_delta] coeff. matrix
    K_wye : numpy.ndarray
        linear abs voltage model, [P_wye,Q_wye] coeff. matrix
    K0 : numpy.ndarray
        linear abs voltage model, constant vector
    line_config_df : pandas.DataFrame
        information on line configurations, columns: ['name','Zaa',
        'Zbb','Zcc','Zab','Zac','Zbc','Baa','Bbb','Bcc','Bab','Bac','Bbc'], 
        'Zxx' in (Ohms/length unit), 'Bxx' in (S/length unit) [base voltage of 
        Vslack]
    line_df : pandas.DataFrame
        line information: ['busA','busB','Zaa','Zbb','Zcc','Zab','Zac','Zbc',
        'Baa','Bbb','Bcc','Bab','Bac','Bbc'], 'Zxx' in (Ohms) 'Bxx' in (S) 
        [base voltage of Vslack]
    line_info_df : 
        matches lines to their configurations: ['busA','busB','config',
        'length','Z_conv_factor','B_conv_factor'], Z,B_conv_factors used to 
        match 'lenth' units to line_config_df 'Zxx' and 'Bxx' values   
    list_Yseries : list of numpy.ndarray
        list of series admittance matrices for the lines
    list_Yshunt : list of numpy.ndarray
        list of shunt admittance matrices for the lines
    M_del : numpy.ndarray
        linear voltage model, [P_delta,Q_delta] coeff. matrix
    M_wye : numpy.ndarray
        linear voltage model, [P_wye,Q_wye] coeff. matrix
    M0 : numpy.ndarray
        linear abs voltage model, constant vector
    N_buses : int
        number of buses
    N_capacitors : int
        number of capacitors
    N_iter : int
        number of Z-Bus power flow solver iterations
    N_phases : int
        number of phases
    N_lines : int
        number of lines
    N_transformers : int
        number of transformers
    res_bus_df : pandas.DataFrame
        power flow result, bus information, columns: ['name','number',
        'load_type','connect','Sa','Sb','Sc','Va','Vb','Vc','Ia','Ib','Ic']
        load_type: 'S' (slack),'PQ','Z' or 'I' (only S and PQ are currently 
        implemented), connect: 'Y' (wye) or 'D' (delta), 'Sx' in (kVA), 
        'Vx' in (V), 'Ix' in (A)
    res_lines_df : pandas.DataFrame
        power flow result, line information, columns: ['busA','busB','Sa','Sb',
        'Sc','Ia','Ib','Ic','VAa','VAb','VAc','VBa','VBb','VBc'], 'Sx' in (VA),
        'Ix' in (A), bus A voltages 'VAx' in (V), bus B voltages 'VBx' in (V).
    S_del_lin0 : numpy.ndarray
        linear model, delta apparent power load (kVA)
    S_PQloads_del_res : numpy.ndarray
        power flow result, delta apparent power load (kVA)
    S_PQloads_wye_res : numpy.ndarray
        power flow result, wye apparent power load (kVA)
    S_net_res : pandas.DataFrame
        power flow result, apparent power load at each phase (VA)
    S_wye_lin0 :
        linear model, delta apparent power load (kVA)    Y :
    transformer_df : pandas.DataFrame
        transformer information, columns: ['busA','busB','typeA','typeB',
        'Zseries','Zshunt'], typex: 'wye-g', 'wye' or 'delta'
    v_abs_min : numpy.ndarray
        min abs bus voltages [bus, phase] (|V|)
    v_abs_max : numpy.ndarray
        max abs bus voltages [bus, phase] (|V|)
    v_iter : numpy.ndarray
        bus phase voltages at each Z-Bus iteration [iteration, phase]
    v_lin_abs_res : numpy.ndarray
        linear model result, bus phase abs voltage (excl. slack) (|V|)
    v_lin_res : numpy.ndarray
        linear model result, bus phase voltage (excl. slack) (V)
    v_net_lin_abs_res : numpy.ndarray
        linear model result, bus phase abs voltage (|V|)
    v_net_lin_res : numpy.ndarray
        linear model result, bus phase voltage (V)
    v_net_lin0 : numpy.ndarray
        linear model, nominal bus phase voltages (V)
    v_net_res : numpy.ndarray
        power flow result, bus phase voltages (V)
    vs : numpy.ndarray
        slack bus phase voltages (V)
    Vslack : float
        slack bus line-to-line voltage magnitude (|V|)
    Vslack_ph : float
        slack bus line-to-phase voltage magnitude (|V|)
    v_res : numpy.ndarray
        power flow result, bus phase voltages (excl. slack) (V)
    Y : numpy.ndarray
        admittance matrix (excl. slack) (S)
    Ynet : numpy.ndarray
        admittance matrix (S)
    Y_non_singular :
        admittance matrix with 1e-20*I added (excl. slack) (S) [base voltage of 
        Vslack]
    Yns : numpy.ndarray
        admittance matrix partition [Yss, Ysn; Yns, Y] (S) [base voltage of 
        Vslack]
        voltage of Vslack
    Ysn : numpy.ndarray
        admittance matrix partition [Yss, Ysn; Yns, Y] (S) [base voltage of 
        Vslack]
    Yss : numpy.ndarray
        admittance matrix partition [Yss, Ysn; Yns, Y] (S) [base voltage of 
        Vslack]
    Z : 
        impedance matrix (Ohm) [base voltage of Vslack]
    Znet :
        impedance matrix (excl. slack) (Ohm) [base voltage of Vslack]

    Returns
    -------
    Network_3ph

    """
    
    def __init__(self):
        #Default constructor, which sets up the IEEE 13 bus network
        self.Vslack = 4.16e3 #slack bus line-to-line voltage
        self.Vslack_ph = self.Vslack/np.sqrt(3) #slack bus phase voltage
        self.N_iter = 10 #Z bus power flow iterations
        self.setup_network_ieee13() #by default set up the IEEE 13 bus network
        self.update_YandZ() #setup the Z and Y bus matricies
        #set bus voltage and line current limits
        self.set_pf_limits(0.8*self.Vslack_ph,
                           1.2*self.Vslack_ph,1000e3/self.Vslack_ph) 

    def linear_model_setup(self, v_net_lin0, S_wye_lin0, S_del_lin0):
        """
        Set up a linear model based on A. Bernstein, et al., “Load Flow in 
        Multiphase Distribution Networks: Existence, Uniqueness, 
        Non-Singularity and Linear Models,” IEEE Transactions on Power Systems,
        2018.

        Parameters
        ----------
        v_net_lin0 : numpy.ndarray
            nominal operating point, bus phase voltages (V)
        S_wye_lin0 : numpy.ndarray
            nominal operating point, apparent wye power loads (kVA)
        S_del_lin0 : numpy.ndarray
            nominal operating point, apparent delta power loads (kVA)

        """

        self.v_net_lin0 = v_net_lin0
        self.S_wye_lin0 = S_wye_lin0
        self.S_del_lin0 = S_del_lin0
        PQ_wye_lin0 = np.concatenate([np.real(S_wye_lin0),
                                      np.imag(S_wye_lin0)])*1e3
        PQ_del_lin0 = np.concatenate([np.real(S_del_lin0),
                                      np.imag(S_del_lin0)])*1e3
        v_lin0 = v_net_lin0[3:]
        v_lin0_diag = np.diag(np.conj(-v_lin0))
        #Matricies for delta loads
        del_mat = np.array([[1,-1,0],
                            [0,1,-1],
                            [-1,0,1]],dtype=np.complex_)#a-b,b-c,c-a
        del_blkmat = block_diag(*([del_mat]*(len(self.bus_df)-1)))
        #print(del_blkmat.shape)
        #del_blkmat = del_mat
        #for i in range(2,len(self.bus_df)):   
            #del_blkmat = block_diag(del_blkmat,del_mat)
            
        Hv_lin0 = np.matmul(del_blkmat,-np.conj(v_lin0))
        v_lin0_diag_inv = np.zeros([v_lin0.shape[0],
                                    v_lin0.shape[0]],dtype=np.complex_)
        Hv_lin0_diag_inv = np.zeros([v_lin0.shape[0],
                                     v_lin0.shape[0]],dtype=np.complex_)
        for bus_i in range(1,len(self.bus_df)):
            aph_index = (bus_i-1)*3
            bph_index = (bus_i-1)*3+1
            cph_index = (bus_i-1)*3+2
            if np.abs(v_lin0_diag[aph_index,aph_index]) > 0:
                v_lin0_diag_inv[aph_index,aph_index] \
                        = 1/v_lin0_diag[aph_index,aph_index]
            if np.abs(v_lin0_diag[bph_index,bph_index]) > 0:
                v_lin0_diag_inv[bph_index,bph_index] \
                        = 1/v_lin0_diag[bph_index,bph_index]
            if np.abs(v_lin0_diag[cph_index,cph_index]) > 0:
                v_lin0_diag_inv[cph_index,cph_index] \
                        = 1/v_lin0_diag[cph_index,cph_index]
            if np.abs(Hv_lin0[aph_index]) > 0:
                Hv_lin0_diag_inv[aph_index,aph_index] = 1/Hv_lin0[aph_index]
            if np.abs(Hv_lin0[bph_index]) > 0:
                Hv_lin0_diag_inv[bph_index,bph_index] = 1/Hv_lin0[bph_index]
            if np.abs(Hv_lin0[cph_index]) > 0:
                Hv_lin0_diag_inv[cph_index,cph_index] = 1/Hv_lin0[cph_index]            
        #Linear model v = M_wye*x + M_del*x + M0 where x =(p1,...,pN,q1,...,qN)
        Mp_wye =     np.matmul(self.Z,v_lin0_diag_inv)
        Mq_wye = -1j*np.matmul(self.Z,v_lin0_diag_inv)
        self.M_wye = np.concatenate([Mp_wye,Mq_wye],1)
        Mp_del =     np.matmul(self.Z,np.matmul(del_blkmat.T,Hv_lin0_diag_inv))
        Mq_del = -1j*np.matmul(self.Z,np.matmul(del_blkmat.T,Hv_lin0_diag_inv))
        self.M_del = np.concatenate([Mp_del,Mq_del],1)
        self.M0 = np.matmul(-self.Z,np.matmul(self.Yns,self.vs))        
        #Lin. |V| model |v| = K_wye*x + K_del*x + K0, x = (p1,...,pN,q1,...,qN)
        self.K_wye = -np.matmul(np.abs(v_lin0_diag_inv),
                                np.real(np.matmul(v_lin0_diag,self.M_wye)))
        self.K_del = -np.matmul(np.abs(v_lin0_diag_inv),
                                np.real(np.matmul(v_lin0_diag,self.M_del)))
        self.K0 = np.abs(v_lin0) - np.matmul(self.K_wye,PQ_wye_lin0) \
                                    - np.matmul(self.K_del,PQ_del_lin0)

        # liner s0 model = G_wye*x + G_del*x + G0 where x = (p1,...,pN,q1,...,qN)
        self.G_wye = np.matmul(np.diag(self.vs), np.matmul(np.conj(self.Ysn), np.conj(self.M_wye)))
        self.G_del = np.matmul(np.diag(self.vs), np.matmul(np.conj(self.Ysn), np.conj(self.M_del)))
        self.G0 = np.matmul(np.diag(self.vs), \
                            np.matmul(np.conj(self.Yss), np.conj(self.vs)) \
                            + np.matmul(np.conj(self.Ysn), np.conj(self.M0)))

        #Linear line current model
        #   i_ij = J_dPQwye*dx + J_dPQdel*dx 
        #           + J_I0 (J_wye*x0 + J_del*x0 + J0) and 
        #Linear line current model 
        #   |i_ij| = Jabs_dPQwye*dx + Jabs_dPQdel*dx 
        #               + Jabs_I0 (Jabs_wye*x + Jabs_del*x + Jabs0)
        self.J_dPQwye_list = []
        self.J_dPQdel_list = []
        self.J_I0_list = []
        self.Jabs_dPQwye_list = []
        self.Jabs_dPQdel_list = []
        self.Jabs_I0_list = []
        for line_ij in range(self.N_lines):
            line_ij_df = self.line_df.iloc[line_ij]
            bus_i = self.bus_df[self.bus_df['name']
                        ==line_ij_df['busA']]['number'].values[0]-1 
                            #-1 since we are removing the slack node
            bus_j = self.bus_df[self.bus_df['name']
                    ==line_ij_df['busB']]['number'].values[0]-1
            E_i = np.zeros([3,3*(self.N_buses-1)])
            E_j = np.zeros([3,3*(self.N_buses-1)])

            ### below is the correction by Yihong
            M0_net = np.concatenate([np.zeros(3), self.M0])

            v_i_abc = M0_net[3*(bus_i+1):3*(bus_i+2)]
            v_j_abc = M0_net[3*(bus_j+1):3*(bus_j+2)]
            ### correction end

            if bus_i >= 0:
                E_i[:,3*bus_i:3*(bus_i+1)] = np.eye(3)
            if bus_j >= 0:
                E_j[:,3*bus_j:3*(bus_j+1)] = np.eye(3)
            Yshunt = self.list_Yshunt[line_ij]
            Yseries = self.list_Yseries[line_ij]
            J_dPQwye = np.matmul(np.matmul(Yshunt+Yseries,E_i)
                            -np.matmul(Yseries,E_j),self.M_wye)
            J_dPQdel = np.matmul(np.matmul(Yshunt+Yseries,E_i)
                            -np.matmul(Yseries,E_j),self.M_del)
            J_I0 = np.matmul(Yshunt+Yseries,v_i_abc)-np.matmul(Yseries,v_j_abc)

            # below is the correction bu Yihong (do not use it)
            # J_I0 = np.matmul(np.matmul(Yshunt+Yseries,E_i)
            #                 -np.matmul(Yseries,E_j),self.M0)

            self.J_dPQwye_list.append(J_dPQwye)
            self.J_dPQdel_list.append(J_dPQdel)
            self.J_I0_list.append(J_I0)
            i_ij_lin0 = J_I0
            
            ### Yihong Modification
            # i_ij_lin0 = np.matmul(self.J_dPQwye_list[line_ij], PQ_wye) \
            #             + np.matmul(self.J_dPQdel_list[line_ij], PQ_del) \
            #             + self.J_I0_list[line_ij]
            ### Yihong Modification end

            i_ij_lin0_inv_diag = np.zeros([i_ij_lin0.size,i_ij_lin0.size],
                                          dtype=np.complex_)
            i_ij_lin0_conj_diag = np.zeros([i_ij_lin0.size,i_ij_lin0.size],
                                          dtype=np.complex_)
            for ph_index in range(i_ij_lin0.size):
                if np.abs(i_ij_lin0[ph_index]) > 0:
                    i_ij_lin0_inv_diag[ph_index,ph_index] \
                                                = 1/i_ij_lin0[ph_index] # Yihong Slight Modification, remove abs here;original is 1/np.abs(i_ij_lin0[ph_index])
                    i_ij_lin0_conj_diag[ph_index,ph_index] \
                                                = np.conj(i_ij_lin0[ph_index])

            ### The original version
            # Jabs_dPQwye = np.matmul(i_ij_lin0_inv_diag,
            #                         np.real(np.matmul(i_ij_lin0_conj_diag, J_dPQwye)))
            # Jabs_dPQdel = np.matmul(i_ij_lin0_inv_diag,
            #                         np.real(np.matmul(i_ij_lin0_conj_diag, J_dPQdel)))
            # Jabs0 = np.abs(i_ij_lin0)

            # Yihong Modification
            Jabs_dPQwye = np.matmul(np.abs(i_ij_lin0_inv_diag),
                            np.real(np.matmul(i_ij_lin0_conj_diag,J_dPQwye)))
            Jabs_dPQdel = np.matmul(np.abs(i_ij_lin0_inv_diag),
                            np.real(np.matmul(i_ij_lin0_conj_diag,J_dPQdel)))
            Jabs0 = np.abs(i_ij_lin0)
            # Modification END
            
            self.Jabs_dPQwye_list.append(Jabs_dPQwye)
            self.Jabs_dPQdel_list.append(Jabs_dPQdel)
            self.Jabs_I0_list.append(Jabs0)
            
            
    def linear_pf(self):
        """
        Solves the linear model based on A. Bernstein, et al., “Load Flow in 
        Multiphase Distribution Networks: Existence, Uniqueness, 
        Non-Singularity and Linear Models,” IEEE Transactions on Power Systems,
        2018. 
        
        Requires running linear_model_setup() first.

        """
        S_loads = np.zeros([self.Y.shape[0]],dtype=np.complex_) 
            #S loads
        S_PQ_wye = np.zeros([self.Y.shape[0]],dtype=np.complex_) 
            #S of PQ loads (wye)
        S_PQ_del = np.zeros([self.Y.shape[0]],dtype=np.complex_) 
            #S of PQ loads (delta)
        for bus_i in range(1,len(self.bus_df)):
            #not including slack bus (bus 0)
            aph_index = (bus_i-1)*3
            bph_index = (bus_i-1)*3+1
            cph_index = (bus_i-1)*3+2
            S_loads[aph_index] = (self.bus_df.iloc[bus_i]['Pa']
                                    + self.bus_df.iloc[bus_i]['Qa']*1j)*1e3
            S_loads[bph_index] = (self.bus_df.iloc[bus_i]['Pb'] 
                                    + self.bus_df.iloc[bus_i]['Qb']*1j)*1e3
            S_loads[cph_index] = (self.bus_df.iloc[bus_i]['Pc'] 
                                    + self.bus_df.iloc[bus_i]['Qc']*1j)*1e3
            if self.bus_df.iloc[bus_i]['connect'] == 'Y': 
                #bus_df.iloc[bus_i]['load_type'] == 'PQ' and 
                S_PQ_wye[aph_index] = S_loads[aph_index]
                S_PQ_wye[bph_index] = S_loads[bph_index]
                S_PQ_wye[cph_index] = S_loads[cph_index]
            if self.bus_df.iloc[bus_i]['connect'] == 'D': 
                #bus_df.iloc[bus_i]['load_type'] == 'PQ' and 
                S_PQ_del[aph_index] = S_loads[aph_index] #[Sab]
                S_PQ_del[bph_index] = S_loads[bph_index] #[Sbc]
                S_PQ_del[cph_index] = S_loads[cph_index] #[Scb]        
        PQ_wye = np.concatenate([np.real(S_PQ_wye),np.imag(S_PQ_wye)])
        PQ_del = np.concatenate([np.real(S_PQ_del),np.imag(S_PQ_del)])
        PQ_wye = np.nan_to_num(PQ_wye)
        PQ_del = np.nan_to_num(PQ_del)

        #Perform a linear v calculation
        self.v_lin_res = np.matmul(self.M_wye,PQ_wye) \
                            + np.matmul(self.M_del,PQ_del) + self.M0
        self.v_net_lin_res = np.concatenate((self.vs,self.v_lin_res))
        #Perform a linear |V| calculation
        self.v_lin_abs_res = np.matmul(self.K_wye,PQ_wye) \
                            + np.matmul(self.K_del,PQ_del) + self.K0
        self.v_net_lin_abs_res = \
            np.concatenate((np.abs(self.vs),self.v_lin_abs_res))

        #below are added by Yihong
        # perform a linear i calculation
        i_lin_res = []
        i_lin_abs_res = []
        for line_ij in range(self.N_lines):
            i_lin_res_ij = np.matmul(self.J_dPQwye_list[line_ij],PQ_wye) \
                          + np.matmul(self.J_dPQdel_list[line_ij], PQ_del) \
                          + self.J_I0_list[line_ij]
            i_lin_res.append(i_lin_res_ij)

            i_lin_abs_res_ij = np.matmul(self.Jabs_dPQwye_list[line_ij], PQ_wye) \
                              + np.matmul(self.Jabs_dPQdel_list[line_ij], PQ_del) \
                              + self.Jabs_I0_list[line_ij]
            i_lin_abs_res.append(i_lin_abs_res_ij)

        i_lin_res = np.concatenate(i_lin_res)
        i_lin_abs_res = np.concatenate(i_lin_abs_res)

        self.i_lin_res = i_lin_res
        self.i_lin_abs_res = i_lin_abs_res

        # perform a linear complex power calculation at the slack bus
        self.Ss = np.matmul(self.G_wye, PQ_wye) \
                  + np.matmul(self.G_del, PQ_del) + self.G0

        # calculate the active power
        self.Ps = np.matmul(np.real(self.G_wye), PQ_wye) \
                  + np.matmul(np.real(self.G_del), PQ_del) + np.real(self.G0)

        # calculate the reactive power
        self.Qs = np.matmul(np.imag(self.G_wye), PQ_wye) \
                  + np.matmul(np.imag(self.G_del), PQ_del) + np.imag(self.G0)

        self.Ss /= 1e3


    def get_parameters(self, assets_nd, assets_flex, t_ems, t0):
        """
        Get linear parameters for central optimisation with network constraints

        Parameters
        ------------
        assets_nd : list
            list of nondispatchable assets
        assets_flex : list
            list of flexible assets (storage and building)
        t_ems : int
            current time slot of optimisation
        t0 : int t0<=t_ems
            starting point of the optimisation

        Returns
        ----------
        A_Pslack_flex : 2d array
            submatrix(G) for which the rows corespond to buses connected to flexible loads
        A_Pslack_nd : 2d array
            submatrix(G) for which the rows corespond to buses connected to nondispatchable loads
        b_Pslack : float
            g_0

        A_vlim_flex : 2d array
            submatrix(K) for which the rows corespond to buses connected to flexible loads
        A_vlim_nd : 2d array
            submatrix(K) for which the rows corespond to buses connected to nondispatchable loads
        b_vlim : float
            k_0
        v_abs_min_vec : 1d array
            vector of maximum voltage limit
        v_abs_max_vec : 1d array
            vector of minimum voltage limit

        A_lines_flex : list
            list of submatrices(J) over lines connected to flexible loads
        A_lines_nd : list
            list of submatrices(J) over lines connected to nondispatchable loads
        
        A_Pslack_wye_flex : 2d array
            submatrix(G) for which the rows corespond to buses connected to flexible loads, type='Y'
        A_Pslack_del_flex : 2d array
            submatrix(G) for which the rows corespond to buses connected to flexible loads, type='Delta'
        A_vlim_wye_flex : 2d array
            submatrix(K) for which the rows corespond to buses connected to flexible loads, type='Y'
        A_vlim_del_flex : 2d array
            submatrix(K) for which the rows corespond to buses connected to flexible loads, type='Delta'
        A_line_wye_flex : list
            list of submatrices(J) over lines connected to flexible loads, type='Y' 
        A_line_del_flex : list
            list of submatrices(J) over lines connected to flexible loads, type='Delta'
        
        A_Pslack_wye_nd : 2d array
            submatrix(G) for which the rows corespond to buses connected to nondispatchable loads, type='Y'
        A_Pslack_del_nd : 2d array
            submatrix(G) for which the rows corespond to buses connected to nondispatchable loads, type='Delta'
        A_vlim_wye_nd : 2d array
            submatrix(K) for which the rows corespond to buses connected to nondispatchable loads, type='Y' 
        A_vlim_del_nd : 2d array
            submatrix(K) for which the rows corespond to buses connected to nondispatchable loads, type='Delta'
        A_line_wye_nd : list
            list of submatrices(J) over lines connected to nondispatchable loads, type='Y'
        A_line_del_nd : list
            list of submatrices(J) over lines connected to nondispatchable loads, type='Delta'

        """

        if len(assets_nd):
            T = assets_nd[0].T
            dt = assets_nd[0].dt
            T_ems = assets_nd[0].T_ems
            dt_ems = assets_nd[0].dt_ems
        elif len(assets_flex):
            T = assets_flex[0].T
            dt = assets_flex[0].dt
            T_ems = assets_flex[0].T_ems
            dt_ems = assets_flex[0].dt_ems

        #Set up Matrix linking nondispatchable assets to their bus and phase
        G_wye_nondispatch,  G_del_nondispatch = self.get_Gs(assets_nd) #nondispatch
        #Set up Matrix linking energy storage assets to their bus and phase
        G_wye_ES, G_del_ES = self.get_Gs(assets_flex)
        #PQ Gs
        G_wye_nondispatch_PQ = np.concatenate((G_wye_nondispatch, G_wye_nondispatch),axis=0)
        G_del_nondispatch_PQ = np.concatenate((G_del_nondispatch, G_del_nondispatch),axis=0)
        G_wye_ES_PQ = np.concatenate((G_wye_ES,G_wye_ES),axis=0)
        G_del_ES_PQ = np.concatenate((G_del_ES,G_del_ES),axis=0)

        P_lin_buses = np.zeros([self.N_buses,self.N_phases])
        Q_lin_buses = np.zeros([self.N_buses,self.N_phases])
        
        #Setup linear power flow model:
        for i in range(len(assets_nd)):
            bus_id = assets_nd[i].bus_id
            phases_i = assets_nd[i].phases
            for ph_i in np.nditer(phases_i):
                bus_ph_index = 3*(bus_id-1) + ph_i
                if t_ems == t0:
                    P_lin_buses[bus_id,ph_i] +=\
                    (G_wye_nondispatch[bus_ph_index,i]+\
                     G_del_nondispatch[bus_ph_index,i])*assets_nd[i].Pnet_ems[t_ems] #P_demand[t_ems-t0, i]
                    Q_lin_buses[bus_id,ph_i] +=\
                    (G_wye_nondispatch[bus_ph_index,i]+\
                     G_del_nondispatch[bus_ph_index,i])*assets_nd[i].Qnet_ems[t_ems] #Q_demand[t_ems-t0, i]
                else:
                    P_lin_buses[bus_id,ph_i] +=\
                    (G_wye_nondispatch[bus_ph_index,i]+\
                     G_del_nondispatch[bus_ph_index,i])*assets_nd[i].Pnet_ems_pred[t_ems] #P_demand[t_ems-t0, i]
                    Q_lin_buses[bus_id,ph_i] +=\
                    (G_wye_nondispatch[bus_ph_index,i]+\
                     G_del_nondispatch[bus_ph_index,i])*assets_nd[i].Qnet_ems_pred[t_ems]

        #set up a copy of the network for MPC interval t
        network_t = copy.deepcopy(self)#.network)
        network_t.clear_loads()  #self.clear
        for bus_id in range(self.N_buses):
            for ph_i in range(self.N_phases):
                Pph_t = P_lin_buses[bus_id,ph_i]
                Qph_t = Q_lin_buses[bus_id,ph_i]
                #add P,Q loads to the network copy
                network_t.set_load(bus_id,ph_i,Pph_t,Qph_t) #self.

        network_t.zbus_pf()
        network_t.linear_model_setup(network_t.v_net_res, network_t.S_PQloads_wye_res, network_t.S_PQloads_del_res)
        # note that phases need to be 120degrees out for good results
        network_t.linear_pf()


        # Note that linear power flow matricies are in units of W (not kW)
        PQ0_wye = np.concatenate((np.real(network_t.S_PQloads_wye_res),\
                                  np.imag(network_t.S_PQloads_wye_res)))\
                                  *1e3
        PQ0_del = np.concatenate((np.real(network_t.S_PQloads_del_res),\
                                  np.imag(network_t.S_PQloads_del_res)))\
                                  *1e3
        A_Pslack_flex = (np.matmul\
                    (np.real(np.matmul\
                             (network_t.vs.T,\
                              np.matmul(np.conj(network_t.Ysn),\
                                        np.conj(network_t.M_wye)))),\
                                  G_wye_ES_PQ)\
                     + np.matmul\
                     (np.real(np.matmul\
                              (network_t.vs.T,\
                               np.matmul(np.conj(network_t.Ysn),\
                                         np.conj(network_t.M_del)))),\
                                  G_del_ES_PQ))

        A_Pslack_nd = (np.matmul\
                    (np.real(np.matmul\
                             (network_t.vs.T,\
                              np.matmul(np.conj(network_t.Ysn),\
                                        np.conj(network_t.M_wye)))),\
                                  G_wye_nondispatch_PQ)\
                     + np.matmul\
                     (np.real(np.matmul\
                              (network_t.vs.T,\
                               np.matmul(np.conj(network_t.Ysn),\
                                         np.conj(network_t.M_del)))),\
                                  G_del_nondispatch_PQ))

        b_Pslack =  np.real(np.matmul\
                            (network_t.vs.T,\
                             np.matmul(np.conj\
                                       (network_t.Ysn),\
                                       np.matmul(np.conj\
                                                 (network_t.M_wye),\
                                                 PQ0_wye))))\
                    +np.real(np.matmul\
                             (network_t.vs.T,\
                              np.matmul(np.conj\
                                        (network_t.Ysn),\
                                        np.matmul(np.conj\
                                                  (network_t.M_del),
                                                  PQ0_del))))\
                    +np.real(np.matmul\
                             (network_t.vs.T,\
                              (np.matmul(np.conj\
                                         (network_t.Yss),\
                                         np.conj(network_t.vs))\
                             + np.matmul(np.conj\
                                         (network_t.Ysn),\
                                         np.conj(network_t.M0)))))

        
        # Voltage magnitude constraints
        A_vlim_flex = np.matmul(network_t.K_wye,G_wye_ES_PQ)\
                + np.matmul(network_t.K_del,G_del_ES_PQ)
        A_vlim_nd = np.matmul(network_t.K_wye,G_wye_nondispatch_PQ)\
                + np.matmul(network_t.K_del,G_del_nondispatch_PQ)
        b_vlim = network_t.v_lin_abs_res

        #get max/min bus voltages, removing slack and reshaping in a column
        v_abs_max_vec = network_t.v_abs_max[1:,:].reshape(-1,1)
        v_abs_min_vec = network_t.v_abs_min[1:,:].reshape(-1,1)

        # current 
        A_lines_flex, A_line_wye_flex, A_line_del_flex, A_lines_nd, A_line_wye_nd, A_line_del_nd  = [], [], [], [], [], []
        for line_ij in range(self.N_lines):
            #if line_ij not in i_unconstrained_lines:

                iabs_max_line_ij = network_t.i_abs_max[line_ij,:] #3 phases
                # maximum current magnitude constraint
                A_line_flex = np.matmul(network_t.Jabs_dPQwye_list[line_ij],\
                                   G_wye_ES_PQ)\
                                   + np.matmul(network_t.\
                                               Jabs_dPQdel_list[line_ij],\
                                               G_del_ES_PQ)
                A_lines_flex.append(A_line_flex)

                A_line_wye_flex.append(np.matmul(network_t.Jabs_dPQwye_list[line_ij],G_wye_ES_PQ))
                A_line_del_flex.append(np.matmul(network_t.Jabs_dPQdel_list[line_ij],G_del_ES_PQ))

                A_line_nd = np.matmul(network_t.Jabs_dPQwye_list[line_ij],\
                                   G_wye_nondispatch_PQ)\
                                   + np.matmul(network_t.\
                                               Jabs_dPQdel_list[line_ij],\
                                               G_del_nondispatch_PQ)
                A_lines_nd.append(A_line_nd)
                A_line_wye_nd.append(np.matmul(network_t.Jabs_dPQwye_list[line_ij],G_wye_nondispatch_PQ))
                A_line_del_nd.append(np.matmul(network_t.Jabs_dPQdel_list[line_ij],G_del_nondispatch_PQ))
        
        A_Pslack_wye_flex = np.matmul(np.real(np.matmul(network_t.vs.T,\
                                                              np.matmul(np.conj(network_t.Ysn),\
                                                              np.conj(network_t.M_wye)))),G_wye_ES_PQ)
        A_Pslack_del_flex = np.matmul(np.real(np.matmul (network_t.vs.T,\
                                                                          np.matmul(np.conj(network_t.Ysn),\
                                                                          np.conj(network_t.M_del)))), G_del_ES_PQ)
        
        A_vlim_wye_flex = np.matmul(network_t.K_wye,G_wye_ES_PQ)
        A_vlim_del_flex = np.matmul(network_t.K_del,G_del_ES_PQ)
        
        A_Pslack_wye_nd = np.matmul(np.real(np.matmul(network_t.vs.T,\
                                                              np.matmul(np.conj(network_t.Ysn),\
                                                              np.conj(network_t.M_wye)))),G_wye_nondispatch_PQ)
        A_Pslack_del_nd = np.matmul(np.real(np.matmul (network_t.vs.T,\
                                                                          np.matmul(np.conj(network_t.Ysn),\
                                                                          np.conj(network_t.M_del)))), G_del_nondispatch_PQ)
        
        A_vlim_wye_nd = np.matmul(network_t.K_wye,G_wye_nondispatch_PQ)
        A_vlim_del_nd = np.matmul(network_t.K_del,G_del_nondispatch_PQ)
        
        
        return A_Pslack_flex, A_Pslack_nd, b_Pslack, A_vlim_flex, A_vlim_nd, b_vlim, v_abs_min_vec, v_abs_max_vec, A_lines_flex, A_lines_nd,\
               A_Pslack_wye_flex, A_Pslack_del_flex, A_vlim_wye_flex, A_vlim_del_flex, A_line_wye_flex, A_line_del_flex, \
               A_Pslack_wye_nd, A_Pslack_del_nd, A_vlim_wye_nd, A_vlim_del_nd, A_line_wye_nd, A_line_del_nd
        
        #return A_Pslack, b_Pslack, 0, 0, 0, 0, 0, A_Pslack_wye, A_Pslack_del, 0, 0, 0, 0

        
    def get_Gs(self, assets):
        """
        Set up matrices G_wye and G_del linking a group of assets to their bus and phase

        Parameters
        -----------
        assets : list
            list of assets

        Returns
        -----------
        G_wye : numpy.ndarray (no unit)
            size (3*(N_buses-1),nbr of assets)
        G_del : numpy.ndarray (no unit)
            size (3*(N_buses-1),nbr of assets)

        """

        N_buses = self.N_buses
        N_phases = self.N_phases
        G_wye = np.zeros([3*(N_buses-1),len(assets)])
        G_del = np.zeros([3*(N_buses-1),len(assets)])
        for i in range(len(assets)):
            asset_N_phases = assets[i].phases.size
            bus_id = assets[i].bus_id
            # check if Wye connected
            wye_flag = self.bus_df[self.bus_df['number']==\
                                           bus_id]['connect'].values[0]=='Y'
            for ph in np.nditer(assets[i].phases):
                bus_ph_index = 3*(bus_id-1) + ph
                if wye_flag is True:
                    G_wye[bus_ph_index,i] = 1/asset_N_phases
                else:
                    G_del[bus_ph_index,i] = 1/asset_N_phases
        return G_wye, G_del

    def get_linear_parameters(self, assets_nd, t):
        """
        Get linear parameters for linear optimisation

        Parameters
        -----------
        assets_nd : list
            list of the non_dispatchable assets in the network
        t : int
            time

        Returns
        --------
        A_Pslack_nd : 2d array
            submatrix(G) for which the rows corespond to buses connected to nondispatchable loads
        b_Pslack : float
            g_0
        A_vlim_nd : 2d array
            submatrix(K) for which the rows corespond to buses connected to nondispatchable loads
        b_vlim : float
            k_0
        v_abs_min_vec :
        v_abs_max_vec : 1d array
            vector of minimum voltage limit
        A_lines : list
            list of submatrices(J) over lines connected to nondispatchable loads
        i_lim : float
            j0
        i_abs_max :  1d array
            vector of maximum thermal limit
        
        """

        P_lin_buses = np.zeros([self.N_buses,self.N_phases])
        Q_lin_buses = np.zeros([self.N_buses,self.N_phases])    
        G_wye_nondispatch,  G_del_nondispatch = self.get_Gs(assets_nd)
        G_wye_nondispatch_PQ = np.concatenate((G_wye_nondispatch, G_wye_nondispatch),axis=0)
        G_del_nondispatch_PQ = np.concatenate((G_del_nondispatch, G_del_nondispatch),axis=0)

        #Setup linear power flow model:
        for i in range(len(assets_nd)):
            bus_id = assets_nd[i].bus_id
            phases_i = assets_nd[i].phases
            for ph_i in np.nditer(phases_i):
                bus_ph_index = 3*(bus_id-1) + ph_i
                P_lin_buses[bus_id,ph_i] +=\
                (G_wye_nondispatch[bus_ph_index,i]+\
                 G_del_nondispatch[bus_ph_index,i])*assets_nd[i].Pnet_ems[t]
                Q_lin_buses[bus_id,ph_i] +=\
                (G_wye_nondispatch[bus_ph_index,i]+\
                 G_del_nondispatch[bus_ph_index,i])*assets_nd[i].Qnet_ems[t]

        network_t = copy.deepcopy(self)#.network)
        network_t.clear_loads()  #self.clear
        for bus_id in range(self.N_buses):
            for ph_i in range(self.N_phases):
                Pph_t = P_lin_buses[bus_id,ph_i]
                Qph_t = Q_lin_buses[bus_id,ph_i]
                #add P,Q loads to the network copy
                network_t.set_load(bus_id,ph_i,Pph_t,Qph_t) #self.

        network_t.zbus_pf()
        network_t.linear_model_setup(network_t.v_net_res, network_t.S_PQloads_wye_res, network_t.S_PQloads_del_res)
        # note that phases need to be 120degrees out for good results
        network_t.linear_pf()


        # Note that linear power flow matricies are in units of W (not kW)

        PQ0_wye = np.concatenate((np.real(network_t.S_PQloads_wye_res),\
                                  np.imag(network_t.S_PQloads_wye_res)))\
                                  *1e3
        PQ0_del = np.concatenate((np.real(network_t.S_PQloads_del_res),\
                                  np.imag(network_t.S_PQloads_del_res)))\
                                  *1e3
        A_Pslack_nd = (np.matmul\
                    (np.real(np.matmul\
                             (network_t.vs.T,\
                              np.matmul(np.conj(network_t.Ysn),\
                                        np.conj(network_t.M_wye)))),\
                                  G_wye_nondispatch_PQ)\
                     + np.matmul\
                     (np.real(np.matmul\
                              (network_t.vs.T,\
                               np.matmul(np.conj(network_t.Ysn),\
                                         np.conj(network_t.M_del)))),\
                                  G_del_nondispatch_PQ))

        b_Pslack =  np.real(np.matmul\
                            (network_t.vs.T,\
                             np.matmul(np.conj\
                                       (network_t.Ysn),\
                                       np.matmul(np.conj\
                                                 (network_t.M_wye),\
                                                 PQ0_wye))))\
                    +np.real(np.matmul\
                             (network_t.vs.T,\
                              np.matmul(np.conj\
                                        (network_t.Ysn),\
                                        np.matmul(np.conj\
                                                  (network_t.M_del),
                                                  PQ0_del))))\
                    +np.real(np.matmul\
                             (network_t.vs.T,\
                              (np.matmul(np.conj\
                                         (network_t.Yss),\
                                         np.conj(network_t.vs))\
                             + np.matmul(np.conj\
                                         (network_t.Ysn),\
                                         np.conj(network_t.M0)))))
        
        # Voltage magnitude constraints
        A_vlim_nd = np.matmul(network_t.K_wye,G_wye_nondispatch_PQ)\
                + np.matmul(network_t.K_del,G_del_nondispatch_PQ)
        b_vlim = network_t.v_lin_abs_res

        #get max/min bus voltages, removing slack and reshaping in a column
        v_abs_max_vec = network_t.v_abs_max[1:,:].reshape(-1,1)
        v_abs_min_vec = network_t.v_abs_min[1:,:].reshape(-1,1)

        A_lines = np.empty((self.N_phases, len(assets_nd)))
        i_abs_max = []
        i_lim = []
        for line_ij in range(self.N_lines):
                #if line_ij not in i_unconstrained_lines:
                iabs_max_line_ij = network_t.i_abs_max[line_ij,:] #3 phases
                #concatenate
                i_abs_max.append(iabs_max_line_ij)

                i_lim.append(network_t.Jabs_I0_list[line_ij])

                # maximum current magnitude constraint
                A_line = np.matmul(network_t.Jabs_dPQwye_list[line_ij],\
                                   G_wye_nondispatch_PQ)\
                                   + np.matmul(network_t.\
                                               Jabs_dPQdel_list[line_ij],\
                                               G_del_nondispatch_PQ)
                #concatenate in rows   Aline (N_phases, N_loads) and we have 1Aline per ij line
                A_lines = np.concatenate((A_lines, A_line), axis=0)                  

        return A_Pslack_nd, b_Pslack, A_vlim_nd, b_vlim, v_abs_min_vec, v_abs_max_vec, A_lines, i_lim, np.array(i_abs_max)

    def zbus_pf(self):
        """
        Solves the nonlinear power flow problem using the Z-bus method 
        from M. Bazrafshan, N. Gatsis, “Comprehensive Modeling of Three-Phase 
        Distribution Systems via the Bus Admittance Matrix,” IEEE Transactions 
        on Power Systems, 2018.

        """
        S_loads = np.zeros([self.Y.shape[0]],dtype=np.complex_) 
        #S of PQ loads (wye)
        S_PQ_wye = np.zeros([self.Y.shape[0]],dtype=np.complex_) 
        #S of PQ loads (delta)
        S_PQ_del = np.zeros([self.Y.shape[0]],dtype=np.complex_) 

        S_loads[0::3]=(self.bus_df['Pa']+ self.bus_df['Qa']*1j)[1:].values*1e3
        S_loads[1::3]=(self.bus_df['Pb']+ self.bus_df['Qb']*1j)[1:].values*1e3
        S_loads[2::3]=(self.bus_df['Pc']+ self.bus_df['Qc']*1j)[1:].values*1e3

        S_loads = np.nan_to_num(S_loads)

        for bus_i in range(1,len(self.bus_df)):
#            #not including slack bus (bus 0)
            aph_index = (bus_i-1)*3
            bph_index = (bus_i-1)*3+1
            cph_index = (bus_i-1)*3+2
#            S_loads[aph_index] = (self.bus_df.iloc[bus_i]['Pa'] 
#                                   + self.bus_df.iloc[bus_i]['Qa']*1j)*1e3
#            S_loads[bph_index] = (self.bus_df.iloc[bus_i]['Pb'] 
#                                   + self.bus_df.iloc[bus_i]['Qb']*1j)*1e3
#            S_loads[cph_index] = (self.bus_df.iloc[bus_i]['Pc'] 
#                                   + self.bus_df.iloc[bus_i]['Qc']*1j)*1e3
            if self.bus_df.iloc[bus_i]['connect'] == 'Y': 
                #bus_df.iloc[bus_i]['load_type'] == 'PQ' and 
                S_PQ_wye[aph_index] = S_loads[aph_index]
                S_PQ_wye[bph_index] = S_loads[bph_index]
                S_PQ_wye[cph_index] = S_loads[cph_index]
            elif self.bus_df.iloc[bus_i]['connect'] == 'D': 
                #bus_df.iloc[bus_i]['load_type'] == 'PQ' and 
                S_PQ_del[aph_index] = S_loads[aph_index] #[Sab]
                S_PQ_del[bph_index] = S_loads[bph_index] #[Sbc]
                S_PQ_del[cph_index] = S_loads[cph_index] #[Scb] 
        
        #elapsed = time.time() - t
        #print('Add loads time: ' + str(elapsed))
        #t = time.time() 
        
        #Z bus method (PQ buses only)
        aaa = np.exp(1j*np.pi*2/3)
        # NB: order matters. Positive sequence order is not [1 aaa aaa**2]
        self.vs = np.array([1,aaa**2,aaa],dtype=np.complex_)*self.Vslack_ph 
        #slack node voltage. 
        v = np.zeros([self.N_iter,self.Y.shape[0]],dtype=np.complex_)

        for phase_i in range(self.Y.shape[0]):    
            v[0,phase_i] = self.vs[phase_i%3]
        dv = np.zeros([self.N_iter])
        dv[0] = np.sum(np.abs(v[0,:]))
        #Get initial current injections
        i_PQ = np.zeros([self.N_iter,self.Y.shape[0]],dtype=np.complex_)
        #del_mat = np.array([[1,-1,0],[0,1,-1],[-1,0,1]],dtype=np.complex_)
        #del_blkmat = block_diag(*([del_mat]*(len(self.bus_df)-1)))

        #del_blkmat = del_mat
        #for i in range(2,len(self.bus_df)):   
        #    del_blkmat = block_diag(del_blkmat,del_mat)
        #Initital current injections
        for phase_i in range(self.Y.shape[0]):    
            if np.abs(S_PQ_wye[phase_i]) > 0:
                i_PQ[0,phase_i] = -np.conj(S_PQ_wye[phase_i]/v[0,phase_i])
            if np.abs(S_PQ_del[phase_i]) > 0:
                phase_i_mod = phase_i%3
                if phase_i_mod == 0:
                    i_PQ[0,phase_i] = \
                    -np.conj(S_PQ_del[phase_i]/(v[0,phase_i]-v[0,phase_i+1]))\
                    -np.conj(S_PQ_del[phase_i+2]/(v[0,phase_i]-v[0,phase_i+2])) 
                elif phase_i_mod == 1:
                    i_PQ[0,phase_i] = \
                -np.conj(S_PQ_del[phase_i-1]/(v[0,phase_i]-v[0,phase_i-1]))\
                    -np.conj(S_PQ_del[phase_i]/(v[0,phase_i]-v[0,phase_i+1])) 
                else:
                    i_PQ[0,phase_i] = \
                    -np.conj(S_PQ_del[phase_i]/(v[0,phase_i]-v[0,phase_i-2]))\
                    -np.conj(S_PQ_del[phase_i-1]/(v[0,phase_i]-v[0,phase_i-1]))         
        di = np.zeros([self.N_iter]) 
        di[0] = np.sum(np.abs(i_PQ[0,:]))        
        #Iteratively:
        v0 = np.matmul(-self.Z,np.matmul(self.Yns,self.vs))
        for k in range(1,self.N_iter):
            #update the bus voltages
            v[k,:] = np.matmul(self.Z,i_PQ[k-1,:]) + v0
            dv[k] = np.sum(np.abs(v[k,:]-v[k-1,:])) 
                #change in |V| sum during the iteration
            #update current injections
            for phase_i in range(self.Y.shape[0]):    
                if np.abs(v[k,phase_i]) > 0:
                    i_PQ[k,phase_i] = i_PQ[k,phase_i] \
                        -np.conj(S_PQ_wye[phase_i]/v[k,phase_i])
                phase_i_mod = phase_i%3
                if phase_i_mod == 0: #i_a
                    if np.abs(v[k,phase_i+1]) > 0:
                        i_PQ[k,phase_i] = i_PQ[k,phase_i] - \
                            np.conj(S_PQ_del[phase_i]/
                                    (v[k,phase_i]-v[k,phase_i+1]))#a,b
                    if np.abs(v[k,phase_i+2]) > 0:
                        i_PQ[k,phase_i] = i_PQ[k,phase_i] - \
                           np.conj(S_PQ_del[phase_i+2]/
                                    (v[k,phase_i]-v[k,phase_i+2]))#a,c
                elif phase_i_mod == 1: #i_b
                    if np.abs(v[k,phase_i-1]) > 0:
                        i_PQ[k,phase_i] = i_PQ[k,phase_i] \
                            - np.conj(S_PQ_del[phase_i-1]/
                                  (v[k,phase_i]-v[k,phase_i-1]))#b,a
                    if np.abs(v[k,phase_i+1]) > 0:
                        i_PQ[k,phase_i] = i_PQ[k,phase_i] \
                            - np.conj(S_PQ_del[phase_i]/
                                (v[k,phase_i]-v[k,phase_i+1]))#b,c 
                else: #i_c
                    if np.abs(v[k,phase_i-2]) > 0:
                        i_PQ[k,phase_i] = i_PQ[k,phase_i] \
                            - np.conj(S_PQ_del[phase_i]/
                                  (v[k,phase_i]-v[k,phase_i-2]))#c,a
                    if np.abs(v[k,phase_i-1]) > 0:
                        i_PQ[k,phase_i] = i_PQ[k,phase_i] \
                            - np.conj(S_PQ_del[phase_i-1]/
                                  (v[k,phase_i]-v[k,phase_i-1]))#c,b 
            di[k] = np.sum(np.abs(i_PQ[k,:]-i_PQ[k-1,:])) #change in |I| sum during the iteration

        #elapsed = time.time() - t
        #print('Solution time: ' + str(elapsed))
        #t = time.time() 

        #Results
        self.i_PQ_iter = i_PQ
        self.v_iter = v
        self.dv_iter = dv
        self.di_iter = di
        self.i_PQ_res = i_PQ[k,:] #current injections
        self.i_slack_res = np.matmul(self.Ysn,v[k,:]) \
                               + np.matmul(self.Yss,self.vs)
        self.v_res = v[k,:]
        self.v_net_res = np.concatenate((self.vs,self.v_res))
        self.i_net_res = np.concatenate((self.i_slack_res,self.i_PQ_res))
        self.S_net_res = self.v_net_res*np.conj(self.i_net_res)/1e3
        self.S_PQloads_wye_res = S_PQ_wye/1e3
        self.S_PQloads_del_res = S_PQ_del/1e3
        #Create the bus results data frame
        res_bus_columns = ['name','number','load_type','connect',
                           'Sa','Sb','Sc','Va','Vb','Vc','Ia','Ib','Ic']
        self.res_bus_df = pd.DataFrame(index = range(self.N_buses),
                                       columns = res_bus_columns)
        self.res_bus_df= pd.DataFrame({'name':self.bus_df['name'],
                                       'number':self.bus_df['number'],
                                       'load_type':self.bus_df['load_type'],
                                       'connect':self.bus_df['connect'],
                                       'Sa':self.S_net_res[0::3],
                                       'Sb':self.S_net_res[1::3],
                                       'Sc':self.S_net_res[2::3],
                                       'Va':self.v_net_res[0::3],
                                       'Vb':self.v_net_res[1::3],
                                       'Vc':self.v_net_res[2::3],
                                       'Ia':self.i_net_res[0::3],
                                       'Ib':self.i_net_res[1::3],
                                       'Ic':self.i_net_res[2::3]})     

        #elapsed = time.time() - t
        #print('Results time: ' + str(elapsed))
        
        
    def update_line_pf_results(self):
        """
        Updates the line power flow results dataframe res_lines_df, based on
        the results of zbus_pf().

        """        
        res_lines_columns = ['busA','busB','Sa','Sb','Sc','Ia','Ib','Ic','VAa',
                             'VAb','VAc','VBa','VBb','VBc']
        self.res_lines_df = pd.DataFrame(index = range(self.N_lines),
                                         columns = res_lines_columns)
        for line_ij in range(self.N_lines):
            line_ij_df = self.line_df.iloc[line_ij] 
            bus_i_id = self.bus_df[self.bus_df['name']==
                                   line_ij_df['busA']]['number'].values[0] 
            bus_j_id = self.bus_df[self.bus_df['name']==
                                   line_ij_df['busB']]['number'].values[0]
            Yseries = self.list_Yseries[line_ij]
            Yshunt = self.list_Yshunt[line_ij]
            v_i_abc = self.v_net_res[3*bus_i_id:3*(bus_i_id+1)]
            v_j_abc = self.v_net_res[3*bus_j_id:3*(bus_j_id+1)]
            i_ij_abc = np.matmul(Yshunt,v_i_abc)+np.matmul(Yshunt,v_j_abc)\
                            +np.matmul(Yseries,v_i_abc-v_j_abc)
            #For apparent power calculations, need to divided 
            #by 1e3 to go from current (A) * voltage (V) to S (kVA)
            self.res_lines_df.iloc[line_ij]= \
                    { 'busA':line_ij_df['busA'],
                      'busB':line_ij_df['busB'],
                      'Sa':np.conj(i_ij_abc[0])*v_i_abc[0]/1e3,
                      'Sb':np.conj(i_ij_abc[1])*v_i_abc[1]/1e3,
                      'Sc':np.conj(i_ij_abc[2])*v_i_abc[2]/1e3,
                      'Ia':i_ij_abc[0],'Ib':i_ij_abc[1],'Ic':i_ij_abc[2],
                      'VAa':v_i_abc[0],'VAb':v_i_abc[1],'VAc':v_i_abc[2],
                      'VBa':v_j_abc[0],'VBb':v_j_abc[1],'VBc':v_j_abc[2]}

        
    def clear_loads(self):
        """
        Removes all real and reactive power loads from the network by clearing
        bus_df.

        """          
        for bus_i in range(0,len(self.bus_df)):
            self.bus_df.at[bus_i,'Pa'] = 0
            self.bus_df.at[bus_i,'Pb'] = 0
            self.bus_df.at[bus_i,'Pc'] = 0
            self.bus_df.at[bus_i,'Qa'] = 0
            self.bus_df.at[bus_i,'Qb'] = 0
            self.bus_df.at[bus_i,'Qc'] = 0
            
    def set_load(self,bus_id,ph_i,Pph,Qph):
        """
        Sets the P and Q load on a particular bus and phase


        Parameters
        ----------
        bus_id : int
            the load bus id
        ph_i : int
            the load phase (either 0, 1 or 2)
        Pph : float
            nominal operating point, apparent wye load (kVA)
        Qph : float
            nominal operating point, apparent delta load (kVA)
        """
        if ph_i == 0:
            self.bus_df.at[bus_id,'Pa'] = Pph
            self.bus_df.at[bus_id,'Qa'] = Qph
        elif ph_i == 1:
            self.bus_df.at[bus_id,'Pb'] = Pph
            self.bus_df.at[bus_id,'Qb'] = Qph
        else:
            self.bus_df.at[bus_id,'Pc'] = Pph
            self.bus_df.at[bus_id,'Qc'] = Qph
            
    def set_pf_limits(self,v_abs_min_val,v_abs_max_val,i_abs_max_val):
        """
        Sets the abs bus phase voltage limits and abs line phase current limits


        Parameters
        ----------
        v_abs_min_val : float
            minimum abs voltage bus phase voltage limit
        v_abs_max_val : float
            maximum abs voltage bus phase voltage limit
        i_abs_max_val : float
            maximum abs line phase current limit
        """        
        self.v_abs_min = v_abs_min_val*np.ones([self.N_buses,self.N_phases])
        self.v_abs_max = v_abs_max_val*np.ones([self.N_buses,self.N_phases])
        self.i_abs_max = i_abs_max_val*np.ones([self.N_lines,self.N_phases])
        for bus_i in range(1,self.N_buses):
            for phase_i in range(self.N_phases):
                ph_index = 3*(bus_i-1)+ phase_i
                if np.abs(self.Y[ph_index,ph_index]) == 0:
                    self.v_abs_min[bus_i,phase_i] = 0

    def v_flat(self):
        """
        Get the vector of 1 p.u. balanced bus phase voltages

        Returns
        -------
        numpy.ndarray
        """        

        #return a vector of 'flat' bus voltages
        v_flat = self.vs
        for i in range(1,self.N_buses):
            v_flat = np.concatenate((v_flat,self.vs))
        return v_flat   
    
    def update_YandZ(self):
        """
        Update the network admittance and impedance matrices

        """        
        #Set the Y bus matrix
        self.N_buses = len(self.bus_df) # update number of buses/lines
        self.N_lines = len(self.line_df)
        self.N_transformers = len(self.transformer_df)
        self.N_capacitors = len(self.capacitor_df)
        
        self.Ynet = np.zeros([(self.N_buses)*3,(self.N_buses)*3],
                              dtype=np.complex_)
        self.list_Yseries = []
        self.list_Yshunt = []
        for line_ij in range(self.N_lines):
            Zaa = self.line_df.iloc[line_ij]['Zaa']
            Zbb = self.line_df.iloc[line_ij]['Zbb']
            Zcc = self.line_df.iloc[line_ij]['Zcc']
            Zab = self.line_df.iloc[line_ij]['Zab']
            Zac = self.line_df.iloc[line_ij]['Zac']
            Zbc = self.line_df.iloc[line_ij]['Zbc']
            Baa = self.line_df.iloc[line_ij]['Baa']
            Bbb = self.line_df.iloc[line_ij]['Bbb']
            Bcc = self.line_df.iloc[line_ij]['Bcc']
            Bab = self.line_df.iloc[line_ij]['Bab']
            Bac = self.line_df.iloc[line_ij]['Bac']
            Bbc = self.line_df.iloc[line_ij]['Bbc']
            Zseries = np.array([[Zaa,Zab,Zac],[Zab,Zbb,Zbc],[Zac,Zbc,Zcc]])
            phases = [Zaa!=0,Zbb!=0,Zcc!=0]
            phases_reduced_indexes = [0,int(Zaa!=0),int(Zaa!=0)+int(Zbb!=0)]

            #i.e. for reduced Y, phase indexes depend on which are present
            Zseries_reduced = Zseries[phases,:][:,phases]
            Yseries_reduced = np.linalg.inv(Zseries_reduced)
            Yseries = np.zeros([3,3],dtype=np.complex_)
            for phase_i in range(3):
                for phase_j in range(3):
                    if phases[phase_i] and phases[phase_j]:
                        Yseries[phase_i,phase_j] = \
                        Yseries_reduced[phases_reduced_indexes[phase_i],
                                        phases_reduced_indexes[phase_j]]
            Yshunt = np.array([[Baa,Bab,Bac],[Bab,Bbb,Bbc],[Bac,Bbc,Bcc]]) 
            self.list_Yseries.append(Yseries)
            self.list_Yshunt.append(Yshunt)
        ##NEW LINE ADDITION METHOD
        for line_ij in range(len(self.line_df)):
            bus_name_i = self.line_df.iloc[line_ij]['busA']
            bus_name_j = self.line_df.iloc[line_ij]['busB']
            bus_i_Yindex = 3*self.bus_df \
                    [self.bus_df['name']==bus_name_i]['number'].values[0]
            bus_j_Yindex = 3*self.bus_df\
                    [self.bus_df['name']==bus_name_j]['number'].values[0]
            Yseries = self.list_Yseries[line_ij]
            Yshunt = self.list_Yshunt[line_ij]
            #Diagonal elements
            self.Ynet[bus_i_Yindex+0:bus_i_Yindex+3,
                      bus_i_Yindex+0:bus_i_Yindex+3] =\
                self.Ynet[bus_i_Yindex+0:bus_i_Yindex+3,
                          bus_i_Yindex+0:bus_i_Yindex+3]+0.5*Yshunt+Yseries
            self.Ynet[bus_j_Yindex+0:bus_j_Yindex+3,
                      bus_j_Yindex+0:bus_j_Yindex+3] =\
                self.Ynet[bus_j_Yindex+0:bus_j_Yindex+3,
                          bus_j_Yindex+0:bus_j_Yindex+3]+0.5*Yshunt+Yseries
            #Off-diagonal elements
            self.Ynet[bus_i_Yindex+0:bus_i_Yindex+3,
                      bus_j_Yindex+0:bus_j_Yindex+3] =-Yseries
            self.Ynet[bus_j_Yindex+0:bus_j_Yindex+3,
                      bus_i_Yindex+0:bus_i_Yindex+3] =-Yseries
        #Add transformers to admittance matrix
            
        for trans_index in range(self.N_transformers):
            busA_name = self.transformer_df.iloc[trans_index]['busA']
            busB_name = self.transformer_df.iloc[trans_index]['busB']
            typeA = self.transformer_df.iloc[trans_index]['typeA']
            typeB = self.transformer_df.iloc[trans_index]['typeB']
            busA = self.bus_df[
                    self.bus_df['name']==busA_name]['number'].values[0]
            busB = self.bus_df[
                    self.bus_df['name']==busB_name]['number'].values[0]
            busA_Yindex = 3*busA
            busB_Yindex = 3*busB
            y_series = 1/self.transformer_df.iloc[trans_index]['Zseries']
            #NOTE: Small shunt impedance added to ensure invertibility of Y
            if np.abs(self.transformer_df.iloc[trans_index]['Zshunt']) == 0:
                y_shunt = 0
            else:
                y_shunt = 1/self.transformer_df.iloc[trans_index]['Zshunt']
            Yshunt = y_shunt*np.eye(3)
            #extra shunt impedances for convergence
            Ysmall = 1e-4*np.eye(3)*np.abs(np.real(y_series)) 
            #+ np.abs(np.imag(y_series))
            #Get Y bus matrix elements
            if typeA == 'wye-g' and typeB == 'wye-g':
                Y_AA = np.array([[y_series,0,0],\
                                 [0,y_series,0],\
                                 [0,0,y_series]])
                Y_BB = Y_AA
                Y_AB = -Y_AA
                Y_BA = -Y_AA
                #Yshunt_e = 0 #don't need extra shunt impedances if wye-g,wye-g
                #Yshunt_ee = 0
            elif (typeA == 'wye' and typeB == 'wye')\
                or (typeA == 'delta' and typeB == 'delta')\
                or (typeA == 'wye-g' and typeB == 'wye')\
                or (typeA == 'wye' and typeB == 'wye-g'):
                Y_AA = 1/3*np.array([[2*y_series,-y_series,-y_series],
                                     [-y_series,2*y_series,-y_series],
                                     [-y_series,-y_series,2*y_series]])\
                                        + Ysmall
                #Y_AA = 1/3*(3*y_series*np.eye(3) - y_series*np.ones([3,3]))
                Y_BB = Y_AA
                Y_AB = -Y_AA
                Y_BA = -Y_AA
            elif typeA == 'wye-g' and typeB == 'delta':
                Y_AA = np.array([[y_series,0,0],\
                                 [0,y_series,0],\
                                 [0,0,y_series]]) + Ysmall
                Y_BB = 1/3*np.array([[2*y_series,-y_series,-y_series],
                                     [-y_series,2*y_series,-y_series],
                                     [-y_series,-y_series,2*y_series]])+ Ysmall
                Y_AB = 1/np.sqrt(3)*np.array([[-y_series,y_series,0],
                                               [0,-y_series,y_series],
                                               [y_series,0,-y_series]])- Ysmall
                Y_BA = Y_AB.T
            elif typeA == 'delta' and typeB == 'wye-g':
                Y_AA = 1/3*np.array([[2*y_series,-y_series,-y_series],
                                     [-y_series,2*y_series,-y_series],
                                     [-y_series,-y_series,2*y_series]])+ Ysmall
                Y_BB = np.array([[y_series,0,0],\
                                 [0,y_series,0],\
                                 [0,0,y_series]]) + Ysmall
                Y_AB = 1/np.sqrt(3)*np.array([[-y_series,y_series,0],
                                               [0,-y_series,y_series],
                                               [y_series,0,-y_series]])- Ysmall

                Y_BA = Y_AB.T
            elif typeA == 'wye' and typeB == 'delta':
                Y_AA = 1/3*np.array([[2*y_series,-y_series,-y_series],
                                     [-y_series,2*y_series,-y_series],
                                     [-y_series,-y_series,2*y_series]])+ Ysmall
                Y_BB = Y_AA
                Y_AB = 1/np.sqrt(3)*np.array([[-y_series,y_series,0],
                                               [0,-y_series,y_series],
                                               [y_series,0,-y_series]])- Ysmall
                Y_BA = Y_AB.T
            else:# typeA == 'delta' and typeB == 'wye':
                Y_AA = 1/3*np.array([[2*y_series,-y_series,-y_series],
                                     [-y_series,2*y_series,-y_series],
                                     [-y_series,-y_series,2*y_series]])+ Ysmall
                Y_BB = Y_AA
                Y_AB = 1/np.sqrt(3)*np.array([[-y_series,y_series,0],
                                               [0,-y_series,y_series],
                                               [y_series,0,-y_series]])- Ysmall
                Y_BA = Y_AB.T
            #Edit Y matricies by turns ratio if included
            if 'kV_A' in self.transformer_df.columns and \
                'kV_B' in self.transformer_df.columns:
                A_i = np.eye(3)/(self.transformer_df.iloc[trans_index]['kV_A']
                            /self.transformer_df.iloc[trans_index]['kV_B'])
            else:
                A_i = np.eye(3)
            Y_AA = np.matmul(np.matmul(A_i,Y_AA),A_i.T)
            Y_AB = np.matmul(A_i,Y_AB)
            Y_BB = Y_BB
            Y_BA = np.matmul(Y_BA,A_i.T)

            #Diagonal elements
            self.Ynet[busA_Yindex+0:busA_Yindex+3,
                      busA_Yindex+0:busA_Yindex+3] = \
                self.Ynet[busA_Yindex+0:busA_Yindex+3,
                          busA_Yindex+0:busA_Yindex+3]+0.5*Yshunt+ Y_AA
            self.Ynet[busB_Yindex+0:busB_Yindex+3,
                      busB_Yindex+0:busB_Yindex+3] = \
                self.Ynet[busB_Yindex+0:busB_Yindex+3,
                          busB_Yindex+0:busB_Yindex+3]+0.5*Yshunt+ Y_BB 
            #Off-diagonal elements
            self.Ynet[busA_Yindex+0:busA_Yindex+3,
                      busB_Yindex+0:busB_Yindex+3] = Y_AB #- Yshunt_ee
            self.Ynet[busB_Yindex+0:busB_Yindex+3,
                      busA_Yindex+0:busA_Yindex+3] = Y_BA #- Yshunt_ee
        #Add capacitors to admittance matrix
        for cap_index in range(len(self.capacitor_df)):
            #Diagonal elements
            #print(cap_index)
            #print(self.capacitor_df)
            bus_name = self.capacitor_df.iloc[cap_index]['bus']
            #print(self.bus_df['name'])
            bus_Yindex = 3*self.bus_df \
                        [self.bus_df['name']==bus_name]['number'].values[0]
            #print(bus_name)
            #print(self.bus_df)
            connection = self.capacitor_df.iloc[cap_index]['connect']
            kVph = self.capacitor_df.iloc[cap_index]['kVln']*1e3
            Qa = self.capacitor_df.iloc[cap_index]['Qa']*1e3
            Qb = self.capacitor_df.iloc[cap_index]['Qb']*1e3
            Qc = self.capacitor_df.iloc[cap_index]['Qc']*1e3
            Zaa = 0
            Zbb = 0
            Zcc = 0
            Zab = 0
            Zac = 0
            Zbc = 0
            if connection == 'Y':
                if Qa != 0:
                    Zaa = -1j*(kVph**2)/Qa
                if Qb != 0:
                    Zbb = -1j*(kVph**2)/Qb
                if Qc != 0:
                    Zcc = -1j*(kVph**2)/Qc
            else:
                if Qa != 0:
                    Zab = -1j*(kVph**2)/Qa
                if Qb != 0:
                    Zac = -1j*(kVph**2)/Qb
                if Qc != 0:
                    Zbc = -1j*(kVph**2)/Qc
            Zmat = np.array([[Zaa,Zab,Zac],[Zab,Zbb,Zbc],[Zac,Zbc,Zcc]])
            phases = [Zaa!=0,Zbb!=0,Zcc!=0]
            phases_reduced_indexes = [0,int(Zaa!=0),int(Zaa!=0)+int(Zbb!=0)]
            #i.e. for reduced Y, phase indexes depend on which are present
            Zmat_reduced = Zmat[phases,:][:,phases]
            Ymat_reduced = np.linalg.inv(Zmat_reduced)
            Ymat = np.zeros([3,3],dtype=np.complex_)
            for phase_i in range(3):
                for phase_j in range(3):
                    if phases[phase_i] and phases[phase_j]:
                        Ymat[phase_i,phase_j] = \
                            Ymat_reduced[phases_reduced_indexes[phase_i],
                                         phases_reduced_indexes[phase_j]]
            # print(Ymat)
            self.Ynet[bus_Yindex+0:bus_Yindex+3,bus_Yindex+0:bus_Yindex+3] =\
                self.Ynet[bus_Yindex+0:bus_Yindex+3,bus_Yindex+0:bus_Yindex+3]\
                            + Ymat
            
            
        self.Y = self.Ynet[3:,3:]#Ynet excluding the z bus rows and columns
        self.Yns = self.Ynet[3:,0:3]
        self.Ysn = self.Ynet[0:3,3:]
        self.Yss = self.Ynet[0:3,0:3]
        self.Y_non_singular = self.Y + 1e-20*np.eye(self.Y.shape[0])
        #Set the Z bus matrix
        self.Z = np.linalg.inv(self.Y_non_singular)
        self.Znet = np.linalg.inv(self.Ynet + 1e-20*np.eye(self.Ynet.shape[0]))
          
    def setup_network_ieee13(self):
        """
        Set up the network as the unloaded IEEE 13 Bus Test Feeder

        """        
        self.N_buses = 13
        self.N_lines = 11
        self.N_phases = 3
        #Create buses dataframe
        bus_columns = ['name','number','load_type','connect',\
                       'Pa','Pb','Pc','Qa','Qb','Qc']
        bus_index = range(self.N_buses)
        self.bus_df = pd.DataFrame(index = bus_index,columns = bus_columns)
        self.bus_df.iloc[0]= \
                {'name':'650','number':0,
                 'v_base': self.Vslack_ph, 'load_type':'S', 'connect':'Y',
                 'Pa':  0,'Pb':  0,'Pc':  0,'Qa':  0,'Qb':  0,'Qc':  0}
        self.bus_df.iloc[1]= \
                {'name':'632','number':1, 
                 'v_base': self.Vslack_ph, 'load_type':'PQ','connect':'Y',
                 'Pa':  0,'Pb':  0,'Pc':  0,'Qa':  0,'Qb':  0,'Qc':  0}
        self.bus_df.iloc[2]= \
                {'name':'645','number':2, 
                 'v_base': self.Vslack_ph, 'load_type':'PQ','connect':'Y',
                 'Pa':  0,'Pb':  170,'Pc':  0,'Qa':  0,'Qb':  125,'Qc':  0}
        self.bus_df.iloc[3]= \
                {'name':'646','number':3, 
                 'v_base': self.Vslack_ph, 'load_type':'Z', 'connect':'D',
                 'Pa':  0,'Pb':  230,'Pc':  0,'Qa':  0,'Qb':  132,'Qc':  0}
        self.bus_df.iloc[4]= \
                {'name':'633','number':4,  
                 'v_base': self.Vslack_ph, 'load_type':'PQ','connect':'Y',
                 'Pa':  0,'Pb':  0,'Pc':  0,'Qa':  0,'Qb':  0,'Qc':  0}
        self.bus_df.iloc[5]= \
                {'name':'634','number':5,  
                 'v_base': self.Vslack_ph, 'load_type':'PQ','connect':'Y',
                 'Pa':  160,'Pb':  120,'Pc':  120,'Qa':  110,'Qb':  90,'Qc':  90}
        self.bus_df.iloc[6]= \
                {'name':'671','number':6,  
                 'v_base': self.Vslack_ph, 'load_type':'PQ','connect':'D',
                 'Pa':  385,'Pb':  385,'Pc':  385,'Qa':  220,'Qb':  220,'Qc':  220}
        self.bus_df.iloc[7]= \
                {'name':'680','number':7,  
                 'v_base': self.Vslack_ph, 'load_type':'PQ','connect':'Y',
                 'Pa':  0,'Pb':  0,'Pc':  0,'Qa':  0,'Qb':  0,'Qc':  0}
        self.bus_df.iloc[8]= \
                {'name':'684','number':8,  
                 'v_base': self.Vslack_ph, 'load_type':'PQ','connect':'Y',
                 'Pa':  0,'Pb':  0,'Pc':  0,'Qa':  0,'Qb':  0,'Qc':  0}
        self.bus_df.iloc[9]= \
                {'name':'611','number':9,  
                 'v_base': self.Vslack_ph, 'load_type':'I', 'connect':'Y',
                 'Pa':  0,'Pb':  0,'Pc':  170,'Qa':  0,'Qb':  0,'Qc':  80}
        self.bus_df.iloc[10]=\
                {'name':'652','number':10, 
                 'v_base': self.Vslack_ph, 'load_type':'Z', 'connect':'Y',
                 'Pa':  128,'Pb':  0,'Pc':  0,'Qa':  86,'Qb':  0,'Qc':  0}
        self.bus_df.iloc[11]=\
                {'name':'692','number':11, 
                 'v_base': self.Vslack_ph, 'load_type':'I', 'connect':'D',
                 'Pa':  0,'Pb':  0,'Pc':  170,'Qa':  0,'Qb':  0,'Qc':  151}
        self.bus_df.iloc[12]=\
                {'name':'675','number':12, 
                 'v_base': self.Vslack_ph, 'load_type':'PQ','connect':'Y',
                 'Pa':  485,'Pb':  68,'Pc':  290,'Qa':  190,'Qb':  60,'Qc':  212}
        #Create line configuration data frame
        line_config_col = ['name','Zaa','Zbb','Zcc','Zab','Zac','Zbc',
                           'Baa','Bbb','Bcc','Bab','Bac','Bbc']
        self.line_config_df = pd.DataFrame(
                    index = range(7),columns =line_config_col)
        self.line_config_df.iloc[0] = \
            {'name':'601',
             'Zaa':0.3465+1.0179j, 'Zbb':0.3375+1.0478j, 'Zcc':0.3414+1.0348j,\
             'Zab':0.1560+0.5017j, 'Zac':0.1580+0.4236j, 'Zbc':0.1535+0.3849j,\
             'Baa':       6.2998j, 'Bbb':       5.9597j, 'Bcc':       5.6386j,\
             'Bab':      -1.9958j, 'Bac':      -1.2595j, 'Bbc':      -0.7417j}
        self.line_config_df.iloc[1] = \
        {'name':'602',
           'Zaa':0.7526+1.1814j, 'Zbb':0.7475+1.1983j, 'Zcc':0.7436+1.2112j,\
           'Zab':0.1580+0.4236j, 'Zac':0.1560+0.5017j, 'Zbc':0.1535+0.3849j,\
           'Baa':       5.6990j, 'Bbb':       5.1795j, 'Bcc':       5.4246j,\
           'Bab':      -1.0817j, 'Bac':      -1.6905j, 'Bbc':      -0.6588j}
        self.line_config_df.iloc[2] = \
        {'name':'603',
           'Zaa':0,              'Zbb':1.3294+1.3471j, 'Zcc':1.3238+1.3569j,\
           'Zab':0,              'Zac':0,              'Zbc':0.2066+0.4591j,\
           'Baa':0,              'Bbb':       4.7097j, 'Bcc':       4.6658j,\
           'Bab':0,              'Bac':0,              'Bbc':      -0.8999j}
        self.line_config_df.iloc[3] = \
        {'name':'604',
           'Zaa':1.3238+1.3569j, 'Zbb':0,              'Zcc':1.3294+1.3471j,\
           'Zab':0,              'Zac':0.2066+0.4591j, 'Zbc':0,\
           'Baa':       4.6658j, 'Bbb':0,              'Bcc':       4.7097j,\
           'Bab':0,              'Bac':      -0.8999j, 'Bbc':0}
        self.line_config_df.iloc[4] = \
        {'name':'605',
           'Zaa':0,              'Zbb':0,              'Zcc':1.3292+1.3475j,\
           'Zab':0,              'Zac':0,              'Zbc':0,\
           'Baa':0,              'Bbb':0,              'Bcc':       4.5193j,\
           'Bab':0,              'Bac':0,              'Bbc':0}
        self.line_config_df.iloc[5] = \
        {'name':'606',
           'Zaa':0.7982+0.4463j, 'Zbb':0.7891+0.4041j, 'Zcc':0.7982+0.4463j,\
           'Zab':0.3192+0.0328j, 'Zac':0.2849-0.0143j, 'Zbc':0.3192+0.0328j,\
           'Baa':      96.8897j, 'Bbb':      96.8897j, 'Bcc':      96.8897j,\
           'Bab':0,              'Bac':0,              'Bbc':0}
        self.line_config_df.iloc[6] = \
        {'name':'607',
           'Zaa':1.3425+0.5124j, 'Zbb':0,              'Zcc':0,\
           'Zab':0,              'Zac':0,              'Zbc':0,\
           'Baa':      88.9912j, 'Bbb':0,              'Bcc':0,\
           'Bab':0,              'Bac':0,              'Bbc':0}        
        #Create line information data frame
        line_info_col = ['busA','busB','config','length',
                         'Z_conv_factor','B_conv_factor'] 
            #Z_conv_factor = ft/mile, B_conv_factor 1e-6*ft/mile
        self.line_info_df = pd.DataFrame(index=range(10),columns=line_info_col)
        self.line_info_df.iloc[0] = \
        {'busA':'632','busB':'645','length': 500,'config':'603',
         'Z_conv_factor':1/(5280),'B_conv_factor':1e-6/(5280)} #5280 ft/mile
        self.line_info_df.iloc[1] = \
        {'busA':'632','busB':'633','length': 500,'config':'602',
         'Z_conv_factor':1/(5280),'B_conv_factor':1e-6/(5280)} #5280 ft/mile
        self.line_info_df.iloc[2] = \
        {'busA':'645','busB':'646','length': 300,'config':'603',
         'Z_conv_factor':1/(5280),'B_conv_factor':1e-6/(5280)} #5280 ft/mile
        self.line_info_df.iloc[3] = \
        {'busA':'650','busB':'632','length':2000,'config':'601',
         'Z_conv_factor':1/(5280),'B_conv_factor':1e-6/(5280)} #5280 ft/mile
        self.line_info_df.iloc[4] = \
        {'busA':'684','busB':'652','length': 800,'config':'607',
         'Z_conv_factor':1/(5280),'B_conv_factor':1e-6/(5280)} #5280 ft/mile
        self.line_info_df.iloc[5] = \
        {'busA':'632','busB':'671','length':2000,'config':'601',
         'Z_conv_factor':1/(5280),'B_conv_factor':1e-6/(5280)} #5280 ft/mile
        self.line_info_df.iloc[6] = \
        {'busA':'671','busB':'684','length': 300,'config':'604',
         'Z_conv_factor':1/(5280),'B_conv_factor':1e-6/(5280)} #5280 ft/mile
        self.line_info_df.iloc[7] = \
        {'busA':'671','busB':'680','length':1000,'config':'601',
         'Z_conv_factor':1/(5280),'B_conv_factor':1e-6/(5280)} #5280 ft/mile
        self.line_info_df.iloc[8] = \
        {'busA':'684','busB':'611','length': 300,'config':'605',
         'Z_conv_factor':1/(5280),'B_conv_factor':1e-6/(5280)} #5280 ft/mile
        self.line_info_df.iloc[9] = \
        {'busA':'692','busB':'675','length': 500,'config':'606',
         'Z_conv_factor':1/(5280),'B_conv_factor':1e-6/(5280)} #5280 ft/mile        
        #Create line data frame
        line_columns = ['busA','busB','Zaa','Zbb','Zcc','Zab','Zac','Zbc',\
                        'Baa','Bbb','Bcc','Bab','Bac','Bbc']
        self.line_df = pd.DataFrame(columns = line_columns)
        for i in range(len(self.line_info_df)):
            line_info_i = self.line_info_df.iloc[i]
            busA_i = line_info_i['busA']
            busB_i = line_info_i['busB']
            Zaa_i = (self.line_config_df[self.line_config_df['name']
                ==line_info_i['config']]['Zaa'].values[0]*\
                    line_info_i['Z_conv_factor']*line_info_i['length'])
            Zbb_i = self.line_config_df[self.line_config_df['name']
                ==line_info_i['config']]['Zbb'].values[0]*\
                line_info_i['Z_conv_factor']*line_info_i['length']
            Zcc_i = self.line_config_df[self.line_config_df['name']
                ==line_info_i['config']]['Zcc'].values[0]*\
                line_info_i['Z_conv_factor']*line_info_i['length']
            Zab_i = self.line_config_df[self.line_config_df['name']
                ==line_info_i['config']]['Zab'].values[0]*\
                line_info_i['Z_conv_factor']*line_info_i['length']
            Zac_i = self.line_config_df[self.line_config_df['name']
                ==line_info_i['config']]['Zac'].values[0]*\
                line_info_i['Z_conv_factor']*line_info_i['length']
            Zbc_i = self.line_config_df[self.line_config_df['name']
                ==line_info_i['config']]['Zbc'].values[0]*\
                line_info_i['Z_conv_factor']*line_info_i['length']
            Baa_i = self.line_config_df[self.line_config_df['name']
                ==line_info_i['config']]['Baa'].values[0]*\
                line_info_i['B_conv_factor']*line_info_i['length']
            Bbb_i = self.line_config_df[self.line_config_df['name']
                ==line_info_i['config']]['Bbb'].values[0]*\
                line_info_i['B_conv_factor']*line_info_i['length']
            Bcc_i = self.line_config_df[self.line_config_df['name']
                ==line_info_i['config']]['Bcc'].values[0]*\
                line_info_i['B_conv_factor']*line_info_i['length']
            Bab_i = self.line_config_df[self.line_config_df['name']
                ==line_info_i['config']]['Bab'].values[0]*\
                line_info_i['B_conv_factor']*line_info_i['length']
            Bac_i = self.line_config_df[self.line_config_df['name']
                ==line_info_i['config']]['Bac'].values[0]*\
                line_info_i['B_conv_factor']*line_info_i['length']
            Bbc_i = self.line_config_df[self.line_config_df['name']
                ==line_info_i['config']]['Bbc'].values[0]*\
                line_info_i['B_conv_factor']*line_info_i['length']
            self.line_df = self.line_df._append({'busA':busA_i,'busB':busB_i,
                            'Zaa':Zaa_i,'Zbb':Zbb_i,'Zcc':Zcc_i,'Zab':Zab_i,
                            'Zac':Zac_i,'Zbc':Zbc_i,\
                            'Baa':Baa_i,'Bbb':Bbb_i,'Bcc':Bcc_i,'Bab':Bab_i,
                            'Bac':Bac_i,'Bbc':Bbc_i},ignore_index=True)
        #Add the switch (low impedance) between buses 671 and 692
        Zswitch = 1e-6+0j
        self.line_df = self.line_df._append({'busA':'671','busB':'692',
                        'Zaa':Zswitch,'Zbb':Zswitch,'Zcc':Zswitch,'Zab':0,
                        'Zac':0,'Zbc':0,\
                        'Baa':0,'Bbb':0,'Bcc':0,'Bab':0,'Bac':0,'Bbc':0},
                        ignore_index=True)
        #Add Transformer XFM-1 between buses 633 and 634  
        #Y-G to Y-G transformer - therefore diagonal Y matrix 
        #500kVA, 4.16kV/0.48kV, R=1.1%, X=2%
        #impedance on the primary side:
        #Zohms_pri = (4.16e3)**2/(500e3)*(0.011+0.02j)
        
        #self.line_df = self.line_df._append({'busA':'633','busB':'634',
        #                    'Zaa':0.381+0.692j,'Zbb':0.381+0.692j,
        #                    'Zcc':0.381+0.692j,'Zab':0,'Zac':0,'Zbc':0,\
        #                    'Baa':0,'Bbb':0,'Bcc':0,'Bab':0,'Bac':0,'Bbc':0},
        #                    ignore_index=True)           
        
        #Create transformer data frame
        #Types: 'wye-g', 'wye', 'delta'
        transformer_columns =['busA','busB','typeA','typeB','Zseries','Zshunt']
        self.transformer_df = pd.DataFrame(columns = transformer_columns)
        self.transformer_df = self.transformer_df._append({'busA':'633',
                                                          'busB':'634',
                                                          'typeA':'wye-g',
                                                          'typeB':'wye-g',
                                                        'Zseries':0.381+0.692j,
                                                          'Zshunt':0},
                                                            ignore_index=True) 
        
        capacitor_columns = ['name','number','bus','kVln','connect',
                             'Qa','Qb','Qc']
        self.capacitor_df = pd.DataFrame(columns = capacitor_columns)
        self.capacitor_df = self.capacitor_df._append({'name':'cap2',
                                                      'number':0,
                                                      'bus':'675',
                                                      'kVln':2.4,
                                                      'connect':'Y',
                                                      'Qa':200,
                                                      'Qb':200,
                                                      'Qc':200},
                                                        ignore_index=True) 
        self.capacitor_df = self.capacitor_df._append({'name':'cap2',
                                                      'number':1,
                                                      'bus':'611',
                                                      'kVln':2.4,
                                                      'connect':'Y',
                                                      'Qa': 0,
                                                      'Qb': 0,
                                                      'Qc':100},
                                                        ignore_index=True)

    def setup_network_eulv_reduced(self, updateYZ=True):
        #path = os.path.dirname(os.path.abspath(__file__))
        #path = os.path.dirname(os.path.dirname(__file__))
        file_location = os.path.join(path, 'Data', 'networks', 'eulv_reduced/')
        buses_csv = pd.read_csv(file_location + 'eulv_bus_df.csv')
        lines_csv = pd.read_csv(file_location + 'Lines2.txt', sep=' ',
                                names=["new", "name", "busA", "busB", "phases", "linecode", "length", "units"])
        #
        self.N_buses = buses_csv.shape[0]
        self.N_lines = lines_csv.shape[0]
        self.N_phases = 3
        ##Create buses dataframe
        self.Vslack_ph = 240
        bus_columns = ['name', 'number', 'load_type', 'connect', 'Pa', 'Pb', 'Pc', 'Qa', 'Qb', 'Qc']
        bus_index = range(self.N_buses)
        self.bus_df = pd.DataFrame(index=bus_index, columns=bus_columns)
        for i in range(self.N_buses):
            self.bus_df.iloc[i] = {'name': buses_csv['name'][i], 'number': i, 'v_base': buses_csv['v_base'][i],
                                   'load_type': buses_csv['load_type'][i], 'connect': buses_csv['connect'][i], \
                                   'Pa': buses_csv['Pa'][i], 'Pb': buses_csv['Pb'][i], 'Pc': buses_csv['Pc'][i],
                                   'Qa': buses_csv['Qa'][i], 'Qb': buses_csv['Qb'][i], 'Qc': buses_csv['Qc'][i]}

        # self.bus_df.iloc[0]= {'name':'650','number':0,  'v_base': self.Vslack_ph, 'load_type':'S', 'connect':'Y','Pa':  0,'Pb':  0,'Pc':  0,'Qa':  0,'Qb':  0,'Qc':  0}
        ##Create line configuration data frame
        line_config_col = ['name', 'Zaa', 'Zbb', 'Zcc', 'Zab', 'Zac', 'Zbc', 'Baa', 'Bbb', 'Bcc', 'Bab', 'Bac', 'Bbc']
        line_config_df = pd.DataFrame(index=range(10), columns=line_config_col)
        line_config_df.iloc[0] = {'name': 'Linecode=2c_.007', 'Zaa': 0.00397 + 0.000099j, 'Zbb': 0.00397 + 0.000099j,
                                  'Zcc': 0.00397 + 0.000099j, \
                                  'Zab': 0j, 'Zac': 0j, 'Zbc': 0j, \
                                  'Baa': 0j, 'Bbb': 0j, 'Bcc': 0j, \
                                  'Bab': 0j, 'Bac': 0j, 'Bbc': 0j}
        line_config_df.iloc[1] = {'name': 'Linecode=2c_.0225', 'Zaa': 0.001257 + 0.000085j, 'Zbb': 0.001257 + 0.000085j,
                                  'Zcc': 0.001257 + 0.000085j, \
                                  'Zab': 0j, 'Zac': 0j, 'Zbc': 0j, \
                                  'Baa': 0j, 'Bbb': 0j, 'Bcc': 0j, \
                                  'Bab': 0j, 'Bac': 0j, 'Bbc': 0j}
        line_config_df.iloc[2] = {'name': 'Linecode=2c_16', 'Zaa': 0.001166667 + 0.000088j,
                                  'Zbb': 0.001166667 + 0.000088j, 'Zcc': 0.001166667 + 0.000088j, \
                                  'Zab': 1.66667e-05 - 4.48339E-22j, 'Zac': 1.66667e-05 - 4.48339E-22j,
                                  'Zbc': 1.66667e-05 - 4.48339E-22j, \
                                  'Baa': 0j, 'Bbb': 0j, 'Bcc': 0j, \
                                  'Bab': 0j, 'Bac': 0j, 'Bbc': 0j}
        line_config_df.iloc[3] = {'name': 'Linecode=35_SAC_XSC', 'Zaa': 0.000832 + 9.2E-05j, 'Zbb': 0.000832 + 9.2E-05j,
                                  'Zcc': 0.000832 + 9.2E-05j, \
                                  'Zab': -3.6E-05 + 1.46264E-21j, 'Zac': -3.6E-05 + 1.46264E-21j,
                                  'Zbc': -3.6E-05 + 1.46264E-21j, \
                                  'Baa': 0j, 'Bbb': 0j, 'Bcc': 0j, \
                                  'Bab': 0j, 'Bac': 0j, 'Bbc': 0j}
        line_config_df.iloc[4] = {'name': 'Linecode=4c_.06', 'Zaa': 0.000839667 + 8.03333E-05j,
                                  'Zbb': 0.000839667 + 8.03333E-05j, 'Zcc': 0.000839667 + 8.03333E-05j, \
                                  'Zab': 0.000370667 + 5.33333E-06j, 'Zac': 0.000370667 + 5.33333E-06j,
                                  'Zbc': 0.000370667 + 5.33333E-06j, \
                                  'Baa': 0j, 'Bbb': 0j, 'Bcc': 0j, \
                                  'Bab': 0j, 'Bac': 0j, 'Bbc': 0j}
        line_config_df.iloc[5] = {'name': 'Linecode=4c_.1', 'Zaa': 0.000502333 + 0.000075j,
                                  'Zbb': 0.000502333 + 0.000075j, 'Zcc': 0.000502333 + 0.000075j, \
                                  'Zab': 0.000228333 + 0.000002j, 'Zac': 0.000228333 + 0.000002j,
                                  'Zbc': 0.000228333 + 0.000002j, \
                                  'Baa': 0j, 'Bbb': 0j, 'Bcc': 0j, \
                                  'Bab': 0j, 'Bac': 0j, 'Bbc': 0j}
        line_config_df.iloc[6] = {'name': 'Linecode=4c_.35', 'Zaa': 0.000165667 + 7.03333E-05j,
                                  'Zbb': 0.000165667 + 7.03333E-05j, 'Zcc': 0.000165667 + 7.03333E-05j, \
                                  'Zab': 7.66667E-05 + 2.83333E-06j, 'Zac': 7.66667E-05 + 2.83333E-06j,
                                  'Zbc': 7.66667E-05 + 2.83333E-06j, \
                                  'Baa': 0j, 'Bbb': 0j, 'Bcc': 0j, \
                                  'Bab': 0j, 'Bac': 0j, 'Bbc': 0j}
        line_config_df.iloc[7] = {'name': 'Linecode=4c_185', 'Zaa': 0.000304 + 7.13333E-05j,
                                  'Zbb': 0.000304 + 7.13333E-05j, 'Zcc': 0.000304 + 7.13333E-05j, \
                                  'Zab': 0.000138 + 3.33333E-06j, 'Zac': 0.000138 + 3.33333E-06j,
                                  'Zbc': 0.000138 + 3.33333E-06j, \
                                  'Baa': 0j, 'Bbb': 0j, 'Bcc': 0j, \
                                  'Bab': 0j, 'Bac': 0j, 'Bbc': 0j}
        line_config_df.iloc[8] = {'name': 'Linecode=4c_70', 'Zaa': 0.000799 + 0.000075j, 'Zbb': 0.000799 + 0.000075j,
                                  'Zcc': 0.000799 + 0.000075j, \
                                  'Zab': 0.000353 + 4E-06j, 'Zac': 0.000353 + 4E-06j, 'Zbc': 0.000353 + 4E-06j, \
                                  'Baa': 0j, 'Bbb': 0j, 'Bcc': 0j, \
                                  'Bab': 0j, 'Bac': 0j, 'Bbc': 0j}
        line_config_df.iloc[9] = {'name': 'Linecode=4c_95_SAC_XC', 'Zaa': 0.000482667 + 8.03333E-05j,
                                  'Zbb': 0.000482667 + 8.03333E-05j, 'Zcc': 0.000482667 + 8.03333E-05j, \
                                  'Zab': 0.000160667 + 6.33333E-06j, 'Zac': 0.000160667 + 6.33333E-06j,
                                  'Zbc': 0.000160667 + 6.33333E-06j, \
                                  'Baa': 0j, 'Bbb': 0j, 'Bcc': 0j, \
                                  'Bab': 0j, 'Bac': 0j, 'Bbc': 0j}

        ##Create line information data frame
        line_info_col = ['busA', 'busB', 'config', 'length', 'Z_conv_factor',
                         'B_conv_factor']  # Z_conv_factor = ft/mile, B_conv_factor 1e-6*ft/mile
        line_info_df = pd.DataFrame(index=range(len(lines_csv)), columns=line_info_col)
        for i in range(len(lines_csv)):
            busA_i = lines_csv['busA'][i][5:]
            busB_i = lines_csv['busB'][i][5:]
            length_i = float(lines_csv['length'][i][7:])
            config_i = lines_csv['linecode'][i]
            line_info_df.iloc[i] = {'busA': busA_i, 'busB': busB_i, \
                                    'config': config_i, 'length': length_i, 'Z_conv_factor': 1, 'B_conv_factor': 1}

        ##Create line data frame
        line_columns = ['busA', 'busB', 'Zaa', 'Zbb', 'Zcc', 'Zab', 'Zac', 'Zbc', 'Baa', 'Bbb', 'Bcc', 'Bab', 'Bac',
                        'Bbc']
        self.line_df = pd.DataFrame(columns=line_columns)
        self.line_df = self.line_df._append({'busA': 'sourcebus', 'busB': 'sourcebusz', \
                                            'Zaa': 0.5743408230315273 + 1.7234619861939902j,
                                            'Zbb': 0.5743408230315273 + 1.7234619861939902j,
                                            'Zcc': 0.5743408230315273 + 1.7234619861939902j, \
                                            'Zab': 0.5736068230292247 + 1.720524986189671j,
                                            'Zac': 0.5736068230292249 + 1.7205249861896714j,
                                            'Zbc': 0.5736068230292249 + 1.7205249861896714j, \
                                            'Baa': 0j, 'Bbb': 0j, 'Bcc': 0j, 'Bab': 0j, 'Bac': 0j, 'Bbc': 0j},
                                           ignore_index=True)
        for i in range(len(line_info_df)):
            line_info_i = line_info_df.iloc[i]
            Zaa_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Zaa'].values[0] * line_info_i[
                'Z_conv_factor'] * line_info_i['length']
            Zbb_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Zbb'].values[0] * line_info_i[
                'Z_conv_factor'] * line_info_i['length']
            Zcc_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Zcc'].values[0] * line_info_i[
                'Z_conv_factor'] * line_info_i['length']
            Zab_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Zab'].values[0] * line_info_i[
                'Z_conv_factor'] * line_info_i['length']
            Zac_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Zac'].values[0] * line_info_i[
                'Z_conv_factor'] * line_info_i['length']
            Zbc_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Zbc'].values[0] * line_info_i[
                'Z_conv_factor'] * line_info_i['length']
            Baa_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Baa'].values[0] * line_info_i[
                'B_conv_factor'] * line_info_i['length']
            Bbb_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Bbb'].values[0] * line_info_i[
                'B_conv_factor'] * line_info_i['length']
            Bcc_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Bcc'].values[0] * line_info_i[
                'B_conv_factor'] * line_info_i['length']
            Bab_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Bab'].values[0] * line_info_i[
                'B_conv_factor'] * line_info_i['length']
            Bac_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Bac'].values[0] * line_info_i[
                'B_conv_factor'] * line_info_i['length']
            Bbc_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Bbc'].values[0] * line_info_i[
                'B_conv_factor'] * line_info_i['length']
            self.line_df = self.line_df._append(
                {'busA': line_info_df.iloc[i]['busA'], 'busB': line_info_df.iloc[i]['busB'], \
                 'Zaa': Zaa_i, 'Zbb': Zbb_i, 'Zcc': Zcc_i, 'Zab': Zab_i, 'Zac': Zac_i, 'Zbc': Zbc_i, \
                 'Baa': Baa_i, 'Bbb': Bbb_i, 'Bcc': Bcc_i, 'Bab': Bab_i, 'Bac': Bac_i, 'Bbc': Bbc_i}, ignore_index=True)
        ##Add Transformer
        ##Y-G to Y-G transformer - therefore diagonal Y matrix (and not rank deficient)
        ##500kVA, 4.16kV/0.48kV, R=1.1%, X=2%
        ##impedance on the primary side: Zohms_pri = (4.16e3)**2/(500e3)*(0.011+0.02j)
        ##
        ##Create transformer data frame
        ##Types: 'wye-g', 'wye', 'delta'
        transformer_columns = ['busA', 'busB', 'typeA', 'typeB', 'Zseries', 'Zshunt']
        self.transformer_df = pd.DataFrame(columns=transformer_columns)
        self.transformer_df = self.transformer_df._append(
            {'busA': 'sourcebusz', 'busB': '1', 'typeA': 'delta', 'typeB': 'wye-g', 'Zseries': 0.00086528 + 0.0086528j,
             'Zshunt': 0j}, ignore_index=True)
        ##e.g. can test other transformer combinations:
        ##self.transformer_df = self.transformer_df._append({'busA':'633','busB':'634','typeA':'wye','typeB':'wye','Zseries':0.381+0.692j,'Zshunt':0},ignore_index=True)

        ### Yihong Modification
        self.capacitor_df = []

        if updateYZ:
            # print('Update Y and Z...\n',time.process_time())
            print('Updating Y and Z...')
            self.update_YandZ()

    def setup_network_ieee123(self, updateYZ=True):

        #path = os.path.dirname(os.path.abspath(__file__))
        #path = os.path.dirname(os.path.dirname(__file__))
        file_location = os.path.join(path, 'Data', 'networks', 'feeder123/')

        bus_load_data = pd.read_excel(file_location + 'spot loads data.xls', header=[2, 3])
        bus_load_data = bus_load_data[:-1]

        lines_data = pd.read_excel(file_location + 'line data.xls', header=2)
        bus_name_list = [150] + list(range(1, 115)) + [135, 149, 151, 152, 160, 197, 250, 300, 350, 450, 610]

        switch_data = pd.read_excel(file_location + 'switch data.xls', header=2)
        # the switches that are open
        switch_data_open = switch_data.loc[switch_data['Normal'] == 'open']
        # the switches that are closed
        switch_data_closed = switch_data.loc[switch_data['Normal'] == 'closed']
        ###keep only lines that connect two connected buses

        self.N_buses = len(bus_name_list)
        self.N_lines = lines_data.shape[0]
        self.N_phases = 3
        ##Create buses dataframe
        self.Vslack = 4.16e3  # slack bus line-to-line voltage
        self.Vslack_ph = self.Vslack / np.sqrt(3)  # slack bus phase voltage

        #### bus_df
        bus_columns = ['name', 'number', 'v_base', 'load_type', 'connect', 'Pa', 'Pb', 'Pc', 'Qa', 'Qb', 'Qc']
        bus_index = range(self.N_buses)
        self.bus_df = pd.DataFrame(index=bus_index, columns=bus_columns)
        for i, bus_name in enumerate(bus_name_list):
            # if there are specified bus load, set it as the bus_load_data
            if bus_name in bus_load_data['Node', 'bus'].values:
                index = bus_load_data.index[bus_name == bus_load_data['Node', 'bus']][0]
                self.bus_df.iloc[i] = {'name': str(bus_name), 'number': i, 'v_base': self.Vslack_ph,
                                       'load_type': bus_load_data['Load', 'Model'][index][2:],
                                       'connect': bus_load_data['Load', 'Model'][index][0], \
                                       'Pa': bus_load_data['Ph-1', 'kW'][index], 'Pb': bus_load_data['Ph-2', 'kW'][index],
                                       'Pc': bus_load_data['Ph-3', 'kW'][index], \
                                       'Qa': bus_load_data['Ph-1', 'kVAr'][index], 'Qb': bus_load_data['Ph-2', 'kVAr'][index],
                                       'Qc': bus_load_data['Ph-4', 'kVAr'][index]}
            # if not, set the bus to some default settings
            else:
                self.bus_df.iloc[i] = {'name': str(bus_name), 'number': i, 'v_base': self.Vslack_ph,
                                       'load_type': 'PQ' if i > 0 else 'S',
                                       'connect': 'D', \
                                       'Pa': 0,
                                       'Pb': 0,
                                       'Pc': 0, \
                                       'Qa': 0,
                                       'Qb': 0,
                                       'Qc': 0}

        #### line config df Z in Ohm/mile and B in micro Siemens per mile
        line_config_col = ['name', 'Zaa', 'Zbb', 'Zcc', 'Zab', 'Zac', 'Zbc', 'Baa', 'Bbb', 'Bcc', 'Bab', 'Bac', 'Bbc']
        line_config_df = pd.DataFrame(index=range(12), columns=line_config_col)

        line_config_df.iloc[0] = {'name': 1, 'Zaa': 0.4576 + 1.0780j, 'Zbb': 0.4666 + 1.0482j, 'Zcc': 0.4615 + 1.0651j, \
                                  'Zab': 0.1560 + 0.5017j, 'Zac': 0.1535 + 0.3849j, 'Zbc': 0.1580 + 0.4236j, \
                                  'Baa': 5.6765j, 'Bbb': 5.9809j, 'Bcc': 5.3971j, \
                                  'Bab': -1.8319j, 'Bac': -0.6982j, 'Bbc': -1.1645j}

        line_config_df.iloc[1] = {'name': 2, 'Zaa': 0.4666 + 1.0482j, 'Zbb': 0.4615 + 1.0651j, 'Zcc': 0.4576 + 1.0780j, \
                                  'Zab': 0.1580 + 0.4236j, 'Zac': 0.1560 + 0.5017j, 'Zbc': 0.1535 + 0.3849j, \
                                  'Baa': 5.9809j, 'Bbb': 5.3971j, 'Bcc': 5.6765j, \
                                  'Bab': -1.1645j, 'Bac': -1.8319j, 'Bbc': -0.6982j}

        line_config_df.iloc[2] = {'name': 3, 'Zaa': 0.4615 + 1.0651j, 'Zbb': 0.4576 + 1.0780j, 'Zcc': 0.4666 + 1.0482j, \
                                  'Zab': 0.1535 + 0.3849j, 'Zac': 0.1580 + 0.4236j, 'Zbc': 0.1560 + 0.5017j, \
                                  'Baa': 5.3971j, 'Bbb': 5.6765j, 'Bcc': 5.9809j, \
                                  'Bab': -0.6982j, 'Bac': -1.1645j, 'Bbc': -1.8319j}

        line_config_df.iloc[3] = {'name': 4, 'Zaa': 0.4615 + 1.0651j, 'Zbb': 0.4666 + 1.0482j, 'Zcc': 0.4576 + 1.0780j, \
                                  'Zab': 0.1580 + 0.4236j, 'Zac': 0.1535 + 0.3849j, 'Zbc': 0.1560 + 0.5017j, \
                                  'Baa': 5.3971j, 'Bbb': 5.9809j, 'Bcc': 5.6765j, \
                                  'Bab': -1.1645j, 'Bac': -0.6982j, 'Bbc': -1.8319j}

        line_config_df.iloc[4] = {'name': 5, 'Zaa': 0.4666 + 1.0482j, 'Zbb': 0.4576 + 1.0780j, 'Zcc': 0.4615 + 1.0651j, \
                                  'Zab': 0.1560 + 0.5017j, 'Zac': 0.1580 + 0.4236j, 'Zbc': 0.1535 + 0.3849j, \
                                  'Baa': 5.9809j, 'Bbb': 5.6765j, 'Bcc': 5.3971j, \
                                  'Bab': -1.8319j, 'Bac': -1.1645j, 'Bbc': -0.6982j}

        line_config_df.iloc[5] = {'name': 6, 'Zaa': 0.4576 + 1.0780j, 'Zbb': 0.4615 + 1.0651j, 'Zcc': 0.4666 + 1.0482j, \
                                  'Zab': 0.1535 + 0.3849j, 'Zac': 0.1560 + 0.5017j, 'Zbc': 0.1580 + 0.4236j, \
                                  'Baa': 5.6765j, 'Bbb': 5.3971j, 'Bcc': 5.9809j, \
                                  'Bab': -0.6982j, 'Bac': -1.8319j, 'Bbc': -1.1645j}

        line_config_df.iloc[6] = {'name': 7, 'Zaa': 0.4576 + 1.0780j, 'Zbb': 0.0000 + 0.0000j, 'Zcc': 0.4615 + 1.0651j, \
                                  'Zab': 0.0000 + 0.0000j, 'Zac': 0.1535 + 0.3849j, 'Zbc': 0.0000 + 0.0000j, \
                                  'Baa': 5.1154j, 'Bbb': 0.0000j, 'Bcc': 5.1704j, \
                                  'Bab': 0.0000j, 'Bac': -1.0549j, 'Bbc': 0.0000j}

        line_config_df.iloc[7] = {'name': 8, 'Zaa': 0.4576 + 1.0780j, 'Zbb': 0.4615 + 1.0651j, 'Zcc': 0.0000 + 0.0000j, \
                                  'Zab': 0.1535 + 0.3849j, 'Zac': 0.0000 + 0.0000j, 'Zbc': 0.0000 + 0.0000j, \
                                  'Baa': 5.1154j, 'Bbb': 5.1704j, 'Bcc': 0.0000j, \
                                  'Bab': -1.0549j, 'Bac': 0.0000j, 'Bbc': 0.0000j}

        line_config_df.iloc[8] = {'name': 9, 'Zaa': 1.3292 + 1.3475j, 'Zbb': 0.0000 + 0.0000j, 'Zcc': 0.0000 + 0.0000j, \
                                  'Zab': 0.0000 + 0.0000j, 'Zac': 0.0000 + 0.0000j, 'Zbc': 0.0000 + 0.0000j, \
                                  'Baa': 4.5193j, 'Bbb': 0.0000j, 'Bcc': 0.0000j, \
                                  'Bab': 0.0000j, 'Bac': 0.0000j, 'Bbc': 0.0000j}

        line_config_df.iloc[9] = {'name': 10, 'Zaa': 0.0000 + 0.0000j, 'Zbb': 1.3292 + 1.3475j, 'Zcc': 0.0000 + 0.0000j, \
                                  'Zab': 0.0000 + 0.0000j, 'Zac': 0.0000 + 0.0000j, 'Zbc': 0.0000 + 0.0000j, \
                                  'Baa': 0.0000j, 'Bbb': 4.5193j, 'Bcc': 0.0000j, \
                                  'Bab': 0.0000j, 'Bac': 0.0000j, 'Bbc': 0.0000j}

        line_config_df.iloc[10] = {'name': 11, 'Zaa': 0.0000 + 0.0000j, 'Zbb': 0.0000 + 0.0000j,
                                   'Zcc': 1.3292 + 1.3475j, \
                                   'Zab': 0.0000 + 0.0000j, 'Zac': 0.0000 + 0.0000j, 'Zbc': 0.0000 + 0.0000j, \
                                   'Baa': 0.0000j, 'Bbb': 0.0000j, 'Bcc': 4.5193j, \
                                   'Bab': 0.0000j, 'Bac': 0.0000j, 'Bbc': 0.0000j}

        line_config_df.iloc[11] = {'name': 12, 'Zaa': 1.5209 + 0.7521j, 'Zbb': 1.5329 + 0.7162j,
                                   'Zcc': 1.5209 + 0.7521j, \
                                   'Zab': 0.5198 + 0.2775j, 'Zac': 0.4924 + 0.2157j, 'Zbc': 0.5198 + 0.2775j, \
                                   'Baa': 67.2242j, 'Bbb': 67.2242j, 'Bcc': 67.2242j, \
                                   'Bab': 0.0000j, 'Bac': 0.0000j, 'Bbc': 0.0000j}

        ##Create line information data frame
        line_info_col = ['busA', 'busB', 'config', 'length', 'Z_conv_factor',
                         'B_conv_factor']  # Z_conv_factor = ft/mile, B_conv_factor 1e-6*ft/mile
        line_info_df = pd.DataFrame(index=range(len(lines_data)), columns=line_info_col)
        for i in range(len(lines_data)):
            line_info_df.iloc[i] = {'busA': str(lines_data['Node A'][i]), 'busB': str(lines_data['Node B'][i]), \
                                    'config': lines_data['Config.'][i], 'length': float(lines_data['Length (ft.)'][i]), \
                                    'Z_conv_factor': 1 / 5280, 'B_conv_factor': 1e-6 / 5280}

        ##Create line data frame
        line_columns = ['busA', 'busB', 'Zaa', 'Zbb', 'Zcc', 'Zab', 'Zac', 'Zbc', 'Baa', 'Bbb', 'Bcc', 'Bab', 'Bac',
                        'Bbc']
        self.line_df = pd.DataFrame(columns=line_columns)

        # create list containing list of disconected buses by switch
        switch_open_list = switch_data_open[['Node A', 'Node B']].values.tolist()
        for i in range(len(line_info_df)):

            line_info_i = line_info_df.iloc[i]

            if ([line_info_i['busA'], line_info_i['busB']] in switch_open_list) or \
                    ([line_info_i['busB'], line_info_i['busA']] in switch_open_list):
                continue

            Zaa_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Zaa'].values[0] * line_info_i[
                'Z_conv_factor'] * line_info_i['length']
            Zbb_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Zbb'].values[0] * line_info_i[
                'Z_conv_factor'] * line_info_i['length']
            Zcc_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Zcc'].values[0] * line_info_i[
                'Z_conv_factor'] * line_info_i['length']
            Zab_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Zab'].values[0] * line_info_i[
                'Z_conv_factor'] * line_info_i['length']
            Zac_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Zac'].values[0] * line_info_i[
                'Z_conv_factor'] * line_info_i['length']
            Zbc_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Zbc'].values[0] * line_info_i[
                'Z_conv_factor'] * line_info_i['length']
            Baa_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Baa'].values[0] * line_info_i[
                'B_conv_factor'] * line_info_i['length']
            Bbb_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Bbb'].values[0] * line_info_i[
                'B_conv_factor'] * line_info_i['length']
            Bcc_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Bcc'].values[0] * line_info_i[
                'B_conv_factor'] * line_info_i['length']
            Bab_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Bab'].values[0] * line_info_i[
                'B_conv_factor'] * line_info_i['length']
            Bac_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Bac'].values[0] * line_info_i[
                'B_conv_factor'] * line_info_i['length']
            Bbc_i = line_config_df[line_config_df['name'] == line_info_i['config']]['Bbc'].values[0] * line_info_i[
                'B_conv_factor'] * line_info_i['length']
            self.line_df = self.line_df._append(
                {'busA': line_info_df.iloc[i]['busA'], 'busB': line_info_df.iloc[i]['busB'], \
                 'Zaa': Zaa_i, 'Zbb': Zbb_i, 'Zcc': Zcc_i, 'Zab': Zab_i, 'Zac': Zac_i, 'Zbc': Zbc_i, \
                 'Baa': Baa_i, 'Bbb': Bbb_i, 'Bcc': Bcc_i, 'Bab': Bab_i, 'Bac': Bac_i, 'Bbc': Bbc_i}, ignore_index=True)

        ## Add Switches that are closed. Note that, this is necessary
        ## since some buses are not connected by line but by switches
        Zswitch = 1e-6 + 0j
        for i in range(len(switch_data_closed)):
            switch_data_closed_i = switch_data_closed.iloc[i]
            self.line_df = self.line_df._append({'busA': str(switch_data_closed_i['Node A']), 'busB': str(switch_data_closed_i['Node B']), \
                                                'Zaa': Zswitch, 'Zbb': Zswitch, 'Zcc': Zswitch, 'Zab': 0, 'Zac': 0,
                                                'Zbc': 0, \
                                                'Baa': 0, 'Bbb': 0, 'Bcc': 0, 'Bab': 0, 'Bac': 0, 'Bbc': 0},
                                               ignore_index=True)

        ##Add Transformer
        transformer_columns = ['busA', 'busB', 'typeA', 'typeB', 'Zseries', 'Zshunt']
        self.transformer_df = pd.DataFrame(columns=transformer_columns)
        self.transformer_df = self.transformer_df._append(
                                    {'busA': '61', 'busB': '610', 'typeA': 'delta', 'typeB': 'delta',
                                     'Zseries': 1.465 + 3.138j, 'Zshunt': 0}, ignore_index=True)

        capacitor_data = pd.read_excel(file_location + 'cap data.xls', header=2).fillna(0)
        capacitor_data = capacitor_data[1:-1]
        capacitor_columns = ['name', 'number', 'bus', 'kVln', 'connect', 'Qa', 'Qb', 'Qc']
        self.capacitor_df = pd.DataFrame(columns=capacitor_columns)
        for i in range(1, len(capacitor_data)):
            self.capacitor_df = self.capacitor_df._append(
                {'name': 'cap2', 'number': i - 1, 'bus': str(capacitor_data['Node'][i]), 'kVln': 2.4, 'connect': 'Y',
                 'Qa': capacitor_data['Ph-A'][i], 'Qb': capacitor_data['Ph-B'][i], 'Qc': capacitor_data['Ph-C'][i]},
                ignore_index=True)

        if updateYZ:
            # print('Update Y and Z...\n',time.process_time())
            print('Updating Y and Z...')
            self.update_YandZ()


    def loadDssNetwork(self,ntwkName,updateYZ=True,
                       testModel=False,testPlot=False):
        print('Loading network ',ntwkName,'\n')
        #sys.path.insert(0, os.path.join(path,'System'))
        #os.path.dirname(os.path.dirname(path))
        #path = os.path.dirname(os.path.abspath(__file__))
        #path = os.path.dirname(os.path.dirname(__file__))
        self.feederLoc = os.path.join(path,'Data','networks',
                                      ntwkName+'_dss',ntwkName)
        
        # LOAD dataframes and correct formatting V V V V
        src_df = pd.read_csv(self.feederLoc+"_src_df.csv",index_col=0)
        self.Vbase = 1e3*src_df.loc[:,'kvBaseLL'][0]/np.sqrt(3)
        
        VsrcLN = self.Vbase*src_df.loc[:,'pu'][0]

        bus_df = pd.read_csv(self.feederLoc+"_bus_df.csv",index_col=0)
        bus_df.index=np.arange(len(bus_df))
        bus_df.loc[:,'load_type'] = 'PQ'
        bus_df.loc[0,'load_type'] = 'S'
        bus_df['name'] = bus_df['name'].astype("str") # <------ required

        line_df = pd.read_csv(self.feederLoc+"_line_df.csv",index_col=0)
        line_df.index=np.arange(len(line_df))
        line_df.loc[:,'busA'] = list(map(str,line_df.loc[:,'busA'])) 
            # cannot save numbers as 'str' type when saving as csv
        line_df.loc[:,'busB'] = list(map(str,line_df.loc[:,'busB']))

        # fix up all of the complex numbers to be complex:
        for i in range(2,line_df.shape[1]):
            for j in range(line_df.shape[0]):
                line_df.iloc[j,i] = np.complex_(np.complex(line_df.iloc[j,i])) 
                # convert to (exactly) same format

        # add the transformers (if there are any)
        if os.path.exists(self.feederLoc+"_trn_df.csv"):
            trn_df = pd.read_csv(self.feederLoc+"_trn_df.csv",index_col=0)
            trn_df.index=np.arange(len(trn_df))
            trn_df.loc[:,'busA'] = list(map(str,trn_df.loc[:,'busA'])) 
            # cannot save numbers as 'str' type when saving as csv
            trn_df.loc[:,'busB'] = list(map(str,trn_df.loc[:,'busB'])) 
            # cannot save numbers as 'str' type when saving as csv

            for i in [4,5]: # convert to (exactly) same format
                for j in range(trn_df.shape[0]):
                    trnNo = np.complex_(np.complex(trn_df.iloc[j,i])) 
                    if i==5 and trnNo==0:
                        trn_df.iloc[j,i] = int(trnNo.real)
                    else:
                        trn_df.iloc[j,i] = complex(trnNo)
        else:
            trn_df = pd.DataFrame(columns=self.transformer_df.columns) 
            # empty but with correct column names

        # add the capacitors (if there are any)
        if os.path.exists(self.feederLoc+"_cap_df.csv"):
            cap_df = pd.read_csv(self.feederLoc+"_cap_df.csv",index_col=0)
            cap_df.loc[:,'bus'] = list(map(str,cap_df.loc[:,'bus'])) 
            # cannot save numbers as 'str' type when saving as csv
        else:
            cap_df = pd.DataFrame(columns=self.capacitor_df.columns) 
            # empty but with correct column names

        # load these into the model
        self.Vslack = VsrcLN*np.sqrt(3)
        self.Vslack_ph = VsrcLN
        self.bus_df = bus_df
        self.line_df = line_df
        self.transformer_df = trn_df
        self.capacitor_df = cap_df
        
        if testPlot or testModel or updateYZ:
            print('Updating Y and Z.')
            self.update_YandZ()
        if testPlot or testModel:
            print('Run Zbus power flow')
            self.zbus_pf()
            RES = self.res_bus_df.copy()
            RES.loc[np.where(RES['Va']==0)[0],'Va'] = np.nan
            RES.loc[np.where(RES['Vb']==0)[0],'Vb'] = np.nan
            RES.loc[np.where(RES['Vc']==0)[0],'Vc'] = np.nan
            
            solution_df = pd.read_csv(self.feederLoc+"_solution_df.csv",
                                      index_col=0)
            YZorder = tuple(solution_df.loc[:,'bus'].astype('str'))
            YNodeV = strCpx2cpx(np.array(
                    solution_df.loc[:,'vLN'])).astype('complex128')

            resMask,dssMask = self.getDssResMasks(RES,YZorder)
            [resMask1,resMask2,resMask3] = resMask
            [dssMask1,dssMask2,dssMask3] = dssMask
            vOxemf = np.r_[ RES['Va'].astype('complex128').values[resMask1],
                           RES['Vb'].astype('complex128').values[resMask2], 
                           RES['Vc'].astype('complex128').values[resMask3]
                               ]/self.Vbase
            vDss = YNodeV[dssMask1+dssMask2+dssMask3]/self.Vbase
            print( '--> Voltage relative error:',
                  100*np.linalg.norm( vOxemf - vDss )/
                  np.linalg.norm( vDss ),' %' )
        
        # TESTING V V V V [if not testing, done here]
        if testPlot: self.dssTestPlot(RES)
        
    def voltageComparison(self,RES=None):
        if RES is None:
            RES=self.res_bus_df
        solution_df = pd.read_csv(self.feederLoc+"_solution_df.csv",
                                  index_col=0)
        YZorder = tuple(solution_df.loc[:,'bus'].astype('str'))
        YNodeV = strCpx2cpx(np.array(
                    solution_df.loc[:,'vLN'])).astype('complex128')
        resMask,dssMask = self.getDssResMasks(RES,YZorder)
        [resMask1,resMask2,resMask3] = resMask
        [dssMask1,dssMask2,dssMask3] = dssMask
        vOxemf = np.r_[ RES['Va'].astype('complex128').values[resMask1],
                        RES['Vb'].astype('complex128').values[resMask2], 
                        RES['Vc'].astype('complex128').values[resMask3] ]
        self.vDssVolts = YNodeV[dssMask1+dssMask2+dssMask3]
        self.vOxemfVolts = vOxemf
        self.dssEmfBusNode = np.r_[RES['name'].values[resMask1]\
            + np.array(['.1']*len(resMask1)), 
              RES['name'].values[resMask2] + np.array(['.2']*len(resMask2)), 
              RES['name'].values[resMask3] + np.array(['.3']*len(resMask3))]
        
    def getDssResMasks(self,RES,YZorder):
        YZbuses = []; YZphase = []
        for bus in YZorder:
           YZbuses = YZbuses + [bus.split('.')[0]]
           YZphase = YZphase + [bus.split('.')[1]]
        
        dssMask1 = []; dssMask2 = []; dssMask3 = []
        resMask1 = []; resMask2 = []; resMask3 = []
        i=0
        lnplt = np.arange(len(RES))
        for bus in YZbuses:
            if YZphase[i]=='1':
                dssMask1.append(i)
                resMask1.append(np.where(RES['name']==bus.lower())[0][0])
            elif YZphase[i]=='2':
                dssMask2.append(i)
                resMask2.append(np.where(RES['name']==bus.lower())[0][0])
            elif YZphase[i]=='3':
                dssMask3.append(i)
                resMask3.append(np.where(RES['name']==bus.lower())[0][0])
            i+=1
        
        resMask = [resMask1,resMask2,resMask3]
        dssMask = [dssMask1,dssMask2,dssMask3]
        return resMask,dssMask
        
    def dssTestPlot(self,RES=None,pltShow=True,resMask=None,
                    dssMask=None,types=None,YNodeV=None):
        if types is None:
            types=['V','S']
        
        if RES is None:
            RES = self.res_bus_df.copy()
            RES.loc[np.where(RES['Va']==0)[0],'Va'] = np.nan
            RES.loc[np.where(RES['Vb']==0)[0],'Vb'] = np.nan
            RES.loc[np.where(RES['Vc']==0)[0],'Vc'] = np.nan
        
        solution_df =pd.read_csv(self.feederLoc+"_solution_df.csv",index_col=0)
        sInj = strCpx2cpx(np.array(
                solution_df.loc[:,'sInjkW'])).astype('complex128')
        
        if resMask is None or dssMask is None:
            YZorder = tuple(solution_df.loc[:,'bus'].astype('str'))
            resMask,dssMask = self.getDssResMasks(RES,YZorder)
        
        if 'V' in types and YNodeV is None:
            solution_df = pd.read_csv(
                    self.feederLoc+"_solution_df.csv",index_col=0)
            YNodeV = strCpx2cpx(np.array(
                    solution_df.loc[:,'vLN'])).astype('complex128')
        
        pInj = -sInj.real
        qInj = -sInj.imag
        
        print(len(resMask))
        [resMask1,resMask2,resMask3] = resMask
        [dssMask1,dssMask2,dssMask3] = dssMask
        if 'S' in types:
            plt.figure(1)
            ax1 = plt.subplot(131); plt.grid(True)
            ax1.plot(RES['Sa'].astype('complex128').real,'rx')
            ax1.plot(RES['Sa'].astype('complex128').imag,'kx')
            plt.ylabel('Real/Reactive Power (kW, kVAr)')
            plt.title('Phase A')
            ax2 = plt.subplot(132); plt.grid(True)
            ax2.plot(RES['Sb'].astype('complex128').real,'rx')
            ax2.plot(RES['Sb'].astype('complex128').imag,'kx')
            plt.title('Phase B')
            ax3 = plt.subplot(133); plt.grid(True)
            ax3.plot(RES['Sc'].astype('complex128').real,'rx')
            ax3.plot(RES['Sc'].astype('complex128').imag,'kx')
            plt.title('Phase C')

            ax1.plot(resMask1,pInj[dssMask1],'r.')
            ax1.plot(resMask1,qInj[dssMask1],'k.')
            ax2.plot(resMask2,pInj[dssMask2],'r.')
            ax2.plot(resMask2,qInj[dssMask2],'k.')
            ax3.plot(resMask3,pInj[dssMask3],'r.')
            ax3.plot(resMask3,qInj[dssMask3],'k.')
            
            ax1.legend(('P, OPEN','Q, OPEN','P, O\'DSS','Q, O\'DSS'))
            plt.tight_layout()
            if pltShow: plt.show()
        if 'V' in types:
            xlm = (-0.5,len(RES)+0.5)

            plt.figure(2)
            ax1 = plt.subplot(231)
            plt.plot(abs(RES['Va'])/self.Vbase,'kx')
            plt.title('Phase A')
            plt.ylabel('Voltage (pu)')
            plt.grid(True)
            plt.ylim((0.85,1.15)) # fix ylims here <---
            plt.xlim(xlm)
            ax2 = plt.subplot(232)
            plt.plot(abs(RES['Vb'])/self.Vbase,'kx')
            plt.title('Phase B')
            plt.grid(True)
            plt.ylim((0.85,1.15)) # fix ylims here <---
            plt.xlim(xlm)
            ax3 = plt.subplot(233)
            plt.plot(abs(RES['Vc'])/self.Vbase,'kx')
            plt.title('Phase C')
            plt.grid(True)
            plt.ylim((0.85,1.15)) # fix ylims here <---
            plt.xlim(xlm)
            i=0
            lnplt = np.arange(len(RES))
            
            ax1.plot(resMask1,abs(YNodeV[dssMask1])
                /self.Vbase,'ko',markerfacecolor='w',zorder=-1)
            ax2.plot(resMask2,abs(YNodeV[dssMask2])
                /self.Vbase,'ko',markerfacecolor='w',zorder=-1)
            ax3.plot(resMask3,abs(YNodeV[dssMask3])
                /self.Vbase,'ko',markerfacecolor='w',zorder=-1)
            
            ax1.legend(('OPEN','OpenDSS'))

            ax1 = plt.subplot(234)
            plt.plot(np.rad2deg(
                np.angle(RES['Va'].get_values().astype('complex128'))),'kx')
            plt.ylabel('Angle (degrees)')
            plt.xlabel('Bus no.'); plt.grid(True)
            plt.ylim((-190,190))
            # xlm = plt.xlim()
            plt.plot(xlm,[180,180],'k--')
            plt.plot(xlm,[-180,-180],'k--')
            # plt.xlim((-0.5,len(RES)+0.5))
            plt.xlim(xlm)
            ax2 = plt.subplot(235)
            plt.plot(np.rad2deg(
                np.angle(RES['Vb'].get_values().astype('complex128'))),'kx')
            plt.xlabel('Bus no.'); plt.grid(True)
            plt.ylim((-190,190))
            # xlm = plt.xlim()
            # plt.xlim((-0.5,len(RES)+0.5))
            plt.plot(xlm,[180,180],'k--')
            plt.plot(xlm,[-180,-180],'k--')
            plt.xlim(xlm)
            ax3 = plt.subplot(236)
            plt.plot(np.rad2deg(
                np.angle(RES['Vc'].get_values().astype('complex128'))),'kx')
            plt.xlabel('Bus no.'); plt.grid(True)
            plt.ylim((-190,190))
            # xlm = plt.xlim()
            plt.plot(xlm,[180,180],'k--')
            plt.plot(xlm,[-180,-180],'k--')
            plt.xlim(xlm)
            plt.xlim((-0.5,len(RES)+0.5))
            i=0
            lnplt = np.arange(len(RES))
            
            ax1.plot(resMask1,np.rad2deg(np.angle(YNodeV[dssMask1])),
                     'ko',markerfacecolor='w',zorder=-1)
            ax2.plot(resMask2,np.rad2deg(np.angle(YNodeV[dssMask2])),
                     'ko',markerfacecolor='w',zorder=-1)
            ax3.plot(resMask3,np.rad2deg(np.angle(YNodeV[dssMask3])),
                     'ko',markerfacecolor='w',zorder=-1)
            
            plt.tight_layout()
            if pltShow: plt.show()
