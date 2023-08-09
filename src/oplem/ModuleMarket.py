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


__version__ = "1.1.0"



class Market:
    """
    Market class
    
    Parameters
    ----------
    participants : list of objects
	Containing details of each participant
	T_market : int
	Market horizon
	dt_market : float
	time interval duration (hours)
	P_import : numpy.ndarray
		max import from the grid (kW)
	price_imp : numpy.ndarray
		import prices from the grid (£/kWh)
	P_export : numpy.ndarray
		max export to the grid (kW)
	price_exp : numpy.ndarray
		export prices to the grid (£/kWh)
    Returns
    -------
    Market 
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

class subclass1(Module):
    """
    A subclass that inherits from the class Module

    Parameters
    ----------
    param1 : float
        param1 is a float
    param2 : list, optional, default [0,1,2]
        description of param2 
        
        description of param2 
        
        description of param2 
    Returns
    -------
    subClass  

    """
    
    def __init__(self, param1, param2=[0, 1, 2]):
        Module.__init__(self, param1, param2=param2)
        self.param1 = param1
        self.param2 = param2 
               
    def func1(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func2(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func3(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func4(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func5(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func6(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func7(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func8(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

class subclass2(Module):
    """
    A subclass that inherits from the class Module

    Parameters
    ----------
    param1 : float
        param1 is a float
    param2 : list, optional, default [0,1,2]
        description of param2 
        
        description of param2 
        
        description of param2 
    Returns
    -------
    subClass  

    """
    
    def __init__(self, param1, param2=[0, 1, 2]):
        Module.__init__(self, param1, param2=param2)
        self.param1 = param1
        self.param2 = param2 
               
    def func1(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func2(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func3(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func4(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func5(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func6(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func7(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func8(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

class subclass3(Module):
    """
    A subclass that inherits from the class Module

    Parameters
    ----------
    param1 : float
        param1 is a float
    param2 : list, optional, default [0,1,2]
        description of param2 
        
        description of param2 
        
        description of param2 
    Returns
    -------
    subClass  

    """
    
    def __init__(self, param1, param2=[0, 1, 2]):
        Module.__init__(self, param1, param2=param2)
        self.param1 = param1
        self.param2 = param2 
               
    def func1(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func2(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func3(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func4(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func5(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func6(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func7(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2

    def func8(self, arg1, arg2):
        """
        func with arguments

        Parameters
        ----------
        arg1 : float
            input1 from user 
        arg2 : int
            input2 from user 

        Returns
        ---------
        out1 : float
            out1 from user 
        out2 : int
            out2 from user
        """

        return arg1+arg2
