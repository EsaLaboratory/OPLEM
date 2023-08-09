#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project  Module 

This module helps debug the documentation issue with readthedocs
"""


__version__ = "1.1.0"


class Module:
    """
    This is the name of the class
    
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
    Class 
    """
    
    def __init__(self, param1, param2=[0, 1, 2]):
        self.param1 = param1
        self.param2 = param2 

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
