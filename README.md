# Open Platform for Local Energy Markets (OPLEM)

This is an extension of Open Platform for Energy Networks (OPEN) available in https://github.com/EPGOxford/OPEN

Overview
=============

Oxford University's Energy and Power Group's Open Platform for Energy Networks (OPEN) provides a Python toolset for the modelling, simulation and optimisation of smart local energy systems.
The framework combines distributed energy resource modelling (e.g. for PV generation sources, battery energy storage systems, and electric vehicles), energy market modelling, power flow simulation and multi-period optimisation for scheduling flexible energy resources.

Energy System Architecture Lab ESAL's OPLEM is an extension of OPEN for modelling and testing LEM designs.  It offers a modular and flexible framework to create and test market designs adapted for distribution networks. 
OPLEM comes with the same features as OPEN, which all combined cannot be found in existing tools, such as the multi-phase distribution network power flow, non-linear energy storage modelling, receding horizon and multi-period optimisation, separate models for control and simulation and the additional key feature of a generic market modelling that incorporates the common LEM designs and allows the user to develop their customised LEM designs. 

OPEN and the methods used are presented in detail in the following publication:

T. Morstyn, K. Collett, A. Vijay, M. Deakin, S. Wheeler, S. M. Bhagavathy, F. Fele and M. D. McCulloch; *"An Open-Source Platform for Developing Smart Local Energy System Applications”*; University of Oxford Working Paper, 2019

The key features of OPLEM are presented in:
> insert paper info

Documentation
-------------
OPLEM documentation can be found [here] (https://open-new.readthedocs.io/en/latest/)


Installation
-------------
1. Create a conda virtual environment:
```
conda create --name <name_env> python
```
and activate it: `conda activate <name_env>`

3. install oplem package and its dependencies by running the following 

```
pip install git+https://github.com/EsaLaboratory/OPEN.git
```

Getting started
----------------

The simplest way to start is to run the notebook `ToU_simple.ipynb` that demonstrate a simple case study.

More advanced case studies can be found under the root directory of the repo:
- test_TOU_Market.py
- test_P2P_Market.py
- test_Central_Market.py

License
--------
For academic and professional use, please provide attribution to the papers describing OPEN/OPLEM.

Contributors
------------
- Chaimaa ESSAYEH
- Thomas MORSTYN



