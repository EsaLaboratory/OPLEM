# Open Platform for Local Energy Markets (OPLEM)

Overview
=============
Energy System Architecture Lab ESAL's OPLEM is a tool for modelling and testing LEM designs. The platform combines distributed energy resource modelling (e.g. for PV generation sources, battery energy storage systems, electric vehicles), power flow simulation, multi-period optimisation for scheduling flexible energy resources, and offers a modular and flexible framework to create and test market designs adapted for distribution networks. OPLEM comes with the same features as [OPEN](https://github.com/EPGOxford/OPEN), which all combined cannot be found in existing tools, such as the multi-phase distribution network power flow, non-linear energy storage modelling, receding horizon and multi-period optimisation, separate models for control and simulation and the additional key feature of a generic market modelling that incorporates the common LEM designs and allows the user to develop their customised LEM designs.

The key features of OPLEM are presented in:
> insert paper info here when ready

Documentation
-------------
OPLEM documentation can be found [here](https://open-new.readthedocs.io/en/latest/)


Installation
-------------
1. Create a conda virtual environment:
```
conda create --name <name_env> python=3.11
```
and activate it: `conda activate <name_env>`

3. install oplem package and its dependencies by running the following 

```
pip install git+https://github.com/EsaLaboratory/OPLEM.git
```

Optimisation algorithms use `mosek` solver, academic license can be requested from [their website](https://www.mosek.com/products/academic-licenses/)

Getting started
----------------

The simplest way to start is to run the notebook `ToU_simple.ipynb` that demonstrates a simple case study.

More advanced case studies can be found under the root directory of the repo:
- test_TOU_Market.py
- test_P2P_Market.py
- test_Central_Market.py

Note that the 3 case studies need to be run under a directory that contains the Data folder in the root of the repo

License
--------
For academic and professional use, please provide attribution to the paper describing OPLEM.

Contributors
------------
- Chaimaa ESSAYEH
- Yihong ZHOU
- Thomas MORSTYN



