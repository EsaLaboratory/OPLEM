OPLEM Examples
===============

For further details on the example case studies, please refer here [1]_

Three case studies are presented to showcase the three markets.
All markets consider the reduced European low voltage (EULV) network with 55 load buses, of which 33 buses have connected PV panels, 16 buses with battery storage systems and 16 buses with heat pumps.

.. note::
   The case studies need to be run under a directory that contains the Data folder in the root of the repo

Time of Use (ToU) Market
-------------------------
ToU market considers a decentralised approach for energy trading. Every customer/participant optimises its resources in response to a market price signal.
The price signal is generally split into two: import prices and export prices, with the export prices lower than the import prices to encentivise self-consumption. the import prices are split into two or three periods on/off/(mid) peak with each period having different prices.


Central Market
---------------
Central market runs centrally and optimise all the resources dwonstream the network. The prices cutomers/participants have to pay are determined by the distributed locational marginal prices (DLMP).


P2P Market
---------------------------------------------------------------------------------
P2P market runs a bilateral peer-to-peer energy trading as was proposed in [2]_. This P2P strategy is a price-adjusting mechanism that returns a stable set of bilateral contracts between peers The strategy considers the peers' preferences and maximises their utility.

.. [1] tbc
.. [2] T. Morstyn, A. Teytelboym and M. D. Mcculloch, "Bilateral Contract Networks for Peer-to-Peer Energy Trading," in IEEE Transactions on Smart Grid, vol. 10, no. 2, pp. 2026-2035, March 2019, doi: 10.1109/TSG.2017.2786668.
