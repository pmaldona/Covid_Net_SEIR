#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime as dt
import Single_dist_ref_SEIR as SDSEIR

#  ---------------------------------- #
#       Find Optimal Parameters       #
# ----------------------------------- #
np.random.seed()
tsim = 400
mov = 0.2
results = SDSEIR.ref_sim_national(mov=mov,qp=0,tsim = tsim,tci=None,movfunct='sawtooth')

Si = results['sim']['S'][0]
Ei = results['sim']['E'][0]
Ii = results['sim']['I'][0]
Ri = results['sim']['R'][0]
ti = 0

beta = results['params'][0]
sigma = results['params'][1]
gamma = results['params'][2]

Ir = pd.DataFrame(results['Ir'],columns=['Ir'])
tr = pd.DataFrame(results['tr'],columns=['tr'])
rd = tr.join(Ir)

params = pd.DataFrame(results['params'].tolist(),columns = ['params'],index=['beta','sigma','gamma','mu'])

print(params)
print('r0 = '+str(beta/gamma))

"""
# --------------- #
#   Save Data     #
# --------------- #
"""

# Saving real Data
rd.to_csv('National_RealData.csv',index=False)

# Saving parameters
params.to_csv('National_params.csv')

# -------------------------- #
#   Saving Simulation Data   #
# -------------------------- #

# No Quarantine
sim = results['sim']
sim.pop('t')
sim = pd.DataFrame(sim)
sim.to_csv('National_sim_NQ.csv')

# Total Quarantine
qp = -1
sim = SDSEIR.intger(Si,Ei,Ii,Ri,ti,tsim,0.01,beta,sigma,gamma,mov,qp=qp,tci=None, movfunct = 'sawtooth')
sim.pop('t')
sim = pd.DataFrame(sim)
sim.to_csv('Nacional_sim_TQ.csv')

# Dynamic Quaratine 7D
qp = 7
sim = SDSEIR.intger(Si,Ei,Ii,Ri,ti,tsim,0.01,beta,sigma,gamma,mov,qp=qp,tci=None, movfunct = 'sawtooth')
sim.pop('t')
sim = pd.DataFrame(sim)
sim.to_csv('Nacional_sim_7D.csv')

# Dynamic Quaratine 14D
qp = 14
sim = SDSEIR.intger(Si,Ei,Ii,Ri,ti,tsim,0.01,beta,sigma,gamma,mov,qp=qp,tci=None, movfunct = 'sawtooth')
sim.pop('t')
sim = pd.DataFrame(sim)
sim.to_csv('Nacional_sim_14D.csv')

# Dynamic Quaratine 28D
qp = 28
sim = SDSEIR.intger(Si,Ei,Ii,Ri,ti,tsim,0.01,beta,sigma,gamma,mov,qp=qp,tci=None, movfunct = 'sawtooth')
sim.pop('t')
sim = pd.DataFrame(sim)
sim.to_csv('Nacional_sim_28D.csv')