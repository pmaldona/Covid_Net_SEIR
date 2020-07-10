#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------- #
#                      #
#     SEIRHDV Paper    #
#                      #
# -------------------- #

from SEIRHVD_local import SEIRHVD_local
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

tstate = ''
initdate = datetime(2020,5,15)

# ------------------- #
#        Plot 1       #
# ------------------- # 

# Camas H totales vs infectados severos vs HcrtoD



# Parametros del modelo
beta = 0.117 # Tasa de contagio
mu = 0.6 # Razon E0/I0
ScaleFactor = 1 # Factor de Escala: Numero de infectados por sobre los reportados
SeroPrevFactor = 1 # Sero Prevalence Factor. Permite ajustar la cantidad de gente que entra en la dinamica
expinfection = 1 # Proporcion en la que contagian los expuestos

tsim = 1000 # Tiempo de simulacion


# Simular
# Activos Iniciales
I_act0 = 100
# Muertos iniciales
dead0 = 0

population = 100000
# Initial Hospitalized
H0 = 0
# Initial VMI 
V0 = 0
# UCI/UTI capacity
Htot = 20
Htot = list(range(21))
# VMI Capacity
Vtot = 10



# Simulation
sims = []
for i in Htot:
    # Creación del objeto de simulación 
    simulation = SEIRHVD_local(beta = beta,mu = mu,ScaleFactor=ScaleFactor,SeroPrevFactor=SeroPrevFactor,expinfection=expinfection,initdate = initdate, tsim = tsim,tstate=tstate)
    quarantines = [[tsim, 0.85, 0.6, 0.0, 0.0, tsim, 0.0]]
    simulation.inputarray = np.array(quarantines)
    simulation.addquarantine()
    simulation.initialvalues(I_act0,dead0,population,H0,V0,i,Vtot,R=0,D=0,H_cr = 0)
    simulation.simulate()
    sims.append(simulation)

simulation.simulate()



# ------------------- #
#        Plot 2       #
# ------------------- # 

# Ventiladores vs infectados criticos vs IcrtoD vs VtoD 





# ------------------- #
#        Plot 3       #
# ------------------- # 

# Contourplot de SHFR=Muertos acumulados/((Ise+Icr) acumulados) considerando movilidad (alpha) en el eje X, 
# y numero de camas en el eje Y


# Parametros del modelo
beta = 0.15 # Tasa de contagio
mu = 0.6 # Razon E0/I0
ScaleFactor = 1 # Factor de Escala: Numero de infectados por sobre los reportados
SeroPrevFactor = 1 # Sero Prevalence Factor. Permite ajustar la cantidad de gente que entra en la dinamica
expinfection = 1 # Proporcion en la que contagian los expuestos

tsim = 1000 # Tiempo de simulacion


# Simular
# Activos Iniciales
I_act0 = 100
# Muertos iniciales
dead0 = 0

population = 100000
# Initial Hospitalized
H0 = 0
# Initial VMI 
V0 = 0
# UCI/UTI capacity per 1000 persons
nm = int(population/1000)
Htot = list(range(0*nm,20*nm,2*nm))
Htot_per1000 = list(range(0,20,2))
# VMI Capacity
Vtot = 10*nm

step = 0.2
alpha = list(np.arange(0,1+step,step))

# Simulation
SHFR = np.zeros((len(Htot),len(alpha)))  
for i in range(len(Htot)):
    for j in range(len(alpha)):
        # Creación del objeto de simulación 
        simulation = SEIRHVD_local(beta = beta,mu = mu,ScaleFactor=ScaleFactor,SeroPrevFactor=SeroPrevFactor,expinfection=expinfection,initdate = initdate, tsim = tsim,tstate=tstate)
        quarantines = [[tsim, 0.85, alpha[j], 0.0, 0.0, tsim, 0.0]]
        simulation.inputarray = np.array(quarantines)
        simulation.addquarantine()
        simulation.initialvalues(I_act0,dead0,population,H0,V0,Htot[i],Vtot,R=0,D=0,H_cr = 0)
        simulation.simulate()
        SHFR[i,j] = simulation.SHFR[0]




fig,ax=plt.subplots(1,1)
cp = ax.contourf(alpha,Htot_per1000,SHFR) 
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('SHFR')
ax.set_xlabel('Mobility')
ax.set_ylabel('Beds per 1000')
plt.show() 