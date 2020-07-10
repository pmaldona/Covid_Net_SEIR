#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import class_SEIRHUVD2 as SD2
#import SEIRHVD
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as LA
import json
import requests
from datetime import datetime
from datetime import timedelta

from functions_SEIRHDV import SEIRHVD_DA

"""
SEIRHVDB Class Initialization

"""


# ------------------..........--------------- #
#        Ingreso de Parámetros Generales      #
# ------------------------------------------- #
# Región Por CUT 



# activos
# Let CFR
# Muertos acumulados

# plot 
# Graficos anteriores con cuarentena dinamica y total 40% para las 2 fechas de inicio. 

"""
# ---------------------------- #
#     Ajuste infectados RM     #
# ---------------------------- #
"""
# Fecha Inicial
initdate = datetime(2020,5,15)

tsim = 500 # Tiempo de simulacion
quarantineinitialdate = datetime(2020,6,26) + timedelta(days=14)


qit = (quarantineinitialdate-initdate).days

quarantineinitialdate = datetime(2020,6,26) #+ timedelta(days=14)
qit2 = (quarantineinitialdate-initdate).days

quarantinefinaldate = datetime(2020,6,26)
qft = 500


# Parametros Epidemicos

tstate = '13'

# Ajuste por ventiladores
# Parametros del modelo
beta = 0.117 #0.25#0.19 0.135
mu = 0.6 #2.6 0.6
ScaleFactor = 1.9 #4.8
SeroPrevFactor = 0.5 # Sero Prevalence Factor. Permite ajustar la cantidad de gente que entra en la dinamica
expinfection = 1 # Proporcion en la que contagian los expuestos



# Case 1
inputarray = [[500.0, 0.85, 0.6, 0.0, qit, 500.0, 0.0],
              [500.0, 0.85, 0.65, 0.0,qit, 500.0, 0.0],
              [500.0, 0.85, 0.7, 0.0, qit, 500.0, 0.0],
              [500.0, 0.85, 0.85, 0.0,qit, 500.0, 0.0],
              [500.0, 0.85, 0.4, 0.0,qit, 500.0, 0.0]]


simulation = SEIRHVD_DA(beta = beta,mu = mu,ScaleFactor=ScaleFactor,SeroPrevFactor=SeroPrevFactor,expinfection=expinfection,initdate = initdate, tsim = tsim,tstate=tstate)
simulation.inputarray = np.array(inputarray)
simulation.addscenario()

simulation.importdata()

# Simular
simulation.simulate()


#simulation.plotactivos(ylim=250,days=-1)
simulation.plotactivos(days=-1)
simulation.plotletalidad(days=-1)
simulation.plotfallecidosacumulados(days=-1,legend=False)




"""
# --------------------------------- #
#     Ajuste infectados Atacama     #
# --------------------------------- #
"""
# Fecha Inicial
initdate = datetime(2020,5,25)

tsim = 500 # Tiempo de simulacion
quarantineinitialdate = datetime(2020,6,26) + timedelta(days=14)


qit = (quarantineinitialdate-initdate).days

quarantineinitialdate = datetime(2020,6,26) #+ timedelta(days=14)
qit2 = (quarantineinitialdate-initdate).days

quarantinefinaldate = datetime(2020,6,26)
qft = 500


# Parametros Epidemicos

tstate = '03'

beta = 0.145 # Tasa de contagio
mu = 0.8 # Razon E0/I0
ScaleFactor = 1.0 # Factor de Escala: Numero de infectados por sobre los reportados
SeroPrevFactor = 0.3 # SeroPrevalence Factor. Permite ajustar la cantidad de gente que entra en la dinamica
expinfection = 1 # Proporcion en la que contagian los expuestos


# Case 1
inputarray = [[500.0, 0.85, 0.6, 0.0, qit, 500.0, 0.0],
              [500.0, 0.85, 0.65, 0.0,qit, 500.0, 0.0],
              [500.0, 0.85, 0.7, 0.0, qit, 500.0, 0.0],
              [500.0, 0.85, 0.85, 0.0,qit, 500.0, 0.0],
              [500.0, 0.85, 0.4, 0.0,qit, 500.0, 0.0]]

# Case 2
inputarray = [[500.0, 0.85, 0.6, 0.0, qit2, 500.0, 0.0],
              [500.0, 0.85, 0.65, 0.0,qit2, 500.0, 0.0],
              [500.0, 0.85, 0.7, 0.0, qit2, 500.0, 0.0],
              [500.0, 0.85, 0.85, 0.0,qit2, 500.0, 0.0],
              [500.0, 0.85, 0.4, 0.0,qit2, 500.0, 0.0]]

# Case 3 
inputarray = [[500.0, 0.50, 0.4,14, -qit, qft, 1],
              [500.0, 0.85, 0.4, 0.0, qit, qft, 0.0],
              [500.0, 0.50, 0.4,14, -qit2, qft, 1],
              [500.0, 0.85, 0.4, 0.0, qit2, qft, 0.0]]


simulation = SEIRHVD_DA(beta = beta,mu = mu,ScaleFactor=ScaleFactor,SeroPrevFactor=SeroPrevFactor,expinfection=expinfection,initdate = initdate, tsim = tsim,tstate=tstate)
simulation.inputarray = np.array(inputarray)
simulation.addscenario()

simulation.importdata()

# Simular
simulation.simulate()


#simulation.plotactivos(ylim=250,days=-1)
simulation.plotactivos(days=-1)
simulation.plotletalidad(days=-1)
simulation.plotfallecidosacumulados(days=-1,legend=False)




"""
# -------------------------- #
#     Ajuste Valparaiso      #
# -------------------------- #
"""
# Parámetros iniciales
# Fecha Inicial
initdate = datetime(2020,4,13)
tsim = 500 # Tiempo de simulacion
quarantineinitialdate = datetime(2020,6,12) #+ timedelta(days=14)
qit = (quarantineinitialdate-initdate).days

quarantinefinaldate = datetime(2020,6,26)
qft = 500

tstate = '05'

# ----------------------------------- #
#    Ajuste por  infectados activos   #
# ----------------------------------- #

beta = 0.125 # Tasa de contagio
mu = 0.8 # Razon E0/I0
ScaleFactor = 1. # Factor de Escala: Numero de infectados por sobre los reportados
SeroPrevFactor = 0.25 # Sero Prevalence Factor. Permite ajustar la cantidad de gente que entra en la dinamica
expinfection = 1 # Proporcion en la que contagian los expuestos

# ---------------------------- #
#    Ajuste por ventiladores   #
# ---------------------------- #

beta = 0.1 # Tasa de contagio
mu = 2 # Razon E0/I0
ScaleFactor = 2. # Factor de Escala: Numero de infectados por sobre los reportados
SeroPrevFactor = 0.25 # Sero Prevalence Factor. Permite ajustar la cantidad de gente que entra en la dinamica
expinfection = 1 # Proporcion en la que contagian los expuestos


# ---------------------------- #
#     Ajuste por Fallecidos    #
# ---------------------------- #
beta = 0.12 # Tasa de contagio
mu = 0.1 # Razon E0/I0
ScaleFactor = 2.5 # Factor de Escala: Numero de infectados por sobre los reportados
SeroPrevFactor = 0.25 # Sero Prevalence Factor. Permite ajustar la cantidad de gente que entra en la dinamica
expinfection = 1 # Proporcion en la que contagian los expuestos




inputarray = [[500.0, 0.85, 0.6, 0.0, qit, 500.0, 0.0],
              [500.0, 0.85, 0.65, 0.0,qit, 500.0, 0.0],
              [500.0, 0.85, 0.7, 0.0, qit, 500.0, 0.0],
              [500.0, 0.85, 0.85, 0.0,qit, 500.0, 0.0]]

simulation = SEIRHVD_DA(beta = beta,mu = mu,ScaleFactor=ScaleFactor,SeroPrevFactor=SeroPrevFactor,expinfection=expinfection,initdate = initdate, tsim = tsim,tstate=tstate)
simulation.inputarray = np.array(inputarray)
simulation.addscenario()

simulation.importdata()
simulation.simulate()
simulation.resume()



simulation.plotventiladores()

simulation.plotactivos(ylim=5000)

simulation.plotfallecidosacumulados(legend=False,ylim=400)




"""
# ------------------------------------ #
#     Ajuste infectados Magallanes     #
# ------------------------------------ #
"""

# Parámetros iniciales
# Fecha Inicial
initdate = datetime(2020,4,13)
tsim = 500 # Tiempo de simulacion
quarantineinitialdate = initdate# datetime(2020,4,1)
qit = (quarantineinitialdate-initdate).days

quarantinefinaldate = datetime(2020,5,7)
qft = 500


tstate = '12'
beta = 0.144 # Tasa de contagio
mu = 0.8 # Razon E0/I0
ScaleFactor = 1. # Factor de Escala: Numero de infectados por sobre los reportados
SeroPrevFactor = 0.25 # Sero Prevalence Factor. Permite ajustar la cantidad de gente que entra en la dinamica
expinfection = 1 # Proporcion en la que contagian los expuestos
inputarray = [[500.0, 0.85, 0.6, 0.0, 0.0, 500.0, 0.0],
              [500.0, 0.85, 0.65, 0.0, 0.0, 500.0, 0.0],
              [500.0, 0.85, 0.7, 0.0, 0.0, 500.0, 0.0],
              [500.0, 0.85, 0.85, 0.0, 0.0, 500.0, 0.0]]
simulation = SEIRHVD_DA(beta = beta,mu = mu,ScaleFactor=ScaleFactor,SeroPrevFactor=SeroPrevFactor,expinfection=expinfection,initdate = initdate, tsim = tsim,tstate=tstate)
simulation.inputarray = np.array(inputarray)
simulation.addscenario()

simulation.importdata()
simulation.simulate()



simulation.plotventiladores()
simulation.plotactivos(ylim=5000)
simulation.plotfallecidosacumulados(legend=False,ylim=400)



"""
# ------------------------------------ #
#     Ajuste infectados Magallanes     #
# ------------------------------------ #
"""

# Parámetros iniciales
# Fecha Inicial
initdate = datetime(2020,4,13)
tsim = 500 # Tiempo de simulacion
quarantineinitialdate = initdate# datetime(2020,4,1)
qit = (quarantineinitialdate-initdate).days

quarantinefinaldate = datetime(2020,5,7)
qft = 500


tstate = '02'
beta = 0.144 # Tasa de contagio
mu = 0.8 # Razon E0/I0
ScaleFactor = 1. # Factor de Escala: Numero de infectados por sobre los reportados
SeroPrevFactor = 0.25 # Sero Prevalence Factor. Permite ajustar la cantidad de gente que entra en la dinamica
expinfection = 1 # Proporcion en la que contagian los expuestos
inputarray = [[500.0, 0.85, 0.6, 0.0, 0.0, 500.0, 0.0],
              [500.0, 0.85, 0.65, 0.0, 0.0, 500.0, 0.0],
              [500.0, 0.85, 0.7, 0.0, 0.0, 500.0, 0.0],
              [500.0, 0.85, 0.85, 0.0, 0.0, 500.0, 0.0]]
simulation = SEIRHVD_DA(beta = beta,mu = mu,ScaleFactor=ScaleFactor,SeroPrevFactor=SeroPrevFactor,expinfection=expinfection,initdate = initdate, tsim = tsim,tstate=tstate)
simulation.inputarray = np.array(inputarray)
simulation.addscenario()

simulation.importdata()
simulation.simulate()

simulation.plotdatosactivos()
simulation.plotdatossochimi()  
simulation.plotdatosfallecidosacumulados()


simulation.plotventiladores()
simulation.plotactivos(ylim=5000)
simulation.plotfallecidosacumulados(legend=False,ylim=400)




# ---------------------------- #
#    Análisis de Resultados    #
# ---------------------------- #

# Infectados Activos  
simulation.tabladedatos(inicio = datetime(2020,5,15), fin = datetime(2020,6,30),variables =['I_cum','I_act','D','L'], path=''))


plt.scatter(simulation.sochimi_tr,simulation.Hr,label='Camas Ocupadas reales')
plt.scatter(simulation.sochimi_tr,simulation.Vr,label='Ventiladores Ocupados reales')
plt.scatter(simulation.sochimi_tr,simulation.Hr_tot,label='Capacidad de Camas')
plt.scatter(simulation.sochimi_tr,simulation.Vr_tot,label='Capacidad de Ventiladores')
plt.xlabel("Días desde "+initdate.strftime("%Y/%m/%d"))
#plt.ylabel('Camas y Ventiladores')
plt.title("Datos Sochimi")
plt.legend(loc=0)
plt.show()



plt.scatter(simulation.sochimi_tr,simulation.Hr,label='Camas Ocupadas reales')
plt.scatter(simulation.sochimi_tr,simulation.Vr,label='Ventiladores Ocupados reales')
plt.scatter(simulation.sochimi_tr,simulation.Hr_tot,label='Capacidad de Camas')
plt.scatter(simulation.sochimi_tr,simulation.Vr_tot,label='Capacidad de Ventiladores')
plt.xlabel("Días desde "+initdate.strftime("%Y/%m/%d"))
#plt.ylabel('Camas y Ventiladores')
plt.title("Datos Sochimi")
plt.legend(loc=0)
plt.show()


# Plot cuarentenas
i = 1
plt.plot(simulation.t[i],simulation.quarantines[i])
plt.show()