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
tstate = '02'

# Fecha Inicial
initdate = datetime(2020,5,25)

# Parametros del modelo
beta = 0.144 # Tasa de contagio
mu = 0.01 # Razon E0/I0
ScaleFactor = 1. # Factor de Escala: Numero de infectados por sobre los reportados
SeroPrevFactor = 0.25 # Sero Prevalence Factor. Permite ajustar la cantidad de gente que entra en la dinamica
expinfection = 1 # Proporcion en la que contagian los expuestos

tsim = 500 # Tiempo de simulacion

# Parametros del modelo
beta = 0.144 # Tasa de contagio
mu = 0.8 # Razon E0/I0
ScaleFactor = 1. # Factor de Escala: Numero de infectados por sobre los reportados
SeroPrevFactor = 0.25 # Sero Prevalence Factor. Permite ajustar la cantidad de gente que entra en la dinamica
expinfection = 1 # Proporcion en la que contagian los expuestos


# Creación del objeto de simulación 
simulation = SEIRHVD_DA(beta = beta,mu = mu,ScaleFactor=ScaleFactor,SeroPrevFactor=SeroPrevFactor,expinfection=expinfection,initdate = initdate, tsim = tsim,tstate=tstate)

# Creación de escenarios
# Opcion 1: Escenarios por defecto 
simulation.defaultescenarios()

# Opción 2: Escenarios personalizados
# [tsim,max_mov,rem_mov,qp,iqt,fqt,movfunct]

inputarray = [[500.0, 0.85, 0.6, 0.0, 0.0, 500.0, 0.0],
              [500.0, 0.85, 0.65, 0.0, 0.0, 500.0, 0.0],
              [500.0, 0.85, 0.7, 0.0, 0.0, 500.0, 0.0],
              [500.0, 0.85, 0.85, 0.0, 0.0, 500.0, 0.0]]
simulation.inputarray = np.array(inputarray)
simulation.addscenario()

simulation.importdata()

# Simular
simulation.simulate()


simulation.plotfallecidosacumulados(ylim=200,days=50)  
simulation.plotventiladores() 
simulation.plotactivos()




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