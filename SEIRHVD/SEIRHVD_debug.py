#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# -------------------- #
#                      #
#     SEIRHDV Debug    #
#                      #
# -------------------- #


Scripts for testing new features and messing around

"""
# -------------------- #
#                      #
#     SEIRHDV Local    #
#                      #
# -------------------- #

from SEIRHVD_local import SEIRHVD_local
import numpy as np
from datetime import datetime

# ------------------..........--------------- #
#        Ingreso de Parámetros Generales      #
# ------------------------------------------- #



# Región Por CUT 
tstate = '13'
# Fecha Inicial Simulación
initdate = datetime(2020,5,15)


# Parámetros Epidemiológicos
beta = 0.25 #0.25#0.19 0.135
mu = 1.0 #2.6 0.6
ScaleFactor = 1#2.25 #4.8
SeroPrevFactor = 0.1 # Sero Prevalence Factor. Permite ajustar la cantidad de gente que entra en la dinamica
expinfection = 1 # Proporcion en la que contagian los expuestos

# Tiempo de simulación
tsim = 500 

# Creación de cuarentenas
quarantines = [[500.0, 0.85, 0.6, 0.0, 0.0, 500.0, 0.0],
              [500.0, 0.85, 0.65, 0.0, 0.0, 500.0, 0.0],
              [500.0, 0.85, 0.7, 0.0, 0.0, 500.0, 0.0]]

quarantines = [[500.0, 0.85, 0.6, 0.0, 0.0, 500.0, 0.0]]

# Creación del objeto de simulación 
simulation = SEIRHVD_local(beta = beta,mu = mu,ScaleFactor=ScaleFactor,SeroPrevFactor=SeroPrevFactor,expinfection=expinfection,initdate = initdate, tsim = tsim,tstate=tstate)
simulation.inputarray = np.array(quarantines)
simulation.addquarantine()


# Simular
simulation.simulate()

simulation.simulate(v=3,intgr=0)

simulation.I_as_ac = [simulation.sims[i][0].I_as_ac for i in range(simulation.numescenarios)]
simulation.I_mi_ac = [simulation.sims[i][0].I_mi_ac for i in range(simulation.numescenarios)]


# ------------------------------------------- #
#            Análisis de resultados:          #
# ------------------------------------------- #
# funcion genérica: 
simulation.pĺotvariable(enddate =  datetime(2020,7,30),days=0, reales= True,ylim = 0,norm=1,scalefactor = False)

#plots
simulation.plotfallecidosacumulados(days=-1)
simulation.plotactivos(days=-1)


# Tablas
simulation.tabladedatos(inicio = datetime(2020,6,15), fin = datetime(2020,7,30),variables =['I_cum','I_act','D','L'], path='') 















##############################################
# Región Por CUT 

import numpy as np
from datetime import datetime
from class_SEIRHUVD2 import SEIRHUDV 
import matplotlib.pyplot as plt  
tstate = '13'
# Fecha Inicial Simulación
initdate = datetime(2020,5,15)


# Parámetros Epidemiológicos
beta = 0.25 #0.25#0.19 0.135
mu = 1.0 #2.6 0.6
# Tiempo de simulación
tsim = 500 
max_mov = 0.8
rem_mov = 0.5
qp = 0
iqt = 0
fqt = 300
movfunct = 0

case = SEIRHUDV(tsim,max_mov,rem_mov,qp,iqt,fqt,movfunct)
case.beta =beta       
case.mu =mu
case.B = 0
case.D = 0
case.V = 0
case.I_act0 = 1000
case.R  = 0
case.Htot = np.poly1d(100)
case.Vtot = np.poly1d(10)
case.H_cr = 10  # Hospitalizados criticos dia 0
case.H0 = 25  # Hospitalizados totales dia 0

case.Hmax = 100
case.Vmax = 10
case.expinfection = 1
case.SeroPrevFactor = 0.5
case.population = 1000000 

# Accumulated Infected
case.I_as_ac = 0
case.I_mi_ac = 0
case.I_se_ac = 0
case.I_cr_ac = 0

# Deaths
case.H_crD = 0
case.VD = 0
case.I_seD = 0
case.I_crD = 0
#Daily infected
case.I_as_d = 0
case.I_mi_d = 0
case.I_se_d = 0
case.I_cr_d = 0

        
case.setrelationalvalues()
sol = case.integr_sci(0,tsim,0.1,False) 


from class_SEIRHUVD3 import SEIRHUDV 
# solver 2
tstate = '13'
# Fecha Inicial Simulación
initdate = datetime(2020,5,15)


# Parámetros Epidemiológicos
beta = 0.25 #0.25#0.19 0.135
mu = 1.0 #2.6 0.6
# Tiempo de simulación
tsim = 500 
max_mov = 0.8
rem_mov = 0.5
qp = 0
iqt = 0
fqt = 300
movfunct = 0

case2 = SEIRHUDV(tsim,max_mov,rem_mov,qp,iqt,fqt,movfunct)
case2.beta =beta       
case2.mu =mu
case2.B = 0
case2.D = 0
case2.V = 0
case2.I_act0 = 1000
case2.R  = 0
case2.Htot = np.poly1d(100)
case2.Vtot = np.poly1d(10)
case2.H_cr = 10  # Hospitalizados criticos dia 0
case2.H0 = 25  # Hospitalizados totales dia 0

case2.Hmax =100
case2.Vmax = 10
case2.expinfection = 1
case2.SeroPrevFactor = 0.5
case2.population = 1000000 

# Accumulated Infected
case2.I_as_ac = 0
case2.I_mi_ac = 0
case2.I_se_ac = 0
case2.I_cr_ac = 0

# Deaths
case2.H_crD = 0
case2.VD = 0
case2.I_seD = 0
case2.I_crD = 0

case2.I_as_d = 0
case2.I_mi_d = 0
case2.I_se_d = 0
case2.I_cr_d = 0

        
case2.setrelationalvalues()

sol2 = case2.integr_sci(0,tsim,0.1,False) 