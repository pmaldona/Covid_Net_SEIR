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
from datetime import date
from datetime import timedelta
from functions_SEIRHDV import SEIRHVD_DA
import requests

"""
SEIRHVDB Class Initialization

"""


# ------------------..........--------------- #
#        Ingreso de Parámetros Generales      #
# ------------------------------------------- #
# Región Por CUT 
tstate = '13'

# Parámetros de tiempo
initdate = datetime(2020,5,15)


# Ajuste por ventiladores
# Parametros del modelo
beta = 0.117 #0.25#0.19 0.135
mu = 0.6 #2.6 0.6
ScaleFactor = 1.9 #4.8
SeroPrevFactor = 0.5 # Sero Prevalence Factor. Permite ajustar la cantidad de gente que entra en la dinamica
expinfection = 1 # Proporcion en la que contagian los expuestos

tsim = 500

simulation = SEIRHVD_DA(beta,mu,ScaleFactor=ScaleFactor,SeroPrevFactor=SeroPrevFactor,expinfection=expinfection,initdate = initdate, tsim = tsim,tstate=tstate)
simulation.importdata()
simulation.escenarios()
simulation.simulate()


endpoint = 'http://192.168.2.248:5003/SEIRHVDsimulate'
data = {
'state': str(tstate),
'beta': str(beta),
'mu': str(mu),
'tsim': str(tsim),
'initdate': str(initdate),
'ScaleFactor': str(ScaleFactor),
'SeroPrevFactor': str(SeroPrevFactor),
'qp': str(0),
'min_mov': str(0.65),
'max_mov': str(0.85),
'movfunct': str(0),
'qit': str(0),
'qft': str(100)}

r = requests.post(url = endpoint, data = data)

# T