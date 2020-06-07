#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEIRHVDB Class Initialization
"""
import class_SEIRHUVD2 as SD2
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



# ------------------------------- #
#        Importar Data Real       #
# ------------------------------- #
# Fecha de inicio de la simulación
initdate = datetime(2020,4,12)
# Región 
tstate = '13'

# ---------------------- # 
#   Infectados Activos   #
# ---------------------- #
# Import cutlist
cutlist = []
cutlistpath = "../Data/cutlist.csv"
cutlist = pd.read_csv(cutlistpath, header = None,dtype=str)

actives = []
for index, row in cutlist.iterrows():    
    state = str(row[0])[0:2]
    comuna = str(row[0])
    if tstate == state:
        endpoint = "http://192.168.2.223:5006/getActiveNewCasesByComuna?comuna="+comuna
        r = requests.get(endpoint) 
        mydict = r.json()
        actives.append(mydict['actives'])
        #data=pd.DataFrame(mydict)
Ir = (np.array(actives)).sum(axis=0)
Ir_dates = [datetime.strptime(mydict['dates'][i],'%Y-%m-%d') for i in range(len(mydict['dates']))]


index = np.where(np.array(Ir_dates) >= initdate)[0][0]     
Ir=Ir[index:]
Ir_dates=Ir_dates[index:]
tr = [(Ir_dates[i]-initdate).days for i in range(len(Ir))]

# ------------------ #
#    Datos Sochimi   #
# ------------------ #
endpoint = "http://192.168.2.223:5006/getBedsAndVentilationByState?state="+tstate
r = requests.get(endpoint) 
mydict = r.json()
sochimi=pd.DataFrame(mydict)

Hr = sochimi['camas_ocupadas']
Vr =  sochimi['vmi_ocupados']
Vr_tot =  sochimi['vmi_totales']
Hr_tot =  sochimi['camas_totales']

sochimi_dates = [datetime.strptime(sochimi['dates'][i][:10],'%Y-%m-%d') for i in range(len(sochimi))]

index = np.where(np.array(sochimi_dates) >= initdate)[0][0] 
Hr=Hr[index:]
Vr=Vr[index:]
Hr_tot=Hr_tot[index:]
Vr_tot=Vr_tot[index:]
sochimi_dates = sochimi_dates[index:]
sochimi_tr = [(sochimi_dates[i]-initdate).days for i in range(len(Hr))]

# -------------------------------- #
#    Datos Fallecidos acumulados   #
# -------------------------------- #
endpoint = 'https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto14/FallecidosCumulativo.csv' 
Br = pd.read_csv(endpoint).iloc[6][1:] 
Br_dates = [datetime.strptime(Br.index[i],'%Y-%m-%d') for i in range(len(Br))]
index = np.where(np.array(Br_dates) >= initdate)[0][0] 
Br = Br[index:]
Br_dates = Br_dates[index:]
Br_tr = [(Br_dates[i]-initdate).days for i in range(len(Br))]


#------------------------------------------------- #
#               Define Scenarios                   #
#------------------------------------------------- #
#tsim,max_mov,rem_mov,qp,iqt,fqt,movfunct
#mov funct: 0 = once 1 = square
tsim = 500
inputarray=np.array([
        [tsim,0.8,0.6,0,0,500,0],
        [tsim,0.6,0.5,0,0,500,0],
        [tsim,0.4,0.4,0,0,500,0]])#,        
        #[tsim,0.8,0.8,0,0,500,0],                
        #[tsim,0.4,0.4,0,0,28,0],
        #[tsim,0.6,0.6,0,0,28,0],
        #[tsim,0.8,0.8,0,0,28,0]

ns = len(inputarray)

"""
Escenarios:
Realistas
0) Cuarentena total 14 movmax = 0.85 mov_rem = 0.6
1) Cuarentena total 14 movmax = 0.85 mov_rem = 0.7
2) Cuarentena total 14 movmax = 0.85 mov_rem = 0.8
3) Cuarentena total 21 movmax = 0.85 mov_rem = 0.6
4) Cuarentena total 21 movmax = 0.85 mov_rem = 0.7
5) Cuarentena total 21 movmax = 0.85 mov_rem = 0.8
6) Cuarentena total de 60 días, cuarentena hiperdinámica de 14 días movmax = 0.85 mov_rem = 0.7
7) Cuarentena total de 60 días, cuarentena hiperdinámica de 21 días movmax = 0.85 mov_rem = 0.7

Optimistas
8) Cuarentena total 14 movmax = 0.55 mov_rem = 0.2
9) Cuarentena total 14 movmax = 0.55 mov_rem = 0.3
10) Cuarentena total 14 movmax = 0.55 mov_rem = 0.4
11) Cuarentena total 21 movmax = 0.55 mov_rem = 0.2
12) Cuarentena total 21 movmax = 0.55 mov_rem = 0.3
13) Cuarentena total 21 movmax = 0.55 mov_rem = 0.4
14) Cuarentena total de 60 días, cuarentena hiperdinámica de 14 días movmax = 0.55 mov_rem = 0.3
15) Cuarentena total de 60 días, cuarentena hiperdinámica de 21 días movmax = 0.55 mov_rem = 0.3
""" 

#B=221
#D=26
#V=758
#I_act0=12642
#cId0=2234
#R=0
#H_cr=80
#H0=1720

# -------------- #
#    Simulate    #
# -------------- #


beta = 0.18#25#0.19
mu = 0.6#2.6
B = Br[0]  # Muertos acumulados al dia de inicio
D = Br[1]-Br[0]  # Muertos en el dia de inicio

fI = 3.8

I_act0 = fI*Ir[0]  # Infectados Activos al dia de inicio
cId0 = fI*(Ir[1]-Ir[0])# cId0 # I Nuevos dia 0
R  = 0   # Recuperados al dia de inicio

# Dinámica del aumento de camas
V = Vr[0]   # Ventilados al dia de inicio
H_cr = 0#Vr[0]*0.01 # Hospitalizados criticos dia 0
H0 = Hr[0] # Hospitalizados totales dia 0

Hcmodel = np.poly1d(np.polyfit(sochimi_tr, Hr_tot, 2)) 
Vcmodel = np.poly1d(np.polyfit(sochimi_tr, Vr_tot, 2)) 
Hc0=Hcmodel[0]#1980
H_incr=Hcmodel[1]
H_incr2 = Hcmodel[2]
Vc0=Vcmodel[0]#1029
V_incr = Vcmodel[1]
V_incr2 = Vcmodel[2]
tsat = (datetime(2020,7,8)-initdate).days
Hmax = Hcmodel(tsat)
Vmax = Vcmodel(tsat)

model = SD2.simSEIRHVD(beta = beta, mu = mu, inputarray= inputarray, B=B,D=D,V=V,I_act0=I_act0,cId0=cId0,R=R,Hc0=Hc0,H_incr=H_incr,H_incr2=H_incr2,Vc0=Vc0,V_incr=V_incr,V_incr2=V_incr2,H_cr=H_cr,H0=H0,tsat=tsat,Hmax=Hmax,Vmax=Vmax)
sims = model.simulate()

Iac15M = 29276

#-------------------------------- #
#       Variables auxiliares      #
#-------------------------------- #

# Poblacion total
T=[sims[i][0].S+sims[i][0].E_as+sims[i][0].E_sy+sims[i][0].I_as+sims[i][0].I_cr+sims[i][0].I_mi+sims[i][0].I_se\
    +sims[i][0].H_in+sims[i][0].H_out+sims[i][0].H_cr+sims[i][0].V+sims[i][0].D+sims[i][0].R+sims[i][0].B for i in range(len(inputarray))]


# Susceptibles
S = [sims[i][0].S for i in range(len(inputarray))]
# Hospitalizados totales diarios
H_sum=[sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V for i in range(len(inputarray))] 
# Hospitalizados camas diarios
H_bed=[sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out for i in range(len(inputarray))] 
# Hospitalizados ventiladores diarios
H_vent=[sims[i][0].V for i in range(len(inputarray))] 
# Infectados Acumulados
Iac=[sims[i][0].I+Iac15M-sims[i][0].I[0] for i in range(len(inputarray))] 
# Infectados activos diarios
I = [sims[i][0].I_as+sims[i][0].I_cr + sims[i][0].I_mi + sims[i][0].I_se + sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V for i in range(len(inputarray))] 
I_act = [sims[i][0].I_mi + sims[i][0].I_cr + sims[i][0].I_se + sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V for i in range(len(inputarray))] 


# Infectados asintomaticos
I_as = [sims[i][0].I_as for i in range(len(inputarray))] 
# Infectados mild
I_mi = [sims[i][0].I_mi for i in range(len(inputarray))] 
# Infectados severos
I_se = [sims[i][0].I_se for i in range(len(inputarray))] 
# Infectados criticos
I_cr = [sims[i][0].I_cr for i in range(len(inputarray))] 


# Expuestos totales diarios
E = [sims[i][0].E_as+sims[i][0].E_sy for i in range(len(inputarray))]
E_as = [sims[i][0].E_as for i in range(len(inputarray))]  
E_sy = [sims[i][0].E_sy for i in range(len(inputarray))]  
# Enterrados/Muertos acumulados
B = [sims[i][0].B for i in range(len(inputarray))] 
# Muertos diarios
D = [sims[i][0].D for i in range(len(inputarray))] 
# Recuperados
R = [sims[i][0].R for i in range(len(inputarray))] 
# Ventiladores diarios
V = [sims[i][0].V for i in range(len(inputarray))] 

# Variables temporales
t = [sims[i][0].t for i in range(len(inputarray))] 
dt = [np.diff(t[i]) for i in range(len(inputarray))] 
#tr = range(tsim)
idx = [np.searchsorted(t[i],tr) for i in range(len(inputarray))] 


# CAMAS
H_crin=[sims[i][0].H_cr for i in range(len(inputarray))] 
H_in=[sims[i][0].H_in for i in range(len(inputarray))] 
H_out=[sims[i][0].H_out for i in range(len(inputarray))] 
H_sum=[sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out for i in range(len(inputarray))]
H_tot=[sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V  for i in range(len(inputarray))]

CH = [sims[i][0].CH for i in range(len(inputarray))]
CV = [sims[i][0].CV for i in range(len(inputarray))]
ACH = [sims[i][0].ACH for i in range(len(inputarray))]
ACV = [sims[i][0].ACV for i in range(len(inputarray))]






# ------------------------------------------------------------------ #
#                         Estudio Resultados                         #
# ------------------------------------------------------------------ #

Hsat = [sims[0][0].h_sat(sims[0][0].H_in[i],sims[0][0].H_cr[i],sims[0][0].H_out[i],t[0][i]) for i in range(len(t[0]))] 
#Htot    

# -------------------------------- #
#        Graficar resultados       #
# -------------------------------- #
def plot(title='',xlabel='',ylabel=''):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=0)
    plt.show()


# ----------------------------------------- #
#       Camas simulados vs reales           #
# ----------------------------------------- #

# Máximo 4000 camas, saturación logística

# ----------- #
#     Time    #
# ----------- #

initday = initdate#date(2020,3,15)
enddate =  datetime(2020,6,30)
days = (enddate-initdate).days
#initD 
endD = [np.searchsorted(t[i],days) for i in range(len(inputarray))]
idx = [np.searchsorted(t[i],sochimi_tr) for i in range(len(inputarray))]

# ----------- #
#     Error   #
# ----------- #
err_bed = [LA.norm(Hr-H_sum[i][idx[i]])/LA.norm(Hr) for i in range(len(inputarray))]
err_vent = [LA.norm(Vr-V[i][idx[i]])/LA.norm(Vr) for i in range(len(inputarray))]


# ----------- #
#     Plot    #
# ----------- #
plt.scatter(sochimi_tr,Hr,label='Camas Ocupadas reales')
plt.scatter(sochimi_tr,Vr,label='Ventiladores Ocupados reales')

#plt.scatter(sochimi_tr,Hr_tot,label='Capacidad de Camas')
#plt.scatter(sochimi_tr,Vr_tot,label='Capacidad de Ventiladores')

Htot = [sims[0][0].Htot(i) for i in t[0][:endD[0]]]
Vtot = [sims[0][0].Vtot(i) for i in t[0][:endD[0]]]

plt.plot(t[0][:endD[0]],Htot,color='lime')
plt.plot(t[0][:endD[0]],Vtot,color='lime')

i = 0
plt.plot(t[i][:endD[i]],H_bed[i][:endD[i]],label='Camas utilizadas mov='+str(inputarray[i][1]),color = 'red' ,linestyle = 'dashed')
plt.plot(t[i][:endD[i]],V[i][:endD[i]],label='VMI Utilizados mov='+str(inputarray[i][1]),color = 'blue' ,linestyle = 'dashed')
i = 1
#plt.plot([], [], ' ', label='err_bed: '+str(round(100*err_bed[i],2))+'%')
#plt.plot([], [], ' ', label='err_vent: '+str(round(100*err_vent[i],2))+'%')
plt.plot(t[i][:endD[i]],H_bed[i][:endD[i]],label='Camas utilizadas mov='+str(inputarray[i][1]),color = 'red' ,linestyle = 'solid')
plt.plot(t[i][:endD[i]],V[i][:endD[i]],label='VMI Utilizados mov='+str(inputarray[i][1]),color = 'blue' ,linestyle = 'solid')
i = 2
plt.plot(t[i][:endD[i]],H_bed[i][:endD[i]],label='Camas utilizadas mov='+str(inputarray[i][1]),color = 'red' ,linestyle = 'dashed')
plt.plot(t[i][:endD[i]],V[i][:endD[i]],label='Camas utilizadas mov='+str(inputarray[i][1]),color = 'blue' ,linestyle = 'dashed')

plot(title = 'Camas',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))



plt.scatter(sochimi_tr,Hr,label='Camas Ocupadas reales')
plt.scatter(sochimi_tr,Vr,label='Ventiladores Ocupados reales')
plt.scatter(sochimi_tr,Hr_tot,label='Capacidad de Camas')
plt.scatter(sochimi_tr,Vr_tot,label='Capacidad de Ventiladores')

i = 3
plt.plot(t[i][:endD[i]],H_bed[i][:endD[i]],label='Camas utilizadas',color = 'red' ,linestyle = 'dashed')
plt.plot(t[i][:endD[i]],V[i][:endD[i]],label='Camas utilizadas' ,color = 'blue' ,linestyle = 'dashed')
i = 4
plt.plot([], [], ' ', label='err_bed: '+str(round(100*err_bed[i],2))+'%')
plt.plot([], [], ' ', label='err_vent: '+str(round(100*err_vent[i],2))+'%')
plt.plot(t[i][:endD[i]],H_bed[i][:endD[i]],label='Camas utilizadas',color = 'red' ,linestyle = 'solid')
plt.plot(t[i][:endD[i]],V[i][:endD[i]],label='Camas utilizadas',color = 'blue' ,linestyle = 'solid')
i = 5
plt.plot(t[i][:endD[i]],H_bed[i][:endD[i]],label='Camas utilizadas',color = 'red' ,linestyle = 'dashed')
plt.plot(t[i][:endD[i]],V[i][:endD[i]],label='Camas utilizadas',color = 'blue' ,linestyle = 'dashed')

plot(title = 'Camas',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))


# ----------------------------- #
#       Hospitalizados          #
# ----------------------------- #
# ----------- #
#     Time    #
# ----------- #

idx = [np.searchsorted(t[i],sochimi_tr) for i in range(len(inputarray))]


# ----------- #
#     Plot    #
# ----------- #

i = 1
plt.plot(t[i][:endD[i]],H_in[i][:endD[i]],label='Hin',linestyle = 'solid')
plt.plot(t[i][:endD[i]],H_out[i][:endD[i]],label='Hout',linestyle = 'solid')
plt.plot(t[i][:endD[i]],H_crin[i][:endD[i]],label='Hcr_in',linestyle = 'solid')

plot(title = 'Camas',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))

#
# Hrate
#

i=1
Hrate = H_in[i]/H_out[i] 
plt.plot(t[i][:endD[i]],Hrate[:endD[i]],label='Hrate',linestyle = 'solid')
plot(title = 'Razón Hin/Hout',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))

i=1
plt.plot(t[i][:endD[i]],(H_in[i]/H_sum[i])[:endD[i]],label='Hin',linestyle = 'solid')
plt.plot(t[i][:endD[i]],(H_out[i]/H_sum[i])[:endD[i]],label='Hout',linestyle = 'solid')
plot(title = 'Rate Hin y Hout / Hsum',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))


# -------------------------------------- #
#       Activos simulados vs reales      #
# -------------------------------------- #
I_shift = [I[i] - I[0][0] + Ir[0] for i in range(len(inputarray))] 
I_act_shift = [I_act[i] - I[0][0] + Ir[0] for i in range(len(inputarray))] 



# ----------- #
#     Time    #
# ----------- #

#initday = date(2020,5,15)
#endday =  date(2020,6,2)
#days = (endday-initday).days
#endD = [np.searchsorted(t[i],days) for i in range(len(inputarray))]
idx = [np.searchsorted(t[i],tr) for i in range(len(inputarray))]

# ----------- #
#     Error   #
# ----------- #
i = 1
err = [LA.norm(Ir-I[i][idx[i]])/LA.norm(Ir) for i in range(len(inputarray))]

# ----------- #
#     Plot    #
# ----------- #
plt.plot([], [], ' ', label='err: '+str(round(100*err[i],2))+'%')
plt.scatter(tr,Ir,label='Infectados Activos reales')
#i = 0
#plt.plot(t[i][:endD[i]],I_shift[i][:endD[i]],label='Infectados Mov = '+str(inputarray[i][1]))
i = 1
plt.plot(t[i][:endD[i]],I[i][:endD[i]],label='Infectados Mov = '+str(inputarray[i][1]))
#i = 2
#plt.plot(t[i][:endD[i]],I_shift[i][:endD[i]],label='Infectados Mov = '+str(inputarray[i][1]))
plt.xlim(0,55)
plot(title = 'Activos',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))




# ----------------------------------------- #
#      Fallecidos simulados vs reales       #
# ----------------------------------------- #
# ----------- #
#     Time    #
# ----------- #

#initday = date(2020,5,15)
#endday =  date(2020,6,30)
#days = (endday-initday).days
#endD = [np.searchsorted(t[i],days) for i in range(len(inputarray))]
idx = [np.searchsorted(t[i],Br_tr) for i in range(len(inputarray))]

# ----------- #
#     Error   #
# ----------- #
i = 1 # Mov 0.6
err = [LA.norm(Br-B[i][idx[i]])/LA.norm(Br) for i in range(len(inputarray))]

# ----------- #
#     Plot    #
# ----------- #
plt.plot([], [], ' ', label='err: '+str(round(100*err[i],2))+'%')
plt.scatter(Br_tr,Br,label='Fallecidos reales')
#i = 0
#plt.plot(t[i][:endD[i]],B[i][:endD[i]],label='Fallecidos Mov = '+str(inputarray[i][1]))
i = 1
plt.plot(t[i][:endD[i]],B[i][:endD[i]],label='Fallecidos Mov = '+str(inputarray[i][1]))
#i = 2
#plt.plot(t[i][:endD[i]],B[i][:endD[i]],label='Fallecidos Mov = '+str(inputarray[i][1]))
plot(title = 'Fallecidos',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))




# --------------------------- #
#      Camas requeridas       #
# --------------------------- #
# ----------- #
#     Time    #
# ----------- #

#initday = date(2020,5,15)
#endday =  date(2020,6,2)
#days = (endday-initday).days
#endD = [np.searchsorted(t[i],days) for i in range(len(inputarray))]
#idx = [np.searchsorted(t[i],Br_tr) for i in range(len(inputarray))]


# Fecha de colapso
i = 1
CH_date = [np.where(CH[i]>0)[0][0] for i in range(len(inputarray))]
CV_date = [np.where(CV[i]>0)[0][0] for i in range(len(inputarray))]

# ----------- #
#     Plot    #
# ----------- #

plt.plot([], [], ' ', label='Fecha de colapso Camas: '+str(round(t[i][CH_date[i]])))
plt.plot([], [], ' ', label='Fecha de colapso Vent: '+str(round(t[i][CV_date[i]])))

i = 0
plt.plot(t[i][:endD[i]],CH[i][:endD[i]],label='Intermedio/Intensivo Mov = '+str(inputarray[i][1]),color = 'red' ,linestyle = 'dashed')
plt.plot(t[i][:endD[i]],CV[i][:endD[i]],label='VMI Mov = '+str(inputarray[i][1]),color = 'blue' ,linestyle = 'dashed')
i = 1
plt.plot(t[i][:endD[i]],CH[i][:endD[i]],label='Intermedio/Intensivo Mov = '+str(inputarray[i][1]),color = 'red' ,linestyle = 'solid')
plt.plot(t[i][:endD[i]],CV[i][:endD[i]],label='VMI Mov = '+str(inputarray[i][1]),color = 'blue' ,linestyle = 'solid')
i = 2
plt.plot(t[i][:endD[i]],CH[i][:endD[i]],label='Intermedio/Intensivo Mov = '+str(inputarray[i][1]),color = 'red' ,linestyle = 'dashed')
plt.plot(t[i][:endD[i]],CV[i][:endD[i]],label='VMI Mov = '+str(inputarray[i][1]),color = 'blue' ,linestyle = 'dashed')

plot(title='Camas Requeridas',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))




# ------------------------------------ #
#      Necesidad total de Camas        #
# ------------------------------------ #
# ----------- #
#     Time    #
# ----------- #

#initday = date(2020,5,15)
#endday =  date(2020,6,2)
#days = (endday-initday).days
#endD = [np.searchsorted(t[i],days) for i in range(len(inputarray))]
#idx = [np.searchsorted(t[i],Br_tr) for i in range(len(inputarray))]


# Fecha de colapso
CH_date = [np.where(CH[i]>0)[0][0] for i in range(len(inputarray))]
CV_date = [np.where(CV[i]>0)[0][0] for i in range(len(inputarray))]

# ----------- #
#     Plot    #
# ----------- #
plt.scatter(sochimi_tr,Hr,label='Camas Ocupadas reales')
plt.scatter(sochimi_tr,Vr,label='Ventiladores Ocupados reales')

plt.scatter(sochimi_tr,Hr_tot,label='Capacidad de Camas')
plt.scatter(sochimi_tr,Vr_tot,label='Capacidad de Ventiladores')

plt.plot([], [], ' ', label='Fecha de colapso Camas: '+str(round(t[i][CH_date[i]])))
plt.plot([], [], ' ', label='Fecha de colapso Vent: '+str(round(t[i][CV_date[i]])))

i = 0
plt.plot(t[i][:endD[i]],np.array(CH[i][:endD[i]])+np.array(H_sum[i][:endD[i]]),label='Intermedio/Intensivo Mov = '+str(inputarray[i][1]),color = 'red' ,linestyle = 'dashed')
plt.plot(t[i][:endD[i]],np.array(CV[i][:endD[i]])+np.array(V[i][:endD[i]]),label='VMI Mov = '+str(inputarray[i][1]),color = 'blue' ,linestyle = 'dashed')
i = 1
plt.plot(t[i][:endD[i]],np.array(CH[i][:endD[i]])+np.array(H_sum[i][:endD[i]]),label='Intermedio/Intensivo Mov = '+str(inputarray[i][1]),color = 'red' ,linestyle = 'solid')
plt.plot(t[i][:endD[i]],np.array(CV[i][:endD[i]])+np.array(V[i][:endD[i]]),label='VMI Mov = '+str(inputarray[i][1]),color = 'blue' ,linestyle = 'solid')
i = 2
plt.plot(t[i][:endD[i]],np.array(CH[i][:endD[i]])+np.array(H_sum[i][:endD[i]]),label='Intermedio/Intensivo Mov = '+str(inputarray[i][1]),color = 'red' ,linestyle = 'dashed')
plt.plot(t[i][:endD[i]],np.array(CV[i][:endD[i]])+np.array(V[i][:endD[i]]),label='VMI Mov = '+str(inputarray[i][1]),color = 'blue' ,linestyle = 'dashed')

plot(title='Necesidad total de Camas',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))







# ------------------------------ #
#     Curvas Epidemiológicas     #
# ------------------------------ #

plt.plot(t[i],I[i],label='Infectados Activos')
plt.plot(t[i],S[i],label='Susceptibles')
plt.plot(t[i],E[i],label='Expuestos')
plt.plot(t[i],R[i],label='Recuperados')
plt.plot(t[i],D[i],label='Muertos por dia')
plt.plot(t[i],B[i],label='Enterrados')
plot(title = 'Curvas Epidemiologicas')




# ------------------------------ #
#       Curvas Infectados        #
# ------------------------------ #
plt.plot(t[i],I_as[i],label='Acumulados asintomáticos')
plt.plot(t[i],I_mi[i],label='Acumulados Mild')
plt.plot(t[i],I_se[i],label='Acumulados Severos')
plt.plot(t[i],I_cr[i],label='Acumulados Criticos')
plot(title='Infectados Simulados')


# ---------------------------------------- #
#       Proporcionalidad Infectados        #
# ---------------------------------------- #
Isum = [sims[i][0].I_as+sims[i][0].I_cr + sims[i][0].I_mi + sims[i][0].I_se for i in range(len(inputarray))] 

plt.plot(t[i],I_as[i]/Isum[i],label='Acumulados asintomáticos')
plt.plot(t[i],I_mi[i]/Isum[i],label='Acumulados Mild')
plt.plot(t[i],I_se[i]/Isum[i],label='Acumulados Severos')
plt.plot(t[i],I_cr[i]/Isum[i],label='Acumulados Criticos')
plot(title='Infectados Simulados')


# ------------------------------ #
#       Curvas Expuestos         #
# ------------------------------ #
initday = initdate#date(2020,3,15)
enddate =  datetime(2020,6,30)
days = (enddate-initdate).days
endD = [np.searchsorted(t[i],days) for i in range(len(inputarray))]
Isum = [sims[i][0].I_as+sims[i][0].I_cr + sims[i][0].I_mi + sims[i][0].I_se for i in range(len(inputarray))] 

i=1
plt.plot(t[i][:endD[i]],Isum[i][:endD[i]],label='Infectados')
plt.plot(t[i][:endD[i]],E[i][:endD[i]],label='Expuestos')
#plt.plot(t[i][:endD[i]],E_sy[i][:endD[i]],label='Expuestos sintomáticos')
plot(title='Expuestos - mu ='+str(mu)+' beta='+str(beta))


# ------------------------------ #
#       Curvas Expuestos         #
# ------------------------------ #
initday = initdate#date(2020,3,15)
enddate =  datetime(2020,6,30)
days = (enddate-initdate).days
endD = [np.searchsorted(t[i],days) for i in range(len(inputarray))]

Isum = [sims[i][0].I_as+sims[i][0].I_cr + sims[i][0].I_mi + sims[i][0].I_se for i in range(len(inputarray))] 

i=1
EIrate = E[i]/Isum[i]
plt.plot(t[i][:endD[i]],EIrate[:endD[i]],label='Tasa Expuestos/Infectados')
plot(title='Expuestos - mu ='+str(mu)+' beta='+str(beta))








# Estudio de Alpha
alpha = alphafunct(1,0,5,-10,80,'square') 
tplot = range(100)
alphaplot = [alpha(i) for i in tplot]
plt.plot(tplot,alphaplot)
plot()




























# ---------------------------- #
#     Estudio de Letalidad     #
# ---------------------------- #

i=0
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='red',linestyle='dotted',linewidth = 3.0)
i=1
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='red',linestyle='solid',linewidth = 3.0)
i=2
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='red',linestyle='dotted',linewidth = 3.0)
i=3
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='blue',linestyle='dotted',linewidth = 3.0)
i=4
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='blue',linestyle='solid',linewidth = 3.0)
i=5
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='blue',linestyle='dotted',linewidth = 3.0)
i=6
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4])+'SQ='+str(inputarray[i][3]),color='lime',linestyle='solid',linewidth = 3.0)
i=7
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4])+'SQ='+str(inputarray[i][3]),color='black',linestyle='solid',linewidth = 3.0)

plot(title='Letalidad Realista',ylabel='Letalidad')

i=8
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='blue',linestyle='dashed')
i=9
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='blue',linestyle='solid')
i=10
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='blue',linestyle='dotted')
i=11
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='red',linestyle='dashed')
i=12
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='red',linestyle='solid')
i=13
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='red',linestyle='dotted')
i=14
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4])+'SQ='+str(inputarray[i][3]),color='lime',linestyle='solid')
i=15
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4])+'SQ='+str(inputarray[i][3]),color='aqua',linestyle='solid')

plot(title='Letalidad Optimista',ylabel='Letalidad %')


# ----------------------------- #
#     Estudio de Mortalidad     #
# ----------------------------- #
i=0
plt.plot(t[i],B[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='blue',linestyle='dashed')
i=1
plt.plot(t[i],B[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='blue',linestyle='solid')
i=2
plt.plot(t[i],B[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='blue',linestyle='dotted')
i=3
plt.plot(t[i],B[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='red',linestyle='dashed')
i=4
plt.plot(t[i],B[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='red',linestyle='solid')
i=5
plt.plot(t[i],B[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='red',linestyle='dotted')
i=6
plt.plot(t[i],B[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4])+'SQ='+str(inputarray[i][3]),color='lime',linestyle='solid')
i=7
plt.plot(t[i],B[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4])+'SQ='+str(inputarray[i][3]),color='aqua',linestyle='solid')

plot(title='Mortalidad Realista',ylabel='Mortalidad')

i=8
plt.plot(t[i],B[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='blue',linestyle='dashed')
i=9
plt.plot(t[i],B[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='blue',linestyle='solid')
i=10
plt.plot(t[i],B[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='blue',linestyle='dotted')
i=11
plt.plot(t[i],B[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='red',linestyle='dashed')
i=12
plt.plot(t[i],B[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='red',linestyle='solid')
i=13
plt.plot(t[i],B[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='red',linestyle='dotted')
i=14
plt.plot(t[i],B[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4])+'SQ='+str(inputarray[i][3]),color='lime',linestyle='solid')
i=15
plt.plot(t[i],B[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4])+'SQ='+str(inputarray[i][3]),color='aqua',linestyle='solid')

plot(title='Mortalidad Optimista',ylabel='Mortalidad')




plt.plot(sims[i][0].t,sims[i][0].B,label='Buried')    
plt.plot(sims[i][0].t,sims[i][0].B,label='Buried')
plot('Mortalidad')




# ------------------------------- #
#      Estudio Hospitalizados     #
# ------------------------------- #
#H_sum=[sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V for i in range(len(inputarray))] 
i=0
plt.plot(t[i],H_sum[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='blue',linestyle='dashed')
i=1
plt.plot(t[i],H_sum[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='blue',linestyle='solid')
i=2
plt.plot(t[i],H_sum[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='blue',linestyle='dotted')
i=3
plt.plot(t[i],H_sum[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='red',linestyle='dashed')
i=4
plt.plot(t[i],H_sum[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='red',linestyle='solid')
i=5
plt.plot(t[i],H_sum[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='red',linestyle='dotted')
i=6
plt.plot(t[i],H_sum[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4])+'SQ='+str(inputarray[i][3]),color='lime',linestyle='solid')
i=7
plt.plot(t[i],H_sum[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4])+'SQ='+str(inputarray[i][3]),color='aqua',linestyle='solid')

plot(title='Hospitalizados Realista',ylabel='Letalidad')

i=8
plt.plot(t[i],H_sum[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='blue',linestyle='dashed')
i=9
plt.plot(t[i],H_sum[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='blue',linestyle='solid')
i=10
plt.plot(t[i],H_sum[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='blue',linestyle='dotted')
i=11
plt.plot(t[i],H_sum[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='red',linestyle='dashed')
i=12
plt.plot(t[i],H_sum[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='red',linestyle='solid')
i=13
plt.plot(t[i],H_sum[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4]),color='red',linestyle='dotted')
i=14
plt.plot(t[i],H_sum[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4])+'SQ='+str(inputarray[i][3]),color='lime',linestyle='solid')
i=15
plt.plot(t[i],H_sum[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+'] Qi='+str(inputarray[i][4])+'SQ='+str(inputarray[i][3]),color='aqua',linestyle='solid')

plot(title='Hospitalizados Optimista',ylabel='Letalidad')










# --------------------------- #
#      Valores Globales       #
# --------------------------- #
plt.plot(sims[i][0].t,sims[i][0].S,label='Susceptible')
plt.plot(sims[i][0].t,E,label='Tot Exposed')
plt.plot(sims[i][0].t,I,label='Tot infected')
plt.plot(sims[i][0].t,sims[i][0].R,label='Recovererd')
plt.plot(sims[i][0].t,sims[i][0].B,label='Buried')
plt.plot(sims[i][0].t,sims[i][0].D,label='Dead')


plt.plot(sims[i][0].t,sims[i][0].E_as,label='Exposed_asyntomatic')
plt.plot(sims[i][0].t,sims[i][0].E_sy,label='Exposed_syntomatic')
plt.plot(sims[i][0].t,sims[i][0].I_as,label='Infected_asyntomatic')
plt.plot(sims[i][0].t,sims[i][0].I_mi,label='Infected_mild')

plt.plot(sims[i][0].t,sims[i][0].I_se,label='Infected_severd')
# plt.plot(test.t,test.I_cr,label='Infected_critical')
# plt.plot(sims[i][0].t[1:],V_D,label='V_D')
# plt.plot(sims[i][0].t[1:],Hcr_D,label='Hcr_D')
# plt.plot(sims[i][0].t[1:],Icr_D,label='Icr_D')

plt.plot(sims[i][0].t,T,label='Total')
plot()




















# 14 dias de cuarentena inicial
i=0
plt.plot(dates,CH_hoy[i],color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(dates,CV_hoy[i],color='lime',linestyle='dotted',linewidth = 3.0)
i=1
plt.plot(dates,CH_hoy[i],color='red',linestyle='solid',linewidth = 3.0)
plt.plot(dates,CV_hoy[i],color='lime',linestyle='solid',linewidth = 3.0)
i=2
plt.plot(dates,CH_hoy[i], color='red',linestyle='dotted',linewidth = 3.0)
plt.plot(dates,CV_hoy[i], color='lime',linestyle='dotted',linewidth = 3.0)
plot(title = 'Camas adicionales 14D',ylabel='Camas Adicionales',xlabel = 'Días desde el 15 de Mayo')

# 21 dias de cuarentena inicial
i=3
plt.plot(dates,CH_hoy[i], color='blue',linestyle='dotted',linewidth = 3.0)
plt.plot(dates,CV_hoy[i], color='lime',linestyle='dotted',linewidth = 3.0)
i=4
plt.plot(dates,CH_hoy[i], color='blue',linestyle='solid',linewidth = 3.0)
plt.plot(dates,CV_hoy[i], color='lime',linestyle='solid',linewidth = 3.0)
i=5
plt.plot(dates,CH_hoy[i], color='blue',linestyle='dotted',linewidth = 3.0)
plt.plot(dates,CV_hoy[i], color='lime',linestyle='dotted',linewidth = 3.0)
plot(title = 'Camas adicionales 21D',ylabel='Camas Adicionales',xlabel = 'Días desde el 15 de Mayo')