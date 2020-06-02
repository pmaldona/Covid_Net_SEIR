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


#------------------------------------------------- #
#               Define Scenarios                   #
#------------------------------------------------- #
inputarray=np.array([
        [500,0.85,0.6,14,14,500,0],
        [500,0.85,0.7,14,14,500,0],
        [500,0.85,0.8,14,14,500,0],
        [500,0.85,0.6,14,21,500,0],
        [500,0.85,0.7,14,21,500,0],
        [500,0.85,0.8,14,21,500,0],
        [500,0.85,0.7,14,60,500,1],
        [500,0.85,0.7,21,60,500,1],
        [500,0.55,0.2,14,14,500,0],
        [500,0.55,0.3,14,14,500,0],
        [500,0.55,0.4,14,14,500,0],
        [500,0.55,0.2,14,21,500,0],
        [500,0.55,0.3,14,21,500,0],
        [500,0.55,0.4,14,21,500,0],        
        [500,0.55,0.3,14,60,500,1],
        [500,0.55,0.3,21,60,500,1]])  

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



# -------------- #
#    Simulate    #
# -------------- #
beta = 0.19
mu = 2.6
sims = SD2.simSEIRHVD(beta = beta, mu = mu, inputarray= inputarray)
sims = SEIRHVD.simulate(beta,mu)



#------------------------------------------------- #
#               Estudio Resultados                 #
#------------------------------------------------- #

# -------------------------- #
#        Importar Data       #
# -------------------------- #
# 15 de Mayo igual dato 32
data = pd.read_excel('/home/samuel/Downloads/Data_SEIRHUVD.xlsx') 
Ir = data['Infectados Activos (extrapolados)'] 
Irac = data['Infectados Acumulados'] 
Br = data['Fallecidos']
Hr_bed = data['Camas Ocupadas']
Hr_vent = data['VMI Ocupados']


#Iac15M = 39542 # infectados acumulados a la fecha 15 de mayo 39542  -  
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
tr = range(tsim)
idx = [np.searchsorted(t[i],tr) for i in range(len(inputarray))] 


# CAMAS
H_crin=[sims[i][0].H_cr for i in range(len(inputarray))] 
H_in=[sims[i][0].H_in for i in range(len(inputarray))] 
H_out=[sims[i][0].H_out for i in range(len(inputarray))] 

# Disponibilizar fechas peak


# ------------------ #
#     Infectados     #
# ------------------ #

# Dias desde el 15 de mayo hasta el final de los datos
datestxt = data['Fechas'][32:]
dates = list(range(15,15+len(datestxt)))
Irac_hoy = Irac[32:]
d15M = len(Irac[32:])

I_as_hoy = I_as[i][idx[i][0:d15M]]
I_mi_hoy = I_mi[i][idx[i][0:d15M]]
I_se_hoy = I_se[i][idx[i][0:d15M]]
I_cr_hoy = I_cr[i][idx[i][0:d15M]]


plt.plot(dates,I_as_hoy,label='Acumulados Asintomaticos')
plt.plot(dates,I_mi_hoy,label='Acumulados Mild')
plt.plot(dates,I_se_hoy,label='Acumulados Severos')
plt.plot(dates,I_cr_hoy,label='Acumulados Criticos')
plt.plot([], [], ' ', label='beta: '+str(beta))
plot(title='Infectados Simulados')


# ----------------------- #
#      Saturacion         #
# ----------------------- #
H_crin=[sims[i][0].H_cr for i in range(len(inputarray))] 
H_in=[sims[i][0].H_in for i in range(len(inputarray))] 
H_out=[sims[i][0].H_out for i in range(len(inputarray))] 
H_in_hoy = H_in[i][idx[i][0:d15M]] 
H_out_hoy = H_out[i][idx[i][0:d15M]] 
H_crin_hoy = H_crin[i][idx[i][0:d15M]]

datestxt = data['Fechas'][32:]
dates = list(range(15,15+len(datestxt)))
Br_hoy = Br[32:]
d15M = len(Ir[32:])
H_sat_hoy = [H_sat(H_in_hoy[i],H_out_hoy[i],H_crin_hoy[i],idx[0][i])for i in range(d15M)] #+ Ir_hoy[32] - I[0]

plt.plot(dates,H_sat_hoy,label='Saturacion')
plot(title='Saturacion')



# -------------------------------------- #
#     Acumulados simulados vs reales     #
# -------------------------------------- #
# Dias desde el 15 de mayo hasta el final de los datos
datestxt = data['Fechas'][32:]
dates = list(range(15,15+len(datestxt)))
datessim = list(range(15,15+len(datestxt)+14))

Irac_hoy = Irac[32:]
d15M = len(Irac[32:])
Iac_hoy = Iac[i][idx[i][0:(d15M+14)]]
err = LA.norm(Irac_hoy-Iac_hoy[0:len(Irac_hoy)])/LA.norm(Irac_hoy)

plt.plot(datessim,Iac_hoy,label='Acumulados Simulados')
plt.scatter(dates,Irac_hoy,label='Acumulados Reales')
plt.plot([], [], ' ', label='beta: '+str(beta))
plt.plot([], [], ' ', label='mu: '+str(mu))
plt.plot([], [], ' ', label='err: '+str(round(100*err,2))+'%')
plot(title='Infectados Acumulados')

# ----------------------------------------- #
#      Fallecidos simulados vs reales       #
# ----------------------------------------- #
#i=1
datestxt = data['Fechas'][32:]
dates = list(range(15,15+len(datestxt)))
Br_hoy = Br[32:]
d15M = len(Br[32:])
datessim = list(range(15,15+len(datestxt)+14))
B_hoy = B[i][idx[i][0:(d15M+14)]] #+ Ir_hoy[32] - I[0]
err = LA.norm(Br_hoy-B_hoy[0:len(Br_hoy)])/LA.norm(Br_hoy)

plt.plot(datessim,B_hoy,label='Fallecidos Simulados')
#plt.plot(dates,Br_hoy,label='Fallecidos Reales')
plt.scatter(dates,Br_hoy,label='Fallecidos Reales',color = 'red')
plt.plot([], [], ' ', label='beta: '+str(beta))
plt.plot([], [], ' ', label='mu: '+str(mu))
plt.plot([], [], ' ', label='err: '+str(round(100*err,2))+'%')
plot(title='Fallecidos')

# ----------------------------------------- #
#       Camas simulados vs reales           #
# ----------------------------------------- #
#H_sum=[sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V for i in range(len(inputarray))] 
H_crin=[sims[i][0].H_cr for i in range(len(inputarray))] 
H_in=[sims[i][0].H_in for i in range(len(inputarray))] 
H_out=[sims[i][0].H_out for i in range(len(inputarray))] 
#H_vent=[sims[i][0].V for i in range(len(inputarray))] 

#Hr_bed = data['Camas Ocupadas']
#Hr_vent = data['VMI Ocupados']


datestxt = data['Fechas'][32:38]
dates = list(range(15,15+len(datestxt)))
datessim = list(range(15,15+len(datestxt)+14))

Hr_bed_hoy = Hr_bed[32:32+len(datestxt)]
Hr_vent_hoy = Hr_vent[32:32+len(datestxt)]
d15M = len(Ir[32:32+len(datestxt)])

H_bed_hoy = H_bed[i][idx[i][0:(d15M+14)]] 
H_in_hoy = H_in[i][idx[i][0:(d15M+14)]] 
H_out_hoy = H_out[i][idx[i][0:(d15M+14)]] 
H_crin_hoy = H_crin[i][idx[i][0:(d15M+14)]] 
H_vent_hoy = H_vent[i][idx[i][0:(d15M+14)]] 

err_bed = LA.norm(Hr_bed_hoy-H_bed_hoy[0:len(Hr_bed_hoy)])/LA.norm(Hr_bed_hoy)
err_vent = LA.norm(Hr_vent_hoy-H_vent_hoy[0:len(Hr_vent_hoy)])/LA.norm(Hr_vent_hoy)

#plt.scatter(

plt.plot(datessim,H_in_hoy,label='Hin')
plt.plot(datessim,H_out_hoy,label='Hout')
plt.plot(datessim,H_crin_hoy,label='Hcri')
plt.plot(datessim,H_bed_hoy,label='Camas Simulados')
plt.plot(datessim,H_vent_hoy,label='Ventiladores Simulados')

#plt.plot(dates,Hr_bed_hoy,label='Camas Reales')
plt.scatter(dates,Hr_bed_hoy,label='Camas Reales')
#plt.plot(dates,Hr_vent_hoy,label='Ventiladores Reales')
plt.scatter(dates,Hr_vent_hoy,label='Ventiladores Reales')
plt.plot([], [], ' ', label='beta: '+str(beta))
plt.plot([], [], ' ', label='mu: '+str(mu))
plt.plot([], [], ' ', label='err_bed: '+str(round(100*err_bed,2))+'%')
plt.plot([], [], ' ', label='err_vent: '+str(round(100*err_vent,2))+'%')
plot(title='Camas y Ventiladores')

# --------------------------- #
#      Camas requeridas       #
# --------------------------- #
tsim = 500
tr = range(tsim)
idx = [np.searchsorted(t[i],tr) for i in range(len(inputarray))] 
endD = idx[i][-1]
CH = [sims[i][0].CH for i in range(len(inputarray))]
CV = [sims[i][0].CV for i in range(len(inputarray))]
ACH = [sims[i][0].ACH for i in range(len(inputarray))]
ACV = [sims[i][0].ACV for i in range(len(inputarray))]

plt.plot(t[i][:endD],CH[i][:endD],label='Intermedio/Intensivo')
plt.plot(t[i][:endD],CV[i][:endD],label='VMI')
#plt.plot(t[i],D[i],label='Muertos Diarios')

#plt.plot(t[i],ACH[i],label='Camas Acumuladas Hospitalarias')
#plt.plot(t[i],ACV[i],label='Camas Acumuladas Ventiladores')
#plt.plot(t[i],ACV[i]+ACH[i],label='Suma Camas Acumuladas Ventiladores')
#plt.plot(t[i],B[i],label='Muertos Acumulados')

plt.plot([], [], ' ', label='beta: '+str(beta))
plt.plot([], [], ' ', label='mu: '+str(mu))
plot(title='Camas Requeridas')




# -------------------------------------- #
#      Activos simulados vs reales       #
# -------------------------------------- #
i = 1
Pdiag = 0.4
I_act = [Pdiag*sims[i][0].I_mi + sims[i][0].I_cr + sims[i][0].I_se + sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V for i in range(len(inputarray))] 
datestxt = data['Fechas'][32:]
dates = list(range(15,15+len(datestxt)))
datessim = list(range(15,15+len(datestxt)+14))

Ir_hoy = Ir[32:40]
d15M = len(Ir[32:40])
I_hoy = I_act[i][idx[i][0:(d15M+14)]] + Ir_hoy[32] - I_act[i][0]
err = LA.norm(Ir_hoy-I_hoy[0:len(Ir_hoy)])/LA.norm(Ir_hoy)

plt.plot(datessim,I_hoy,label='Activos Simulados')
plt.scatter(dates,Ir_hoy,label='Activos Reales')
plt.plot([], [], ' ', label='beta: '+str(beta))
plt.plot([], [], ' ', label='mu: '+str(mu))
plt.plot([], [], ' ', label='err: '+str(round(100*err,2))+'%')
plot(title='Infectados Activos - err')


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