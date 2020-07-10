#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEIRHVDB Class Initialization
"""
import class_SEIRHUVD as SD
import multiprocessing
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy import signal
import pandas as pd
from numpy import linalg as LA
import datetime


# Reporte Piñera
#  Importar datos Reales
infectadosreales = pd.read_csv('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/Reporte Piñera 28-05/infectadosacumulados.csv')
IacRM = infectadosreales.iloc[6] 
infectadosactivosreales = pd.read_csv('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/Reporte Piñera 28-05/infectadosactivos.csv')
IactivosRM=infectadosactivosreales.iloc[140][6:] 

#Importar datos simulacion SEIR
simcaso1 = pd.read_csv('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/Reporte Piñera 28-05/Sim-caso1-400-I.csv')   
simcaso2 = pd.read_csv('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/Reporte Piñera 28-05/Sim-caso2-400-I.csv')   
simcaso3 = pd.read_csv('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/Reporte Piñera 28-05/Sim-caso3-400-I.csv')   

totalcaso1 = [sum(simcaso1.iloc[i])/1000 for i in range(simcaso1.shape[0])]
totalcaso2 = [sum(simcaso2.iloc[i])/1000 for i in range(simcaso1.shape[0])]
totalcaso3 = [sum(simcaso3.iloc[i])/1000 for i in range(simcaso1.shape[0])]

tcaso1 = range(simcaso1.shape[0])    
tcaso2 = range(simcaso2.shape[0])    
tcaso3 = range(simcaso3.shape[0])    

peaktotalcaso1 = np.where(np.array(totalcaso1) == max(totalcaso1))  
peaktotalcaso2 = np.where(np.array(totalcaso2) == max(totalcaso2))  
peaktotalcaso3 = np.where(np.array(totalcaso3) == max(totalcaso3))  

peaktotalcaso2 = '2020-06-18'
peaktotalcaso3 = '2020-06-18'

plt.plot(tcaso1,totalcaso1, color = 'black',linewidth = 3.0) 
plt.plot(tcaso2,totalcaso2, color = 'C0',linewidth = 3.0)
plt.plot(tcaso3,totalcaso3, color = 'C3',linewidth = 3.0 )


# Vector temporal de contagiados activos:
initdate = datetime.datetime(2020,3,17)
t_Iactivos = []
for i in range(len(IactivosRM)):
    aux = datetime.datetime.strptime(IactivosRM.index[i], '%Y-%m-%d')
    t_Iactivos.append((aux-initdate).days)



#plot()
# Realista
Iact15M = IactivosRM['2020-05-15']  
i = 4 # movilidad 70%
I1 = (sims[i][0].I_cr + sims[i][0].I_mi + sims[i][0].I_se)*0.8 +Iact15M-((sims[i][0].I_cr[0] + sims[i][0].I_mi[0] + sims[i][0].I_se[0])*0.8)
#I1 = (sims[i][0].I_cr + sims[i][0].I_mi + sims[i][0].I_se)*0.8 +Iac15M-(sims[i][0].I_cr[0] + sims[i][0].I_mi[0] + sims[i][0].I_se[0])
plt.plot(t[i]+59, I1/1000,color='C1',linestyle='solid',linewidth = 3.0)
# Optimista
i = 12 # movilidad 40%
I2 = (sims[i][0].I_cr + sims[i][0].I_mi + sims[i][0].I_se)*0.9 +Iact15M-((sims[i][0].I_cr[0] + sims[i][0].I_mi[0] + sims[i][0].I_se[0])*0.9)
plt.plot(t[i]+59, I2/1000,color='grey',linestyle='solid',linewidth = 3.0)

# Peaks
#SEIR
initdateseir = 1
peaktotalcaso1 = np.where(np.array(totalcaso1) == max(totalcaso1))  
peaktotalcaso2 = np.where(np.array(totalcaso2) == max(totalcaso2))  
peaktotalcaso3 = np.where(np.array(totalcaso3) == max(totalcaso3))  

#SEIRHVD
initdateseirhvd = 1
peakrealista = np.where(I1==max(I1))[0][0] 
peakoptimista = np.where(I2==max(I2))[0][0] 

# Datos reales desde el 17 de Marzo
#IacRM_plot = IactivosRM
plt.scatter(t_Iactivos,IactivosRM/1000)

plt.title('Curva Epidemica')
plt.xlabel('Dias desde el 17 de Marzo')
plt.ylabel('Infectados')
plt.xlim(0,100)
plt.ylim(0,250)
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/Reporte Piñera 28-05/CurvaEpidemica100D.png', dpi=300)
plot(ylabel='Infectados Activos',xlabel='Días desde el 17 de Marzo')



# ---------------------------- #
# Datos vs simulacion
# ---------------------------- #

D15M = 59

i = 1 # movilidad 70%
I1 = I[i]#(sims[i][0].I_cr + sims[i][0].I_mi + sims[i][0].I_se)*0.8

tsim_2 = range(15,45)

datestxt = data['Fechas'][32:]
dates = list(range(15,15+len(datestxt)))
Irac_hoy = Irac[32:]

datessim = t[i][:idx[i][30]]
Iac_hoy = Iac[i][:idx[i][30]]
err = LA.norm(Irac_hoy-Iac_hoy[0:len(Irac_hoy)])/LA.norm(Irac_hoy)

plt.plot(range(15,45),totalcaso1[D15M:(D15M+30)], color = 'black',linewidth = 3.0) 
plt.plot(range(15,45),totalcaso2[D15M:(D15M+30)], color = 'C0',linewidth = 3.0) 
plt.plot(range(15,45),totalcaso3[D15M:(D15M+30)], color = 'C3',linewidth = 3.0) 


#plt.plot(tcaso2,totalcaso2, color = 'C2')
#plt.plot(tcaso3,totalcaso3, color = 'C3' )

plt.plot(datessim+15,Iac_hoy, color = 'C1',linewidth = 3.0)
i = 9
datessim = t[i][:idx[i][30]]
Iac_hoy = Iac[i][:idx[i][30]]
plt.plot(datessim+15,Iac_hoy, color = 'grey',linewidth = 3.0)
plt.scatter(dates,Irac_hoy)
#plt.plot([], [], ' ', label='beta: '+str(beta))
#plt.plot([], [], ' ', label='mu: '+str(mu))
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/Reporte Piñera 28-05/CEpidemicaDatosVsSimulacion.png', dpi=300)
plot(title='Infectados Acumulados')




# --------------------------- #
#   Letalidad a 500 dias      #
# --------------------------- #
# *Grafico 1

i=4
plt.plot(t[i],100*B[i]/Iac[i],color='blue',linestyle='solid',linewidth = 3.0)

i=6
plt.plot(t[i],100*B[i]/Iac[i],color='lime',linestyle='solid',linewidth = 3.0)
i=7
plt.plot(t[i],100*B[i]/Iac[i],color='black',linestyle='solid',linewidth = 3.0)

plt.ylim((0, 15))
plt.title('Letalidad')
plt.xlabel('Dias desde el 15 de Mayo')
plt.ylabel('Letalidad(%)')
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/Reporte Piñera 28-05/Letalidad500DRealista.png', dpi=300)
plot(ylabel='Letalidad (%)',xlabel='dias', title = 'Letalidad' )

# Optimista
i=12
plt.plot(t[i],100*B[i]/Iac[i],color='blue',linestyle='solid',linewidth = 3.0)

i=14
plt.plot(t[i],100*B[i]/Iac[i],color='lime',linestyle='solid',linewidth = 3.0)
i=15
plt.plot(t[i],100*B[i]/Iac[i],color='black',linestyle='solid',linewidth = 3.0)

plt.ylim((0, 15))
plt.title('Letalidad')
plt.xlabel('Dias desde el 15 de Mayo')
plt.ylabel('Letalidad(%)')
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/Reporte Piñera 28-05/etalidad500DOptimista.png', dpi=300)
plot(ylabel='Letalidad (%)', xlabel = 'Días desde el 15 de Mayo', title = 'Letalidad' )



# --------------------------- #
#    Letalidad a 30 dias      #
# --------------------------- #
# Grafico 2
# 70 y 30
time = 30
endD = [np.searchsorted(t[i],range(time)) for i in range(len(input))]

i=4
plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='blue',linestyle='solid',linewidth = 3.0)
i=6
plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='lime',linestyle='solid',linewidth = 3.0)
i=7
plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='black',linestyle='solid',linewidth = 3.0)

plt.ylim((0, 8))
plt.title('Letalidad')
plt.xlabel('Dias desde el 15 de Mayo')
plt.ylabel('Letalidad(%)')
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/Reporte Piñera 28-05/Letalidad30DRealista.png', dpi=300)
plot(ylabel='Letalidad (%)')

i=12
plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='blue',linestyle='solid',linewidth = 3.0)
i=14
plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='lime',linestyle='solid',linewidth = 3.0)
i=15
plt.plot(t[i][:endD[i][-1]],100*B[i][:endD[i][-1]]/Iac[i][:endD[i][-1]], color='black',linestyle='solid',linewidth = 3.0)
plt.ylim((0, 8))

plt.title('Letalidad')
plt.xlabel('Dias desde el 15 de Mayo')
plt.ylabel('Letalidad(%)')
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/Reporte Piñera 28-05/Letalidad30DOptimista.png', dpi=300)
plot(ylabel='Letalidad (%)')




# --------------------------- #
#   Curva Epidemica 500D      #
# --------------------------- #
# Grafico 3:
# Infectados Activos a 500 dias

I = [sims[i][0].I_as+sims[i][0].I_cr + sims[i][0].I_mi + sims[i][0].I_se + sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V for i in range(len(input))] 
I_act = [sims[i][0].I_mi + sims[i][0].I_cr + sims[i][0].I_se + sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V for i in range(len(input))] 

# Buscar maximo
maxI = max([max(I_act[i]) for i in range(len(input))])

# Realista
#i=0
#plt.plot(t[i], I_act[i]/maxI,color='red',linestyle='dotted',linewidth = 3.0)
#i=1
#plt.plot(t[i], I_act[i]/maxI,color='red',linestyle='solid',linewidth = 3.0)
#i=2
#plt.plot(t[i], I_act[i]/maxI,color='red',linestyle='dotted',linewidth = 3.0)
#i=3
#plt.plot(t[i], I_act[i]/maxI,color='blue',linestyle='dotted',linewidth = 3.0)
i=4
plt.plot(t[i], I_act[i]/maxI,color='blue',linestyle='solid',linewidth = 3.0)
#i=5
#plt.plot(t[i], I_act[i]/maxI,color='blue',linestyle='dotted',linewidth = 3.0)
i=6
plt.plot(t[i], I_act[i]/maxI,color='lime',linestyle='solid',linewidth = 3.0)
i=7
plt.plot(t[i], I_act[i]/maxI,color='black',linestyle='solid',linewidth = 3.0)

plt.title('Curva Epidemica')
plt.xlabel('Dias desde el 15 de Mayo')
plt.ylabel('Infectados')
plt.ylim((0, 1.1))
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/Reporte Piñera 28-05/CurvaEpidemicaRealista.png', dpi=300)
plot(ylabel='Infectados Activos')

# Optimista
#i=8
#plt.plot(t[i], I_act[i]/maxI,color='red',linestyle='dashed',linewidth = 3.0)
#i=9
#plt.plot(t[i], I_act[i]/maxI,color='red',linestyle='solid',linewidth = 3.0)
#i=10
#plt.plot(t[i], I_act[i]/maxI,color='red',linestyle='dotted',linewidth = 3.0)
#i=11
#plt.plot(t[i], I_act[i]/maxI,color='blue',linestyle='dashed',linewidth = 3.0)
i=12
plt.plot(t[i], I_act[i]/maxI,color='blue',linestyle='solid',linewidth = 3.0)
#i=13
#plt.plot(t[i], I_act[i]/maxI,color='blue',linestyle='dotted',linewidth = 3.0)
i=14
plt.plot(t[i], I_act[i]/maxI,color='lime',linestyle='solid',linewidth = 3.0)
i=15
plt.plot(t[i], I_act[i]/maxI,color='black',linestyle='solid',linewidth = 3.0)

plt.title('Curva Epidemica')
plt.xlabel('Dias desde el 15 de Mayo')
plt.ylabel('Infectados')
plt.ylim((0, 1.1))
plt.savefig('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/Reporte Piñera 28-05/CurvaEpidemicaOptimista.png', dpi=300)
plot(ylabel='Infectados Activos')