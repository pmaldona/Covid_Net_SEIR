#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import class_SEIRHUVD2 as SD2
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

"""
Analisis SEIRHVD


N=1.000.000
Ej numerico 1
alpha=0.3,0.5,0.8, Vtot constante Htot constante
Ej. numerico 2.
alpha dyn=[0.3, 0.8] Estatica 0.8, 14D, 7D, 21D (comienen 30 dias, movilidad base 0.8), Vtot constante, Htot constante, probar valores para ver colapso
Ej. numerico 3
alpha dyn=[0.3, 0.8] 14D, 7D, 21D (comienen 30 dias, movilidad base 0.8), Vtot variable, Htot variable, probar valores para ver colapso
Aplicacion
Caso Chile: 15 mayo cuarentena toda region, curvas conocidas. Normalizado.

Infectados activos por tipo
Letalidad

Poblacion 1.000.000
Infectados iniciales: 100 (0.1%)

Graficar:
Infectados totales activos
Infectados totales acumulados
Necesidad total de Camas con una rallita que muestre cuantas tenemos
Muertos
Letalidad

Letalidad hospitalaria M/(Icr + Ise) vs Numero de camas, para distintas movilidades

Definir rango de numero de camas proporcional a la población 

Letalidad hospitalaria M/(Icr + Ise) vs Numero de ventiladores, para distintas movilidades

Análisis de sensibilidad camas vs ventiladores  (cuanta gente salva cada uno)

Máximo de letalidad hospitalaria para numero de camas vs movilidad

Letalidad hospitalaria vs tiempo para distintos incrementos de camas. Definir proporción de crecimiento entre ventiladores y camas
"""


#----------------------------------------------- #
#               Ej Numerico 1                    #
#----------------------------------------------- #
# alpha=0.3,0.5,0.8, Vtot constante Htot constante
# Sin Cuarentena


tsim = 500
#mov = np.linspace(0.2,0.35,15)
mov = [0.2,0.5,0.8]
fqt = 0
iqt = 0
qp = 0
movfunct = 'once'

# Infe

# Create cases
case = []
for i in range(len(mov)):
    case.append(SD2.SEIRHUDV(tsim,mov[i],mov[i],qp,iqt,fqt,movfunct))

# Set Params:
beta = 0.19
mu = 2.6

for i in range(len(mov)):
    case[i].beta = beta
    case[i].mu = mu


# Set Initial Values:
totpop = 1000000
I_act0 = 100
I_new0 = 20

for i in range(len(mov)):
    case[i].H_incr = 0
    case[i].V_incr = 0
    case[i].D = 0
    case[i].B = 0
    case[i].V = 0
    case[i].R = 0
    case[i].H0 = 0
    case[i].H_cr = 0
    case[i].pop = totpop
    case[i].cIEx0 = I_act0/1.5
    case[i].cI0S = I_act0/2.5
    case[i].Id0 = I_new0 
    case[i].setrelationalvalues()


# Simular:
sims = []
for i in range(len(mov)):
    case[i].integr_sci(0,tsim,0.1,False)

# -------------------- #
#       Analisis       #
# -------------------- #

# Infectados: 
Itot = [case[i].I_as+case[i].I_mi+case[i].I_se+case[i].I_cr for i in range(len(mov))]
color = ['red','blue','green']
for i in range(len(mov)):
    plt.plot(case[i].t,Itot[i],label='Infectados Activos Totales mov = '+str(mov[i]),linestyle = 'solid')
    #plt.plot(case[i].t,case[i].I_as,label='Infectados asintomaticos',color = color[i],linestyle = 'dashed')
    #plt.plot(case[i].t,case[i].I_mi,label='Infectados mild',color = color[i],linestyle = 'dotted')
    #plt.plot(case[i].t,case[i].I_se,label='Infectados severos',color = color[i],linestyle = 'dashdot')
    #plt.plot(case[i].t,case[i].I_cr,label='Infectados criticos',color = color[i],linestyle = (0, (5,10)))
plot(title='Infectados Simulados')


i=0
plt.plot(case[i].t,Itot[i],label='Infectados Activos Totales',color = color[i],linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_as,label='Infectados asintomaticos',color = color[i],linestyle = 'dashed')
plt.plot(case[i].t,case[i].I_mi,label='Infectados mild',color = color[i],linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_se,label='Infectados severos',color = color[i],linestyle = 'dashdot')
plt.plot(case[i].t,case[i].I_cr,label='Infectados criticos',color = color[i],linestyle = (0, (5,10)))
plot(title='Infectados Simulados, mov = 0.3')

i=1
plt.plot(case[i].t,Itot[i],label='Infectados Activos Totales',color = color[i],linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_as,label='Infectados asintomaticos',color = color[i],linestyle = 'dashed')
plt.plot(case[i].t,case[i].I_mi,label='Infectados mild',color = color[i],linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_se,label='Infectados severos',color = color[i],linestyle = 'dashdot')
plt.plot(case[i].t,case[i].I_cr,label='Infectados criticos',color = color[i],linestyle = (0, (5,10)))
plot(title='Infectados Simulados, mov = 0.5')

i=2
plt.plot(case[i].t,Itot[i],label='Infectados Activos Totales',color = color[i],linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_as,label='Infectados asintomaticos',color = color[i],linestyle = 'dashed')
plt.plot(case[i].t,case[i].I_mi,label='Infectados mild',color = color[i],linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_se,label='Infectados severos',color = color[i],linestyle = 'dashdot')
plt.plot(case[i].t,case[i].I_cr,label='Infectados criticos',color = color[i],linestyle = (0, (5,10)))
plot(title='Infectados Simulados, mov = 0.8')


# Infectados Activos totales por movilidad Normalizados
for i in range(len(mov)):
    plt.plot(case[i].t,Itot[i]/totpop,label='Infectados activos mov = '+str(mov[i]),color = color[i],linestyle = 'solid')
plot(title='Infectados Activos totales por movilidad')


# Infectados Acumulados Normalizados
for i in range(len(mov)):
    plt.plot(case[i].t,case[i].I/totpop,label='Infectados acumulados mov = '+str(mov[i]),color = color[i],linestyle = 'solid')
plot(title='Infectados Acumulados')

# Letalidad 
for i in range(len(mov)):
    plt.plot(case[i].t,100*case[i].B/case[i].I,label='Letalidad % mov = '+str(mov[i]),color = color[i],linestyle = 'solid')
plot(title='Letalidad')


# Mortalidad Instantanea Normalizada
for i in range(len(mov)):
    plt.plot(case[i].t,case[i].D/totpop,label='Infectados Activos Totales',color = color[i],linestyle = 'solid')
plot(title='Infectados Acumulados)

# Mortalidad acumulada Normalizada
for i in range(len(mov)):
    plt.plot(case[i].t,case[i].B/totpop,label='Infectados Activos Totales',color = color[i],linestyle = 'solid')
plot(title='Infectados Acumulados)


#----------------------------------------------- #
#               Ej Numerico 2                    #
#----------------------------------------------- #
# alpha dyn=[0.3, 0.8] Estatica 0.8, 14D, 7D, 21D 
# (comienen 30 dias, movilidad base 0.8) 
# Vtot constante, Htot constante, probar valores para ver colapso

tsim = 500
max_mov = 0.8
rem_mov = 0.3
fqt = 500
iqt = 0
qp = [0,7,14,21]
movfunct = 'square2'

# Infe

# Create cases
case = []
for i in range(len(qp)):
    case.append(SD2.SEIRHUDV(tsim,max_mov,rem_mov,qp[i],iqt,fqt,movfunct))

# Set Params:
beta = 0.19
mu = 2.6

for i in range(len(qp)):
    case[i].beta = beta
    case[i].mu = mu


# Set Initial Values:
totpop = 1000000
I_act0 = 100
I_new0 = 20

for i in range(len(qp)):
    case[i].H_incr = 0
    case[i].V_incr = 0
    case[i].D = 0
    case[i].B = 0
    case[i].V = 0
    case[i].R = 0
    case[i].H0 = 0
    case[i].H_cr = 0
    case[i].pop = totpop
    case[i].cIEx0 = I_act0/1.5
    case[i].cI0S = I_act0/2.5
    case[i].Id0 = I_new0
    case[i].H0=200
    case[i].H_cr=10.0 
    case[i].setrelationalvalues()


# Simular:
sims = []
for i in range(len(qp)):
    case[i].integr_sci(0,tsim,0.1,False)

# -------------------- #
#       Analisis       #
# -------------------- #
Itot = [case[i].I_as+case[i].I_mi+case[i].I_se+case[i].I_cr for i in range(len(qp))]
color = ['red','blue','green','black']

# Cuarentena
for i in range(len(qp)):
    alphadata = [case[i].alpha(j) for j in case[i].t] 
    plt.plot(case[i].t,alphadata,label='Periodo cuarentena '+str(qp[i])+' dias',color = color[i],linestyle = 'dashed')
plt.xlim(0,100)
plt.ylim(0,1)
plot(title='Movilidad')

# Para comparar por tipo de Infectados por movilidad: 
Itot = [case[i].I_as+case[i].I_mi+case[i].I_se+case[i].I_cr for i in range(len(qp))]
color = ['red','blue','green','black']

for i in range(len(qp)):
    plt.plot(case[i].t,Itot[i]/totpop,label='Infectados Activos Totales '+str(qp[i])+' dias',color = color[i],linestyle = 'dotted')
    #plt.plot(case[i].t,case[i].I_as/totpop,label='Asintomaticos Periodo cuarentena '+str(qp[i])+' dias',color = color[i],linestyle = 'dashed')
    #plt.plot(case[i].t,case[i].I_mi/totpop,label='Mild',color = color[i],linestyle = 'dotted')
    #plt.plot(case[i].t,case[i].I_se/totpop,label='Severos Periodo cuarentena '+str(qp[i])+' dias',color = color[i],linestyle = 'dashdot')
    #plt.plot(case[i].t,case[i].I_cr/totpop,label='Criticos Periodo cuarentena '+str(qp[i])+' dias',color = color[i],linestyle = (0, (5,10)))
plot(title='Infectados Activos Normalizados')

# Infectados Disgregados por movilidad
i=0
plt.plot(case[i].t,Itot[i],label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_as,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dashed')
plt.plot(case[i].t,case[i].I_mi,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_se,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dashdot')
plt.plot(case[i].t,case[i].I_cr,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = (0, (5,10)))
plot(title='Infectados Simulados, QP = 0')

i=1
plt.plot(case[i].t,Itot[i],label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_as,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dashed')
plt.plot(case[i].t,case[i].I_mi,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_se,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dashdot')
plt.plot(case[i].t,case[i].I_cr,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = (0, (5,10)))
plot(title='Infectados Simulados, QP = 7')

i=2
plt.plot(case[i].t,Itot[i],label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_as,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dashed')
plt.plot(case[i].t,case[i].I_mi,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_se,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dashdot')
plt.plot(case[i].t,case[i].I_cr,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = (0, (5,10)))
plot(title='Infectados Simulados, QP = 14')

i=3
plt.plot(case[i].t,Itot[i],label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_as,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dashed')
plt.plot(case[i].t,case[i].I_mi,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_se,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dashdot')
plt.plot(case[i].t,case[i].I_cr,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = (0, (5,10)))
plot(title='Infectados Simulados, QP = 21')



# Infectados Activos totales por movilidad Normalizados
for i in range(len(qp)):
    plt.plot(case[i].t,Itot[i]/totpop,label='Infectados activos qp = '+str(qp[i]),color = color[i],linestyle = 'solid')
plot(title='Infectados Activos totales por movilidad')


# Infectados Acumulados Normalizados
for i in range(len(qp)):
    plt.plot(case[i].t,case[i].I/totpop,label='Infectados acumulados qp = '+str(qp[i]),color = color[i],linestyle = 'solid')
plot(title='Infectados Acumulados')

# Letalidad 
for i in range(len(qp)):
    plt.plot(case[i].t,100*case[i].B/case[i].I,label='Letalidad % qp = '+str(qp[i]),color = color[i],linestyle = 'solid')
plot(title='Letalidad')


# Mortalidad Instantanea Normalizada
for i in range(len(qp)):
    plt.plot(case[i].t,case[i].D/totpop,label='Mortalidad Instantanea qp = '+str(qp[i]),color = color[i],linestyle = 'solid')
plot(title='Mortalidad Instantanea Normalizada')

# Mortalidad acumulada Normalizada
for i in range(len(qp)):
    plt.plot(case[i].t,case[i].B/totpop,label='Mortalidad Acumulada qp = '+str(qp[i]),color = color[i],linestyle = 'solid')
plot(title='Mortalidad Acumulada Normalizada')



#----------------------------------------------- #
#               Ej Numerico 3                    #
#----------------------------------------------- #
# alpha dyn=[0.3, 0.8] 14D, 7D, 21D (comienen 30 dias, movilidad base 0.8)
# Vtot variable
# Htot variable 
# probar valores para ver colapso
tsim = 500
max_mov = 0.8
rem_mov = 0.3
fqt = 500
iqt = 0
qp = [0,7,14,21]
movfunct = 'square2'

# Infe

# Create cases
case = []
for i in range(len(qp)):
    case.append(SD2.SEIRHUDV(tsim,max_mov,rem_mov,qp[i],iqt,fqt,movfunct))

# Set Params:
beta = 0.19
mu = 2.6

for i in range(len(qp)):
    case[i].beta = beta
    case[i].mu = mu


# Set Initial Values:
totpop = 1000000
I_act0 = 100
I_new0 = 20

for i in range(len(qp)):
    case[i].H_incr = 23
    case[i].V_incr = 18
    case[i].D = 0
    case[i].B = 0
    case[i].V = 0
    case[i].R = 0
    case[i].H0 = 0
    case[i].H_cr = 0
    case[i].pop = totpop
    case[i].cIEx0 = I_act0/1.5
    case[i].cI0S = I_act0/2.5
    case[i].Id0 = I_new0 
    case[i].setrelationalvalues()


# Simular:
sims = []
for i in range(len(qp)):
    case[i].integr_sci(0,tsim,0.1,False)

# -------------------- #
#       Analisis       #
# -------------------- #
Itot = [case[i].I_as+case[i].I_mi+case[i].I_se+case[i].I_cr for i in range(len(qp))]
color = ['red','blue','green','yellow']

# Cuarentena
for i in range(len(qp)):
    alphadata = [case[i].alpha(j) for j in case[i].t] 
    plt.plot(case[i].t,alphadata,label='Periodo cuarentena '+str(qp[i])+' dias',color = color[i],linestyle = 'dashed')
plt.xlim(0,100)
plt.ylim(0,1)
plot(title='Movilidad')

# Para comparar por tipo de Infectados por movilidad: 
for i in range(len(qp)):
    plt.plot(case[i].t,Itot[i],label='Periodo cuarentena '+str(qp[i])+' dias',color = color[i],linestyle = 'dotted')
    #plt.plot(case[i].t,case[i].I_as,label='Asintomaticos Periodo cuarentena '+str(qp[i])+' dias',color = color[i],linestyle = 'dashed')
    #plt.plot(case[i].t,case[i].I_mi,label='Mild',color = color[i],linestyle = 'dotted')
    #plt.plot(case[i].t,case[i].I_se,label='Severos Periodo cuarentena '+str(qp[i])+' dias',color = color[i],linestyle = 'dashdot')
    #plt.plot(case[i].t,case[i].I_cr,label='Criticos Periodo cuarentena '+str(qp[i])+' dias',color = color[i],linestyle = (0, (5,10)))
plot(title='Infectados Activos')

for i in range(len(qp)):
    plt.plot(case[i].t,Itot[i]/totpop,label='Infectados Activos Totales',color = color[i],linestyle = 'dotted')
    #plt.plot(case[i].t,case[i].I_as/totpop,label='Asintomaticos Periodo cuarentena '+str(qp[i])+' dias',color = color[i],linestyle = 'dashed')
    #plt.plot(case[i].t,case[i].I_mi/totpop,label='Mild',color = color[i],linestyle = 'dotted')
    #plt.plot(case[i].t,case[i].I_se/totpop,label='Severos Periodo cuarentena '+str(qp[i])+' dias',color = color[i],linestyle = 'dashdot')
    #plt.plot(case[i].t,case[i].I_cr/totpop,label='Criticos Periodo cuarentena '+str(qp[i])+' dias',color = color[i],linestyle = (0, (5,10)))
plot(title='Infectados Activos Normalizados')

# Infectados Disgregados por movilidad
i=0
plt.plot(case[i].t,Itot[i],label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_as,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dashed')
plt.plot(case[i].t,case[i].I_mi,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_se,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dashdot')
plt.plot(case[i].t,case[i].I_cr,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = (0, (5,10)))
plot(title='Infectados Simulados, QP = 0')

i=1
plt.plot(case[i].t,Itot[i],label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_as,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dashed')
plt.plot(case[i].t,case[i].I_mi,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_se,label=label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dashdot')
plt.plot(case[i].t,case[i].I_cr,label=label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = (0, (5,10)))
plot(title='Infectados Simulados, QP = 7')

i=2
plt.plot(case[i].t,Itot[i],label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_as,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dashed')
plt.plot(case[i].t,case[i].I_mi,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_se,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dashdot')
plt.plot(case[i].t,case[i].I_cr,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = (0, (5,10)))
plot(title='Infectados Simulados, QP = 14')

i=3
plt.plot(case[i].t,Itot[i],label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_as,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dashed')
plt.plot(case[i].t,case[i].I_mi,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dotted')
plt.plot(case[i].t,case[i].I_se,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = 'dashdot')
plt.plot(case[i].t,case[i].I_cr,label='Periodo cuarentena '+str(qp[i])+' dias',linestyle = (0, (5,10)))
plot(title='Infectados Simulados, QP = 21')



# Infectados Activos totales por movilidad Normalizados
for i in range(len(qp)):
    plt.plot(case[i].t,Itot[i]/totpop,label='Infectados activos qp = '+str(qp[i]),color = color[i],linestyle = 'solid')
plot(title='Infectados Activos totales por movilidad')


# Infectados Acumulados Normalizados
for i in range(len(qp)):
    plt.plot(case[i].t,case[i].I/totpop,label='Infectados acumulados qp = '+str(qp[i]),color = color[i],linestyle = 'solid')
plot(title='Infectados Acumulados')

# Letalidad 
for i in range(len(qp)):
    plt.plot(case[i].t,100*case[i].B/case[i].I,label='Letalidad % qp = '+str(qp[i]),color = color[i],linestyle = 'solid')
plot(title='Letalidad')


# Mortalidad Instantanea Normalizada
for i in range(len(qp)):
    plt.plot(case[i].t,case[i].D/totpop,label='Mortalidad Instantanea qp = '+str(qp[i]),color = color[i],linestyle = 'solid')
plot(title='Mortalidad Instantanea Normalizada')

# Mortalidad acumulada Normalizada
for i in range(len(qp)):
    plt.plot(case[i].t,case[i].B/totpop,label='Mortalidad Acumulada qp = '+str(qp[i]),color = color[i],linestyle = 'solid')
plot(title='Mortalidad Acumulada Normalizada')


#----------------------------------------------- #
#                 Caso Chile                     #
#----------------------------------------------- #
# 15 mayo cuarentena toda region, curvas conocidas. Normalizado.