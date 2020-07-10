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

"""
To Do:
   - Permitir ejecución remota desde el cluster
   - Habilitar el uso del código para otras regiones
   - Simplificar la creación de cuarentenas hiperdinámicas
   - Facilitar la creación de escenarios
   - Creación de funciones para simplificar el uso del código
      - Ingreso de Parámetros Generales
      - Ingreso de parámetros del Modelo
      - Ingreso de escenarios
      - Extender importación de datos Sochimi a otras regiones y otras unidades territoriales según corresponda
    - Arreglar el ingreso del modelo de ingreso de capacidad total de camas para que sea una funcion
    - Resolver como se haría el ploteo para varios escenarios
    - Generar automaticamente nombres de columnas en las tablas
    - Incorporar 
    - Generar Vector de estilos
    - Agregar comparacion fallecidos diarios vs reales
    - Agregar opción de trabajar con comunas

    - Agregar gráfico casos nuevos por día
      
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




# Ajuste por Muertos minciencia (SEIRHVD Class 2)
# Parametros del modelo
beta = 0.117 #0.25#0.19 0.135
mu = 0.15 #2.6 0.6
ScaleFactor = 2.25 #4.8
SeroPrevFactor = 0.5 # Sero Prevalence Factor. Permite ajustar la cantidad de gente que entra en la dinamica
expinfection = 1 # Proporcion en la que contagian los expuestos



# Ajuste por Muertos minciencia para adelantar peak(SEIRHVD Class 2)
# Parametros del modelo
beta = 0.115 #0.25#0.19 0.135
mu = 0.15 #2.6 0.6
ScaleFactor = 4 #4.8
SeroPrevFactor = 0.3 # Sero Prevalence Factor. Permite ajustar la cantidad de gente que entra en la dinamica
expinfection = 1 # Proporcion en la que contagian los expuestos



# Ajuste por exceso de Muertos (SEIRHVD Class 2)
# Parametros del modelo
beta = 0.14 #0.25#0.19 0.135
mu = 0.01 #2.6 0.6
ScaleFactor = 2.0 #4.8
SeroPrevFactor = 0.5 # Sero Prevalence Factor. Permite ajustar la cantidad de gente que entra en la dinamica
expinfection = 1 # Proporcion en la que contagian los expuestos





beta = 0.119 #0.25#0.19 0.135
mu = 0.7#2.6 0.6
ScaleFactor = 2.5 #4.8
SeroPrevFactor = 0.5 # Sero Prevalence Factor. Permite ajustar la cantidad de gente que entra en la dinamica
expinfection = 1 # Proporcion en la que contagian los expuestos


# 
#   Valores de ajuste de parametros Metodologia MRV
#
beta = 0.117 
mu = 0.67 
ScaleFactor = 2.666
SeroPrevFactor = 0.5 # Sero Prevalence Factor. Permite ajustar la cantidad de gente que entra en la dinamica
expinfection = 1 # Proporcion en la que contagian los expuestos







importdata()
May15 = (datetime(2020,5,15)-initdate).days
escenarios()
#initialconditions(dead='exceso')
simulate()
#infectedpop = []
auxvar() 



plotventilados()
plotcamas()
plotcamasyventiladores()
plotmuertosvsreales()


""" 
# ------------------------------- #
#      Creación de Funciones      #
# ------------------------------- #
"""



#------------------------------------------------- #
#              Definir Escenarios                  #
#------------------------------------------------- #
#tsim,max_mov,rem_mov,qp,iqt,fqt,movfunct
def escenarios():
    global tsim
    global inputarray
    global numescenarios
    tsim = 500
    inputarray=np.array([
            [tsim,0.85,0.6,0,May15,500,0],
            [tsim,0.85,0.65,0,May15,500,0],
            [tsim,0.85,0.7,0,May15,500,0]])        
    numescenarios = len(inputarray)

def addscenario(tsim,max_mov,rem_mov,qp,iqt,fqt,movfunct):
    global inputarray
    inputarray.append([tsim,max_mov,rem_mov,qp,iqt,fqt,movfunct])
    return()

# traspasarlo a lenguaje humao
def showscenarios():
    global inputarray
    print(inputarray)
    return()

# -------------- #
#    Simulate    #
# -------------- #


def initialconditions(dead='minsal'):
    global B
    global D
    global V
    global R
    global Vcmodel
    global Hcmodel
    global sims
    global Vmax
    global Hmax


    # Valores iniciales
    if dead=='minsal':        
        B = Br[0]  # Muertos acumulados al dia de inicio
        D = Br[1]-Br[0]  # Muertos en el dia de inicio
    elif dead == 'exceso':
        B = ED_RM_ac[0]
        D = ED_RM_ac[1] - ED_RM_ac[0]

    #elif dead == 'deis':     
    #    B = ED_RM_ac[0]
    #    D = ED_RM_ac[1] - ED_RM_ac[0]


    I_act0 = ScaleFactor*Ir[0]  # Infectados Activos al dia de inicio
    cId0 = ScaleFactor*(Ir[1]-Ir[0])# cId0 # I Nuevos dia 0
    R  = 0   # Recuperados al dia de inicio


    # Dinámica del aumento de camas
    V = Vr[0]   # Ventilados al dia de inicio
    H_cr = 0#Vr[0]*0.01 # Hospitalizados criticos dia 0
    H0 = Hr[0] # Hospitalizados totales dia 0

    bedmodelorder = 2
    Hcmodel = np.poly1d(np.polyfit(sochimi_tr, Hr_tot, bedmodelorder)) 
    Vcmodel = np.poly1d(np.polyfit(sochimi_tr, Vr_tot, bedmodelorder)) 
    Hc0=Hcmodel[0]#1980
    H_incr=Hcmodel[1]
    H_incr2 = Hcmodel[2]
    H_incr3 = Hcmodel[3]
    Vc0=Vcmodel[0]#1029
    V_incr = Vcmodel[1]
    V_incr2 = Vcmodel[2]
    V_incr3 = Vcmodel[3] 
    Vmax = 1500# Vcmodel(tsat)
    vents = [Vcmodel(t) for t in range(tsim)]  
    tsat = int(np.where(np.array(vents)>=1500)[0][0])
    Hmax = Hcmodel(tsat)
    return

def simulate():    
    global sims
    global model
    global B
    global D
    global V
    global R
    global Vcmodel
    global Hcmodel
    global sims
    global Vmax
    global Hmax
    
    dead = 'minsal'
    # Valores iniciales
    if dead=='minsal':        
        B = Br[0]  # Muertos acumulados al dia de inicio
        D = Br[1]-Br[0]  # Muertos en el dia de inicio
    elif dead == 'exceso':
        B = ED_RM_ac[0]
        D = ED_RM_ac[1] - ED_RM_ac[0]


    I_act0 = ScaleFactor*Ir[0]  # Infectados Activos al dia de inicio
    cId0 = ScaleFactor*(Ir[1]-Ir[0])# cId0 # I Nuevos dia 0
    R  = 0   # Recuperados al dia de inicio


    # Dinámica del aumento de camas
    V = Vr[0]   # Ventilados al dia de inicio
    H_cr = 0#Vr[0]*0.01 # Hospitalizados criticos dia 0
    H0 = Hr[0] # Hospitalizados totales dia 0

    bedmodelorder = 2
    Hcmodel = np.poly1d(np.polyfit(sochimi_tr, Hr_tot, bedmodelorder)) 
    Vcmodel = np.poly1d(np.polyfit(sochimi_tr, Vr_tot, bedmodelorder)) 
    Hc0=Hcmodel[0]#1980
    H_incr=Hcmodel[1]
    H_incr2 = Hcmodel[2]
    H_incr3 = Hcmodel[3]
    Vc0=Vcmodel[0]#1029
    V_incr = Vcmodel[1]
    V_incr2 = Vcmodel[2]
    V_incr3 = Vcmodel[3] 
    Vmax = 1500# Vcmodel(tsat)
    vents = [Vcmodel(t) for t in range(tsim)]  
    tsat = int(np.where(np.array(vents)>=1500)[0][0])
    Hmax = Hcmodel(tsat)
                    
    model = SD2.simSEIRHVD(beta = beta, mu = mu, inputarray= inputarray, B=B,D=D,V=V,I_act0=I_act0,cId0=cId0,R=R,Hc0=Hc0,H_incr=H_incr,H_incr2=H_incr2,H_incr3=H_incr3,Vc0=Vc0,V_incr=V_incr,V_incr2=V_incr2,V_incr3=V_incr3,H_cr=H_cr,H0=H0,tsat=tsat,Hmax=Hmax,Vmax=Vmax, expinfection=expinfection, SeroPrevFactor= SeroPrevFactor)
    sims = model.simulate()
    auxvar()
    return

#-------------------------------- #
#       Variables auxiliares      #
#-------------------------------- #
# Creacion de variables auxiliares para los analisis

def auxvar():
    global T
    global S
    global H_sum
    global H_bed
    global H_vent
    global Iac
    global I
    global I_act
    global I_as
    global I_mi
    global I_se
    global I_cr
    global I_sum
    global E
    global E_as
    global E_sy
    global B
    global D
    global R
    global V
    global t
    global dt
    global idx
    global H_crin
    global H_in
    global H_out
    global H_sum
    global H_tot
    global CH
    global CV
    global ACH
    global ACV
    global peakindex
    global peak
    global peak_t
    global peak_date
    global population
    global infectedsusc
    global infectedpop
    global err_bed
    global err_vent
    global err_Iactives
    global H_colapsedate
    global V_colapsedate

    # Poblacion total
    T=[sims[i][0].S+sims[i][0].E_as+sims[i][0].E_sy+sims[i][0].I_as+sims[i][0].I_cr+sims[i][0].I_mi+sims[i][0].I_se\
        +sims[i][0].H_in+sims[i][0].H_out+sims[i][0].H_cr+sims[i][0].V+sims[i][0].D+sims[i][0].R+sims[i][0].B for i in range(numescenarios)]


    # Susceptibles
    S = [sims[i][0].S for i in range(numescenarios)]
    # Hospitalizados totales diarios
    H_sum=[sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V for i in range(numescenarios)] 
    # Hospitalizados camas diarios
    H_bed=[sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out for i in range(numescenarios)] 
    # Hospitalizados ventiladores diarios
    H_vent=[sims[i][0].V for i in range(numescenarios)] 
    # Infectados Acumulados
    Iac=[sims[i][0].I for i in range(numescenarios)] 
    # Infectados activos diarios
    I = [sims[i][0].I_as+sims[i][0].I_cr + sims[i][0].I_mi + sims[i][0].I_se + sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V for i in range(numescenarios)] 
    I_act = [sims[i][0].I_mi + sims[i][0].I_cr + sims[i][0].I_se + sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V for i in range(numescenarios)] 


    # Infectados asintomaticos
    I_as = [sims[i][0].I_as for i in range(numescenarios)] 
    # Infectados mild
    I_mi = [sims[i][0].I_mi for i in range(numescenarios)] 
    # Infectados severos
    I_se = [sims[i][0].I_se for i in range(numescenarios)] 
    # Infectados criticos
    I_cr = [sims[i][0].I_cr for i in range(numescenarios)] 
    # suma de infectados "sueltos"
    I_sum = [sims[i][0].I_as+sims[i][0].I_cr + sims[i][0].I_mi + sims[i][0].I_se for i in range(numescenarios)] 


    # Expuestos totales diarios
    E = [sims[i][0].E_as+sims[i][0].E_sy for i in range(numescenarios)]
    E_as = [sims[i][0].E_as for i in range(numescenarios)]  
    E_sy = [sims[i][0].E_sy for i in range(numescenarios)]  
    # Enterrados/Muertos acumulados
    B = [sims[i][0].B for i in range(numescenarios)] 
    # Muertos diarios
    D = [sims[i][0].D for i in range(numescenarios)] 
    # Recuperados
    R = [sims[i][0].R for i in range(numescenarios)] 
    # Ventiladores diarios
    V = [sims[i][0].V for i in range(numescenarios)] 

    # Variables temporales
    t = [sims[i][0].t for i in range(numescenarios)] 
    dt = [np.diff(t[i]) for i in range(numescenarios)] 
    #tr = range(tsim)
    idx = [np.searchsorted(t[i],tr) for i in range(numescenarios)] 


    # CAMAS
    H_crin=[sims[i][0].H_cr for i in range(numescenarios)] 
    H_in=[sims[i][0].H_in for i in range(numescenarios)] 
    H_out=[sims[i][0].H_out for i in range(numescenarios)] 
    H_sum=[sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out for i in range(numescenarios)]
    H_tot=[sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V  for i in range(numescenarios)]

    CH = [sims[i][0].CH for i in range(numescenarios)]
    CV = [sims[i][0].CV for i in range(numescenarios)]
    ACH = [sims[i][0].ACH for i in range(numescenarios)]
    ACV = [sims[i][0].ACV for i in range(numescenarios)]
    
    #Cálculo de la fecha del Peak  
    peakindex = [np.where(I[i]==max(I[i]))[0][0] for i in range((numescenarios))]
    peak = [max(I[i]) for i in range((numescenarios))]
    peak_t = [t[i][peakindex[i]] for i in range((numescenarios))]
    peak_date = [initdate+timedelta(days=round(peak_t[i])) for i in range((numescenarios))]

    #proporcion de la poblacion que entra en la dinamica de infeccion
    population = sims[0][0].pop
    infectedsusc = [100*((S[i][0] - S[i][-1])/S[i][0]) for i in range(numescenarios)] 
    infectedpop = [100*((S[i][0] - S[i][-1]))/population for i in range(numescenarios)] 

    # -------------- #
    #     Errores    #
    # -------------- #
    # Camas
    idx = [np.searchsorted(t[i],sochimi_tr) for i in range(numescenarios)]
    err_bed = [LA.norm(Hr-H_sum[i][idx[i]])/LA.norm(Hr) for i in range(numescenarios)]
    err_vent = [LA.norm(Vr-V[i][idx[i]])/LA.norm(Vr) for i in range(numescenarios)]  
    
    # Infecatos Activos
    idx = [np.searchsorted(t[i],tr) for i in range(numescenarios)]
    err_Iactives = [LA.norm(Ir-I[i][idx[i]])/LA.norm(Ir) for i in range(numescenarios)]    
    
    # Infectados acumulados
    #idx = [np.searchsorted(t[i],tr) for i in range(numescenarios)]
    #err_Iactives = [LA.norm(Ir-I[i][idx[i]])/LA.norm(Ir) for i in range(numescenarios)]    
    
    # Fallecidos
    idx = [np.searchsorted(t[i],Br_tr) for i in range(numescenarios)]
    err_dead = [LA.norm(Br-B[i][idx[i]])/LA.norm(Br) for i in range(numescenarios)]

    # -------------------- #
    #   Fecha de Colapso   #
    # -------------------- #
    H_colapsedate = [np.where(CH[i]>0)[0][0] for i in range(numescenarios)]
    V_colapsedate = [np.where(CV[i]>0)[0][0] for i in range(numescenarios)]



# ------------------------------- #
#        Importar Data Real       #
# ------------------------------- #

def importinfectadosactivos():
    # ---------------------- # 
    #   Infectados Activos   #
    # ---------------------- #
    global Ir
    global Ir_dates
    global tr
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
    print('Infectados Activos')
    return

def importsochimi():
    # ------------------ #
    #    Datos Sochimi   #
    # ------------------ #
    global sochimi
    global Hr
    global Vr 
    global Vr_tot
    global Hr_tot 
    global sochimi_dates
    global sochimi_tr

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
    Hr=list(Hr[index:])
    Vr=list(Vr[index:])
    Hr_tot=list(Hr_tot[index:])
    Vr_tot=(list(Vr_tot[index:]))
    sochimi_dates = sochimi_dates[index:]
    sochimi_tr = [(sochimi_dates[i]-initdate).days for i in range(len(Hr))]
    print('Sochimi')
    return

# -------------------------------- #
#    Datos Fallecidos acumulados   #
# -------------------------------- #
def importfallecidosacumuados():
    global Br
    global Br_dates
    global Br_tr
    endpoint = 'https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto14/FallecidosCumulativo.csv' 
    Br = pd.read_csv(endpoint).iloc[6][1:] 
    Br_dates = [datetime.strptime(Br.index[i],'%Y-%m-%d') for i in range(len(Br))]
    index = np.where(np.array(Br_dates) >= initdate)[0][0] 
    Br = Br[index:]
    Br_dates = Br_dates[index:]
    Br_tr = [(Br_dates[i]-initdate).days for i in range(len(Br))]
    print('Fallecidos Acumulados')
    return



# -------------------------- #
#    Fallecidos excesivos    #
# -------------------------- #
def importfallecidosexcesivos():
    global ED_RM
    global ED_RM_dates
    global ED_tr
    global ED_RM_ac
    path = '/home/samuel/Documents/Dlab/data/Excess_dead_daily.csv'
    excess_dead = pd.read_csv(path)

    ED_RM_df = excess_dead.loc[excess_dead['Codigo region']==13]      
    ED_RM = [ED_RM_df['Defunciones Covid'].iloc[i] + ED_RM_df['Exceso de muertes media poderada'].iloc[i] for i in range(len(ED_RM_df))]       

    ED_RM_dates = [datetime.strptime(ED_RM_df['Fecha'].iloc[i], '%Y-%m-%d')  for i in range(len(ED_RM_df))]
    index = np.where(np.array(ED_RM_dates) >= initdate)[0][0]
    enddate = max(ED_RM_dates)
    indexend = np.where(np.array(ED_RM_dates) >= enddate)[0][0]
    ED_RM_dates = ED_RM_dates[index:indexend]  
    ED_RM = ED_RM[index:indexend]
    ED_RM_ac = np.cumsum(ED_RM)
    ED_tr = [(ED_RM_dates[i]-initdate).days for i in range(len(ED_RM))]
    print('Fallecidos Excesivos')
    return

# --------------------------- #
#    Importar toda la data    #
# --------------------------- #

def importdata():
    print('Importando Datos')
    importfallecidosacumuados()
    importfallecidosexcesivos()
    importinfectadosactivos()
    importsochimi()
    print('Done')





"""
    # ---------------------------------- #
    #          Estudio Resultados        #
    # ---------------------------------- # 
"""


# -------------------------- #
#        Plot function       #
# -------------------------- #
def plot(title = '',xlabel='',ylabel=''):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=0)
    plt.show()


# -------------------------------------------------------- #
#                  Uso Hospitalario                        #
# -------------------------------------------------------- #

# -------------------------------------- #
#       Hospitalizados desagregados      #
# -------------------------------------- #
# Hospitalizados desagregados
def plothospitalizados(enddate =  datetime(2020,7,30)):
    # -------- #
    #   Time   #
    # -------- #
    days = (enddate-initdate).days
    endD = [np.searchsorted(t[i],days) for i in range(numescenarios)]

    for i in range(numescenarios):
        plt.plot(t[i][:endD[i]],H_in[i][:endD[i]],label='Hin',linestyle = 'solid')
        plt.plot(t[i][:endD[i]],H_out[i][:endD[i]],label='Hout',linestyle = 'solid')
        plt.plot(t[i][:endD[i]],H_crin[i][:endD[i]],label='Hcr_in',linestyle = 'solid')

    plot(title = 'Hospitalizados',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))



# ------------------ #
#     Ventiladores   #
# ------------------ #
def plotventiladores(enddate =  datetime(2020,7,30)):
    # -------- #
    #   Time   #
    # -------- # 
    days = (enddate-initdate).days
    endD = [np.searchsorted(t[i],days) for i in range(numescenarios)]
    idx = [np.searchsorted(t[i],sochimi_tr) for i in range(numescenarios)]

    # Inicio cuarentena general
    plt.axvline(x=May15,linestyle = 'dashed',color = 'grey')

    # Ploteo datos reales
    plt.scatter(sochimi_tr,Vr,label='Ventiladores Ocupados reales')
    plt.scatter(sochimi_tr,Vr_tot,label='Capacidad de Ventiladores')

    
    # Error y parámetros
    for i in range(numescenarios):        
        plt.plot([], [], ' ', label='err_vent: '+str(round(100*err_vent[i],2))+'%')
    plt.plot([], [], ' ', label='beta: '+str(beta))
    plt.plot([], [], ' ', label='mu: '+str(mu))
    plt.plot([], [], ' ', label='factor B-Y: '+str(ScaleFactor))

    # Fecha de peaks
    for i in range(numescenarios):        
        plt.plot([], [], ' ', label='Mov='+str(inputarray[i][2])+'Peak='+peak_date[i].strftime('%Y-%m-%d'))
    

    # funcion de ventiladores totales
    Vtot = [sims[0][0].Vtot(i) for i in t[0][:endD[0]]]    
    plt.plot(t[0][:endD[0]],Vtot,color='lime')

    for i in range(numescenarios):            
        plt.plot(t[i][:endD[i]],V[i][:endD[i]],label='VMI Utilizados mov='+str(inputarray[i][2]),color = 'blue' ,linestyle = 'dashed')
  

    plt.xlim(0,days)
    plot(title = 'Ventiladores',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))



# ------------ #
#     Camas    #
# ------------ #
def plotcamas(enddate =  datetime(2020,7,30)):
    # -------- #
    #   Time   #
    # -------- #
    days = (enddate-initdate).days    
    endD = [np.searchsorted(t[i],days) for i in range(numescenarios)]
    idx = [np.searchsorted(t[i],sochimi_tr) for i in range(numescenarios)]

    # Inicio cuarentena general
    plt.axvline(x=May15,linestyle = 'dashed',color = 'grey')

    # Ploteo datos reales
    plt.scatter(sochimi_tr,Hr,label='Camas Ocupadas reales')
    plt.scatter(sochimi_tr,Hr_tot,label='Capacidad de Camas')


    # Display de Parametros y errores
    for i in range(numescenarios):
        plt.plot([], [], ' ', label='err_bed: '+str(round(100*err_bed[i],2))+'%')
        plt.plot([], [], ' ', label='Mov='+str(inputarray[i][2])+'Peak='+peak_date[i].strftime('%Y-%m-%d'))
    plt.plot([], [], ' ', label='beta: '+str(beta))
    plt.plot([], [], ' ', label='mu: '+str(mu))
    plt.plot([], [], ' ', label='fScale: '+str(ScaleFactor))
        
    
    # funcion de camas totales
    Htot = [sims[0][0].Htot(i) for i in t[0][:endD[0]]]
    plt.plot(t[0][:endD[0]],Htot,color='lime')
    
    for i in range(numescenarios):
        plt.plot(t[i][:endD[i]],H_bed[i][:endD[i]],label='Camas utilizadas mov='+str(inputarray[i][2]),color = 'red' ,linestyle = 'dashed')
    
    #plt.plot(t[i][:endD[i]],H_bed[i][:endD[i]],label='Camas utilizadas mov='+str(inputarray[i][2]),color = 'red' ,linestyle = 'solid')
    
    plt.xlim(0,days)
    plot(title = 'Camas',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))



# -------------------------- #
#     Camas y Ventiladores   #
# -------------------------  #
def plotcamasyventiladores(enddate =  datetime(2020,7,30),days=0):
    # -------- #
    #   Time   #
    # -------- #
    if days == 0:
        days = (enddate-initdate).days
    if days < 0:
        days = tsim
    endD = [np.searchsorted(t[i],days) for i in range(numescenarios)]
    idx = [np.searchsorted(t[i],sochimi_tr) for i in range(numescenarios)]

    # inicio cuarentena
    plt.axvline(x=May15,linestyle = 'dashed',color = 'grey')

    #ploteo de datos reales
    plt.scatter(sochimi_tr,Hr,label='Camas Ocupadas reales')
    plt.scatter(sochimi_tr,Vr,label='Ventiladores Ocupados reales')
    plt.scatter(sochimi_tr,Hr_tot,label='Capacidad de Camas')
    plt.scatter(sochimi_tr,Vr_tot,label='Capacidad de Ventiladores')

    for i in range(numescenarios):
        plt.plot([], [], ' ', label='err_bed: '+str(round(100*err_bed[i],2))+'%')
        plt.plot([], [], ' ', label='Mov='+str(inputarray[i][2])+'Peak='+peak_date[i].strftime('%Y-%m-%d'))
    plt.plot([], [], ' ', label='beta: '+str(beta))
    plt.plot([], [], ' ', label='mu: '+str(mu))
    plt.plot([], [], ' ', label='factor de escala: '+str(ScaleFactor))
 

    # Camas y ventiladores totales
    Htot = [sims[0][0].Htot(i) for i in t[0][:endD[0]]]
    Vtot = [sims[0][0].Vtot(i) for i in t[0][:endD[0]]]
    plt.plot(t[0][:endD[0]],Htot,color='lime')
    plt.plot(t[0][:endD[0]],Vtot,color='lime')

    
    for i in range(numescenarios): 
        plt.plot(t[i][:endD[i]],H_bed[i][:endD[i]],label='Camas utilizadas mov='+str(inputarray[i][2]),color = 'red' ,linestyle = 'dashed')
        plt.plot(t[i][:endD[i]],V[i][:endD[i]],label='VMI Utilizados mov='+str(inputarray[i][2]),color = 'blue' ,linestyle = 'dashed')
    #plt.plot(t[i][:endD[i]],H_crin[i][:endD[i]],label='Camas críticas mov='+str(inputarray[i][2]),color = 'black' ,linestyle = 'dashed')
    
    plt.xlim(0,days)
    plot(title = 'Camas',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))


# ---------- #
#    Hrate   #
# ---------- #
def plothrate(enddate =  datetime(2020,7,30)):
    # -------- #
    #   Time   #
    # -------- #
    days = (enddate-initdate).days
    endD = [np.searchsorted(t[i],days) for i in range(numescenarios)]
    idx = [np.searchsorted(t[i],sochimi_tr) for i in range(numescenarios)]
    
    Hrate = [H_in[i]/H_out[i] for i in range(numescenarios)]

    title = 'Rate Hin y Hout / Hsum'
    xlabel = 'Dias desde '+datetime.strftime(initdate,'%Y-%m-%d')
    fig, axs = plt.subplots(3)
    #fig.suptitle(title)
    
    
    colors = ['red','blue','green']
    
    
    for i in range(numescenarios):
        axs[0].plot(t[i][:endD[i]],Hrate[i][:endD[i]], label='Mov='+str(inputarray[i][2]),linestyle = 'solid',color = colors[i])
        axs[0].legend()
       
    
    for i in range(numescenarios):
        axs[1].plot(t[i][:endD[i]],(H_in[i]/H_sum[i])[:endD[i]],label='Hin '+'Mov='+str(inputarray[i][2]),linestyle = 'solid',color = colors[i])        
        axs[1].legend()        
 
    for i in range(numescenarios):        
        axs[2].plot(t[i][:endD[i]],(H_out[i]/H_sum[i])[:endD[i]],label='Hout '+'Mov='+str(inputarray[i][2]),linestyle = 'solid',color = colors[i])
        axs[2].legend()

    axs[0].set_title('Rate Hin/Hout')
    axs[1].set_title('Rate Hin/ Hsum')
    axs[1].set_title('Rate Hout/Hsum')
    for ax in axs.flat:
        ax.label_outer()    
    plt.xlabel(xlabel)
    plt.xlim=days                
    plt.show()
 

# --------------------------- #
#      Camas requeridas       #
# --------------------------- #
def plotcamasrequeridas(enddate =  datetime(2020,7,30),days=0):
    # ----------- #
    #     Time    #
    # ----------- #    
    if days ==0:
        days = (enddate-initday).days
    endD = [np.searchsorted(t[i],days) for i in range(numescenarios)]
    
    # ----------- #
    #     Plot    #
    # ----------- #
    # Fechas de colapso
    plt.plot([], [], ' ', label='Fecha colapso Camas: '+str(round(t[i][H_colapsedate[i]])))
    plt.plot([], [], ' ', label='Fecha colapso Ventiladores: '+str(round(t[i][V_colapsedate[i]])))

    linestyle = ['dashed','solid','dotted']
    for i in range(numescenarios):
        plt.plot(t[i][:endD[i]],CH[i][:endD[i]],label='Intermedio/Intensivo Mov = '+str(inputarray[i][1]),color = 'red' ,linestyle = linestyle[i])
        plt.plot(t[i][:endD[i]],CV[i][:endD[i]],label='VMI Mov = '+str(inputarray[i][1]),color = 'blue' ,linestyle = linestyle[i])
    
    plot(title='Camas Requeridas',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))


# ------------------------------------ #
#      Necesidad total de Camas        #
# ------------------------------------ #
def plotnecesidadtotcamas(enddate =  datetime(2020,7,30),days=0):
    # ----------- #
    #     Time    #
    # ----------- #    
    if days ==0:
        days = (enddate-initday).days
    endD = [np.searchsorted(t[i],days) for i in range(numescenarios)]

    # ----------- #
    #     Plot    #
    # ----------- #
    
    # Fechas de colapso
    plt.plot([], [], ' ', label='Fecha colapso Camas: '+str(round(t[i][H_colapsedate[i]])))
    plt.plot([], [], ' ', label='Fecha colapso Ventiladores: '+str(round(t[i][V_colapsedate[i]])))

    # Datos reales
    plt.scatter(sochimi_tr,Hr,label='Camas Ocupadas reales')
    plt.scatter(sochimi_tr,Vr,label='Ventiladores Ocupados reales')
    plt.scatter(sochimi_tr,Hr_tot,label='Capacidad de Camas')
    plt.scatter(sochimi_tr,Vr_tot,label='Capacidad de Ventiladores')

    linestyle = ['dashed','solid','dotted']
    for i in range(numescenarios):    
        plt.plot(t[i][:endD[i]],np.array(CH[i][:endD[i]])+np.array(H_sum[i][:endD[i]]),label='UCI/UTI Mov = '+str(inputarray[i][1]),color = 'red' ,linestyle = linestyle[i])
        plt.plot(t[i][:endD[i]],np.array(CV[i][:endD[i]])+np.array(V[i][:endD[i]]),label='VMI Mov = '+str(inputarray[i][1]),color = 'blue' ,linestyle = linestyle[i])
    plot(title='Necesidad total de Camas',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))




# -------------------------------------------------------- #
#                       Infectados                         #
# -------------------------------------------------------- #

# ------------------------------ #
#       Infectados Activos       #
# ------------------------------ #
def plotactivos(enddate =  datetime(2020,7,30), days = 0,reales= True,scalefactor = False):
    # -------- #
    #   Time   #
    # -------- #
    if days == 0:
        days = (enddate-initdate).days
    if days < 0:
        days = tsim
    endD = [np.searchsorted(t[i],days) for i in range(numescenarios)]
    idx = [np.searchsorted(t[i],tr) for i in range(numescenarios)]

    Isf = 1    
    if scalefactor:
        Isf = ScaleFactor


    # ----------- #
    #     Plot    #
    # ----------- #
    # Error
    for i in range(numescenarios):
        plt.plot([], [], ' ', label='err: '+str(round(100*err_Iactives[i],2))+'%')    

    # Reales
    if reales:
        plt.scatter(tr,Ir,label='Infectados Activos reales')

    # Infectados
    for i in range(numescenarios):        
        plt.plot(t[i],I[i]/Isf,label='Infectados Mov = '+str(inputarray[i][2]))

    if days >0:
        plt.xlim(0,days)
    plot(title = 'Activos',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))

# -------------------------------- #
#       Infectados Acumulados      #
# -------------------------------- #
# No esta listo
def plotacumulados(enddate =  datetime(2020,7,30), days = 0,reales= True,scalefactor = False):
    # -------- #
    #   Time   #
    # -------- #
    if days == 0:
        days = (enddate-initdate).days
    if days < 0:
        days = tsim
    endD = [np.searchsorted(t[i],days) for i in range(numescenarios)]
    idx = [np.searchsorted(t[i],tr) for i in range(numescenarios)]

    # ----------- #
    #     Plot    #
    # ----------- #
    # Error
    for i in range(numescenarios):
        plt.plot([], [], ' ', label='err: '+str(round(100*err[i],2))+'%')    

    # Reales
    #if reales:
    #    plt.scatter(tr,Ir,label='Infectados Activos reales')

    # Infectados
    for i in range(numescenarios):        
        plt.plot(t[i],I[i],label='Infectados Mov = '+str(inputarray[i][2]))

    if days >0:
        plt.xlim(0,days)
    plot(title = 'Activos',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))  


def plotactivosdesagregados(enddate =  datetime(2020,7,30), days = 0,reales= True,scalefactor = False):
    # -------- #
    #   Time   #
    # -------- #
    if days == 0:
        days = (enddate-initdate).days
    if days < 0:
        days = tsim
    endD = [np.searchsorted(t[i],days) for i in range(numescenarios)]
    idx = [np.searchsorted(t[i],tr) for i in range(numescenarios)]

    # ----------- #
    #     Plot    #
    # ----------- #
    # Error
    #for i in range(numescenarios):
    #    plt.plot([], [], ' ', label='err: '+str(round(100*err[i],2))+'%')    

    # Reales
    if reales:
        plt.scatter(tr,Ir,label='Infectados Activos reales')

    # Infectados
    for i in range(numescenarios):        
        #plt.plot(t[i],I[i],label='Infectados )
        plt.plot(t[i],I_as[i],label='Acumulados asintomáticos Mov = '+str(inputarray[i][2]))
        plt.plot(t[i],I_mi[i],label='Acumulados Mild Mov = '+str(inputarray[i][2]))
        plt.plot(t[i],I_se[i],label='Acumulados Severos Mov = '+str(inputarray[i][2]))
        plt.plot(t[i],I_cr[i],label='Acumulados Criticos Mov = '+str(inputarray[i][2]))        

    if days >0:
        plt.xlim(0,days)
    plot(title = 'Infectados Activos desagregados',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))



# ------------------------------------------------------------------------------------------------------- #
#                                            Fallecidos                                                   #
# ------------------------------------------------------------------------------------------------------- #

# --------------------------------- #
#      Fallecidos  acumulados       #
# --------------------------------- #
def plotfallecidosacumulados(enddate =  datetime(2020,7,30),days=0, reales= True,ylim = 0,norm=1,scalefactor = False):
    # -------- #
    #   Time   #
    # -------- #
    if days == 0:
        days = (enddate-initdate).days      
    elif days < 0:
        days = tsim     
    endD = [np.searchsorted(t[i],days) for i in range(numescenarios)]
    idx = [np.searchsorted(t[i],Br_tr) for i in range(numescenarios)]

    if norm <1:
        norm = ScaleFactor
    #Isf = 1    
    #if scalefactor:
    #    Isf = ScaleFactor

    # ----------- #
    #     Error   #
    # ----------- #
    i = 1 # Mov 0.6
    err = [LA.norm(Br[:21]-B[i][idx[i][:21]])/LA.norm(Br[:21]) for i in range(numescenarios)]
    err2 = [LA.norm(Br[23:]-B[i][idx[i][23:]])/LA.norm(Br[23:]) for i in range(numescenarios)]

    # ----------- #
    #     Plot    #
    # ----------- #
    # Parametros 
    plt.plot([], [], ' ', label='beta: '+str(beta))
    plt.plot([], [], ' ', label='mu: '+str(mu))
    plt.plot([], [], ' ', label='factor de escala: '+str(ScaleFactor))

    # Fecha de Peak
    for i in range(numescenarios):
        plt.plot([], [], ' ', label='Mov='+str(inputarray[i][2])+'Peak='+peak_date[i].strftime('%Y-%m-%d'))
    
    # Error
    for i in range(numescenarios):
        plt.plot([], [], ' ', label='Mov='+str(inputarray[i][2])+'err: '+str(round(100*err[i],2))+'%')


    # Datos reales
    if reales:
        plt.scatter(Br_tr,Br,label='Fallecidos reales')
        plt.scatter(ED_tr,ED_RM_ac,label='Fallecidos excesivos proyectados')

    linestyle = ['dashed','solid','dashed']
    for i in range(numescenarios):
        plt.plot(t[i][:endD[i]],B[i][:endD[i]]/norm,label='Fallecidos Mov = '+str(inputarray[i][2]),color = 'blue',linestyle=linestyle[i])

    plt.xlim(0,days)   
    if ylim >0:
        plt.ylim(0,ylim)

    plot(title = 'Fallecidos',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))
    


# ----------------------------- #
#       Fallecidos diarios      #
# ----------------------------- #
def plotfallecidosdiarios(enddate =  datetime(2020,7,30),days=0,scalefactor = False,reales= False):
    # -------- #
    #   Time   #
    # -------- #
    if days == 0:
        days = (enddate-initdate).days      
    elif days < 0:
        days = tsim     
    endD = [np.searchsorted(t[i],days) for i in range(numescenarios)]
    idx = [np.searchsorted(t[i],Br_tr) for i in range(numescenarios)]

    Isf = 1    
    if scalefactor:
        Isf = ScaleFactor

    linestyle = ['dashed','solid','dotted']
    for i in range(numescenarios):
        plt.plot(t[i],D[i]/Isf,label='Mov = '+str(inputarray[i][2]),color = 'black' ,linestyle = linestyle[i])
    plot(title = 'Fallecidos diarios',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))



# ---------------------- #
#       Letalidad        #
# ---------------------- #

def plotletalidad(enddate =  datetime(2020,7,30),days=0, reales= True,ylim = 0,norm=1,scalefactor = False):
    # -------- #
    #   Time   #
    # -------- #
    if days == 0:
        days = (enddate-initdate).days      
    elif days < 0:
        days = tsim     
    endD = [np.searchsorted(t[i],days) for i in range(numescenarios)]
    idx = [np.searchsorted(t[i],Br_tr) for i in range(numescenarios)]

    if norm <1:
        norm = ScaleFactor
    #Isf = 1    
    #if scalefactor:
    #    Isf = ScaleFactor

    # ----------- #
    #     Error   #
    # ----------- #
    i = 1 # Mov 0.6
    err = [LA.norm(Br[:21]-B[i][idx[i][:21]])/LA.norm(Br[:21]) for i in range(numescenarios)]
    err2 = [LA.norm(Br[23:]-B[i][idx[i][23:]])/LA.norm(Br[23:]) for i in range(numescenarios)]

    # ----------- #
    #     Plot    #
    # ----------- #
    # Parametros 
    plt.plot([], [], ' ', label='beta: '+str(beta))
    plt.plot([], [], ' ', label='mu: '+str(mu))
    plt.plot([], [], ' ', label='factor de escala: '+str(ScaleFactor))

    # Fecha de Peak
    for i in range(numescenarios):
        plt.plot([], [], ' ', label='Mov='+str(inputarray[i][2])+'Peak='+peak_date[i].strftime('%Y-%m-%d'))
    
    # Error
    for i in range(numescenarios):
        plt.plot([], [], ' ', label='Mov='+str(inputarray[i][2])+'err: '+str(round(100*err[i],2))+'%')


    # Datos reales
    #if reales:
    #    plt.scatter(Br_tr,Br,label='Fallecidos reales')
    #    plt.scatter(ED_tr,ED_RM_ac,label='Fallecidos excesivos proyectados')

    linestyle = ['dashed','solid','dashed']
    for i in range(numescenarios):
        plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(inputarray[i][2])+','+str(inputarray[i][1])+']' ,color='blue',linestyle=linestyle[i])
        
    plt.xlim(0,days)   
    if ylim >0:
        plt.ylim(0,ylim)

    plot(title = 'Letalidad',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))
    




# ------------------ #
#     Expuestos      #
# ------------------ #
def plotexpuestos(enddate =  datetime(2020,7,30),days=0, reales= True,ylim = 0,norm=1,scalefactor = False):
    # -------- #
    #   Time   #
    # -------- #
    if days == 0:
        days = (enddate-initdate).days      
    elif days < 0:
        days = tsim     
    endD = [np.searchsorted(t[i],days) for i in range(numescenarios)]
    idx = [np.searchsorted(t[i],Br_tr) for i in range(numescenarios)]

    if norm <1:
        norm = ScaleFactor
    #Isf = 1    
    #if scalefactor:
    #    Isf = ScaleFactor

    # ----------- #
    #     Plot    #
    # ----------- #
    # Parametros 
    plt.plot([], [], ' ', label='beta: '+str(beta))
    plt.plot([], [], ' ', label='mu: '+str(mu))
    plt.plot([], [], ' ', label='factor de escala: '+str(ScaleFactor))

    # Fecha de Peak
    for i in range(numescenarios):
        plt.plot([], [], ' ', label='Mov='+str(inputarray[i][2])+'Peak='+peak_date[i].strftime('%Y-%m-%d'))
    

    linestyle = ['dashed','solid','dashed']
    for i in range(numescenarios):
        plt.plot(t[i][:endD[i]],E[i][:endD[i]],label='Expuestos Mov = '+str(inputarray[i][2]),color = 'blue',linestyle=linestyle[i])
        plt.plot(t[i][:endD[i]],E_sy[i][:endD[i]],label='Expuestos sintomáticos Mov = '+str(inputarray[i][2]),color = 'red',linestyle=linestyle[i])
        
    plt.xlim(0,days)   
    if ylim >0:
        plt.ylim(0,ylim)

    plot(title = 'Expuestos',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))
    


# -------------------- #
#     Curvas SEIR      #
# -------------------- #
def plotseird(enddate =  datetime(2020,7,30),days=-1, reales= True,ylim = 0,norm=1,scalefactor = False,seird = [1,1,1,1,1]):
    # -------- #
    #   Time   #
    # -------- #
    if days == 0:
        days = (enddate-initdate).days      
    elif days < 0:
        days = tsim     
    endD = [np.searchsorted(t[i],days) for i in range(numescenarios)]
    idx = [np.searchsorted(t[i],Br_tr) for i in range(numescenarios)]

    if norm <1:
        norm = ScaleFactor
    #Isf = 1    
    #if scalefactor:
    #    Isf = ScaleFactor

    # ----------- #
    #     Plot    #
    # ----------- #
    # Parametros 
    plt.plot([], [], ' ', label='beta: '+str(beta))
    plt.plot([], [], ' ', label='mu: '+str(mu))
    plt.plot([], [], ' ', label='factor de escala: '+str(ScaleFactor))

    # Fecha de Peak
    for i in range(numescenarios):
        plt.plot([], [], ' ', label='Mov='+str(inputarray[i][2])+'Peak='+peak_date[i].strftime('%Y-%m-%d'))
    

    linestyle = ['dashed','solid','dashed']
    for i in range(numescenarios):        
        plt.plot(t[i],S[i],label='Susceptibles Mov = '+str(inputarray[i][2]),linestyle=linestyle[i])
        plt.plot(t[i],I[i],label='Infectados Mov = '+str(inputarray[i][2]),linestyle=linestyle[i])        
        plt.plot(t[i],E[i],label='Expuestos Mov = '+str(inputarray[i][2]),linestyle=linestyle[i])
        plt.plot(t[i],R[i],label='Recuperados Mov = '+str(inputarray[i][2]),linestyle=linestyle[i])
        #plt.plot(t[i],D[i],label='Muertos diarios Mov = '+str(inputarray[i][2]),linestyle=linestyle[i])
        plt.plot(t[i],B[i],label='Enterrados Mov = '+str(inputarray[i][2]),linestyle=linestyle[i])
        
    plt.xlim(0,days)   
    if ylim >0:
        plt.ylim(0,ylim)

    plot(title = 'Curvas SEIR',xlabel='Dias desde '+datetime.strftime(initdate,'%Y-%m-%d'))
    


"""
# ------------------------------------------ #
#       Graficos para parametrización        #
# ------------------------------------------ #
"""
# ----------------------------------------- #
#       Curvas Expuestos/Infectados         #
# ----------------------------------------- #
def plotexpuestosinfectados(enddate =  datetime(2020,7,30),days=-1, reales= True,ylim = 0,norm=1,scalefactor = False,seird = [1,1,1,1,1]):
    initday = initdate#date(2020,3,15)
    enddate =  datetime(2020,6,30)
    days = (enddate-initdate).days
    endD = [np.searchsorted(t[i],days) for i in range(numescenarios)]  
    EIrate = [E[i]/I_sum[i] for i in range(numescenarios)]

    for i in range(numescenarios):    
        plt.plot(t[i][:endD[i]],EIrate[:endD[i]],label='Tasa Expuestos/Infectados')
    plot(title='Expuestos/infectados - mu ='+str(mu)+' beta='+str(beta))




# ------------------------ #
#       Curvas H/I         #
# ------------------------ #
def plothospitalizadosinfectados(enddate =  datetime(2020,7,30),days=-1, reales= True,ylim = 0,norm=1,scalefactor = False,seird = [1,1,1,1,1]):
    initday = initdate#date(2020,3,15)
    enddate =  datetime(2020,6,30)
    days = (enddate-initdate).days
    endD = [np.searchsorted(t[i],days) for i in range(numescenarios)]

    HIrate = [H_sum[i]/I_sum[i] for i in range(numescenarios)]
    for i in range(numescenarios):    
        plt.plot(t[i][:endD[i]],HIrate[:endD[i]],label='Tasa Expuestos/Infectados')
    plot(title='H/I - mu ='+str(mu)+' beta='+str(beta))


# ------------------------ #
#       Curvas V/I         #
# ------------------------ #
def plotventiladosinfectados(enddate =  datetime(2020,7,30),days=-1, reales= True,ylim = 0,norm=1,scalefactor = False,seird = [1,1,1,1,1]):
    initday = initdate#date(2020,3,15)
    enddate =  datetime(2020,6,30)
    days = (enddate-initdate).days
    days = 200
    endD = [np.searchsorted(t[i],days) for i in range(numescenarios)]
    
    VIrate = [V[i]/I_sum[i] for i in range(numescenarios)]
    for i in range(numescenarios):     
        plt.plot(t[i][:endD[i]],VIrate[:endD[i]],label='Tasa Expuestos/Infectados')
    plot(title='V/I - mu ='+str(mu)+' beta='+str(beta))


# ------------------------ #
#       Curvas V/H         #
# ------------------------- #
def plotventiladosinfectados(enddate =  datetime(2020,7,30),days=-1, reales= True,ylim = 0,norm=1,scalefactor = False,seird = [1,1,1,1,1]):
    initday = initdate#date(2020,3,15)
    enddate =  datetime(2020,6,30)
    days = (enddate-initdate).days
    endD = [np.searchsorted(t[i],days) for i in range(numescenarios)]
    VHrate = [V[i]/H_sum[i] for i in range(numescenarios)]
    for i in range(numescenarios):        
        plt.plot(t[i][:endD[i]],VHrate[:endD[i]],label='Tasa Expuestos/Infectados')
    plot(title='V/H - mu ='+str(mu)+' beta='+str(beta))




"""
# -------------------------- #
#    Generación de Tablas    #
# -------------------------- #


# Variables: 
# I_cum: Infectados acumulados
# I_act: Infectados activos
# D: Muertos acumulados
# D_d: Muertos Diarios
# L: Letalidad
# H: Uso de Camas hospitalarias
# V: Uso de VMI
# H_tot: Necesidad total de camas (incluidas las que se necesitan por sobre la capacidad)
# V_tot: Necesidad total de VMI (incluidos las que se necesitan por sobre la capacidad)


"""
# ------------------- #
#    Tabla Muertos    #
# ------------------- #
# Muertos hasta el 30 de Junio


#from datetime import timedelta
def tabladedatos(inicio = datetime(2020,5,15), fin = datetime(2020,6,30),variables =['I_cum','I_act','D','L'], path=''):

    # Time
    tr_i = (inicio-initdate).days
    tr_f = (fin-initdate).days
    days = (fin-inicio).days
    index = [(inicio+timedelta(days=i)).strftime("%d/%m/%Y") for i in range(days+1)]
    idx = [np.searchsorted(t[i],range(tr_i,tr_f+1)) for i in range(numescenarios)]

    #data
    data = []
    # --------------------------- #
    #    Fallecidos acumulados    #
    # --------------------------- #
    if 'D' in variables:
        # Interpolacion para datos faltantes
        Bdata = dict()
        namelist = ['Muertos-60','Muertos-65','Muertos-70']
        for i in range(len(namelist)):        
            B_hoy = [round(B[i][idx[i][0]])]
            for j in range(1,len(idx[i])-1):
                if idx[i][j-1]== idx[i][j]:
                    B_hoy.extend([round((B[i][idx[i][j-1]]+B[i][idx[i][j+1]])/2)])
                else:        
                    B_hoy.extend([round(B[i][idx[i][j]])])
            B_hoy.extend([round(B[i][idx[i][-1]])])
            Bdata[namelist[i]]=B_hoy

        Bdata = pd.DataFrame(Bdata)
        data.append(Bdata)


    # Infectados Acumulados
    if 'I_cum' in variables:
        Iacdata = dict()
        namelist = ['Infectados-60','Infectados-65','Infectados-70']
        for i in range(numescenarios):        
            Iac_hoy = [round(Iac[i][idx[i][0]])]
            for j in range(1,len(idx[i])-1):
                if idx[i][j-1]== idx[i][j]:
                    Iac_hoy.extend([round((Iac[i][idx[i][j-1]]+Iac[i][idx[i][j+1]])/2)])
                else:        
                    Iac_hoy.extend([round(Iac[i][idx[i][j]])])
            Iac_hoy.extend([round(Iac[i][idx[i][-1]])])
            Iacdata[namelist[i]]=Iac_hoy
        Iacdata = pd.DataFrame(Iacdata)
        data.append(Iacdata)


    # Letalidad
    #let = [100*B[i]/Iac[i] for i in range(numescenarios)
    #Iacdata = dict()
    #namelist = ['Infectados-60','Infectados-65','Infectados-70']
    #for i in range(numescenarios):        
    #    Iac_hoy = [round(Iac[i][idx[i][0]])]
    #    for j in range(1,len(idx[i])-1):
    #        if idx[i][j-1]== idx[i][j]:
    #            Iac_hoy.extend([round((Iac[i][idx[i][j-1]]+Iac[i][idx[i][j+1]])/2)])
    #        else:        
    #            Iac_hoy.extend([round(Iac[i][idx[i][j]])])
    #    Iac_hoy.extend([round(Iac[i][idx[i][-1]])])
    #    Iacdata[namelist[i]]=Iac_hoy
    #
    #Iacdata = pd.DataFrame(Iacdata)




    # ------------------ #
    #    Uso de Camas    #
    # ------------------ #

    #H_bed_hoy = [H_bed[i][idx[i][0:days]] for i in range(len(input))]
    if 'H' in variables:
        UsoCamas = dict()
        namelist = ['UsoCamas-60','UsoCamas-65','UsoCamas-70']
        for i in range(3):        
            UsoCamas_hoy = [round(H_bed[i][idx[i][0]])]
            for j in range(1,len(idx[i])-1):
                if idx[i][j-1]== idx[i][j]:
                    UsoCamas_hoy.extend([round((H_bed[i][idx[i][j-1]]+H_bed[i][idx[i][j+1]])/2)])
                else:        
                    UsoCamas_hoy.extend([round(H_bed[i][idx[i][j]])])
            UsoCamas_hoy.extend([round(H_bed[i][idx[i][-1]])])
            UsoCamas[namelist[i-3]]=UsoCamas_hoy

        Hbed = pd.DataFrame(UsoCamas)
        data.append(Hbed)

    # ------------------------- #
    #    Uso de Ventiladores    #
    # ------------------------- #
    #H_vent_hoy = [H_vent[i][idx[i][0:days]] for i in range(len(input))]
    if 'V' in variables:
        namelist = ['UsoVMI-60','UsoVMI-65','UsoVMI-70']
        UsoVMI = dict()
        for i in range(3):        
            UsoVMI_hoy = [round(H_vent[i][idx[i][0]])]
            for j in range(1,len(idx[i])-1):
                if idx[i][j-1]== idx[i][j]:
                    UsoVMI_hoy.extend([round((H_vent[i][idx[i][j-1]]+H_vent[i][idx[i][j+1]])/2)])
                else:        
                    UsoVMI_hoy.extend([round(H_vent[i][idx[i][j]])])
            UsoVMI_hoy.extend([round(H_vent[i][idx[i][-1]])])
            UsoVMI[namelist[i-3]]=UsoVMI_hoy

        Hvent = pd.DataFrame(UsoVMI)
        data.append(Hvent)


    # ---------------------------------- #
    #    Camas adicionales Requeridas    #
    # ---------------------------------- #
    if 'H_ad' in variables:
        CH_d = dict()
        namelist = ['CamaAdicional-60','CamaAdicional-65','CamaAdicional-70']
        for i in range(3):        
            CH_hoy = [round(CH[i][idx[i][0]])]
            for j in range(1,len(idx[i])-1):
                if idx[i][j-1]== idx[i][j]:
                    CH_hoy.extend([round((CH[i][idx[i][j-1]]+CH[i][idx[i][j+1]])/2)])
                else:        
                    CH_hoy.extend([round(CH[i][idx[i][j]])])
            CH_hoy.extend([round(CH[i][idx[i][-1]])])
            CH_d[namelist[i-3]]=CH_hoy

        CH_d = pd.DataFrame(CH_d)
        data.append(CH_d)

    if 'V_ad' in variables:
        CV_d = dict()
        namelist = ['VMIAdicional-60','VMIAdicional-65','VMIAdicional-70']
        for i in range(3):        
            CV_hoy = [round(CV[i][idx[i][0]])]
            for j in range(1,len(idx[i])-1):
                if idx[i][j-1]== idx[i][j]:
                    CV_hoy.extend([round((CV[i][idx[i][j-1]]+CV[i][idx[i][j+1]])/2)])
                else:        
                    CV_hoy.extend([round(CV[i][idx[i][j]])])
            CV_hoy.extend([round(CV[i][idx[i][-1]])])
            CV_d[namelist[i-3]]=CV_hoy

        CV_d = pd.DataFrame(CV_d)
        data.append(CV_d)


    # ------------------------------ #
    #    Necesidad total de Camas    #
    # ------------------------------ #
    if False:
        namelistUsoC = ['UsoCamas-60','UsoCamas-65','UsoCamas-70']
        namelistUsoV = ['UsoVMI-60','UsoVMI-65','UsoVMI-70']

        namelistCH = ['CamaAdicional-60','CamaAdicional-65','CamaAdicional-70']
        namelistCV = ['VMIAdicional-60','VMIAdicional-65','VMIAdicional-70']

        namelistcamas = ['NecesidadTotalCamas-60','NecesidadTotalCamas-65','NecesidadTotalCamas-70']
        namelistvmi = ['NecesidadTotalVMI-60','NecesidadTotalVMI-65','NecesidadTotalVMI-70']

        totbed_d = pd.DataFrame()

        totvmi_d = pd.DataFrame()

        for i in range(len(namelistCH)):
            totbed_d[namelistcamas[i]] = CH_d[namelistCH[i]] + Hbed[namelistUsoC[i]]
            totvmi_d[namelistvmi[i]] = CV_d[namelistCV[i]] + Hvent[namelistUsoV[i]]
        data.append(totbed_d)
        data.append(totvmi_d )


    # ------------------------- #
    #     Create Data Frame     #
    # ------------------------- #
    index = pd.DataFrame(dict(dates=index))
    data = pd.concat(data, axis=1, sort=False)
    data = pd.concat([index,data], axis=1, sort=False) 
    data = data.set_index('dates')
    if path:
        data.to_excel(path)
        #
    return(data)


#datosochimi.to_excel('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/Fiteo por Muertos/TabladeMuertos.xls') 
#datosochimi.to_excel('/home/samuel/Covid_Net_SEIR/SEIRHVDB/Plots/ReporteSochimi-11-06/datosReporteSochimi.xls') 

