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
from scipy import signal
import pandas as pd
from numpy import linalg as LA


# --------- #
#   To Do   #
# --------- #

# i=3 caga  enm el uso de camas
# Automatizar generación de gráficos para distintos rangos de fechas
# Implementar el smooth para todos los datos

def plot(title='',xlabel='dias',ylabel='Personas'):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=0)
    plt.show()

# ------------- #
#   Parametros  #
# ------------- #
inputarray = []

def inputlen():
    return(len(inputarray))

def simulate(beta,mu, input = [],inputmovfunct = []):
    inputarray = input
    beta = beta # (*probabilidad de transmision por contacto con contagiados*)
    betaD = 0.0 #(*probabilidad de transmision por contacto con muertos*)

    pSas = 0.3 # Transicion de Susceptible a Expuesto Asintomatico
    tSas = 1.0

    pSsy = 0.7 # Transicion de Susceptible a Expuesto sintomatico
    tSsy = 1.0

    pasas = 1.0# Transicion de Expuesto asintomatico a Infectado asintomatico
    tasas = 5.0

    psymi = 0.78 # Transicion de Expuesto Sintomatico a Infectado Mild
    tsymi = 5.0

    psycr = 0.08 # Transicion de Expuesto Sintomatico a Infectado critico
    tsycr = 5.0

    psyse = 0.14 # Transicion de Expuesto Sintomatico a Infectado Severo
    tsyse = 5.0

    pasR = 1.0   # Transicion de Infectado asintomatico a Recuperado
    tasR = 15.0 

    pmiR = 1.0  # Transicion de Infectado mild a Recuperado
    tmiR = 15.0

    psein = 1.0  # Transicion de Infectado serio a Hospitalizado (si no ha colapsado Htot)
    tsein = 1.0 

    pincrin = 0.01 # Transicion de Hospitalizado a Hospitalizado Critico (si no ha colapsado Htot)
    tincrin = 3.0

    pcrcrin = 1.0 # Transicion de Infectado critico  a Hopsitalizado Critico (si no ha colapsado Htot)
    tcrcrin = 1.0 

    pcrinV = 1.0 # Transicion de Hospitalizado critico a Ventilado (si no ha colapsado V)
    tcrinV = 1.0 

    pcrinD = 1.0 # Muerte de hospitalizado critico (Cuando V colapsa)
    tcrinD = 3.0 #

    pcrD = 1.0 # Muerte de Infectado critico (si ha colapsado Htot)
    tcrD = 3.0 #(*Hin+H_cr_in+Hout colapsa*)

    pseD = 1.0 # Muerte de Infectado severo (si ha colapsado Htot)
    tseD = 3.0

    pinout = 0.99 # Mejora de paciente severo hospitalizado, transita a Hout
    tinout = 4.0

    pVout = 0.5 # Mejora de ventilado hospitalizado, transita a Hout
    tVout = 15.0

    pVD = 0.5 # Muerte de ventilado
    tVD = 15.0

    poutR = 1.0 # Mejora del paciente hospitalizado, Hout a R
    toutR = 6.0

    pDB = 1.0 # Entierro del finado
    tDB = 1.0 

    eta = 0.0 # tasa de perdida de inmunidad (1/periodo)


    # ------------------- #
    #  Valores Iniciales  #
    # ------------------- #
        
    #I_act0 = 12642
    #cIEx0 =I_act0/1.5 # 3879 # Cantidad de infectados para calcular los expuestos iniciales
    #cId0 = 2060 # Infectados nuevos de ese día
    #cI0S =I_act0/2.5 # Cantidad de infectados para calcular los Infectados iniciales
    #muS=mu


    res=1
    cIEx0=4000
    cId0=2234
    cI0S = res*4842
    muS=mu

    I_as= 0.3*cI0S 
    I_mi= 0.6*cI0S 
    I_cr= 0.03*cId0 
    I_se = 0.07*cId0
    E_as=0.3*muS*cIEx0
    E_sy=0.7*muS*cIEx0
    Htot=lambda t: 1997.0+23*t
    H0=1731#1903.0
    H_cr=80.0
    H_in=H0*0.5-H_cr/2
    H_out=H0*0.5-H_cr/2
    Vtot=lambda t:1029.0+18*t
    gw=5
    D=26.0
    B=221.0
    R=0.0
    V=758.0#846.0
    mu=1.4
    t=400.0
    CV=0
    CH=0
    ACV=0
    ACH=0
    I_crD=0
    I_seD=0
    VD=0
    H_crD=0
    S=8125072.0-H0-V-D-(E_as+E_sy)-(I_as+I_cr+I_se+I_mi)



    def alphafunct(max_mov,rem_mov,qp,tci=0,dtc=300,movfunct = 'once'):
        """    
        # max_mov: Movilidad sin cuarentena
        # rem_mov: Movilidad con cuarentena
        # qp: Periodo cuarentena dinamica 
        #          - qp >0 periodo Qdinamica 
        #          - qp = 0 sin qdinamica
        # tci: tiempo inicial antes de cuarentena dinamica
        #          - tci>0 inicia con cuarentena total hasta tci
        #          - tci<0 sin cuarentena hasta tci
        # dtc: duracion tiempo cuarentena 
        # movfunct: Tipo de cuarentena dinamica desde tci
        #          - once: una vez durante qp dias 
        #          - total: total desde tci
        #          - sawtooth: diente de cierra
        #          - square: onda cuadrada
        """
        def alpha(t):
            if movfunct=='total':
                # dtc no sirve
                if t < -tci:
                    return(max_mov)
                else:
                    return(rem_mov)

            elif movfunct =='once':        
                if t<tci:
                    return(rem_mov)
                else:
                    return(max_mov)
                #if t<-tci:
                #    return(max_mov)
                #elif t >dtc:
                #    return(max_mov)
                #elif tci> 0 and t>tci:
                #    return(max_mov)
                #else:
                #    return(rem_mov)

            elif movfunct =='sawtooth':
                def f(t): 
                    return signal.sawtooth(t)
                if t<abs(tci):
                    if tci>0:
                        return(rem_mov)
                    else:
                        return(max_mov)
                else:
                    if t<dtc:
                        return((max_mov-rem_mov)/2*(f(np.pi / qp * t - np.pi))+(max_mov+rem_mov)/2)
                    else:
                        return(max_mov)

            elif movfunct =='square':
                def f(t): 
                    return signal.square(t)
                if t<abs(tci):
                    if tci>0:
                        return(rem_mov)
                    else:
                        return(max_mov)
                else:
                    if t<dtc:
                        return((max_mov-rem_mov)/2*(f(np.pi / qp * t - np.pi))+(max_mov+rem_mov)/2)
                    else:
                        return(max_mov)
        return(alpha)


    def sim_run(tsim,max_mov,rem_mov,qp,tci=0,dtc = 300,movfunct = 'once'):    
        alpha = alphafunct(max_mov,rem_mov,qp,tci,dtc,movfunct)
        
        case=SD.SEIRHUDV(alpha,Htot,Vtot,gw,mu,
                S,E_as,E_sy,
                I_as,I_mi,I_se,I_cr,
                H_in,H_cr,H_out,V,D,B,R,CV,CH,ACH,ACV,
                beta,betaD,eta,pSas,tSas,pSsy,tSsy,
                pasas,tasas,psymi,tsymi,psyse,tsyse,psycr,tsycr,
                pasR,tasR,pmiR,tmiR,psein,tsein,pseD,tseD,
                pcrcrin,tcrcrin,pcrD,tcrD,
                pincrin,tincrin,pinout,tinout,
                pcrinV,tcrinV,pcrinD,tcrinD,pVout,tVout,poutR,toutR,
                pVD,tVD,pDB,tDB)

        # sol=test.integr(0,20,0.1,False)
        case.integr_sci(0,tsim,0.1,False)
        out=[case,max_mov,rem_mov,qp,tsim]
        return(out)   


    tsim = 500
    

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
    
    Input: 
    [tsim,max_mov,rem_mov,qp,iqt,fqt]  
    
    inmputmovfunct: []  {once, square, sawtooth}
    tsim: Tiempo Simulación
    max_mov: Movilidad Máxima
    rem_mov: Movilidad remanente
    qp: Período cuarentena
    iqt: Tiempo inicial cuarentena
    fqt: Tiempo final de cuarentena
    """   
    
    if len(inputarray)==0:
        inputarray=np.array([
                [tsim,0.85,0.6,14,14,500],
                [tsim,0.85,0.7,14,14,500],
                [tsim,0.85,0.8,14,14,500],
                [tsim,0.85,0.6,14,21,500],
                [tsim,0.85,0.7,14,21,500],
                [tsim,0.85,0.8,14,21,500],
                [tsim,0.85,0.7,14,60,500],
                [tsim,0.85,0.7,21,60,500],
                [tsim,0.55,0.2,14,14,500],
                [tsim,0.55,0.3,14,14,500],
                [tsim,0.55,0.4,14,14,500],
                [tsim,0.55,0.2,14,21,500],
                [tsim,0.55,0.3,14,21,500],
                [tsim,0.55,0.4,14,21,500],        
                [tsim,0.55,0.3,14,60,500],
                [tsim,0.55,0.3,21,60,500]])
    inputmovfunct= ['once','once','once','once','once','once','square','square','once','once','once','once','once','once','square','square',]

    num_cores = multiprocessing.cpu_count()
    #params=Parallel(n_jobs=num_cores, verbose=50)(delayed(ref_test.refinepso_all)(Ir,tr,swarmsize=200,maxiter=50,omega=0.5, phip=0.5, phig=0.5,eta_r=[0,1],Q_r=[0,1],obj_func='IN')for i in range(int(rep)))
    sims=Parallel(n_jobs=num_cores, verbose=50)(delayed(sim_run)(inputarray[i,0],inputarray[i,1],inputarray[i,2],inputarray[i,3],inputarray[i,4],inputarray[i,5],inputmovfunct[i])for i in range(inputarray.shape[0]))
    return(sims)
