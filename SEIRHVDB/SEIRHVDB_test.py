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


# ------------- #
#   Parametros  #
# ------------- #
def sim(beta,mu):
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
    Pesimistas
    1) Cuarentena total 14 movmax = 1 mov_rem = 0.9
    2) Cuarentena total 14 movmax = 1 mov_rem = 0.7
    3) Cuarentena total 14 movmax = 1 mov_rem = 0.5
    4) Cuarentena total 28 movmax = 1 mov_rem = 0.9
    5) Cuarentena total 28 movmax = 1 mov_rem = 0.7
    6) Cuarentena total 28 movmax = 1 mov_rem = 0.5

    Optimistas
    7) Cuarentena total 28 movmax = 0.5 mov_rem = 0.2 + din 14p d200
    8) Cuarentena total 28 movmax = 0.5 mov_rem = 0.2 + din 28p d200
    9) Cuarentena total 40 movmax = 0.5 mov_rem = 0.2 + din 14p d200
    10) Cuarentena total 40 movmax = 0.5 mov_rem = 0.2 + din 28p d200

    Realistas:
    11) Cuarentena total 28 movmax = 0.75 mov_rem = 0.3 + din 14p d200
    12) Cuarentena total 28 movmax = 0.75 mov_rem = 0.3 + din 28p d200
    13) Cuarentena total 40 movmax = 0.75 mov_rem = 0.3 + din 14p d200
    14) Cuarentena total 40 movmax = 0.75 mov_rem = 0.3 + din 28p d200


    """
    # tsim,max_mov,rem_mov,qp,tci,dtc)  (movfunct)
    input=np.array([
            [tsim,0.85,0.6,14,14,500],
            [tsim,0.85,0.7,14,14,500],
            [tsim,0.85,0.8,14,14,500],
            [tsim,0.85,0.6,14,21,500],
            [tsim,0.85,0.7,14,21,500],
            [tsim,0.85,0.8,14,21,500],
            [tsim,0.85,0.6,14,60,500],
            [tsim,0.85,0.6,21,60,500],
            [tsim,0.55,0.2,14,14,500],
            [tsim,0.55,0.3,14,14,500],
            [tsim,0.55,0.4,14,14,500],
            [tsim,0.55,0.2,14,21,500],
            [tsim,0.55,0.3,14,21,500],
            [tsim,0.55,0.4,14,21,500],        
            [tsim,0.55,0.2,14,60,500],
            [tsim,0.55,0.2,21,60,500]])
    inputmovfunct= ['once','once','once','once','once','once','square','square','once','once','once','once','once','once','square','square',]

    num_cores = multiprocessing.cpu_count()
    #params=Parallel(n_jobs=num_cores, verbose=50)(delayed(ref_test.refinepso_all)(Ir,tr,swarmsize=200,maxiter=50,omega=0.5, phip=0.5, phig=0.5,eta_r=[0,1],Q_r=[0,1],obj_func='IN')for i in range(int(rep)))
    sims=Parallel(n_jobs=num_cores, verbose=50)(delayed(sim_run)(input[i,0],input[i,1],input[i,2],input[i,3],input[i,4],input[i,5],inputmovfunct[i])for i in range(input.shape[0]))
    return(sims)

# -------------- #
#    Simulate    #
# -------------- #
beta = 0.19
mu = 2.6
sims = sim(beta,mu)



# Como incorporar los que se van removiendo: 

# Removerlos desde el 1 de Mayo
#

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
T=sims[i][0].S+sims[i][0].E_as+sims[i][0].E_sy+sims[i][0].I_as+sims[i][0].I_cr+sims[i][0].I_mi+sims[i][0].I_se\
    +sims[i][0].H_in+sims[i][0].H_out+sims[i][0].H_cr+sims[i][0].V+sims[i][0].D+sims[i][0].R+sims[i][0].B


# Susceptibles
S = [sims[i][0].S for i in range(len(input))]
# Hospitalizados totales diarios
H_sum=[sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V for i in range(len(input))] 
# Hospitalizados camas diarios
H_bed=[sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out for i in range(len(input))] 
# Hospitalizados ventiladores diarios
H_vent=[sims[i][0].V for i in range(len(input))] 
# Infectados Acumulados
Iac=[sims[i][0].I+Iac15M-sims[i][0].I[0] for i in range(len(input))] 
# Infectados activos diarios
I = [sims[i][0].I_as+sims[i][0].I_cr + sims[i][0].I_mi + sims[i][0].I_se + sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V for i in range(len(input))] 
I_act = [sims[i][0].I_mi + sims[i][0].I_cr + sims[i][0].I_se + sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V for i in range(len(input))] 


# Infectados asintomaticos
I_as = [sims[i][0].I_as for i in range(len(input))] 
# Infectados mild
I_mi = [sims[i][0].I_mi for i in range(len(input))] 
# Infectados severos
I_se = [sims[i][0].I_se for i in range(len(input))] 
# Infectados criticos
I_cr = [sims[i][0].I_cr for i in range(len(input))] 


# Expuestos totales diarios
E = [sims[i][0].E_as+sims[i][0].E_sy for i in range(len(input))] 
# Enterrados/Muertos acumulados
B = [sims[i][0].B for i in range(len(input))] 
# Muertos diarios
D = [sims[i][0].D for i in range(len(input))] 
# Recuperados
R = [sims[i][0].R for i in range(len(input))] 
# Ventiladores diarios
V = [sims[i][0].V for i in range(len(input))] 

# Variables temporales
t = [sims[i][0].t for i in range(len(input))] 
dt = [np.diff(t[i]) for i in range(len(input))] 
tr = range(tsim)
idx = [np.searchsorted(t[i],tr) for i in range(len(input))] 





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
H_crin=[sims[i][0].H_cr for i in range(len(input))] 
H_in=[sims[i][0].H_in for i in range(len(input))] 
H_out=[sims[i][0].H_out for i in range(len(input))] 
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
#H_sum=[sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V for i in range(len(input))] 
H_crin=[sims[i][0].H_cr for i in range(len(input))] 
H_in=[sims[i][0].H_in for i in range(len(input))] 
H_out=[sims[i][0].H_out for i in range(len(input))] 
#H_vent=[sims[i][0].V for i in range(len(input))] 

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
idx = [np.searchsorted(t[i],tr) for i in range(len(input))] 
endD = idx[i][-1]
CH = [sims[i][0].CH for i in range(len(input))]
CV = [sims[i][0].CV for i in range(len(input))]
ACH = [sims[i][0].ACH for i in range(len(input))]
ACV = [sims[i][0].ACV for i in range(len(input))]

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
Pdiag = 0.4
I_act = [Pdiag*sims[i][0].I_mi + sims[i][0].I_cr + sims[i][0].I_se + sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V for i in range(len(input))] 
datestxt = data['Fechas'][32:]
dates = list(range(15,15+len(datestxt)))
datessim = list(range(15,15+len(datestxt)+14))

Ir_hoy = Ir[32:]
d15M = len(Ir[32:])
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
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='red',linestyle='dotted',linewidth = 3.0)
i=1
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='red',linestyle='solid',linewidth = 3.0)
i=2
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='red',linestyle='dotted',linewidth = 3.0)
i=3
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='blue',linestyle='dotted',linewidth = 3.0)
i=4
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='blue',linestyle='solid',linewidth = 3.0)
i=5
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='blue',linestyle='dotted',linewidth = 3.0)
i=6
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4])+'SQ='+str(input[i][3]),color='lime',linestyle='solid',linewidth = 3.0)
i=7
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4])+'SQ='+str(input[i][3]),color='black',linestyle='solid',linewidth = 3.0)

plot(title='Letalidad Realista',ylabel='Letalidad')

i=8
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='blue',linestyle='dashed')
i=9
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='blue',linestyle='solid')
i=10
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='blue',linestyle='dotted')
i=11
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='red',linestyle='dashed')
i=12
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='red',linestyle='solid')
i=13
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='red',linestyle='dotted')
i=14
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4])+'SQ='+str(input[i][3]),color='lime',linestyle='solid')
i=15
plt.plot(t[i],100*B[i]/Iac[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4])+'SQ='+str(input[i][3]),color='aqua',linestyle='solid')

plot(title='Letalidad Optimista',ylabel='Letalidad %')


# ----------------------------- #
#     Estudio de Mortalidad     #
# ----------------------------- #
i=0
plt.plot(t[i],B[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='blue',linestyle='dashed')
i=1
plt.plot(t[i],B[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='blue',linestyle='solid')
i=2
plt.plot(t[i],B[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='blue',linestyle='dotted')
i=3
plt.plot(t[i],B[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='red',linestyle='dashed')
i=4
plt.plot(t[i],B[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='red',linestyle='solid')
i=5
plt.plot(t[i],B[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='red',linestyle='dotted')
i=6
plt.plot(t[i],B[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4])+'SQ='+str(input[i][3]),color='lime',linestyle='solid')
i=7
plt.plot(t[i],B[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4])+'SQ='+str(input[i][3]),color='aqua',linestyle='solid')

plot(title='Mortalidad Realista',ylabel='Mortalidad')

i=8
plt.plot(t[i],B[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='blue',linestyle='dashed')
i=9
plt.plot(t[i],B[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='blue',linestyle='solid')
i=10
plt.plot(t[i],B[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='blue',linestyle='dotted')
i=11
plt.plot(t[i],B[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='red',linestyle='dashed')
i=12
plt.plot(t[i],B[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='red',linestyle='solid')
i=13
plt.plot(t[i],B[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='red',linestyle='dotted')
i=14
plt.plot(t[i],B[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4])+'SQ='+str(input[i][3]),color='lime',linestyle='solid')
i=15
plt.plot(t[i],B[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4])+'SQ='+str(input[i][3]),color='aqua',linestyle='solid')

plot(title='Mortalidad Optimista',ylabel='Mortalidad')




plt.plot(sims[i][0].t,sims[i][0].B,label='Buried')    
plt.plot(sims[i][0].t,sims[i][0].B,label='Buried')
plot('Mortalidad')




# ------------------------------- #
#      Estudio Hospitalizados     #
# ------------------------------- #
#H_sum=[sims[i][0].H_in+sims[i][0].H_cr+sims[i][0].H_out+sims[i][0].V for i in range(len(input))] 
i=0
plt.plot(t[i],H_sum[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='blue',linestyle='dashed')
i=1
plt.plot(t[i],H_sum[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='blue',linestyle='solid')
i=2
plt.plot(t[i],H_sum[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='blue',linestyle='dotted')
i=3
plt.plot(t[i],H_sum[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='red',linestyle='dashed')
i=4
plt.plot(t[i],H_sum[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='red',linestyle='solid')
i=5
plt.plot(t[i],H_sum[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='red',linestyle='dotted')
i=6
plt.plot(t[i],H_sum[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4])+'SQ='+str(input[i][3]),color='lime',linestyle='solid')
i=7
plt.plot(t[i],H_sum[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4])+'SQ='+str(input[i][3]),color='aqua',linestyle='solid')

plot(title='Hospitalizados Realista',ylabel='Letalidad')

i=8
plt.plot(t[i],H_sum[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='blue',linestyle='dashed')
i=9
plt.plot(t[i],H_sum[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='blue',linestyle='solid')
i=10
plt.plot(t[i],H_sum[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='blue',linestyle='dotted')
i=11
plt.plot(t[i],H_sum[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='red',linestyle='dashed')
i=12
plt.plot(t[i],H_sum[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='red',linestyle='solid')
i=13
plt.plot(t[i],H_sum[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4]),color='red',linestyle='dotted')
i=14
plt.plot(t[i],H_sum[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4])+'SQ='+str(input[i][3]),color='lime',linestyle='solid')
i=15
plt.plot(t[i],H_sum[i],label='Mov=['+str(input[i][2])+','+str(input[i][1])+'] Qi='+str(input[i][4])+'SQ='+str(input[i][3]),color='aqua',linestyle='solid')

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

def plot(title='',xlabel='dias',ylabel='Personas'):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=0)
    plt.show()




















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