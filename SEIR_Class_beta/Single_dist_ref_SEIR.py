#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import json
import requests
from pyswarm import pso
import datetime as dt
from scikits.odes.odeint import odeint
from numpy import linalg as LA
import multiprocessing
from joblib import Parallel, delayed
from scipy import signal


#endpoint = "http://192.168.2.220:8080/covid19/selectComunas"
#r = requests.get(endpoint) #, params = {"w":"774508"})
#mydict = r.json()
#comunas = pd.DataFrame(mydict)


def intger(Si,Ei,Ii,Ri,ti,T,h,beta,sigma,gamma,mov,qp=0,tci=None, movfunct = 'sawtooth'):
    # Movility function shape
    if movfunct == 'sawtooth':
        def f(t): 
            return signal.sawtooth(t)
    elif movfunct == 'square':
        def f(t):
            return signal.square(t)
    
    if qp <=0:
        def e_fun(t):
            return(1)
    
    elif tci is None:        
        def e_fun(t):
            return((1-mov)/2*(f(np.pi / qp * t - np.pi))+(1+mov)/2)
    else:
        def e_fun(t):
                if t<tci:
                    return(1)
                else:
                    return((1-mov)/2*(f(np.pi / qp * t - np.pi))+(1+mov)/2)
    
    def model_SEIR(t,y,ydot):
        S0 = y[0]
        E0 = y[1]
        I0 = y[2]
        R0 = y[3]
        N=S0+E0+I0+R0
        ydot[0] = -beta * I0 * S0 /N * e_fun(t)
        ydot[1] = beta * I0 * S0 /N * e_fun(t) - sigma * E0 
        ydot[2] = sigma * E0  - (gamma * I0)
        ydot[3] = gamma * I0 
       
    
    t=np.arange(ti,T+h,h) 
    y0 = [Si, Ei, Ii, Ri]          
    
    soln = odeint(model_SEIR, t, y0)
    t=soln[1][0] 
    soln=soln[1][1]
    S = soln[:, 0]
    E = soln[:, 1]
    I = soln[:, 2]
    R = soln[:, 3]

    # Return info just per day
    tout = range(int(T))
    idx=np.searchsorted(t,tout)
    Sout = S[idx]
    Eout = E[idx]
    Iout = I[idx]
    Rout = R[idx]
    
    return({"S":Sout,"E":Eout,"I":Iout,"R":Rout,"t":tout})
   

def objective_funct(Ir,tr,I,t):
    idx=np.searchsorted(t,tr)
    return LA.norm(Ir-I[idx])


def simulate(state,comuna,beta,sigma,gamma,mu,qp=0,mov=0.2,tsim=300,tci=None,movfunct='sawtooth'):
    #print(state)
    #print(comuna)    
    endpoint="http://192.168.2.220:8080/covid19/findComunaByIdState?idState="+state+"&&comuna="+comuna
    r = requests.get(endpoint) #, params = {"w":"774508"})
    mydict = r.json()
    info=pd.DataFrame(mydict)
    
    endpoint="http://192.168.2.220:8080/covid19/getDatosMinSalSummary?state="+state+"&comuna="+comuna
    r = requests.get(endpoint) #, params = {"w":"774508"})
    mydict = r.json()
    data=pd.DataFrame(mydict)
    data=data[data.data != 0]
    data=data.reset_index()    
    if not data.data.any():
        return

    tr=np.zeros(len(data.data))
    for i in range(1,len(data.data),1):
        diff=dt.datetime.strptime(data.labels[i], '%d/%m')-dt.datetime.strptime(data.labels[i-1], '%d/%m')
        tr[i]=diff.days+tr[i-1]
    
    if(len(tr)==1):
        return
    
    Cn=np.zeros(data.data.shape[0])
    Cn[0]=data.data.iloc[0]
    for i in range(1,len(Cn),1):
        Cn[i]=data.data[i]-data.data[i-1]
    
    Ir=np.zeros(data.data.shape[0])
    Ir[np.where(tr-14<=0)]=data.data.iloc[np.where(tr-14<=0)]
    for i in np.where(tr-14>0)[0]:
        ind=np.where(tr-tr[i]+14<0)[0]
        Ir[i]=data.data[i]-sum(Cn[0:ind[-1]])
    
    
    S0 = info[info['cut']==comuna].numPopulation.iloc[0]
    I0 = Ir[0]
    R0 = 0
    h=0.01
    #Tfinal = 300 #1.2*max(tr)
    b_date=(dt.datetime.strptime(data.labels.loc[0], '%d/%m') + dt.timedelta(days=43830)).strftime("%d-%b-%Y")
    result = intger(S0,mu*I0,I0,R0,min(tr),tsim,h,beta,sigma,gamma,mov,qp,tr[-1],movfunct)
    result['init_date'] = b_date
    return(result)


def ref_sim_all(state,comuna,mov=0.2,qp=0,tsim = 300,tci=None,movfunct='sawtooth'):
    # Region, comuna, movilidad durante cuarentena, periodo cuarenetena, tiempo simulacion, tiempo inicial cuarentena
 
    # Total number of inhabitants
    endpoint="http://192.168.2.220:8080/covid19/findComunaByIdState?idState="+state+"&&comuna="+comuna
    r = requests.get(endpoint) #, params = {"w":"774508"})
    mydict = r.json()
    info=pd.DataFrame(mydict)
    
    # Total number of infected people
    endpoint="http://192.168.2.220:8080/covid19/getDatosMinSalSummary?state="+state+"&comuna="+comuna
    r = requests.get(endpoint) #, params = {"w":"774508"})
    mydict = r.json()
    data=pd.DataFrame(mydict)
    data=data[data.data != 0]
    data=data.reset_index()
    if not data.data.any():
        return

    tr=np.zeros(len(data.data))
    for i in range(1,len(data.data),1):
        diff=dt.datetime.strptime(data.labels[i], '%d/%m')-dt.datetime.strptime(data.labels[i-1], '%d/%m')
        tr[i]=diff.days+tr[i-1]
    
    if(len(tr)==1):
        return
    
    Cn=np.zeros(data.data.shape[0])
    Cn[0]=data.data.iloc[0]
    for i in range(1,len(Cn),1):
        Cn[i]=data.data[i]-data.data[i-1]
    
    Ir=np.zeros(data.data.shape[0])
    Ir[np.where(tr-14<=0)]=data.data.iloc[np.where(tr-14<=0)]
    for i in np.where(tr-14>0)[0]:
        ind=np.where(tr-tr[i]+14<0)[0]
        Ir[i]=data.data[i]-sum(Cn[0:ind[-1]])
    
    
    S0 = info[info['cut']==comuna].numPopulation.iloc[0]
    I0 = Ir[0]
    R0 = 0
    h=0.01
    
    lb=[0.01,0.1,0.05,1.5]
    ub=[3.5,0.3,0.1,5.5]
    
    def opti(x):
        E0=0
        E0=x[3]*I0
        sol=pd.DataFrame(intger(S0,E0,I0,R0,min(tr),max(tr),h,x[0],x[1],x[2],mov,qp,tr[-1],movfunct))
        return(objective_funct(Ir,tr,sol.I,sol.t))
        
    
    xopt, fopt = pso(opti, lb, ub, minfunc=1e-8, omega=0.5, phip=0.5, phig=0.5,swarmsize=100,maxiter=50)
    #print('rel_error '+str(fopt/LA.norm(Ir)))
    print('error '+str(fopt))
    sim=intger(S0,xopt[3]*I0,I0,R0,min(tr),tsim,0.01,xopt[0],xopt[1],xopt[2],mov,qp,tr[-1],movfunct)
    b_date=dt.datetime.strptime(data.labels.loc[0], '%d/%m')
    return({'Ir':Ir,'tr':tr, 'params':xopt, 'err':fopt,'sim':sim, 'init_date':b_date})
    # I reales t real, parametros optimos, error, diccionario con resultado simulacion, fecha primer contagiado



def ref_sim_national(mov=0.2,qp=0,tsim = 300,tci=None,movfunct='sawtooth'):
    # Region, comuna, movilidad durante cuarentena, periodo cuarenetena, tiempo simulacion, tiempo inicial cuarentena
 
    # Total number of inhabitants
    endpoint="http://192.168.2.220:8080/covid19/findComunaByIdState?idState=""&&comuna="
    r = requests.get(endpoint) #, params = {"w":"774508"})
    mydict = r.json()
    info=pd.DataFrame(mydict)
    
    # Total number of infected people
    endpoint="http://192.168.2.220:8080/covid19/getDatosMinSalSummary?state=""&comuna="
    r = requests.get(endpoint) #, params = {"w":"774508"})
    mydict = r.json()
    data=pd.DataFrame(mydict)
    data=data[data.data != 0]
    data=data.reset_index()
    if not data.data.any():
        return

    tr=np.zeros(len(data.data))
    for i in range(1,len(data.data),1):
        diff=dt.datetime.strptime(data.labels[i], '%d/%m')-dt.datetime.strptime(data.labels[i-1], '%d/%m')
        tr[i]=diff.days+tr[i-1]
    
    if(len(tr)==1):
        return
    
    Cn=np.zeros(data.data.shape[0])
    Cn[0]=data.data.iloc[0]
    for i in range(1,len(Cn),1):
        Cn[i]=data.data[i]-data.data[i-1]
    
    Ir=np.zeros(data.data.shape[0])
    Ir[np.where(tr-14<=0)]=data.data.iloc[np.where(tr-14<=0)]
    for i in np.where(tr-14>0)[0]:
        ind=np.where(tr-tr[i]+14<0)[0]
        Ir[i]=data.data[i]-sum(Cn[0:ind[-1]])
    
    
    S0 = np.sum(info.numPopulation)
    I0 = Ir[0]
    R0 = 0
    h=0.01
    
    lb=[0.01,0.1,0.05,1.5]
    ub=[3.5,0.3,0.1,5.5]
    
    def opti(x):
        E0=0
        E0=x[3]*I0
        sol=pd.DataFrame(intger(S0,E0,I0,R0,min(tr),max(tr),h,x[0],x[1],x[2],mov,qp,tr[-1],movfunct))
        return(objective_funct(Ir,tr,sol.I,sol.t))
        
    
    xopt, fopt = pso(opti, lb, ub, minfunc=1e-8, omega=0.5, phip=0.5, phig=0.5,swarmsize=200,maxiter=100)
    #print('rel_error '+str(fopt/LA.norm(Ir)))
    print('error '+str(fopt))
    sim=intger(S0,xopt[3]*I0,I0,R0,min(tr),tsim,0.01,xopt[0],xopt[1],xopt[2],mov,qp,tr[-1],movfunct)
    b_date=dt.datetime.strptime(data.labels.loc[0], '%d/%m')
    return({'Ir':Ir,'tr':tr, 'params':xopt, 'err':fopt,'sim':sim, 'init_date':b_date})
    # I reales t real, parametros optimos, error, diccionario con resultado simulacion, fecha primer contagiado

