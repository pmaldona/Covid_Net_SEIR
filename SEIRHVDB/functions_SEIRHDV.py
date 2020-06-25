#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEIRHVDB Class Initialization
"""
import class_SEIRHUVD3 as SD3
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
import dill as pickle

"""
To Do:
    - Permitir ejecución remota desde el cluster:
        1. Crear funciones en el backend que llamen a las funciones del simulador. Deben pasar:        
            - parámetros
            - Cuarentenas
            - Opcional de tasas

    - Crear Objeto de Cuarentenas
        - Simplificar la creación de cuarentenas hiperdinámicas
        - Facilitar la creación de escenarios
        - Funcion de grafico de cuarentenas
        - Cambiar nombre de inputarray
    - Habilitar el uso del código para otras regiones
    - Extender importación de datos Sochimi a otras regiones y otras unidades territoriales según corresponda
     - Arreglar el ingreso del modelo de ingreso de capacidad total de camas para que sea una funcion
     - Resolver como se haría el ploteo para varios escenarios
     - Generar automaticamente nombres de columnas en las tablas

     - Generar Vector de estilos
     - Agregar comparacion fallecidos diarios vs reales
     - Agregar opción de trabajar con comunas
     - reducir parametros enviados al inicio a los necesarios 
     - Enviar la funcion de aproximacion de camas y ventiladores en vez de los parametros
     - Agregar opción de qué seirhudv usar (2 o 3)

     - Terminar funcion de remote simulatew
     - Ordenar funcion de estimacion de uso y capacidad de camas y ventiladores
     - Homologar las funciones actuaizadas en remote y local SEIRHDV
     - Mirar convertir en arregplo el inputarray
     - Falta la función de camas en el remote

"""



""" 
# ------------------------------------ #
#      Inicializacion de Variables     #
# ------------------------------------ #
"""
class SEIRHVD_DA:
        
    def __init__(self,beta,mu,ScaleFactor=1,SeroPrevFactor=1,expinfection=1,initdate = datetime(2020,5,15), tsim = 500,tstate='13'):
        self.beta = beta
        self.mu = mu
        self.ScaleFactor = ScaleFactor
        self.SeroPrevFactor = SeroPrevFactor
        self.expinfection = expinfection
        self.tstate = tstate
        self.initdate = initdate
        self.tsim = tsim
        self.May15 = (datetime(2020,5,15)-initdate).days
        return



    """ 
    # ------------------------------- #
    #      Creación de Funciones      #
    # ------------------------------- #
    """

    #------------------------------------------------- #
    #              Definir Escenarios                  #
    #------------------------------------------------- #
    #tsim,max_mov,rem_mov,qp,iqt,fqt,movfunct
    def defaultescenarios(self):
        self.inputarray=np.array([
                [self.tsim,0.85,0.6,0,self.May15,500,0],
                [self.tsim,0.85,0.65,0,self.May15,500,0],
                [self.tsim,0.85,0.7,0,self.May15,500,0]])        
        self.numescenarios = len(self.inputarray)

    def addscenario(self,tsim=None,max_mov=None,rem_mov=None,qp=None,iqt=None,fqt=None,movfunct=None):        
        if tsim:
            self.inputarray.append([tsim,max_mov,rem_mov,qp,iqt,fqt,movfunct])
        self.numescenarios= len(self.inputarray)
        return()

    # traspasarlo a lenguaje humao
    def showscenarios(self):        
        print(self.inputarray)
        return()

    # -------------- #
    #    Simulate    #
    # -------------- #

    def simulate(self,v=2):                   
        dead = 'minsal'
        # Valores iniciales
        if dead=='minsal':        
            self.B = self.Br[0]  # Muertos acumulados al dia de inicio
            self.D = self.Br[1]-self.Br[0]  # Muertos en el dia de inicio
        elif dead == 'exceso':
            self.B = self.ED_RM_ac[0]
            self.D = self.ED_RM_ac[1] - self.ED_RM_ac[0]


        I_act0 = self.ScaleFactor*self.Ir[0]  # Infectados Activos al dia de inicio
        cId0 = self.ScaleFactor*(self.Ir[1]-self.Ir[0])# cId0 # I Nuevos dia 0
        self.R  = 0   # Recuperados al dia de inicio


        # Dinámica del aumento de camas
        self.V = self.Vr[0]   # Ventilados al dia de inicio
        H_cr = 0#Vr[0]*0.01 # Hospitalizados criticos dia 0
        H0 = self.Hr[0] # Hospitalizados totales dia 0

        bedmodelorder = 2
        self.Hcmodel = np.poly1d(np.polyfit(self.sochimi_tr, self.Hr_tot, bedmodelorder)) 
        self.Vcmodel = np.poly1d(np.polyfit(self.sochimi_tr, self.Vr_tot, bedmodelorder)) 
        Hc0=self.Hcmodel[0]#1980
        H_incr=self.Hcmodel[1]
        H_incr2 = self.Hcmodel[2]
        H_incr3 = self.Hcmodel[3]
        Vc0=self.Vcmodel[0]#1029
        V_incr = self.Vcmodel[1]
        V_incr2 = self.Vcmodel[2]
        V_incr3 = self.Vcmodel[3] 
        Vmax = np.mean(self.Vr_tot[-7:])*1.01# 1500 Vcmodel(tsat)
        vents = [self.Vcmodel(t) for t in range(self.tsim)]          
        Hmax = np.mean(self.Hr_tot[-7:])*1.01# self.Hcmodel(tsat)

        try:
            tsat = int(np.where(np.array(vents)>=Vmax)[0][0])
        except:
            tsat = self.sochimi_tr[-1]+7
        if v==2:
            model = SD2.simSEIRHVD(beta = self.beta, mu = self.mu, inputarray= self.inputarray, B=self.B,D=self.D,V=self.V,I_act0=I_act0,cId0=cId0,R=self.R,Hc0=Hc0,H_incr=H_incr,H_incr2=H_incr2,H_incr3=H_incr3,Vc0=Vc0,V_incr=V_incr,V_incr2=V_incr2,V_incr3=V_incr3,H_cr=H_cr,H0=H0,tsat=tsat,Hmax=Hmax,Vmax=Vmax, expinfection=self.expinfection, SeroPrevFactor= self.SeroPrevFactor)
        elif v==3:
            model = SD3.simSEIRHVD(beta = self.beta, mu = self.mu, inputarray= self.inputarray, B=self.B,D=self.D,V=self.V,I_act0=I_act0,cId0=cId0,R=self.R,Hc0=Hc0,H_incr=H_incr,H_incr2=H_incr2,H_incr3=H_incr3,Vc0=Vc0,V_incr=V_incr,V_incr2=V_incr2,V_incr3=V_incr3,H_cr=H_cr,H0=H0,tsat=tsat,Hmax=Hmax,Vmax=Vmax, expinfection=self.expinfection, SeroPrevFactor= self.SeroPrevFactor)    
        else:
            raise('Version Error')
        self.sims = model.simulate()
        self.auxvar()
        return

    def simulate_remote(self):
        endpoint = 'http://192.168.2.248:5003/SEIRHVDsimulate'
        #auxinputarray = [list(self.inputarray[i]) for i in range(self.numescenarios)]
        data = {
        'state': str(self.tstate),
        'beta': str(self.beta),
        'mu': str(self.mu),        
        'initdate': self.initdate.strftime('%Y/%m/%d'),
        'ScaleFactor': str(self.ScaleFactor),
        'SeroPrevFactor': str(self.SeroPrevFactor),
        'inputarray': str(inputarray)}
        #,
        #'tsim': str(self.inputarray[0]),
        #'qp': str(self.inputarray[3]),
        #'min_mov': str(self.inputarray[2]),
        #'max_mov': str(self.inputarray[1]),
        #'movfunct': str(self.inputarray[0]),
        #'qit': str(0),
        #'qft': str(100)}

        r = requests.post(url = endpoint, data = data)

        self.sims = pickle.loads(r.content)                    


        bedmodelorder = 2
        self.Hcmodel = np.poly1d(np.polyfit(self.sochimi_tr, self.Hr_tot, bedmodelorder)) 
        self.Vcmodel = np.poly1d(np.polyfit(self.sochimi_tr, self.Vr_tot, bedmodelorder)) 
        Hc0=self.Hcmodel[0]#1980
        H_incr=self.Hcmodel[1]
        H_incr2 = self.Hcmodel[2]
        H_incr3 = self.Hcmodel[3]
        Vc0=self.Vcmodel[0]#1029
        V_incr = self.Vcmodel[1]
        V_incr2 = self.Vcmodel[2]
        V_incr3 = self.Vcmodel[3] 
        Vmax = 1500# Vcmodel(tsat)
        vents = [self.Vcmodel(t) for t in range(self.tsim)]  
        tsat = int(np.where(np.array(vents)>=1500)[0][0])
        Hmax = self.Hcmodel(tsat)
                        
        #model = SD2.simSEIRHVD(beta = self.beta, mu = self.mu, inputarray= self.inputarray, B=self.B,D=self.D,V=self.V,I_act0=I_act0,cId0=cId0,R=self.R,Hc0=Hc0,H_incr=H_incr,H_incr2=H_incr2,H_incr3=H_incr3,Vc0=Vc0,V_incr=V_incr,V_incr2=V_incr2,V_incr3=V_incr3,H_cr=H_cr,H0=H0,tsat=tsat,Hmax=Hmax,Vmax=Vmax, expinfection=self.expinfection, SeroPrevFactor= self.SeroPrevFactor)
        self.sims = model.simulate()
        self.auxvar()
        return

    # ------------------------------- #
    #        Importar Data Real       #
    # ------------------------------- #

    def importinfectadosactivos(self):
        # ---------------------- # 
        #   Infectados Activos   #
        # ---------------------- #
        cutlist = []
        cutlistpath = "../Data/cutlist.csv"
        cutlist = pd.read_csv(cutlistpath, header = None,dtype=str)

        actives = []
        mydict = None
        for index, row in cutlist.iterrows():    
            state = str(row[0])[0:2]
            comuna = str(row[0])
            if self.tstate == state:
                endpoint = "http://192.168.2.223:5006/getActiveNewCasesByComuna?comuna="+comuna
                r = requests.get(endpoint) 
                mydict = r.json()
                actives.append(mydict['actives'])
                #data=pd.DataFrame(mydict)
        self.Ir = (np.array(actives)).sum(axis=0)
        self.Ir_dates = [datetime.strptime(mydict['dates'][i],'%Y-%m-%d') for i in range(len(mydict['dates']))]

        index = np.where(np.array(self.Ir_dates) >= self.initdate)[0][0]     
        self.Ir=self.Ir[index:]
        self.Ir_dates=self.Ir_dates[index:]
        self.tr = [(self.Ir_dates[i]-self.initdate).days for i in range(len(self.Ir))]
        print('Infectados Activos')
        return

    def importsochimi(self,endpoint = "http://192.168.2.223:5006/getBedsAndVentilationByState?state="):
        # ------------------ #
        #    Datos Sochimi   #
        # ------------------ #
        endpoint = endpoint+self.tstate
        r = requests.get(endpoint) 
        mydict = r.json()
        self.sochimi=pd.DataFrame(mydict)
        sochimi = self.sochimi
        self.Hr = sochimi['camas_ocupadas']
        self.Vr =  sochimi['vmi_ocupados']
        self.Vr_tot =  sochimi['vmi_totales']
        self.Hr_tot =  sochimi['camas_totales']
        self.sochimi_dates = [datetime.strptime(sochimi['dates'][i][:10],'%Y-%m-%d') for i in range(len(sochimi))]

        index = np.where(np.array(self.sochimi_dates) >= self.initdate)[0][0] 
        self.Hr=list(self.Hr[index:])
        self.Vr=list(self.Vr[index:])
        self.Hr_tot=list(self.Hr_tot[index:])
        self.Vr_tot=(list(self.Vr_tot[index:]))
        self.sochimi_dates = self.sochimi_dates[index:]
        self.sochimi_tr = [(self.sochimi_dates[i]-self.initdate).days for i in range(len(self.Hr))]
        print('Sochimi')
        return(sochimi)

    # -------------------------------- #
    #    Datos Fallecidos acumulados   #
    # -------------------------------- #
    def importfallecidosacumuados(self,endpoint = 'https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto14/FallecidosCumulativo.csv' ):     
        cut =  ['15','01','02','03','04','05','13','06','07','16','08','09','14','10','11','12','00']
        index = cut.index(self.tstate)
        self.Br = pd.read_csv(endpoint).iloc[index][1:] 
        self.Br_dates = [datetime.strptime(self.Br.index[i],'%Y-%m-%d') for i in range(len(self.Br))]
        index = np.where(np.array(self.Br_dates) >= self.initdate)[0][0] 
        self.Br = self.Br[index:]
        self.Br_dates = self.Br_dates[index:]
        self.Br_tr = [(self.Br_dates[i]-self.initdate).days for i in range(len(self.Br))]
        print('Fallecidos Acumulados')
        return


    # -------------------------- #
    #    Fallecidos excesivos    #
    # -------------------------- #
    def importfallecidosexcesivos(self,path = '/home/samuel/Documents/Dlab/data/Excess_dead_daily.csv'):
        #path = '/home/samuel/Documents/Dlab/data/Excess_dead_daily.csv'
        
        excess_dead = pd.read_csv(path)
        self.ED_RM_df = excess_dead.loc[excess_dead['Codigo region']==int(self.tstate)]      
        self.ED_RM = [self.ED_RM_df['Defunciones Covid'].iloc[i] + self.ED_RM_df['Exceso de muertes media poderada'].iloc[i] for i in range(len(self.ED_RM_df))]       

        self.ED_RM_dates = [datetime.strptime(self.ED_RM_df['Fecha'].iloc[i], '%Y-%m-%d')  for i in range(len(self.ED_RM_df))]
        index = np.where(np.array(self.ED_RM_dates) >= self.initdate)[0][0]
        enddate = max(self.ED_RM_dates)
        indexend = np.where(np.array(self.ED_RM_dates) >= enddate)[0][0]
        self.ED_RM_dates = self.ED_RM_dates[index:indexend]  
        self.ED_RM = self.ED_RM[index:indexend]
        self.ED_RM_ac = np.cumsum(self.ED_RM)
        self.ED_tr = [(self.ED_RM_dates[i]-self.initdate).days for i in range(len(self.ED_RM))]
        print('Fallecidos Excesivos')
        return

    # --------------------------- #
    #    Importar toda la data    #
    # --------------------------- #

    def importdata(self):
        print('Importando Datos')
        self.importfallecidosacumuados()
        self.importfallecidosexcesivos()
        self.importinfectadosactivos()
        self.importsochimi()
        print('Done')



    #-------------------------------- #
    #       Variables auxiliares      #
    #-------------------------------- #
    # Creacion de variables auxiliares para los analisis

    def auxvar(self):
        # Poblacion total
        self.T=[self.sims[i][0].S+self.sims[i][0].E_as+self.sims[i][0].E_sy+self.sims[i][0].I_as+self.sims[i][0].I_cr+self.sims[i][0].I_mi+self.sims[i][0].I_se\
            +self.sims[i][0].H_in+self.sims[i][0].H_out+self.sims[i][0].H_cr+self.sims[i][0].V+self.sims[i][0].D+self.sims[i][0].R+self.sims[i][0].B for i in range(self.numescenarios)]


        # Susceptibles
        self.S = [self.sims[i][0].S for i in range(self.numescenarios)]
        # Hospitalizados totales diarios
        self.H_sum=[self.sims[i][0].H_in+self.sims[i][0].H_cr+self.sims[i][0].H_out+self.sims[i][0].V for i in range(self.numescenarios)] 
        # Hospitalizados camas diarios
        self.H_bed=[self.sims[i][0].H_in+self.sims[i][0].H_cr+self.sims[i][0].H_out for i in range(self.numescenarios)] 
        # Hospitalizados ventiladores diarios
        self.H_vent=[self.sims[i][0].V for i in range(self.numescenarios)] 
        # Infectados Acumulados
        self.Iac=[self.sims[i][0].I for i in range(self.numescenarios)] 
        # Infectados activos diarios
        self.I = [self.sims[i][0].I_as+self.sims[i][0].I_cr + self.sims[i][0].I_mi + self.sims[i][0].I_se + self.sims[i][0].H_in+self.sims[i][0].H_cr+self.sims[i][0].H_out+self.sims[i][0].V for i in range(self.numescenarios)] 
        self.I_act = [self.sims[i][0].I_mi + self.sims[i][0].I_cr + self.sims[i][0].I_se + self.sims[i][0].H_in+self.sims[i][0].H_cr+self.sims[i][0].H_out+self.sims[i][0].V for i in range(self.numescenarios)] 


        # Infectados asintomaticos
        self.I_as = [self.sims[i][0].I_as for i in range(self.numescenarios)] 
        # Infectados mild
        self.I_mi = [self.sims[i][0].I_mi for i in range(self.numescenarios)] 
        # Infectados severos
        self.I_se = [self.sims[i][0].I_se for i in range(self.numescenarios)] 
        # Infectados criticos
        self.I_cr = [self.sims[i][0].I_cr for i in range(self.numescenarios)] 
        # suma de infectados "sueltos"
        self.I_sum = [self.sims[i][0].I_as+self.sims[i][0].I_cr + self.sims[i][0].I_mi + self.sims[i][0].I_se for i in range(self.numescenarios)] 


        # Expuestos totales diarios
        self.E = [self.sims[i][0].E_as+self.sims[i][0].E_sy for i in range(self.numescenarios)]
        self.E_as = [self.sims[i][0].E_as for i in range(self.numescenarios)]  
        self.E_sy = [self.sims[i][0].E_sy for i in range(self.numescenarios)]  
        # Enterrados/Muertos acumulados
        self.B = [self.sims[i][0].B for i in range(self.numescenarios)] 
        # Muertos diarios
        self.D = [self.sims[i][0].D for i in range(self.numescenarios)] 
        # Recuperados
        self.R = [self.sims[i][0].R for i in range(self.numescenarios)] 
        # Ventiladores diarios
        self.V = [self.sims[i][0].V for i in range(self.numescenarios)] 

        # Variables temporales
        self.t = [self.sims[i][0].t for i in range(self.numescenarios)] 
        self.dt = [np.diff(self.t[i]) for i in range(self.numescenarios)] 
        
        
        # CAMAS
        self.H_crin=[self.sims[i][0].H_cr for i in range(self.numescenarios)] 
        self.H_in=[self.sims[i][0].H_in for i in range(self.numescenarios)] 
        self.H_out=[self.sims[i][0].H_out for i in range(self.numescenarios)] 
        self.H_sum=[self.sims[i][0].H_in+self.sims[i][0].H_cr+self.sims[i][0].H_out for i in range(self.numescenarios)]
        self.H_tot=[self.sims[i][0].H_in+self.sims[i][0].H_cr+self.sims[i][0].H_out+self.sims[i][0].V  for i in range(self.numescenarios)]

        self.CH = [self.sims[i][0].CH for i in range(self.numescenarios)]
        self.CV = [self.sims[i][0].CV for i in range(self.numescenarios)]
        self.ACH = [self.sims[i][0].ACH for i in range(self.numescenarios)]
        self.ACV = [self.sims[i][0].ACV for i in range(self.numescenarios)]
        
        #Cálculo de la fecha del Peak  
        self.peakindex = [np.where(self.I[i]==max(self.I[i]))[0][0] for i in range((self.numescenarios))]
        self.peak = [max(self.I[i]) for i in range((self.numescenarios))]
        self.peak_t = [self.t[i][self.peakindex[i]] for i in range((self.numescenarios))]
        self.peak_date = [self.initdate+timedelta(days=round(self.peak_t[i])) for i in range((self.numescenarios))]

        #proporcion de la poblacion que entra en la dinamica de infeccion
        self.population = self.sims[0][0].pop
        self.infectedsusc = [100*((self.S[i][0] - self.S[i][-1])/self.S[i][0]) for i in range(self.numescenarios)] 
        self.infectedpop = [100*((self.S[i][0] - self.S[i][-1]))/self.population for i in range(self.numescenarios)] 

        # -------------- #
        #     Errores    #
        # -------------- #
        # Camas
        idx = [np.searchsorted(self.t[i],self.sochimi_tr) for i in range(self.numescenarios)]
        self.err_bed = [LA.norm(self.Hr-self.H_sum[i][idx[i]])/LA.norm(self.Hr) for i in range(self.numescenarios)]
        self.err_vent = [LA.norm(self.Vr-self.V[i][idx[i]])/LA.norm(self.Vr) for i in range(self.numescenarios)]  
        
        # Infecatos Activos
        idx = [np.searchsorted(self.t[i],self.tr) for i in range(self.numescenarios)]
        self.err_Iactives = [LA.norm(self.Ir-self.I[i][idx[i]])/LA.norm(self.Ir) for i in range(self.numescenarios)]    
        
        # Infectados acumulados
        #idx = [np.searchsorted(t[i],tr) for i in range(self.numescenarios)]
        #err_Iactives = [LA.norm(Ir-I[i][idx[i]])/LA.norm(Ir) for i in range(self.numescenarios)]    
        
        # Fallecidos
        idx = [np.searchsorted(self.t[i],self.Br_tr) for i in range(self.numescenarios)]
        self.err_dead = [LA.norm(self.Br-self.B[i][idx[i]])/LA.norm(self.Br) for i in range(self.numescenarios)]

        # -------------------- #
        #   Fecha de Colapso   #
        # -------------------- #
        try:
            self.H_colapsedate = [np.where(self.CH[i]>0)[0][0] for i in range(self.numescenarios)]
        except:
            self.H_colapsedate = [self.tsim for i in range(self.numescenarios)]
        try:
            self.V_colapsedate = [np.where(self.CV[i]>0)[0][0] for i in range(self.numescenarios)]
        except:
            self.V_colapsedate = [self.tsim for i in range(self.numescenarios)]



    """
        # ---------------------------------- #
        #          Estudio Resultados        #
        # ---------------------------------- # 
    """


    # -------------------------- #
    #        Plot function       #
    # -------------------------- #
    def plot(self,title = '',xlabel='',ylabel=''):
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
    def plothospitalizados(self,enddate =  datetime(2020,7,30)):
        # -------- #
        #   Time   #
        # -------- #
        days = (enddate-self.initdate).days
        endD = [np.searchsorted(self.t[i],days) for i in range(self.numescenarios)]

        for i in range(self.numescenarios):
            plt.plot(self.t[i][:endD[i]],self.H_in[i][:endD[i]],label='Hin',linestyle = 'solid')
            plt.plot(self.t[i][:endD[i]],self.H_out[i][:endD[i]],label='Hout',linestyle = 'solid')
            plt.plot(self.t[i][:endD[i]],self.H_crin[i][:endD[i]],label='Hcr_in',linestyle = 'solid')

        self.plot(title = 'Hospitalizados',xlabel='Dias desde '+datetime.strftime(self.initdate,'%Y-%m-%d'))



    # ------------------ #
    #     Ventiladores   #
    # ------------------ #
    def plotventiladores(self,enddate =  datetime(2020,7,30)):
        # -------- #
        #   Time   #
        # -------- # 
        days = (enddate-self.initdate).days
        endD = [np.searchsorted(self.t[i],days) for i in range(self.numescenarios)]
        idx = [np.searchsorted(self.t[i],self.sochimi_tr) for i in range(self.numescenarios)]

        # Inicio cuarentena general
        plt.axvline(x=self.May15,linestyle = 'dashed',color = 'grey')

        # Ploteo datos reales
        plt.scatter(self.sochimi_tr,self.Vr,label='Ventiladores Ocupados reales')
        plt.scatter(self.sochimi_tr,self.Vr_tot,label='Capacidad de Ventiladores')

        
        # Error y parámetros
        for i in range(self.numescenarios):        
            plt.plot([], [], ' ', label='err_vent: '+str(round(100*self.err_vent[i],2))+'%')
        plt.plot([], [], ' ', label='beta: '+str(self.beta))
        plt.plot([], [], ' ', label='mu: '+str(self.mu))
        plt.plot([], [], ' ', label='factor B-Y: '+str(self.ScaleFactor))

        # Fecha de peaks
        for i in range(self.numescenarios):        
            plt.plot([], [], ' ', label='Mov='+str(self.inputarray[i][2])+'Peak='+self.peak_date[i].strftime('%Y-%m-%d'))
        

        # funcion de ventiladores totales
        Vtot = [self.sims[0][0].Vtot(i) for i in self.t[0][:endD[0]]]    
        plt.plot(self.t[0][:endD[0]],Vtot,color='lime')

        linestyle = ['dashed','solid','dashed','dotted']
        for i in range(self.numescenarios):            
            plt.plot(self.t[i][:endD[i]],self.V[i][:endD[i]],label='VMI Utilizados mov='+str(self.inputarray[i][2]),color = 'blue' ,linestyle = linestyle[i])
    

        plt.xlim(0,days)
        self.plot(title = 'Ventiladores',xlabel='Dias desde '+datetime.strftime(self.initdate,'%Y-%m-%d'))



    # ------------ #
    #     Camas    #
    # ------------ #
    def plotcamas(self,enddate =  datetime(2020,7,30)):
        # -------- #
        #   Time   #
        # -------- #
        days = (enddate-self.initdate).days    
        endD = [np.searchsorted(self.t[i],days) for i in range(self.numescenarios)]
        idx = [np.searchsorted(self.t[i],self.sochimi_tr) for i in range(self.numescenarios)]

        # Inicio cuarentena general
        plt.axvline(x=self.May15,linestyle = 'dashed',color = 'grey')

        # Ploteo datos reales
        plt.scatter(self.sochimi_tr,self.Hr,label='Camas Ocupadas reales')
        plt.scatter(self.sochimi_tr,self.Hr_tot,label='Capacidad de Camas')


        # Display de Parametros y errores
        for i in range(self.numescenarios):
            plt.plot([], [], ' ', label='err_bed: '+str(round(100*self.err_bed[i],2))+'%')
            plt.plot([], [], ' ', label='Mov='+str(self.inputarray[i][2])+'Peak='+self.peak_date[i].strftime('%Y-%m-%d'))
        plt.plot([], [], ' ', label='beta: '+str(self.beta))
        plt.plot([], [], ' ', label='mu: '+str(self.mu))
        plt.plot([], [], ' ', label='fScale: '+str(self.ScaleFactor))
            
        
        # funcion de camas totales
        Htot = [self.sims[0][0].Htot(i) for i in self.t[0][:endD[0]]]
        plt.plot(self.t[0][:endD[0]],Htot,color='lime')
        
        for i in range(self.numescenarios):
            plt.plot(self.t[i][:endD[i]],self.H_bed[i][:endD[i]],label='Camas utilizadas mov='+str(self.inputarray[i][2]),color = 'red' ,linestyle = 'dashed')
        
        #plt.plot(self.t[i][:endD[i]],self.H_bed[i][:endD[i]],label='Camas utilizadas mov='+str(inputarray[i][2]),color = 'red' ,linestyle = 'solid')
        
        plt.xlim(0,days)
        self.plot(title = 'Camas',xlabel='Dias desde '+datetime.strftime(self.initdate,'%Y-%m-%d'))



    # -------------------------- #
    #     Camas y Ventiladores   #
    # -------------------------  #
    def plotcamasyventiladores(self,enddate =  datetime(2020,7,30),days=0):
        # -------- #
        #   Time   #
        # -------- #
        if days == 0:
            days = (enddate-self.initdate).days
        if days < 0:
            days = self.tsim
        endD = [np.searchsorted(self.t[i],days) for i in range(self.numescenarios)]
        idx = [np.searchsorted(self.t[i],self.sochimi_tr) for i in range(self.numescenarios)]

        # inicio cuarentena
        plt.axvline(x=self.May15,linestyle = 'dashed',color = 'grey')

        #ploteo de datos reales
        plt.scatter(self.sochimi_tr,self.Hr,label='Camas Ocupadas reales')
        plt.scatter(self.sochimi_tr,self.Vr,label='Ventiladores Ocupados reales')
        plt.scatter(self.sochimi_tr,self.Hr_tot,label='Capacidad de Camas')
        plt.scatter(self.sochimi_tr,self.Vr_tot,label='Capacidad de Ventiladores')

        for i in range(self.numescenarios):
            plt.plot([], [], ' ', label='err_bed: '+str(round(100*self.err_bed[i],2))+'%')
            plt.plot([], [], ' ', label='Mov='+str(self.inputarray[i][2])+'Peak='+self.peak_date[i].strftime('%Y-%m-%d'))
        plt.plot([], [], ' ', label='beta: '+str(self.beta))
        plt.plot([], [], ' ', label='mu: '+str(self.mu))
        plt.plot([], [], ' ', label='factor de escala: '+str(self.ScaleFactor))
    

        # Camas y ventiladores totales
        Htot = [self.sims[0][0].Htot(i) for i in self.t[0][:endD[0]]]
        Vtot = [self.sims[0][0].Vtot(i) for i in self.t[0][:endD[0]]]
        plt.plot(self.t[0][:endD[0]],Htot,color='lime')
        plt.plot(self.t[0][:endD[0]],Vtot,color='lime')

        
        for i in range(self.numescenarios): 
            plt.plot(self.t[i][:endD[i]],self.H_bed[i][:endD[i]],label='Camas utilizadas mov='+str(self.inputarray[i][2]),color = 'red' ,linestyle = 'dashed')
            plt.plot(self.t[i][:endD[i]],self.V[i][:endD[i]],label='VMI Utilizados mov='+str(self.inputarray[i][2]),color = 'blue' ,linestyle = 'dashed')
        #plt.plot(self.t[i][:endD[i]],H_crin[i][:endD[i]],label='Camas críticas mov='+str(inputarray[i][2]),color = 'black' ,linestyle = 'dashed')
        
        plt.xlim(0,days)
        self.plot(title = 'Camas',xlabel='Dias desde '+datetime.strftime(self.initdate,'%Y-%m-%d'))


    # ---------- #
    #    Hrate   #
    # ---------- #
    def plothrate(self,enddate =  datetime(2020,7,30)):
        # -------- #
        #   Time   #
        # -------- #
        days = (enddate-self.initdate).days
        endD = [np.searchsorted(self.t[i],days) for i in range(self.numescenarios)]
        idx = [np.searchsorted(self.t[i],self.sochimi_tr) for i in range(self.numescenarios)]
        
        Hrate = [self.H_in[i]/self.H_out[i] for i in range(self.numescenarios)]

        xlabel = 'Dias desde '+datetime.strftime(self.initdate,'%Y-%m-%d')
        fig, axs = plt.subplots(3)
        #fig.suptitle(title)
        
        
        colors = ['red','blue','green','lime']
        
        
        for i in range(self.numescenarios):
            axs[0].plot(self.t[i][:endD[i]],Hrate[i][:endD[i]], label='Mov='+str(self.inputarray[i][2]),linestyle = 'solid',color = colors[i])
            axs[0].legend()
        
        
        for i in range(self.numescenarios):
            axs[1].plot(self.t[i][:endD[i]],(self.H_in[i]/self.H_sum[i])[:endD[i]],label='Hin '+'Mov='+str(self.inputarray[i][2]),linestyle = 'solid',color = colors[i])        
            axs[1].legend()        
    
        for i in range(self.numescenarios):        
            axs[2].plot(self.t[i][:endD[i]],(self.H_out[i]/self.H_sum[i])[:endD[i]],label='Hout '+'Mov='+str(self.inputarray[i][2]),linestyle = 'solid',color = colors[i])
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
    def plotcamasrequeridas(self,enddate =  datetime(2020,7,30),days=0):
        # ----------- #
        #     Time    #
        # ----------- #    
        if days ==0:
            days = (enddate-self.initdate).days
        endD = [np.searchsorted(self.t[i],days) for i in range(self.numescenarios)]
        
        # ----------- #
        #     Plot    #
        # ----------- #
        # Fechas de colapso
        i=1
        plt.plot([], [], ' ', label='Fecha colapso Camas: '+str(round(self.t[i][self.H_colapsedate[i]])))
        plt.plot([], [], ' ', label='Fecha colapso Ventiladores: '+str(round(self.t[i][self.V_colapsedate[i]])))

        linestyle = ['dashed','solid','dashed','dotted']
        for i in range(self.numescenarios):
            plt.plot(self.t[i][:endD[i]],self.CH[i][:endD[i]],label='Intermedio/Intensivo Mov = '+str(self.inputarray[i][1]),color = 'red' ,linestyle = linestyle[i])
            plt.plot(self.t[i][:endD[i]],self.CV[i][:endD[i]],label='VMI Mov = '+str(self.inputarray[i][1]),color = 'blue' ,linestyle = linestyle[i])
        
        self.plot(title='Camas Requeridas',xlabel='Dias desde '+datetime.strftime(self.initdate,'%Y-%m-%d'))


    # ------------------------------------ #
    #      Necesidad total de Camas        #
    # ------------------------------------ #
    def plotnecesidadtotcamas(self,enddate =  datetime(2020,7,30),days=0):
        # ----------- #
        #     Time    #
        # ----------- #    
        if days ==0:
            days = (enddate-self.initdate).days
        endD = [np.searchsorted(self.t[i],days) for i in range(self.numescenarios)]

        # ----------- #
        #     Plot    #
        # ----------- #
        
        # Fechas de colapso
        i=1
        plt.plot([], [], ' ', label='Fecha colapso Camas: '+str(round(self.t[i][self.H_colapsedate[i]])))
        plt.plot([], [], ' ', label='Fecha colapso Ventiladores: '+str(round(self.t[i][self.V_colapsedate[i]])))

        # Datos reales
        plt.scatter(self.sochimi_tr,self.Hr,label='Camas Ocupadas reales')
        plt.scatter(self.sochimi_tr,self.Vr,label='Ventiladores Ocupados reales')
        plt.scatter(self.sochimi_tr,self.Hr_tot,label='Capacidad de Camas')
        plt.scatter(self.sochimi_tr,self.Vr_tot,label='Capacidad de Ventiladores')

        linestyle = ['dashed','solid','dashed','dotted']
        for i in range(self.numescenarios):    
            plt.plot(self.t[i][:endD[i]],np.array(self.CH[i][:endD[i]])+np.array(self.H_sum[i][:endD[i]]),label='UCI/UTI Mov = '+str(self.inputarray[i][1]),color = 'red' ,linestyle = linestyle[i])
            plt.plot(self.t[i][:endD[i]],np.array(self.CV[i][:endD[i]])+np.array(self.V[i][:endD[i]]),label='VMI Mov = '+str(self.inputarray[i][1]),color = 'blue' ,linestyle = linestyle[i])
        self.plot(title='Necesidad total de Camas',xlabel='Dias desde '+datetime.strftime(self.initdate,'%Y-%m-%d'))




    # -------------------------------------------------------- #
    #                       Infectados                         #
    # -------------------------------------------------------- #

    # ------------------------------ #
    #       Infectados Activos       #
    # ------------------------------ #
    def plotactivos(self,enddate =  datetime(2020,7,30),days=0, reales= True,ylim = 0,norm=1,scalefactor = False):
        # -------- #
        #   Time   #
        # -------- #
        if days == 0:
            days = (enddate-self.initdate).days
        if days < 0:
            days = self.tsim
        endD = [np.searchsorted(self.t[i],days) for i in range(self.numescenarios)]
        idx = [np.searchsorted(self.t[i],self.tr) for i in range(self.numescenarios)]

        Isf = 1    
        if scalefactor:
            Isf = self.ScaleFactor


        # ----------- #
        #     Plot    #
        # ----------- #
        # Error
        for i in range(self.numescenarios):
            plt.plot([], [], ' ', label='err: '+str(round(100*self.err_Iactives[i],2))+'%')    

        # Reales
        if reales:
            plt.scatter(self.tr,self.Ir,label='Infectados Activos reales')

        # Infectados
        linestyle = ['dashed','solid','dashed','dotted']
        for i in range(self.numescenarios):        
            plt.plot(self.t[i],self.I[i]/Isf,label='Infectados Mov = '+str(self.inputarray[i][2]),linestyle=linestyle[i])

        if days >0:
            plt.xlim(0,days)
        if ylim >0:
            plt.ylim(0,ylim)            
        self.plot(title = 'Activos',xlabel='Dias desde '+datetime.strftime(self.initdate,'%Y-%m-%d'))

    # -------------------------------- #
    #       Infectados Acumulados      #
    # -------------------------------- #
    # No esta listo
    def plotacumulados(self,enddate =  datetime(2020,7,30),days=0, reales= True,ylim = 0,norm=1,scalefactor = False):
        # -------- #
        #   Time   #
        # -------- #
        if days == 0:
            days = (enddate-self.initdate).days
        if days < 0:
            days = self.tsim
        endD = [np.searchsorted(self.t[i],days) for i in range(self.numescenarios)]
        idx = [np.searchsorted(self.t[i],self.tr) for i in range(self.numescenarios)]

        # ----------- #
        #     Plot    #
        # ----------- #
        # Error
        #for i in range(self.numescenarios):
        #    plt.plot([], [], ' ', label='err: '+str(round(100*self.err[i],2))+'%')    

        # Reales
        #if reales:
        #    plt.scatter(tr,Ir,label='Infectados Activos reales')

        # Infectados
        for i in range(self.numescenarios):        
            plt.plot(self.t[i],self.I[i],label='Infectados Mov = '+str(self.inputarray[i][2]))

        if days >0:
            plt.xlim(0,days)
        self.plot(title = 'Activos',xlabel='Dias desde '+datetime.strftime(self.initdate,'%Y-%m-%d'))  


    def plotactivosdesagregados(self,enddate =  datetime(2020,7,30),days=0, reales= True,ylim = 0,norm=1,scalefactor = False):
        # -------- #
        #   Time   #
        # -------- #
        if days == 0:
            days = (enddate-self.initdate).days
        if days < 0:
            days = self.tsim
        endD = [np.searchsorted(self.t[i],days) for i in range(self.numescenarios)]
        idx = [np.searchsorted(self.t[i],self.tr) for i in range(self.numescenarios)]

        # ----------- #
        #     Plot    #
        # ----------- #
        # Error
        #for i in range(self.numescenarios):
        #    plt.plot([], [], ' ', label='err: '+str(round(100*err[i],2))+'%')    

        # Reales
        if reales:
            plt.scatter(self.tr,self.Ir,label='Infectados Activos reales')

        # Infectados
        for i in range(self.numescenarios):        
            #plt.plot(self.t[i],I[i],label='Infectados )
            plt.plot(self.t[i],self.I_as[i],label='Acumulados asintomáticos Mov = '+str(self.inputarray[i][2]))
            plt.plot(self.t[i],self.I_mi[i],label='Acumulados Mild Mov = '+str(self.inputarray[i][2]))
            plt.plot(self.t[i],self.I_se[i],label='Acumulados Severos Mov = '+str(self.inputarray[i][2]))
            plt.plot(self.t[i],self.I_cr[i],label='Acumulados Criticos Mov = '+str(self.inputarray[i][2]))        

        if days >0:
            plt.xlim(0,days)
        if ylim >0:
            plt.ylim(0,ylim)
        self.plot(title = 'Infectados Activos desagregados',xlabel='Dias desde '+datetime.strftime(self.initdate,'%Y-%m-%d'))



    # ------------------------------------------------------------------------------------------------------- #
    #                                            Fallecidos                                                   #
    # ------------------------------------------------------------------------------------------------------- #

    # --------------------------------- #
    #      Fallecidos  acumulados       #
    # --------------------------------- #
    def plotfallecidosacumulados(self,enddate =  datetime(2020,7,30),days=0, reales= True,ylim = 0,norm=1,scalefactor = False):
        # -------- #
        #   Time   #
        # -------- #
        if days == 0:
            days = (enddate-self.initdate).days      
        elif days < 0:
            days = self.tsim     
        endD = [np.searchsorted(self.t[i],days) for i in range(self.numescenarios)]
        idx = [np.searchsorted(self.t[i],self.Br_tr) for i in range(self.numescenarios)]

        if norm <1:
            norm = self.ScaleFactor
        #Isf = 1    
        #if scalefactor:
        #    Isf = ScaleFactor

        # ----------- #
        #     Error   #
        # ----------- #
        i = 1 # Mov 0.6
        err = [LA.norm(self.Br[:21]-self.B[i][idx[i][:21]])/LA.norm(self.Br[:21]) for i in range(self.numescenarios)]
        err2 = [LA.norm(self.Br[23:]-self.B[i][idx[i][23:]])/LA.norm(self.Br[23:]) for i in range(self.numescenarios)]

        # ----------- #
        #     Plot    #
        # ----------- #
        # Parametros 
        plt.plot([], [], ' ', label='beta: '+str(self.beta))
        plt.plot([], [], ' ', label='mu: '+str(self.mu))
        plt.plot([], [], ' ', label='factor de escala: '+str(self.ScaleFactor))

        # Fecha de Peak
        for i in range(self.numescenarios):
            plt.plot([], [], ' ', label='Mov='+str(self.inputarray[i][2])+'Peak='+self.peak_date[i].strftime('%Y-%m-%d'))
        
        # Error
        for i in range(self.numescenarios):
            plt.plot([], [], ' ', label='Mov='+str(self.inputarray[i][2])+'err: '+str(round(100*err[i],2))+'%')


        # Datos reales
        if reales:
            plt.scatter(self.Br_tr,self.Br,label='Fallecidos reales')
            plt.scatter(self.ED_tr,self.ED_RM_ac,label='Fallecidos excesivos proyectados')

        linestyle = ['dashed','solid','dashed','dotted']
        for i in range(self.numescenarios):
            plt.plot(self.t[i][:endD[i]],self.B[i][:endD[i]]/norm,label='Fallecidos Mov = '+str(self.inputarray[i][2]),color = 'blue',linestyle=linestyle[i])

        plt.xlim(0,days)   
        if ylim >0:
            plt.ylim(0,ylim)

        self.plot(title = 'Fallecidos',xlabel='Dias desde '+datetime.strftime(self.initdate,'%Y-%m-%d'))
        


    # ----------------------------- #
    #       Fallecidos diarios      #
    # ----------------------------- #
    def plotfallecidosdiarios(self,enddate =  datetime(2020,7,30),days=0,scalefactor = False,reales= False):
        # -------- #
        #   Time   #
        # -------- #
        if days == 0:
            days = (enddate-self.initdate).days      
        elif days < 0:
            days = self.tsim     
        endD = [np.searchsorted(self.t[i],days) for i in range(self.numescenarios)]
        idx = [np.searchsorted(self.t[i],self.Br_tr) for i in range(self.numescenarios)]

        Isf = 1    
        if scalefactor:
            Isf = self.ScaleFactor

        linestyle = ['dashed','solid','dashed','dotted']
        for i in range(self.numescenarios):
            plt.plot(self.t[i],self.D[i]/Isf,label='Mov = '+str(self.inputarray[i][2]),color = 'black' ,linestyle = linestyle[i])
        self.plot(title = 'Fallecidos diarios',xlabel='Dias desde '+datetime.strftime(self.initdate,'%Y-%m-%d'))



    # ---------------------- #
    #       Letalidad        #
    # ---------------------- #

    def plotletalidad(self,enddate =  datetime(2020,7,30),days=0, reales= True,ylim = 0,norm=1,scalefactor = False):
        # -------- #
        #   Time   #
        # -------- #
        if days == 0:
            days = (enddate-self.initdate).days      
        elif days < 0:
            days = self.tsim     
        endD = [np.searchsorted(self.t[i],days) for i in range(self.numescenarios)]
        idx = [np.searchsorted(self.t[i],self.Br_tr) for i in range(self.numescenarios)]

        if norm <1:
            norm = self.ScaleFactor
        #Isf = 1    
        #if scalefactor:
        #    Isf = ScaleFactor

        # ----------- #
        #     Error   #
        # ----------- #
        i = 1 # Mov 0.6
        err = [LA.norm(self.Br[:21]-self.B[i][idx[i][:21]])/LA.norm(self.Br[:21]) for i in range(self.numescenarios)]
        err2 = [LA.norm(self.Br[23:]-self.B[i][idx[i][23:]])/LA.norm(self.Br[23:]) for i in range(self.numescenarios)]

        # ----------- #
        #     Plot    #
        # ----------- #
        # Parametros 
        plt.plot([], [], ' ', label='beta: '+str(self.beta))
        plt.plot([], [], ' ', label='mu: '+str(self.mu))
        plt.plot([], [], ' ', label='factor de escala: '+str(self.ScaleFactor))

        # Fecha de Peak
        for i in range(self.numescenarios):
            plt.plot([], [], ' ', label='Mov='+str(self.inputarray[i][2])+'Peak='+self.peak_date[i].strftime('%Y-%m-%d'))
        
        # Error
        for i in range(self.numescenarios):
            plt.plot([], [], ' ', label='Mov='+str(self.inputarray[i][2])+'err: '+str(round(100*err[i],2))+'%')


        # Datos reales
        #if reales:
        #    plt.scatter(Br_tr,Br,label='Fallecidos reales')
        #    plt.scatter(self.ED_tr,self.ED_RM_ac,label='Fallecidos excesivos proyectados')

        linestyle = ['dashed','solid','dashed','dotted']
        for i in range(self.numescenarios):
            plt.plot(self.t[i],100*self.B[i]/self.Iac[i],label='Mov=['+str(self.inputarray[i][2])+','+str(self.inputarray[i][1])+']' ,color='blue',linestyle=linestyle[i])
            
        plt.xlim(0,days)   
        if ylim >0:
            plt.ylim(0,ylim)

        self.plot(title = 'Letalidad',xlabel='Dias desde '+datetime.strftime(self.initdate,'%Y-%m-%d'))
        




    # ------------------ #
    #     Expuestos      #
    # ------------------ #
    def plotexpuestos(self,enddate =  datetime(2020,7,30),days=0, reales= True,ylim = 0,norm=1,scalefactor = False):
        # -------- #
        #   Time   #
        # -------- #
        if days == 0:
            days = (enddate-self.initdate).days      
        elif days < 0:
            days = self.tsim     
        endD = [np.searchsorted(self.t[i],days) for i in range(self.numescenarios)]
        idx = [np.searchsorted(self.t[i],self.Br_tr) for i in range(self.numescenarios)]

        if norm <1:
            norm = self.ScaleFactor
        #Isf = 1    
        #if scalefactor:
        #    Isf = ScaleFactor

        # ----------- #
        #     Plot    #
        # ----------- #
        # Parametros 
        plt.plot([], [], ' ', label='beta: '+str(self.beta))
        plt.plot([], [], ' ', label='mu: '+str(self.mu))
        plt.plot([], [], ' ', label='factor de escala: '+str(self.ScaleFactor))

        # Fecha de Peak
        for i in range(self.numescenarios):
            plt.plot([], [], ' ', label='Mov='+str(self.inputarray[i][2])+'Peak='+self.peak_date[i].strftime('%Y-%m-%d'))
        

        linestyle = ['dashed','solid','dashed','dotted']
        for i in range(self.numescenarios):
            plt.plot(self.t[i][:endD[i]],self.E[i][:endD[i]],label='Expuestos Mov = '+str(self.inputarray[i][2]),color = 'blue',linestyle=linestyle[i])
            plt.plot(self.t[i][:endD[i]],self.E_sy[i][:endD[i]],label='Expuestos sintomáticos Mov = '+str(self.inputarray[i][2]),color = 'red',linestyle=linestyle[i])
            
        plt.xlim(0,days)   
        if ylim >0:
            plt.ylim(0,ylim)

        self.plot(title = 'Expuestos',xlabel='Dias desde '+datetime.strftime(self.initdate,'%Y-%m-%d'))
        


    # -------------------- #
    #     Curvas SEIR      #
    # -------------------- #
    def plotseird(self,enddate =  datetime(2020,7,30),days=-1, reales= True,ylim = 0,norm=1,scalefactor = False,seird = [1,1,1,1,1]):
        # -------- #
        #   Time   #
        # -------- #
        if days == 0:
            days = (enddate-self.initdate).days      
        elif days < 0:
            days = self.tsim     
        endD = [np.searchsorted(self.t[i],days) for i in range(self.numescenarios)]
        idx = [np.searchsorted(self.t[i],self.Br_tr) for i in range(self.numescenarios)]

        if norm <1:
            norm = self.ScaleFactor
        #Isf = 1    
        #if scalefactor:
        #    Isf = ScaleFactor

        # ----------- #
        #     Plot    #
        # ----------- #
        # Parametros 
        plt.plot([], [], ' ', label='beta: '+str(self.beta))
        plt.plot([], [], ' ', label='mu: '+str(self.mu))
        plt.plot([], [], ' ', label='factor de escala: '+str(self.ScaleFactor))

        # Fecha de Peak
        for i in range(self.numescenarios):
            plt.plot([], [], ' ', label='Mov='+str(self.inputarray[i][2])+'Peak='+self.peak_date[i].strftime('%Y-%m-%d'))
        

        linestyle = ['dashed','solid','dashed','dotted']
        for i in range(self.numescenarios):        
            plt.plot(self.t[i],self.S[i],label='Susceptibles Mov = '+str(self.inputarray[i][2]),linestyle=linestyle[i])
            plt.plot(self.t[i],self.I[i],label='Infectados Mov = '+str(self.inputarray[i][2]),linestyle=linestyle[i])        
            plt.plot(self.t[i],self.E[i],label='Expuestos Mov = '+str(self.inputarray[i][2]),linestyle=linestyle[i])
            plt.plot(self.t[i],self.R[i],label='Recuperados Mov = '+str(self.inputarray[i][2]),linestyle=linestyle[i])
            #plt.plot(self.t[i],D[i],label='Muertos diarios Mov = '+str(inputarray[i][2]),linestyle=linestyle[i])
            plt.plot(self.t[i],self.B[i],label='Enterrados Mov = '+str(self.inputarray[i][2]),linestyle=linestyle[i])
            
        plt.xlim(0,days)   
        if ylim >0:
            plt.ylim(0,ylim)

        self.plot(title = 'Curvas SEIR',xlabel='Dias desde '+datetime.strftime(self.initdate,'%Y-%m-%d'))
        


    """
    # ------------------------------------------ #
    #       Graficos para parametrización        #
    # ------------------------------------------ #
    """
    # ----------------------------------------- #
    #       Curvas Expuestos/Infectados         #
    # ----------------------------------------- #
    def plotexpuestosinfectados(self,enddate =  datetime(2020,7,30),days=-1, reales= True,ylim = 0,norm=1,scalefactor = False,seird = [1,1,1,1,1]):
        enddate =  datetime(2020,6,30)
        days = (enddate-self.initdate).days
        endD = [np.searchsorted(self.t[i],days) for i in range(self.numescenarios)]  
        EIrate = [self.E[i]/self.I_sum[i] for i in range(self.numescenarios)]

        for i in range(self.numescenarios):    
            plt.plot(self.t[i][:endD[i]],EIrate[:endD[i]],label='Tasa Expuestos/Infectados')
        self.plot(title='Expuestos/infectados - mu ='+str(self.mu)+' beta='+str(self.beta))




    # ------------------------ #
    #       Curvas H/I         #
    # ------------------------ #
    def plothospitalizadosinfectados(self,enddate =  datetime(2020,7,30),days=-1, reales= True,ylim = 0,norm=1,scalefactor = False,seird = [1,1,1,1,1]):
        #initday = self.initdate#date(2020,3,15)
        enddate =  datetime(2020,6,30)
        days = (enddate-self.initdate).days
        endD = [np.searchsorted(self.t[i],days) for i in range(self.numescenarios)]

        HIrate = [self.H_sum[i]/self.I_sum[i] for i in range(self.numescenarios)]
        for i in range(self.numescenarios):    
            plt.plot(self.t[i][:endD[i]],HIrate[:endD[i]],label='Tasa Expuestos/Infectados')
        self.plot(title='H/I - mu ='+str(self.mu)+' beta='+str(self.beta))


    # ------------------------ #
    #       Curvas V/I         #
    # ------------------------ #
    def plotventiladosinfectados(self,enddate =  datetime(2020,7,30),days=-1, reales= True,ylim = 0,norm=1,scalefactor = False,seird = [1,1,1,1,1]):
        #initday = self.initdate#date(2020,3,15)
        enddate =  datetime(2020,6,30)
        days = (enddate-self.initdate).days
        days = 200
        endD = [np.searchsorted(self.t[i],days) for i in range(self.numescenarios)]
        
        VIrate = [self.V[i]/self.I_sum[i] for i in range(self.numescenarios)]
        for i in range(self.numescenarios):     
            plt.plot(self.t[i][:endD[i]],VIrate[:endD[i]],label='Tasa Expuestos/Infectados')
        self.plot(title='V/I - mu ='+str(self.mu)+' beta='+str(self.beta))


    # ------------------------ #
    #       Curvas V/H         #
    # ------------------------- #
    def plotventiladoshospitalizados(self,enddate =  datetime(2020,7,30),days=-1, reales= True,ylim = 0,norm=1,scalefactor = False,seird = [1,1,1,1,1]):
        initday = self.initdate#date(2020,3,15)
        enddate =  datetime(2020,6,30)
        days = (enddate-self.initdate).days
        endD = [np.searchsorted(self.t[i],days) for i in range(self.numescenarios)]
        VHrate = [self.V[i]/self.H_sum[i] for i in range(self.numescenarios)]
        for i in range(self.numescenarios):        
            plt.plot(self.t[i][:endD[i]],VHrate[:endD[i]],label='Tasa Expuestos/Infectados')
        self.plot(title='V/H - mu ='+str(self.mu)+' beta='+str(self.beta))




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
    def tabladedatos(self,inicio = datetime(2020,5,15), fin = datetime(2020,6,30),variables =['I_cum','I_act','D','L'], path=''):

        # Time
        tr_i = (inicio-self.initdate).days
        tr_f = (fin-self.initdate).days
        days = (fin-inicio).days
        index = [(inicio+timedelta(days=i)).strftime("%d/%m/%Y") for i in range(days+1)]
        idx = [np.searchsorted(self.t[i],range(tr_i,tr_f+1)) for i in range(self.numescenarios)]

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
                B_hoy = [round(self.B[i][idx[i][0]])]
                for j in range(1,len(idx[i])-1):
                    if idx[i][j-1]== idx[i][j]:
                        B_hoy.extend([round((self.B[i][idx[i][j-1]]+self.B[i][idx[i][j+1]])/2)])
                    else:        
                        B_hoy.extend([round(self.B[i][idx[i][j]])])
                B_hoy.extend([round(self.B[i][idx[i][-1]])])
                Bdata[namelist[i]]=B_hoy

            Bdata = pd.DataFrame(Bdata)
            data.append(Bdata)


        # Infectados Acumulados
        if 'I_cum' in variables:
            Iacdata = dict()
            namelist = ['Infectados-60','Infectados-65','Infectados-70']
            for i in range(self.numescenarios):        
                Iac_hoy = [round(self.Iac[i][idx[i][0]])]
                for j in range(1,len(idx[i])-1):
                    if idx[i][j-1]== idx[i][j]:
                        Iac_hoy.extend([round((self.Iac[i][idx[i][j-1]]+self.Iac[i][idx[i][j+1]])/2)])
                    else:        
                        Iac_hoy.extend([round(self.Iac[i][idx[i][j]])])
                Iac_hoy.extend([round(self.Iac[i][idx[i][-1]])])
                Iacdata[namelist[i]]=Iac_hoy
            Iacdata = pd.DataFrame(Iacdata)
            data.append(Iacdata)


        # Letalidad
        #let = [100*B[i]/Iac[i] for i in range(self.numescenarios)
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
                UsoCamas_hoy = [round(self.H_bed[i][idx[i][0]])]
                for j in range(1,len(idx[i])-1):
                    if idx[i][j-1]== idx[i][j]:
                        UsoCamas_hoy.extend([round((self.H_bed[i][idx[i][j-1]]+self.H_bed[i][idx[i][j+1]])/2)])
                    else:        
                        UsoCamas_hoy.extend([round(self.H_bed[i][idx[i][j]])])
                UsoCamas_hoy.extend([round(self.H_bed[i][idx[i][-1]])])
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
                UsoVMI_hoy = [round(self.H_vent[i][idx[i][0]])]
                for j in range(1,len(idx[i])-1):
                    if idx[i][j-1]== idx[i][j]:
                        UsoVMI_hoy.extend([round((self.H_vent[i][idx[i][j-1]]+self.H_vent[i][idx[i][j+1]])/2)])
                    else:        
                        UsoVMI_hoy.extend([round(self.H_vent[i][idx[i][j]])])
                UsoVMI_hoy.extend([round(self.H_vent[i][idx[i][-1]])])
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
                CH_hoy = [round(self.CH[i][idx[i][0]])]
                for j in range(1,len(idx[i])-1):
                    if idx[i][j-1]== idx[i][j]:
                        CH_hoy.extend([round((self.CH[i][idx[i][j-1]]+self.CH[i][idx[i][j+1]])/2)])
                    else:        
                        CH_hoy.extend([round(self.CH[i][idx[i][j]])])
                CH_hoy.extend([round(self.CH[i][idx[i][-1]])])
                CH_d[namelist[i-3]]=CH_hoy

            CH_d = pd.DataFrame(CH_d)
            data.append(CH_d)

        if 'V_ad' in variables:
            CV_d = dict()
            namelist = ['VMIAdicional-60','VMIAdicional-65','VMIAdicional-70']
            for i in range(3):        
                CV_hoy = [round(self.CV[i][idx[i][0]])]
                for j in range(1,len(idx[i])-1):
                    if idx[i][j-1]== idx[i][j]:
                        CV_hoy.extend([round((self.CV[i][idx[i][j-1]]+self.CV[i][idx[i][j+1]])/2)])
                    else:        
                        CV_hoy.extend([round(self.CV[i][idx[i][j]])])
                CV_hoy.extend([round(self.CV[i][idx[i][-1]])])
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