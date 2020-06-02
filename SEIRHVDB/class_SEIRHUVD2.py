
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEIRHVD Model
"""

import numpy as np
from scikits.odes.odeint import odeint
from scipy.integrate import solve_ivp
from scipy.special import expit
from joblib import Parallel, delayed
from scipy import signal
import pandas as pd
from numpy import linalg as LA 
import multiprocessing  

"""
To do:
  - Create reports function inside simSAEIRHVD class


SEIRHVD Implementation
Instructions: 
    Init a simSEIRHVD objecting giving the simulation condictions:
        - tsim: Simulation time
        - max_mov:
        - rem_mov:
        - qp:
        - iqt:
        - fqt:
        - movfunct:

"""

class simSEIRHVD:
    definputarray=np.array([
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
    def __init__(self,beta = 0.19, mu =2.6 , inputarray = definputarray):
        self.mu = mu
        self.beta = beta 
        self.sims = []
        self.inputarray=inputarray
        self.simulated = False
    
    def sim_run(self,tsim,max_mov,rem_mov,qp,iqt=0,fqt = 300,movfunct = 0):       
        case = SEIRHUDV(tsim,max_mov,rem_mov,qp,iqt,fqt,movfunct)
        case.beta = self.beta       
        case.mu = self.mu
        # sol=test.integr(0,20,0.1,False)
        case.integr_sci(0,tsim,0.1,False)
        out=[case,max_mov,rem_mov,qp,tsim]
        return(out)   


        # Agregar un codigo numerico para que solo haya 1 vector de simulacion
        #['once','once','once','once','once','once','square','square','once','once','once','once','once','once','square','square']
        

    def simulate(self):
        num_cores = multiprocessing.cpu_count()
        #params=Parallel(n_jobs=num_cores, verbose=50)(delayed(ref_test.refinepso_all)(Ir,tr,swarmsize=200,maxiter=50,omega=0.5, phip=0.5, phig=0.5,eta_r=[0,1],Q_r=[0,1],obj_func='IN')for i in range(int(rep)))
        self.sims=Parallel(n_jobs=num_cores, verbose=50)(delayed(self.sim_run)(self.inputarray[i,0],self.inputarray[i,1],self.inputarray[i,2],self.inputarray[i,3],self.inputarray[i,4],self.inputarray[i,5],self.inputarray[i,6]) for i in range(self.inputarray.shape[0]))
        self.simulated = True
        return(self.sims)

    def getscenarios(self):
        return()
    def addscenario(self,inputarray):
        return()





class SEIRHUDV :  
    def __init__(self,tsim,max_mov,rem_mov,qp,iqt,fqt,movfunct):
        
        self.setparams()
        self.setinitvalues()  
        self.setscenario(tsim,max_mov,rem_mov,qp,iqt,fqt,movfunct)        
        #global mobility reduction parameters
        #self.alpha=alpha
        # total Hostipatl beds

        
        # --------------------------- #
        #    Diferential Ecuations    #
        # --------------------------- #
        
        # dVariable/dt = sum(prob_i/in_time_i*in_State_i,i in in states) - sum(prob_i/out_time_i*out_State_i,i in in states) 
        
        # Susceptibles
        # dS/dt:
        self.dS=lambda t,S,E_as,E_sy,I_as,I_mi,D,R: -self.alpha(t)*self.beta*S*(E_as+E_sy+I_as+I_mi)/self.N-self.betaD*D+self.eta*R
        
        # Exposed
        # dE_as/dt
        self.dE_as=lambda t,S,E_as,E_sy,I_as,I_mi: self.pSas/self.tSas*self.alpha(t)*self.beta*S*(E_as+E_sy+I_as+I_mi)/self.N\
            -self.pasas/self.tasas*E_as
        # dE_sy/dt
        self.dE_sy=lambda t,S,E_as,E_sy,I_as,I_mi: self.pSsy/self.tSsy*self.alpha(t)*self.beta*S*(E_as+E_sy+I_as+I_mi)/self.N\
            -self.psymi/self.tsymi*E_sy-self.psyse/self.tsyse*E_sy-self.psycr/self.tsycr*E_sy
        
        # Infected
        # dI_as/dt
        self.dI_as=lambda t,E_as,I_as: self.pasas/self.tasas*E_as-self.pasR/self.tasR*I_as
        # dI_mi/dt
        self.dI_mi=lambda t,E_sy,I_mi: self.psymi/self.tsymi*E_sy-self.pmiR/self.tmiR*I_mi
        # dI_se/dt: Esy -  
        self.dI_se=lambda t,E_sy,I_se,H_in,H_cr,H_out: self.psyse/self.tsyse*E_sy-self.psein/self.tsein*I_se*(self.h_sat(H_in,H_cr,H_out,t))\
            -self.pseD/self.tseD*I_se*(1-self.h_sat(H_in,H_cr,H_out,t))            
        # dI_cr/dt
        self.dI_cr=lambda t,E_sy,I_cr,H_in,H_cr,H_out: self.psycr/self.tsycr*E_sy-self.pcrcrin/self.tcrcrin*I_cr*(self.h_sat(H_in,H_cr,H_out,t))\
            -self.pcrD/self.tcrD*I_cr*(1-self.h_sat(H_in,H_cr,H_out,t))
        
        # Hospitalized
        # dH_in/dt: Hospitalized arriving to the hospital
        self.dH_in=lambda t,I_se,H_in,H_cr,H_out: self.psein/self.tsein*I_se*(self.h_sat(H_in,H_cr,H_out,t))\
            -self.pinout/self.tinout*H_in-self.pincrin/self.tincrin*H_in
        # dH_cr/dt: Hospitalized in critical conditions            
        self.dH_cr=lambda t,I_cr,H_in,H_cr,H_out,V: self.pcrcrin/self.tcrcrin*I_cr*(self.h_sat(H_in,H_cr,H_out,t))\
            -self.pcrinV/self.tcrinV*H_cr*self.v_sat(V,t)-self.pcrinD/self.tcrinD*H_cr*(1-self.v_sat(V,t))+self.pincrin/self.tincrin*H_in
        # dH_out/dt: Hospitalized getting better
        self.dH_out=lambda t,H_in,H_out,V: self.pVout/self.tVout*V+self.pinout/self.tinout*H_in-self.poutR/self.toutR*H_out
        # dV/dt: Ventilator necessity rates
        self.dV=lambda t,H_cr,V: self.pcrinV/self.tcrinV*H_cr*self.v_sat(V,t)-self.pVout/self.tVout*V-self.pVD/self.tVD*V
        
        
        # Deaths
        # dD/dt: Death Rate
        self.dD=lambda t,I_se,I_cr,H_in,H_cr,H_out,V,D: self.pVD/self.tVD*V+self.pcrD/self.tcrD*I_cr*(1-self.h_sat(H_in,H_cr,H_out,t))+\
            self.pseD/self.tseD*I_se*(1-self.h_sat(H_in,H_cr,H_out,t))+self.pcrinD/self.tcrinD*H_cr*(1-self.v_sat(V,t))-self.pDB/self.tDB*D
        # dB/dt: Bury rate
        self.dB=lambda t,D: self.pDB/self.tDB*D
        
        # Recovered
        # dR/dt
        self.dR=lambda t,I_as,I_mi,H_out,R: self.pasR/self.tasR*I_as+self.pmiR/self.tmiR*I_mi+self.poutR/self.toutR*H_out-self.eta*R

        #Auxiliar functions:
        self.dI=lambda t,E_as,E_sy: self.pasas/self.tasas*E_as+self.psymi/self.tsymi*E_sy+self.psyse/self.tsyse*E_sy+self.psycr/self.tsycr*E_sy
        self.dCV=lambda t,I_cr,H_in,H_cr,H_out,V,CV:self.pcrD/self.tcrD*I_cr*(1-self.h_sat(H_in,H_cr,H_out,t))+self.pcrinD/self.tcrinD*H_cr*(1-self.v_sat(V,t))-CV
        self.dCH=lambda t,I_se,H_in,H_cr,H_out,CH:self.pseD/self.tseD*I_se*(1-self.h_sat(H_in,H_cr,H_out,t))-CH
        self.dACV=lambda t,CV: CV
        self.dACH=lambda t,CH: CH
        self.dI_crD=lambda t,I_cr,H_in,H_cr,H_out: self.pcrD/self.tcrD*I_cr*(1-self.h_sat(H_in,H_cr,H_out,t))
        self.dI_seD=lambda t,I_se,H_in,H_cr,H_out: self.pseD/self.tseD*I_se*(1-self.h_sat(H_in,H_cr,H_out,t))
        self.dVD=lambda t,V: self.pVD/self.tVD*V
        self.dH_crD=lambda t,H_cr,V: self.pcrinD/self.tcrinD*H_cr*(1-self.v_sat(V,t))


    # UCI and UTI beds saturation function
    def h_sat(self,H_in,H_cr,H_out,t):
        return(expit(-self.gw*(H_in+H_cr+H_out-self.Htot(t))))
    # Ventilators Saturation Function    
    def v_sat(self,V,t):
        return(expit(-self.gw*(V-self.Vtot(t))))

    def setparams(self):
        self.mu = 2.6

        self.beta = 0.19 # (*probabilidad de transmision por contacto con contagiados*)
        self.betaD = 0.0 #(*probabilidad de transmision por contacto con muertos*)

        self.pSas = 0.3 # Transicion de Susceptible a Expuesto Asintomatico
        self.tSas = 1.0

        self.pSsy = 0.7 # Transicion de Susceptible a Expuesto sintomatico
        self.tSsy = 1.0

        self.pasas = 1.0# Transicion de Expuesto asintomatico a Infectado asintomatico
        self.tasas = 5.0

        self.psymi = 0.78 # Transicion de Expuesto Sintomatico a Infectado Mild
        self.tsymi = 5.0

        self.psycr = 0.08 # Transicion de Expuesto Sintomatico a Infectado critico
        self.tsycr = 5.0

        self.psyse = 0.14 # Transicion de Expuesto Sintomatico a Infectado Severo
        self.tsyse = 5.0

        self.pasR = 1.0   # Transicion de Infectado asintomatico a Recuperado
        self.tasR = 15.0 

        self.pmiR = 1.0  # Transicion de Infectado mild a Recuperado
        self.tmiR = 15.0

        self.psein = 1.0  # Transicion de Infectado serio a Hospitalizado (si no ha colapsado Htot)
        self.tsein = 1.0 

        self.pincrin = 0.01 # Transicion de Hospitalizado a Hospitalizado Critico (si no ha colapsado Htot)
        self.tincrin = 3.0

        self.pcrcrin = 1.0 # Transicion de Infectado critico  a Hopsitalizado Critico (si no ha colapsado Htot)
        self.tcrcrin = 1.0 

        self.pcrinV = 1.0 # Transicion de Hospitalizado critico a Ventilado (si no ha colapsado V)
        self.tcrinV = 1.0 

        self.pcrinD = 1.0 # Muerte de hospitalizado critico (Cuando V colapsa)
        self.tcrinD = 3.0 #

        self.pcrD = 1.0 # Muerte de Infectado critico (si ha colapsado Htot)
        self.tcrD = 3.0 #(*Hin+H_cr_in+Hout colapsa*)

        self.pseD = 1.0 # Muerte de Infectado severo (si ha colapsado Htot)
        self.tseD = 3.0

        self.pinout = 0.99 # Mejora de paciente severo hospitalizado, transita a Hout
        self.tinout = 4.0

        self.pVout = 0.5 # Mejora de ventilado hospitalizado, transita a Hout
        self.tVout = 15.0

        self.pVD = 0.5 # Muerte de ventilado
        self.tVD = 15.0

        self.poutR = 1.0 # Mejora del paciente hospitalizado, Hout a R
        self.toutR = 6.0

        self.pDB = 1.0 # Entierro del finado
        self.tDB = 1.0 

        self.eta = 0.0 # tasa de perdida de inmunidad (1/periodo)


        # ------------------- #
        #  Valores Iniciales  #
        # ------------------- #
            
        #I_act0 = 12642
        #cIEx0 =I_act0/1.5 # 3879 # Cantidad de infectados para calcular los expuestos iniciales
        #cId0 = 2060 # Infectados nuevos de ese día
        #cI0S =I_act0/2.5 # Cantidad de infectados para calcular los Infectados iniciales
        #muS=mu

    def setinitvalues(self):
        self.res=1
        self.cIEx0=4000
        self.cId0=2234
        self.cI0S = self.res*4842
        self.muS=self.mu

        self.I_as= 0.3*self.cI0S 
        self.I_mi= 0.6*self.cI0S 
        self.I_cr= 0.03*self.cId0 
        self.I_se = 0.07*self.cId0
        self.E_as=0.3*self.muS*self.cIEx0
        self.E_sy=0.7*self.muS*self.cIEx0
        self.Htot=lambda t: 1997.0+23*t
        self.H0=1731#1903.0
        self.H_cr=80.0
        self.H_in=self.H0*0.5-self.H_cr/2
        self.H_out=self.H0*0.5-self.H_cr/2
        self.Vtot=lambda t:1029.0+18*t
        self.gw=5
        self.D=26.0
        self.B=221.0
        self.R=0.0
        self.V=758.0#846.0
        self.mu=1.4
        self.t=400.0
        self.CV=0
        self.CH=0
        self.ACV=0
        self.ACH=0
        self.I_crD=0
        self.I_seD=0
        self.VD=0
        self.H_crD=0
        self.S=8125072.0-self.H0-self.V-self.D-(self.E_as+self.E_sy)-(self.I_as+self.I_cr+self.I_se+self.I_mi)
        self.N=(self.S+self.E_as+self.E_sy+self.I_as+self.I_mi+self.I_se+self.I_cr+self.H_in+self.H_cr+self.H_out+self.V+self.D+self.R)
        self.I=self.I_cr+self.I_as+self.I_se+self.I_mi


        #constructor of SEIR class elements, it's initialized when a parameter
        #miminization is performed to adjust the best setting of the actual infected

    def setscenario(self,tsim,max_mov,rem_mov,qp,iqt,fqt,movfunct):
        self.tsim = tsim
        self.max_mov = max_mov
        self.rem_mov = rem_mov
        self.qp = qp
        self.iqt = iqt
        self.fqt = fqt
        if movfunct == 0:
            self.movfunct = 'once'
        elif movfunct == 1:
            self.movfunct = 'square'
        elif movfunct == 2:
            self.movfunct = 'sawtooth'
        else:
            self.movfunct = 'once'
        
        self.alpha = self.alphafunct(self.max_mov,self.rem_mov,self.qp,self.iqt,self.fqt,self.movfunct)
        return() 
        


    def alphafunct(self,max_mov,rem_mov,qp,iqt=0,fqt=300,movfunct = 'once'):
        """    
        # max_mov: Movilidad sin cuarentena
        # rem_mov: Movilidad con cuarentena
        # qp: Periodo cuarentena dinamica 
        #          - qp >0 periodo Qdinamica 
        #          - qp = 0 sin qdinamica
        # iqt: Initial quarantine time. Tiempo inicial antes de cuarentena dinamica
        #          - iqt>0 inicia con cuarentena total hasta iqt
        #          - iqt<0 sin cuarentena hasta iqt
        # fqt: Final quarantine time. Duracion tiempo cuarentena 
        # movfunct: Tipo de cuarentena dinamica desde iqt
        #          - once: una vez durante qp dias 
        #          - total: total desde iqt
        #          - sawtooth: diente de cierra
        #          - square: onda cuadrada
        """
        def alpha(t):
            if movfunct=='total':
                if t < -iqt:
                    return(max_mov)
                else:
                    return(rem_mov)

            elif movfunct =='once':        
                if t<iqt:
                    return(rem_mov)
                else:
                    return(max_mov)
                #if t<-iqt:
                #    return(max_mov)
                #elif t >fqt:
                #    return(max_mov)
                #elif iqt> 0 and t>iqt:
                #    return(max_mov)
                #else:
                #    return(rem_mov)

            elif movfunct =='sawtooth':
                def f(t): 
                    return signal.sawtooth(t)
                if t<abs(iqt):
                    if iqt>0:
                        return(rem_mov)
                    else:
                        return(max_mov)
                else:
                    if t<fqt:
                        return((max_mov-rem_mov)/2*(f(np.pi / qp * t - np.pi))+(max_mov+rem_mov)/2)
                    else:
                        return(max_mov)

            elif movfunct =='square':
                def f(t): 
                    return signal.square(t)
                if t<abs(iqt):
                    if iqt>0:
                        return(rem_mov)
                    else:
                        return(max_mov)
                else:
                    if t<fqt:
                        return((max_mov-rem_mov)/2*(f(np.pi / qp * t - np.pi))+(max_mov+rem_mov)/2)
                    else:
                        return(max_mov)
        return(alpha)



    def get_conditions(self):
        # This function will return a text explaining the different simulation scenarios currently placed as inputs            
        return

 
    def integr(self,t0,T,h,E0init=False):
        #integrator function that star form t0 and finish with T with h as
        #timestep. If there aren't inital values in [t0,T] function doesn't
        #start. Or it's start if class object is initialze.


        if(not isinstance(self.S, np.ndarray)):
            #pass if object is initalized
            if(E0init):
                E_as0=self.mu*(self.I_as)
                E_sy0=self.mu*(self.I_mi+self.I_se+self.I_cr)
            else:
                E_as0=self.E_as
                E_sy0=self.E_sy
            S0=self.S
            I_as0=self.I_as
            I_mi0=self.I_mi
            I_se0=self.I_se
            I_cr0=self.I_cr
            H_in0=self.H_in
            H_cr0=self.H_cr
            H_out0=self.H_out
            V0=self.V
            D0=self.D
            B0=self.B
            R0=self.R
            I0=self.I
            CV0=self.CV
            CH0=self.CH
            ACV0=self.ACV
            ACH0=self.ACH
            I_crD0=self.I_crD
            I_seD0=self.I_seD
            VD0=self.VD
            H_crD0=self.H_crD
            print(self.H_out)
            self.t=np.arange(t0,T+h,h)
            
        elif((min(self.t)<=t0) & (t0<=max(self.t))):
            #Condition over exiting time in already initialized object

            #Search fot initial time
            idx=np.searchsorted(self.t,t0)

            #set initial condition

            S0=self.S[idx]
            E_as0=self.E_as
            E_sy0=self.E_sy
            I_as0=self.I_as[idx]
            I_mi0=self.I_mi[idx]
            I_se0=self.I_se[idx]
            I_cr0=self.I_cr[idx]
            H_in0=self.H_in[idx]
            H_cr0=self.H_cr[idx]
            H_out0=self.H_out[idx]
            V0=self.V[idx]
            D0=self.D[idx]
            B0=self.B[idx]
            R0=self.R[idx]
            I0=self.I[idx]
            CV0=self.CV[idx]
            CH0=self.CH[idx]
            ACV0=self.ACV[idx]
            ACH0=self.ACH[idx]
            I_crD0=self.I_crD[idx]
            I_seD0=self.I_seD[idx]
            VD0=self.VD[idx]
            H_crD0=self.H_crD[idx]

            #set time grid
            self.t=np.arange(self.t[idx],T+h,h)


        else:
            return()
            
        #            dim=self.S.shape[0]
        
        def model_SEIR_graph(t,y,ydot):
            
            ydot[0]=self.dS(t,y[0],y[1],y[2],y[3],y[4],y[11],y[13])
            ydot[1]=self.dE_as(t,y[0],y[1],y[2],y[3],y[4])
            ydot[2]=self.dE_sy(t,y[0],y[1],y[2],y[3],y[4])
            ydot[3]=self.dI_as(t,y[1],y[3])
            ydot[4]=self.dI_mi(t,y[2],y[4])
            ydot[5]=self.dI_se(t,y[2],y[5],y[7],y[8],y[9])
            ydot[6]=self.dI_cr(t,y[2],y[6],y[7],y[8],y[9]) 
            ydot[7]=self.dH_in(t,y[5],y[7],y[8],y[9])
            ydot[8]=self.dH_cr(t,y[6],y[7],y[8],y[9],y[10])
            ydot[9]=self.dH_out(t,y[7],y[9],y[10])
            ydot[10]=self.dV(t,y[8],y[10])
            ydot[11]=self.dD(t,y[5],y[6],y[7],y[8],y[9],y[10],y[11])
            ydot[12]=self.dB(t,y[11])
            ydot[13]=self.dR(t,y[3],y[4],y[9],y[13])
            ydot[14]=self.dI(t,y[1],y[2])
            ydot[15]=self.dCV(t,y[6],y[7],y[8],y[9],y[10],y[15])
            ydot[16]=self.dCH(t,y[5],y[7],y[8],y[9],y[16])
            ydot[17]=self.dACH(t,y[15])
            ydot[18]=self.dACV(t,y[16])
            ydot[19]=self.dI_crD(t,y[6],y[7],y[8],y[9])               
            ydot[20]=self.dI_seD(t,y[5],y[7],y[8],y[9])               
            ydot[21]=self.dVD(t,y[10])                      
            ydot[22]=self.dH_crD(t,y[8],y[10])                  

            
        initcond = np.array([S0,E_as0,E_sy0,I_as0,I_mi0,I_se0,I_cr0,
                                H_in0,H_cr0,H_out0,V0,D0,B0,R0,I0,CV0,CH0,ACV0,ACH0,I_crD0,I_seD0,VD0,H_crD0])

        # initcond = initcond.reshape(4*dim)

        sol = odeint(model_SEIR_graph, self.t, initcond,method='admo')
        
        self.t=sol.values.t 
        #            soln=np.transpose(np.array(soln[1][1]))
        
        self.S=sol.values.y[:,0]
        self.E_as=sol.values.y[:,1]
        self.E_sy=sol.values.y[:,2]
        self.I_as=sol.values.y[:,3]
        self.I_mi=sol.values.y[:,4]
        self.I_se=sol.values.y[:,5]
        self.I_cr=sol.values.y[:,6]                
        self.H_in=sol.values.y[:,7]
        self.H_cr=sol.values.y[:,8]
        self.H_out=sol.values.y[:,9]
        self.V=sol.values.y[:,10]
        self.D=sol.values.y[:,11]
        self.B=sol.values.y[:,12]
        self.R=sol.values.y[:,13]
        self.I=sol.values.y[:,14]
        self.CV=sol.values.y[:,15]
        self.CH=sol.values.y[:,16]
        self.ACV=sol.values.y[:,17]
        self.ACH=sol.values.y[:,18]
        self.I_crD=sol.values.y[:,19]
        self.I_seD=sol.values.y[:,20]               
        self.VD=sol.values.y[:,21]
        self.H_crD=sol.values.y[:,22]                   
        return(sol)

    def integr_sci(self,t0,T,h,E0init=False):
        #integrator function that star form t0 and finish with T with h as
        #timestep. If there aren't inital values in [t0,T] function doesn't
        #start. Or it's start if class object is initialze.

        if(not isinstance(self.S, np.ndarray)):
            #pass if object is initalized
            if(E0init):
                E_as0=self.mu*(self.I_as)
                E_sy0=self.mu*(self.I_mi+self.I_se+self.I_cr)
            else:
                E_as0=self.E_as
                E_sy0=self.E_sy
            S0=self.S
            I_as0=self.I_as
            I_mi0=self.I_mi
            I_se0=self.I_se
            I_cr0=self.I_cr
            H_in0=self.H_in
            H_cr0=self.H_cr
            H_out0=self.H_out
            V0=self.V
            D0=self.D
            B0=self.B
            R0=self.R
            I0=self.I
            CV0=self.CV
            CH0=self.CH
            ACV0=self.ACV
            ACH0=self.ACH
            I_crD0=self.I_crD
            I_seD0=self.I_seD
            VD0=self.VD
            H_crD0=self.H_crD                
            self.t=np.arange(t0,T+h,h)
            
        elif((min(self.t)<=t0) & (t0<=max(self.t))):
            #Condition over exiting time in already initialized object

            #Search fot initial time
            idx=np.searchsorted(self.t,t0)

            #set initial condition

            S0=self.S[idx]
            E_as0=self.E_as
            E_sy0=self.E_sy
            I_as0=self.I_as[idx]
            I_mi0=self.I_mi[idx]
            I_se0=self.I_se[idx]
            I_cr0=self.I_cr[idx]
            H_in0=self.H_in[idx]
            H_cr0=self.H_cr[idx]
            H_out0=self.H_out[idx]
            V0=self.V[idx]
            D0=self.D[idx]
            B0=self.B[idx]
            R0=self.R[idx]
            I0=self.I[idx]
            CV0=self.CV[idx]
            CH0=self.CH[idx]                
            ACV0=self.ACV[idx]
            ACH0=self.ACH[idx]
            I_crD0=self.I_crD[idx]
            I_seD0=self.I_seD[idx]
            VD0=self.VD[idx]
            H_crD0=self.H_crD[idx]

            #set time grid
            self.t=np.arange(self.t[idx],T+h,h)


        else:
            return()
            
        #            dim=self.S.shape[0]
            
        # dim=self.S.shape[0]
        
        def model_SEIR_graph(t,y):
            ydot=np.zeros(len(y))
            ydot[0]=self.dS(t,y[0],y[1],y[2],y[3],y[4],y[11],y[13])
            ydot[1]=self.dE_as(t,y[0],y[1],y[2],y[3],y[4])
            ydot[2]=self.dE_sy(t,y[0],y[1],y[2],y[3],y[4])
            ydot[3]=self.dI_as(t,y[1],y[3])
            ydot[4]=self.dI_mi(t,y[2],y[4])
            ydot[5]=self.dI_se(t,y[2],y[5],y[7],y[8],y[9])
            ydot[6]=self.dI_cr(t,y[2],y[6],y[7],y[8],y[9]) 
            ydot[7]=self.dH_in(t,y[5],y[7],y[8],y[9])
            ydot[8]=self.dH_cr(t,y[6],y[7],y[8],y[9],y[10])
            ydot[9]=self.dH_out(t,y[7],y[9],y[10])
            ydot[10]=self.dV(t,y[8],y[10])
            ydot[11]=self.dD(t,y[5],y[6],y[7],y[8],y[9],y[10],y[11])
            ydot[12]=self.dB(t,y[11])
            ydot[13]=self.dR(t,y[3],y[4],y[9],y[13])
            ydot[14]=self.dI(t,y[1],y[2])
            ydot[15]=self.dCV(t,y[6],y[7],y[8],y[9],y[10],y[15])
            ydot[16]=self.dCH(t,y[5],y[7],y[8],y[9],y[16])
            ydot[17]=self.dACH(t,y[15])
            ydot[18]=self.dACV(t,y[16])
            ydot[19]=self.dI_crD(t,y[6],y[7],y[8],y[9])               
            ydot[20]=self.dI_seD(t,y[5],y[7],y[8],y[9])               
            ydot[21]=self.dVD(t,y[10])                      
            ydot[22]=self.dH_crD(t,y[8],y[10])                                    
            return(ydot)
        initcond = np.array([S0,E_as0,E_sy0,I_as0,I_mi0,I_se0,I_cr0,
                                H_in0,H_cr0,H_out0,V0,D0,B0,R0,I0,CV0,CH0,ACV0,ACH0,I_crD0,I_seD0,VD0,H_crD0])

        # initcond = initcond.reshape(4*dim)
        
        sol = solve_ivp(model_SEIR_graph,(t0,T), initcond,method='LSODA')
        
        self.t=sol.t 
        # #            soln=np.transpose(np.array(soln[1][1]))
        
        self.S=sol.y[0,:]
        self.E_as=sol.y[1,:]
        self.E_sy=sol.y[2,:]
        self.I_as=sol.y[3,:]
        self.I_mi=sol.y[4,:]
        self.I_se=sol.y[5,:]
        self.I_cr=sol.y[6,:]
        self.H_in=sol.y[7,:]
        self.H_cr=sol.y[8,:]
        self.H_out=sol.y[9,:]
        self.V=sol.y[10,:]
        self.D=sol.y[11,:]
        self.B=sol.y[12,:]
        self.R=sol.y[13,:]
        self.I=sol.y[14,:]
        self.CV=sol.y[15,:]
        self.CH=sol.y[16,:]
        self.ACV=sol.y[17,:]
        self.ACH=sol.y[18,:]
        return(sol)






















    # def integr_SciML(self,t0,T,h,E0init=False):
    #     #integrator function that star form t0 and finish with T with h as
    #     #timestep. If there aren't inital values in [t0,T] function doesn't
    #     #start. Or it's start if class object is initialze.


    #     if(not isinstance(self.S, np.ndarray)):
    #         #pass if object is initalized
    #         if(E0init):
    #             E_as0=self.mu*(self.I_as)
    #             E_sy0=self.mu*(self.I_mi+self.I_se+self.I_cr)
    #         else:
    #             E_as0=self.E_as
    #             E_sy0=self.E_sy
    #         S0=self.S
    #         I_as0=self.I_as
    #         I_mi0=self.I_mi
    #         I_se0=self.I_se
    #         I_cr0=self.I_cr
    #         H_in0=self.H_in
    #         H_cr0=self.H_cr
    #         H_out0=self.H_out
    #         V0=self.V
    #         D0=self.D
    #         B0=self.B
    #         R0=self.R
    #         print(self.H_out)
    #         self.t=np.arange(t0,T+h,h)
            
    #     elif((min(self.t)<=t0) & (t0<=max(self.t))):
    #         #Condition over exiting time in already initialized object

    #         #Search fot initial time
    #         idx=np.searchsorted(self.t,t0)

    #         #set initial condition

    #         S0=self.S[idx]
    #         E_as0=self.E_as
    #         E_sy0=self.E_sy
    #         I_as0=self.I_as[idx]
    #         I_mi0=self.I_mi[idx]
    #         I_se0=self.I_se[idx]
    #         I_cr0=self.I_cr[idx]
    #         H_in0=self.H_in[idx]

    #         #set time grid
    #         self.t=np.arange(self.t[idx],T+h,h)


    #     else:
    #         return()
            
    #     # dim=self.S.shape[0]
        
    #     def model_SEIR_graph(t,y):
    #         ydot=np.zeros(len(y))
    #         ydot[0]=self.dS(t,y[0],y[1],y[2],y[3],y[4],y[11],y[13])
    #         ydot[1]=self.dE_as(t,y[0],y[1],y[2],y[3],y[4])
    #         ydot[2]=self.dE_sy(t,y[0],y[1],y[2],y[3],y[4])
    #         ydot[3]=self.dI_as(t,y[1],y[3])
    #         ydot[4]=self.dI_mi(t,y[2],y[4])
    #         ydot[5]=self.dI_se(t,y[2],y[5],y[7],y[8],y[9])
    #         ydot[6]=self.dI_cr(t,y[2],y[6],y[7],y[8],y[9]) 
    #         ydot[7]=self.dH_in(t,y[5],y[7],y[8],y[9])
    #         ydot[8]=self.dH_cr(t,y[6],y[7],y[8],y[9],y[10])
    #         ydot[9]=self.dH_out(t,y[7],y[9],y[10])
    #         ydot[10]=self.dV(t,y[8],y[10])
    #         ydot[11]=self.dD(t,y[5],y[6],y[8],y[10],y[11])
    #         ydot[12]=self.dB(t,y[11])
    #         ydot[13]=self.dR(t,y[3],y[4],y[9],y[13])
    #         return(ydot)
    #     initcond = np.array([S0,E_as0,E_sy0,I_as0,I_mi0,I_se0,I_cr0,
    #                          H_in0,H_cr0,H_out0,V0,D0,B0,R0])

    #     # initcond = initcond.reshape(4*dim)
    #     numba_f = numba.jit(model_SEIR_graph)
    #     tspan=(float(t0),float(T))
    #     print(tspan)
    #     prob = de.ODEProblem(numba_f, initcond,(t0,T))
    #     sol = de.solve(prob,de.TRBDF2())
    #     # self.t=sol.t 

    #     # self.S=sol.y[0,:]
    #     # self.E_as=sol.y[1,:]
    #     # self.E_sy=sol.y[2,:]
    #     # self.I_as=sol.y[3,:]
    #     # self.I_mi=sol.y[4,:]
    #     # self.I_se=sol.y[5,:]
    #     # self.I_cr=sol.y[6,:]
    #     # self.H_in=sol.y[7,:]
    #     # self.H_cr=sol.y[8,:]
    #     # self.H_out=sol.y[9,:]
    #     # self.V=sol.y[10,:]
    #     # self.D=sol.y[11,:]
    #     # self.B=sol.y[12,:]
    #     # self.R=sol.y[13,:]

    #     return(sol)