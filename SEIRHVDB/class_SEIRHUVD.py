
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:28:59 2020

@author: pmaldona
"""

import numpy as np
from scikits.odes.odeint import odeint
from scipy.integrate import solve_ivp
from scipy.special import expit   
# from julia.api import Julia
# # l = Julia(compiled_modules=False)
# from diffeqpy import de
import numba

class SEIRHUDV :

        #constructor of SEIR class elements, it's initialized when a parameter
        #miminization is performed to adjust the best setting of the actual infected

        def __init__(self,alpha,Htot,Vtot,gw,mu,
                    S,E_as,E_sy,
                    I_as,I_mi,I_se,I_cr,
                    H_in,H_cr,H_out,V,D,B,R,CV,CH,ACH,ACV,
                    beta,betaD,eta,pSas,tSas,pSsy,tSsy,
                    pasas,tasas,psymi,tsymi,psyse,tsyse,psycr,tsycr,
                    pasR,tasR,pmiR,tmiR,psein,tsein,pseD,tseD,
                    pcrcrin,tcrcrin,pcrD,tcrD,
                    pincrin,tincrin,pinout,tinout,
                    pcrinV,tcrinV,pcrinD,tcrinD,pVout,tVout,poutR,toutR,
                    pVD,tVD,pDB,tDB):
            
            #init response values
            self.S=S
            self.E_as=E_as
            self.E_sy=E_sy
            self.I_as=I_as
            self.I_mi=I_mi
            self.I_se=I_se
            self.I_cr=I_cr
            self.H_in=H_in
            self.H_cr=H_cr
            self.H_out=H_out
            self.V=V
            self.D=D
            self.B=B
            self.R=R
            self.N=(S+E_as+E_sy+I_as+I_mi+I_se+I_cr+H_in+H_cr+H_out+V+D+R)
            self.I=I_cr+I_as+I_se+I_mi
            self.CV=CV
            self.CH=CH
            self.ACV=ACV
            self.ACH=ACH
            print(type(self.N))
            
            
            
            #global mobility reduction parameters
            self.alpha=alpha
            # total Hostipatl beds
            self.Htot=Htot
            # total Ventilators
            self.Vtot=Vtot
            # gate width
            self.gw=gw
            # 
            #infection rate
            self.beta=beta
            self.betaD=betaD
            self.eta=eta
            
            self.pSas=pSas
            self.tSas=tSas
            
            self.pSsy=pSsy
            self.tSsy=tSsy
            
            self.pasas=pasas
            self.tasas=tasas
            
            self.psymi=psymi
            self.tsymi=tsymi
            
            self.psyse=psyse
            self.tsyse=tsyse
            
            self.psycr=psycr
            self.tsycr=tsycr
            
            self.pasR=pasR
            self.tasR=tasR
            
            self.pmiR=pmiR
            self.tmiR=tmiR
            
            self.psein=psein
            self.tsein=tsein
            
            self.pseD=pseD
            self.tseD=tseD
            
            self.pcrcrin=pcrcrin
            self.tcrcrin=tcrcrin
            
            self.pcrD=pcrD
            self.tcrD=tcrD
            
            self.pincrin=pincrin
            self.tincrin=tincrin
            
            self.pinout=pinout
            self.tinout=tinout
            
            self.pcrinV=pcrinV
            self.tcrinV=tcrinV
            
            self.pcrinD=pcrinD
            self.tcrinD=tcrinD
            
            self.pVout=pVout
            self.tVout=tVout
            
            self.poutR=poutR
            self.toutR=toutR
            
            self.pVD=pVD
            self.tVD=tVD
            
            self.pDB=pDB
            self.tDB=tDB
            
            
            # ------------------------------------ #
            #   Diferential function definitions   #
            # ------------------------------------ #
            
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


        # UCI and UTI beds saturation function
        def h_sat(self,H_in,H_cr,H_out,t):
            return(expit(-self.gw*(H_in+H_cr+H_out-self.Htot(t))))
        # Ventilators Saturation Function    
        def v_sat(self,V,t):
            return(expit(-self.gw*(V-self.Vtot(t))))
        
        # def h_sat(self,H_in,H_cr,H_out,t):
        #     if((H_in+H_cr+H_out)>self.Htot(t)):
        #         return(0)
        #     else:
        #         return(1)
        
        # # Ventilators Saturation Function    
        # def v_sat(self,V,t):
        #     if(V>self.Vtot(t)):
        #         return(0)
        #     else:
        #         return(1)

        # Non zero        
        def non_z(self,x):
            if(x>0):
                return(x)
            else:
                return(0)

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
                
            initcond = np.array([S0,E_as0,E_sy0,I_as0,I_mi0,I_se0,I_cr0,
                                 H_in0,H_cr0,H_out0,V0,D0,B0,R0,I0,CV0,CH0,ACV0,ACH0])

            # initcond = initcond.reshape(4*dim)

            sol = odeint(model_SEIR_graph, self.t, initcond,method='admo')
            
            self.t=sol.values.t 
            #            soln=np.transpose(np.array(soln[1][1]))
            #            
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
                               
                return(ydot)
            initcond = np.array([S0,E_as0,E_sy0,I_as0,I_mi0,I_se0,I_cr0,
                                 H_in0,H_cr0,H_out0,V0,D0,B0,R0,I0,CV0,CH0,ACV0,ACH0])

            # initcond = initcond.reshape(4*dim)
            
            sol = solve_ivp(model_SEIR_graph,(t0,T), initcond,method='LSODA')
            
            self.t=sol.t 
            # #            soln=np.transpose(np.array(soln[1][1]))
            # #            
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
