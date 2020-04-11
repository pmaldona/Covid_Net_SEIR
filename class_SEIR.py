#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:28:59 2020

@author: pmaldona
"""

import numpy as np


class SEIR :
        
        #constructor of SEIR class elements, it's initialized when a parameter 
        #miminization is performed to adjust the best setting of the actual infected
        
        def __init__(self,P,eta,alpha,S0,E0,I0,R0,Ir,tr,h,b_r,g_r,s_r,mu_r):

            #init response values
            self.S=S0
            self.E=E0
            self.I=I0
            self.R=R0
            self.N=(S0+E0+I0+R0).astype("float64")
            
            #init of strategy used for the dynamics
            self.strat_prop(P,alpha,eta)
            
            #saved stragegy functions
            self.alpha=alpha
            self.eta=eta
            
            #saved parameter ranges for later optimization
            self.b_r=b_r
            self.g_r=g_r
            self.s_r=s_r
            self.mu_r=mu_r
            
            # init values for the coeficients
            self.beta=np.mean(b_r)
            self.gamma=np.mean(g_r)
            self.sigma=np.mean(s_r)
            self.mu=np.mean(mu_r)
            
            #diferential function definitions
            self.dSdt = lambda beta,t,G,N,S,I: -beta*np.diag(np.reciprocal(N)*S).dot(G(t)).dot(I);
            self.dEdt = lambda beta,sigma,t,G,N,S,I,E: beta*np.diag(np.reciprocal(N)*S).dot(G(t)).dot(I) - sigma*E;
            self.dIdt = lambda sigma,gamma,t,E,I: sigma*E - gamma*I;
            self.dRdt = lambda gamma,t,I: gamma*I;
            
            #definition of timegrid 
            self.t0=min(tr)
            self.T=max(tr)
            self.h=h
            self.t=np.arange(self.t0,self.T+self.h,self.h)
            
            
        def strat_prop(self,P,alpha,eta):
            
            #partial definition of strategy, must be improved
            def G(t):
                return((np.diag(eta(t))+alpha(t))*(np.eye(P.shape[1])+P))
            
            self.G=G
            
      
        def integr_RK4(self,t0,T,h,beta,sigma,gamma,fix=False):
            #integrator function that star form t0 and finish with T with h as 
            #timestep. If there aren't inital values in [t0,T] function doesn't
            #start. Or it's start if class object is initialze.
            
            
            if(len(self.S.shape)==1):
                #pass if object is initalized
                t=self.t
                S0=self.S
                E0=self.E
                I0=self.I
                R0=self.R
                
            elif((self.t0<=t0) & (t0<=self.T)):
                #Condition over exiting time in already initialized object
                
                #Search fot initial time 
                np.where((10-h<t) & (t<10+h))
                i=np.min(np.where((10-h<t.self) & (t.self<10+h)))
                
                #set initial condition
                S0=self.S[:,i]
                E0=self.E[:,i]
                I0=self.I[:,i]
                R0=self.R[:,i]
                
                #set time grid
                t=np.arange(self.t[i],T+h,h)

                
            else:
                return(0)
            dim=self.S.shape[0]
            S=np.zeros((dim,len(t)))
            E=np.zeros((dim,len(t)))
            I=np.zeros((dim,len(t)))
            R=np.zeros((dim,len(t)))
            S[:,0]=S0
            E[:,0]=E0
            I[:,0]=I0
            R[:,0]=R0
            
            for j in range(len(t)-1):
                k1A = h*self.dSdt(beta, t[j], self.G, self.N, S[:,j], I[:,j])
                k1B = h*self.dEdt(beta, sigma, t[j], self.G, self.N, S[:,j], I[:,j], E[:,j])
                k1C = h*self.dIdt(sigma, gamma, t[j], E[:,j], I[:,j])
                k1D = h*self.dRdt(gamma, t[j], I[:,j])
                k2A = h*self.dSdt(beta, t[j] + h/2, self.G, self.N, S[:,j] + 0.5*k1A, I[:,j] + 0.5*k1C)
                k2B = h*self.dEdt(beta, sigma, t[j] + h/2, self.G, self.N, S[:,j] + 0.5*k1A, I[:,j] + 0.5*k1C, E[:,j] + 0.5*k1B)
                k2C = h*self.dIdt(sigma, gamma, t[j] + h/2, E[:,j] + 0.5*k1B, I[:,j] + 0.5*k1C)
                k2D = h*self.dRdt(gamma, t[j] + h/2, I[:,j] + 0.5*k1C)      
                k3A = h*self.dSdt(beta, t[j] + h/2, self.G, self.N, S[:,j] + 0.5*k2A, I[:,j] + 0.5*k2C)
                k3B = h*self.dEdt(beta, sigma, t[j] + h/2, self.G, self.N, S[:,j] + 0.5*k2A, I[:,j] + 0.5*k2C, E[:,j] + 0.5*k2B)
                k3C = h*self.dIdt(sigma, gamma, t[j] + h/2, E[:,j] + 0.5*k2B, I[:,j] + 0.5*k2C)
                k3D = h*self.dRdt(gamma, t[j] + h/2, I[:,j] + 0.5*k2C)
                k4A = h*self.dSdt(beta, t[j] + h, self.G, self.N, S[:,j] + k3A, I[:,j] + k3C)
                k4B = h*self.dEdt(beta, sigma, t[j] + h, self.G, self.N, S[:,j] + k3A, I[:,j] + k3C, E[:,j] + k3B)
                k4C = h*self.dIdt(sigma, gamma, t[j] + h, E[:,j] + k3B, I[:,j] + k3C)
                k4D = h*self.dRdt(gamma, t[j] + h, I[:,j] + k3C)
                S[:,j+1]=S[:,j] +1/6*(k1A + 2*k2A + 2*k3A + k4A)
                E[:,j+1]=E[:,j] +1/6*(k1B + 2*k2B + 2*k3B + k4B)
                I[:,j+1]=I[:,j] +1/6*(k1C + 2*k2C + 2*k3C + k4C)
                R[:,j+1]=R[:,j] +1/6*(k1D + 2*k2D + 2*k3D + k4D)
                
            if(fix==True):
                self.S=S
                self.E=E
                self.I=I
                self.R=R
                self.t0=t[0]
                self.T=t[-1]
                self.t=t
                self.h=h
                self.beta=beta
                self.sigma=sigma
                self.gamma=gamma
                return(0)
            else:
                return({"S":S,"E":E,"I":I,"R":R,"t0":t[0],"T":t[-1],"h":h,
                        "beta":beta,"sigma":sigma,"gamma":gamma})

    
    