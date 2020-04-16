#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEIR Model param finder
Implementation of a Metropolis-Hasting model
"""
from class_SEIR import SEIR
import numpy as np
import pandas
from numpy import linalg as LA
from time import sleep
from timeit import default_timer as timer
from pyswarm import pso

class SEIRrefiner:
    """
    constructor of SEIR class elements, it's initialized when a parameter
    miminization is performed to adjust the best setting of the actual infected
    """
    def __init__(self,P,eta,alpha,S0,E0,I0,R0,t0,T,h,beta_r,sigma_r,gamma_r,mu_r):

        #init response values
        self.S0=S0
        self.E0=E0
        self.I0=I0
        self.R0=R0
        self.N=(S0+E0+I0+R0).astype("float64")
        self.beta_r=beta_r
        self.sigma_r=sigma_r
        self.gamma_r=gamma_r
        self.mu_r=mu_r
        self.P = P
        
        # Output Params
        
        # Metropolis-Hastings  
        self.paramsMH = []
        self.optimalMH = []
        self.errorMH = []

        # Random Walk
        self.paramsRW = []
        self.optimalRW = []
        self.errorRW = []

        # PSO
        self.paramsPSO = []
        self.optimalPSO = []
        self.errorPSO = []

        self.error=None
        self.SEIR=None

        #saved stragegy functions
        self.alpha=alpha
        self.eta=eta

        #definition of timegrid
        self.t0=t0
        self.T=T
        self.h=h
        self.t=np.arange(self.t0,self.T+self.h,self.h)

    def refineMH(self,I_r,tr,r0,Npoints,steps,err):
        # find the optimal parameter using metropolis-hastings
        # desnity of phase space parameter for the optimizacion of calculeted infected vs. real infected
        mesh = self.mesh(Npoints)
        # tr=np.arange(I_r.shape[1])
        results = []
        for i in range(Npoints):            
            aux = self.met_hast(I_r,tr,mesh[i][0],mesh[i][1],mesh[i][2],mesh[i][3],r0,steps,err)
            results.append(aux)
            
        self.paramsMH = np.array(results)
        optindex = np.where(self.paramsMH[:,4]==np.amin(self.paramsMH[:,4]))[0][0]
        self.optimalMH =self.paramsMH[optindex,:]
        self.errorMH=self.optimalMH[-1]
        # should define an exit protocol
        self.SEIR = SEIR(self.P,self.eta,self.alpha,self.S0,self.E0,self.I0,self.R0,self.optimalMH[0],self.optimalMH[1],self.optimalMH[2],self.optimalMH[3])
        return self.optimalMH

    def refineRW(self,I_r,tr,r0,Npoints,steps,err):
        # Refine using Montecarlo - Markov Chain
        mesh = self.mesh(Npoints)
        # tr=np.arange(I_r.shape[1])
        results = []
        for i in range(Npoints):
            # print("Mesh point number "+str(i))
            aux = self.LocMinRW(I_r,tr,mesh[i][0],mesh[i][1],mesh[i][2],mesh[i][3],r0,steps,err)
            results.append(aux)
            # print("Error: "+str(aux[-1]))
        self.paramsRW = np.array(results)
        optindex = np.where(self.paramsRW[:,4]==np.amin(self.paramsRW[:,4]))[0][0]
        self.optimalRW =self.paramsRW[optindex,:]
        self.errorRW=self.optimalRW[-1]
        # should define an exit protocol
        self.SEIR = SEIR(self.P,self.eta,self.alpha,self.S0,self.E0,self.I0,self.R0,self.optimalRW[0],self.optimalRW[1],self.optimalRW[2],self.optimalRW[3])
        return self.optimalRW

    def refinepso_steps(self,Ir,swarmsize=5,maxiter=25,omega=0.5, phip=0.5, phig=0.5,iter=2):
        tr=np.arange(Ir.shape[1])
        #mu = np.random.uniform(min(self.mu_r),max(self.mu_r))
        mu = 2
        self.paramsPSO = self.pso_opt_coef(Ir,tr,mu,omega=omega, phip=phip, phig=phig,swarmsize=swarmsize,maxiter=maxiter)
            
        for i in range(iter):
            self.paramsPSO = self.pso_opt_mu(Ir,tr,self.paramsPSO[0],self.paramsPSO[1],self.paramsPSO[2],omega=omega, phip=phip, phig=phig,swarmsize=swarmsize,maxiter=maxiter)
            self.paramsPSO = self.pso_opt_coef(Ir,tr,self.paramsPSO[3],omega=omega, phip=phip, phig=phig,swarmsize=swarmsize,maxiter=maxiter)
        return self.paramsPSO 


    def pso_opt_mu(self,Ir,tr,beta,sigma,gamma,omega=0.5, phip=0.5, phig=0.5,swarmsize=5,maxiter=25):
        # _(self,P,eta,alpha,S0,E0,I0,R0,beta,gamma,sigma,mu):
        # def integr(self,t0,T,h,E0init=False):        
        def opti(x):
            model = SEIR(self.P,self.eta,self.alpha,self.S0,self.E0,self.I0,self.R0,beta,sigma,gamma,x)
            model.integr(self.t0,self.T,self.h,True)
            
            # dim=model.S.shape
            # n=np.zeros((4*dim[0],dim[1]))
            # n[0:dim[0],:]=model.S
            # n[dim[0]:2*dim[0],:]=model.E
            # n[2*dim[0]:3*dim[0],:]=model.I
            # n[3*dim[0]:4*dim[0],:]=model.R
            return(self.objective_funct(Ir,tr,model.I,model.t,'fro')) 

        lb=[min(self.mu_r)]
        ub=[max(self.mu_r)]
        
        xopt, fopt = pso(opti, lb, ub, minfunc=1, omega=omega, phip=phip, phig=phig, debug=True,swarmsize=swarmsize,maxiter=maxiter)
        xopt = np.append([beta,sigma,gamma],xopt)
        return [xopt,fopt]
        

    def pso_opt_coef(self,Ir,tr,mu,omega=0.5, phip=0.5, phig=0.5,swarmsize=5,maxiter=25):
        # _(self,P,eta,alpha,S0,E0,I0,R0,beta,gamma,sigma,mu):
        # def integr(self,t0,T,h,E0init=False):        
        def opti(x):
            model = SEIR(self.P,self.eta,self.alpha,self.S0,self.E0,self.I0,self.R0,x[0],x[1],x[2],mu)
            model.integr(self.t0,self.T,self.h,True)
            
            # dim=model.S.shape
            # n=np.zeros((4*dim[0],dim[1]))
            # n[0:dim[0],:]=model.S
            # n[dim[0]:2*dim[0],:]=model.E
            # n[2*dim[0]:3*dim[0],:]=model.I
            # n[3*dim[0]:4*dim[0],:]=model.R
            return(self.objective_funct(Ir,tr,model.I,model.t,'fro')) 

        lb=[min(self.beta_r),min(self.sigma_r),min(self.gamma_r)]
        ub=[max(self.beta_r),max(self.sigma_r),max(self.gamma_r)]
        
        xopt, fopt = pso(opti, lb, ub, minfunc=1, omega=omega, phip=phip, phig=phig, debug=True,swarmsize=swarmsize,maxiter=maxiter)
        xopt = np.append(xopt,mu)
        return [xopt,fopt]

    def refinepso(self,Ir,tr,swarmsize=5,maxiter=25,omega=0.5, phip=0.5, phig=0.5):
        # tr=np.arange(Ir.shape[1])
        self.paramsPSO = self.pso_opt(Ir,tr,omega=omega, phip=phip, phig=phig,swarmsize=swarmsize,maxiter=maxiter)
        return self.paramsPSO 


    def pso_opt(self,Ir,tr,omega=0.5, phip=0.5, phig=0.5,swarmsize=5,maxiter=25):
        # _(self,P,eta,alpha,S0,E0,I0,R0,beta,gamma,sigma,mu):
        # def integr(self,t0,T,h,E0init=False):          
        def opti(x):
            model = SEIR(self.P,self.eta,self.alpha,self.S0,self.E0,self.I0,self.R0,x[0],x[1],x[2],x[3])
            model.integr(self.t0,self.T,self.h,True)
            # dim=model.S.shape
            # n=np.zeros((4*dim[0],dim[1]))
            # n[0:dim[0],:]=model.S
            # n[dim[0]:2*dim[0],:]=model.E
            # n[2*dim[0]:3*dim[0],:]=model.I
            # n[3*dim[0]:4*dim[0],:]=model.R
            return(self.objective_funct(Ir,tr,model.I,model.t,'fro'))  
            
        lb=[min(self.beta_r),min(self.sigma_r),min(self.gamma_r),min(self.mu_r)]
        ub=[max(self.beta_r),max(self.sigma_r),max(self.gamma_r),max(self.mu_r)]
        
        xopt, fopt = pso(opti, lb, ub, minfunc=1, omega=omega, phip=phip, phig=phig, debug=True,swarmsize=swarmsize,maxiter=maxiter)
        return [xopt,fopt]


    def mesh(self,Npoints):
        print("Mesh")
        mesh = []
        for i in range(Npoints):
            beta_i=np.random.uniform(self.beta_r[0],self.beta_r[1])
            sigma_i=np.random.uniform(self.sigma_r[0],self.sigma_r[1])
            gamma_i=np.random.uniform(self.gamma_r[0],self.gamma_r[1])
            mu_i=np.random.uniform(self.mu_r[0],self.mu_r[1])
            mesh.append([beta_i,sigma_i,gamma_i,mu_i])
        return mesh

    ## Definir un objeto SEIR que solo se inicialice con las variables que no cambian
    ## Luego definir uno heredado que se defina con esas variables y los parametros
    def met_hast(self,I_r,tr,beta_i,sigma_i,gamma_i,mu_i,r0,steps,err):
        #print("Build SEIR")
        x=SEIR(self.P,self.eta,self.alpha,self.S0,self.E0,self.I0,self.R0,beta_i,gamma_i,sigma_i,mu_i)
        #print("RK4")
        if x.scikitsimport:
            x.integr(self.t0,self.T,self.h,True)
        else:
            #print("RK4")
            x.integr_RK4(self.t0,self.T,self.h,True)
        e_0=self.objective_funct(I_r,tr,x.I,x.t,2)
        e_o=e_0
        params = [[beta_i,sigma_i,gamma_i,mu_i,e_o]]
        i=0
        #print("Met-Hast")
        while i <steps:
            [b_p,s_p,g_p,m_p]=self.transition_model(x.beta,x.sigma,x.gamma,x.mu,r0)
            x_new=SEIR(self.P,self.eta,self.alpha,self.S0,self.E0,self.I0,self.R0,b_p,s_p,g_p,m_p)
            if x_new.scikitsimport:
                x_new.integr(self.t0,self.T,self.h,True)
            else:
                x_new.integr_RK4(self.t0,self.T,self.h,True)
            e_n=self.objective_funct(I_r,tr,x_new.I,x_new.t,2)
     
            # Acceptance
            if(e_n/e_o<1):
                x=x_new
                e_o = e_n
                params.append([b_p,s_p,g_p,m_p,e_n])
                i+=1
                print("------------------")
                print(b_p,s_p,g_p,m_p)
                print(e_o,e_n)
                # print("time: "+str(end-start))
                # print("Step "+str(i))
            else:
                u=np.random.uniform(0,1)
                if(e_n/e_0>u):
                    x=x_new
                    e_o = e_n
                    params.append([b_p,s_p,g_p,m_p,e_n])
                    i+=1
                    print("------------------")
                    print(b_p,s_p,g_p,m_p)
                    print(e_o,e_n)
        #sleep(0.01)
        # Dejo los params historicos por si hay que debuggear esta parte
        return params


    # Local Minimum Random Walk"
    # Like a RW with bias towards the optimum
    def LocMinRW(self,I_r,tr,beta_i,sigma_i,gamma_i,mu_i,r0,steps,err):
        #print("Build SEIR")
        x=SEIR(self.P,self.eta,self.alpha,self.S0,self.E0,self.I0,self.R0,beta_i,gamma_i,sigma_i,mu_i)
        #print("RK4")
        if x.scikitsimport:
            x.integr(self.t0,self.T,self.h,True)
        else:
            #print("RK4")
            x.integr_RK4(self.t0,self.T,self.h,True)
        e_0=self.objective_funct(I_r,tr,x.I,x.t,2)
        e_o=e_0
        params = [[beta_i,sigma_i,gamma_i,mu_i,e_o]]
        i=0
        k=0
        #print("Met-Hast")
        while i <steps:
            [b_p,s_p,g_p,m_p]=self.transition_model(x.beta,x.sigma,x.gamma,x.mu,r0)
            x_new=SEIR(self.P,self.eta,self.alpha,self.S0,self.E0,self.I0,self.R0,b_p,s_p,g_p,m_p)
            if x_new.scikitsimport:
                x_new.integr(self.t0,self.T,self.h,True)
            else:
                x_new.integr_RK4(self.t0,self.T,self.h,True)
            e_n=self.objective_funct(I_r,tr,x_new.I,x_new.t,2)
     
            # Acceptance
            if(e_n/e_o<1):
                x=x_new
                e_o = e_n
                params.append([b_p,s_p,g_p,m_p,e_n])
                i+=1
                k=0
                print("------------------")
                print(b_p,s_p,g_p,m_p)
                print(e_o,e_n)
                # print("time: "+str(end-start))
                # print("Step "+str(i))
            if(e_n<err):
                break
            k+=1
            if k>=100:
                break
            #sleep(0.01)
        # Dejo los params historicos por si hay que debuggear esta parte
        return params[-1]

    # objective function to minimize for any cases
    def objective_funct(self,Ir,tr,I,t,l):
        idx=np.searchsorted(t,tr)
        #print(idx)
        return LA.norm(Ir-I[:,idx],l)


    #The tranistion model defines how to move from current to new parameters
    def transition_model(self,beta,sigma,gamma,mu,r0):
        #print("Entering transition model")
        rb=r0*(max(self.beta_r)-min(self.beta_r))
        rg=r0*(max(self.gamma_r)-min(self.gamma_r))
        rs=r0*(max(self.sigma_r)-min(self.sigma_r))
        rm=r0*(max(self.mu_r)-min(self.mu_r))
        b_p = np.random.normal(beta,rb)
        s_p = np.random.normal(sigma,rs)
        g_p = np.random.normal(gamma,rg)
        m_p = np.random.normal(mu,rm)

        while b_p > max(self.beta_r) or b_p < min(self.beta_r):
            b_p = np.random.normal(beta,rb)
        while s_p > max(self.sigma_r) or s_p < min(self.sigma_r):
            s_p = np.random.normal(sigma,rs)
        while g_p > max(self.gamma_r) or g_p < min(self.beta_r):
            g_p = np.random.normal(gamma,rg)
        while m_p > max(self.mu_r) or m_p < min(self.mu_r):
            m_p = np.random.normal(mu,rm)
        return [b_p,s_p,g_p,m_p]
