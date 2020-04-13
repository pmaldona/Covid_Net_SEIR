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

    def refine(self,I_r,r0,Npoints,steps,err):
        # find the optimal parameter
        # Return a SEIR object
        mesh = self.mesh(Npoints)
        tr=np.arange(I_r.shape[1])
        results = []
        for i in range(Npoints):
            print("Mesh point number "+str(i))
            aux = self.met_hast(I_r,tr,mesh[i][0],mesh[i][1],mesh[i][2],mesh[i][3],r0,steps,err)
            results.append(aux)
            print("Error: "+str(aux[-1]))
        results = np.array(results)
        optindex = np.where(results[:,4]==np.amin(results[:,4]))[0][0]
        optimal=results[optindex,:]
        self.error=optimal[-1]
        # define an exit protocol
        self.SEIR = SEIR(self.P,self.eta,self.alpha,self.S0,self.E0,self.I0,self.R0,optimal[0],optimal[1],optimal[2],optimal[3])
        return optimal


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
        x.integr_RK4(self.t0,self.T,self.h,True)
        e_0=self.objective_funct(I_r,tr,x.I,x.t,2)
        e_o=e_0
        params = [[beta_i,sigma_i,gamma_i,mu_i,e_o]]
        i=0
        k=0
        print("Met-Hast")
        while i <steps:
            start = timer()
            [b_p,s_p,g_p,m_p]=self.transition_model(x.beta,x.sigma,x.gamma,x.mu,r0)
            x_new=SEIR(self.P,self.eta,self.alpha,self.S0,self.E0,self.I0,self.R0,b_p,s_p,g_p,m_p)
            x_new.integr_RK4(self.t0,self.T,self.h,True)
            e_n=self.objective_funct(I_r,tr,x_new.I,x_new.t,2)
            end = timer()
            print("------------------")
            print(b_p,s_p,g_p,m_p)
            print(e_o,e_n)
            print("time: "+str(end-start))
            # Acceptance
            if(e_n/e_o<1):
                x=x_new
                e_o = e_n
                params.append([b_p,s_p,g_p,m_p,e_n])
                i+=1
                print("Step "+str(i))
            if(e_n<err):
                break
            k+=1
            if k>=50:
                print("Por aca no va")
                print("Solo %i pasos" % (i))
                break
            #sleep(0.01)
        # Dejo los params historicos por si hay que debuggear esta parte
        return params[-1]

    # objective function to minimize for any cases
    def objective_funct(self,Ir,tr,I,t,l):
        idx=np.searchsorted(t,tr)
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
