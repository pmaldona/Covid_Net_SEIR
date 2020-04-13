#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEIR Model param finder
Implementation of a Metropolis-Hasting model
"""
import class_SEIR
import numpy as np
import pandas
from numpy import linalg as LA
from pyswarm import pso



def mesh(model,Ir,tr,Npoints,steps,r0,err):
    params = []
    for i in range(Npoints):
        print(i)
        beta_i=np.random.uniform(min(model.beta_r),max(model.beta_r))
        sigma_i=np.random.uniform(min(model.sigma_r),max(model.sigma_r))
        gamma_i=np.random.uniform(min(model.gamma_r),max(model.gamma_r))
        mu_i=np.random.uniform(min(model.mu_r),max(model.mu_r))
        #print([beta_i,sigma_i,gamma_i,mu_i])
        params.append(pso_opt(model,Ir,tr,beta_i,sigma_i,gamma_i,mu_i))
    return params


def pso_opt(model,Ir,tr,omega=0.5, phip=0.5, phig=0.5,swarmsize=5,maxiter=25):
    np.random.seed()    
    def opti(x):
         din=model.integr_RK4(model.t0,model.T,model.h,x[0],x[1],x[2],x[3],False,True)
         return(objective_funct(Ir,tr,din["I"],din["t"],2))
         
    lb=[min(model.beta_r),min(model.sigma_r),min(model.gamma_r),min(model.mu_r)]
    ub=[max(model.beta_r),max(model.sigma_r),max(model.gamma_r),max(model.mu_r)]
    
    xopt, fopt = pso(opti, lb, ub, minfunc=1, omega=omega, phip=phip, phig=phig, debug=True,swarmsize=swarmsize,maxiter=maxiter)
    return(np.append(xopt,fopt))

# objective function to minimize for any cases
def objective_funct(Ir,tr,I,t,l):
    idx=np.searchsorted(t,tr)
    return LA.norm(Ir-I[:,idx],l)


#The tranistion model defines how to move from current to new parameters
def transition_model(beta,sigma,gamma,mu,r0,model):
    rb=r0*(max(model.beta_r)-min(model.beta_r))
    rg=r0*(max(model.gamma_r)-min(model.gamma_r))
    rs=r0*(max(model.sigma_r)-min(model.sigma_r))
    rm=r0*(max(model.mu_r)-min(model.mu_r))
    b_p = np.random.normal(beta,2*rb)
    s_p = np.random.normal(sigma,2*rs)
    g_p = np.random.normal(gamma,2*rg)
    m_p = np.random.normal(mu,2*rm)

    while b_p > max(model.beta_r) or b_p < min(model.beta_r):
        b_p = np.random.normal(beta,rb)
    while s_p > max(model.sigma_r) or s_p < min(model.sigma_r):
        s_p = np.random.normal(sigma,rs)
    while g_p > max(model.gamma_r) or g_p < min(model.gamma_r):
        g_p = np.random.normal(gamma,rg)
    while m_p > max(model.mu_r) or m_p < min(model.mu_r):
        m_p = np.random.normal(mu,rb)
    return [b_p,s_p,g_p,m_p]


#if __name__ == "__main__":
    # init model
    #model = SEIR(self,P,eta,alpha,S0,E0,I0,R0,Ir,tr,h,beta_r,gamma_r,sigma_r,mu_r)
    # Le sacaria vairables de inicializacion al modelo SeIR
    # no se usan beta, gamma, sigma, mu,y h
    # I_r se usa en todos los modelos, lo abstraeria un nivel
    #   def __init__(self,P,eta,alpha,S0,E0,I0,R0,Ir,tr,h,beta_r,gamma_r,sigma_r,mu_r):
#           n = 5 # Number of initial parameters
    # create a mesh of initial parameters


    # execute met_hast for each parameter

    # Output comparison
